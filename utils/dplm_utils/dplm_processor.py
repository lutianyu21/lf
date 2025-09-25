import transformers
import numpy as np
from transformers import ProcessorMixin, PreTrainedTokenizerFast
from transformers.feature_extraction_utils import BatchFeature
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput
from utils.openfold_utils import OpenfoldProtein
from .tokeniers import DPLMTokenizer

from typing import Any, Dict, Optional, List, Tuple
import torch
import re



class DPLMProcessor(ProcessorMixin):
    
    attributes = ["tokenizer"]
    tokenizer_class = "PreTrainedTokenizerFast"

    def __init__(
        self,
        tokenizer: Any,
        structure_tokenizer: DPLMTokenizer,
        structure_template: Tuple[str, str] = ("<|struct{token_id:0>4d}|>", r"<\|struct(\d{4})\|>"),
    ):
        self.structure_tokenizer = structure_tokenizer
        self.structure_template = structure_template
        self.structure_vocab_size = 8192
        self.tokenizer = tokenizer
        super().__init__(tokenizer)
    
    @torch.no_grad()
    def __call__(
        self,
        protein_text: TextInput | PreTokenizedInput,
        protein_struct: OpenfoldProtein | List[OpenfoldProtein],
        train_task: str = 'folding',
        eval_task: str = 'folding',
        **kwargs,
    ) -> Tuple[BatchFeature, BatchFeature]:
        
        if isinstance(protein_text, str): protein_text = [protein_text]
        if isinstance(protein_struct, OpenfoldProtein): protein_struct = [protein_struct]
        protein_struct_text: List[str] = self.tokenize_structure(protein_struct, as_str=True) # type: ignore
            
        assert train_task in ['folding', 'seq-only', 'struct-only'] and \
                eval_task in ['folding', 'seq-only', 'struct-only']
        
        if train_task == 'folding':
            prompt_train: List[str] = list(map(
                lambda t, s: ''.join([
                    self.tokenizer.bos_token,
                    self.tokenizer.boseq_token, t, self.tokenizer.eoseq_token,
                    self.tokenizer.bostruct_token, s, self.tokenizer.eostruct_token,
                    self.tokenizer.eos_token
                ]),
                protein_text,
                protein_struct_text
            ))
        
        elif train_task == 'seq-only':
            prompt_train: List[str] = list(map(
                lambda t: ''.join([
                    self.tokenizer.bos_token,
                    self.tokenizer.boseq_token, t, self.tokenizer.eoseq_token,
                    self.tokenizer.eos_token
                ]),
                protein_text,
            ))
            
        elif train_task == 'struct-only':
            prompt_train: List[str] = list(map(
                lambda s: ''.join([
                    self.tokenizer.bos_token,
                    self.tokenizer.bostruct_token, s, self.tokenizer.eostruct_token,
                    self.tokenizer.eos_token
                ]),
                protein_struct_text
            ))
        
        # eval_templates
        if eval_task == 'folding':
            # for evaluation, prompt will be shorter than label
            prompt_eval: List[str] = list(map(
            lambda t: ''.join([
                self.tokenizer.bos_token,
                self.tokenizer.boseq_token,
                t,
                self.tokenizer.eoseq_token,
                self.tokenizer.bostruct_token,
            ]),
            protein_text
        ))
        else:
            raise NotImplementedError()
        
        return_type = kwargs.pop('return_tensors')
        constant = self.constant_helper()
        
        inputs_train = self.tokenizer(prompt_train, return_tensors='pt', **kwargs)        
        input_ids_train: torch.Tensor = inputs_train['input_ids']   # [B, L]
        seq_section_masks = (
            (input_ids_train == constant["boseq_token"]).cumsum(dim=1) - \
            (input_ids_train == constant["eoseq_token"]).cumsum(dim=1)
        ).bool()
        struct_section_masks = (
            (input_ids_train == constant["bostruct_token"]).cumsum(dim=1) - \
            (input_ids_train == constant["eostruct_token"]).cumsum(dim=1)
        ).bool()
        inputs_train.update({
            'lengths':              torch.tensor([len(t) for t in protein_text]),
            'seq_section_masks':    seq_section_masks,
            'struct_section_masks': struct_section_masks
        })
        
        inputs_eval = self.tokenizer(prompt_eval, return_tensors='pt', **kwargs)
        inputs_eval.update({
            'lengths': torch.tensor([len(t) for t in protein_text])  
        })
        
        if return_type != 'pt':
            raise NotImplementedError() # TODO
        
        return BatchFeature(inputs_train), BatchFeature(inputs_eval)

    def tokenize_structure(self, protein_structure: List[OpenfoldProtein], as_str: bool = False) -> List[str | torch.Tensor]:
        structure_tokens = []
        for ps in protein_structure:
            ps = ps.to(self.structure_tokenizer.device) 
            input = {
                'residue_atom37_coord': ps.residue_atom37_coord.unsqueeze(0),    # [B, L, 37]
                'residue_missing_mask': 1 - ps.residue_mask.unsqueeze(0),        # [B, L] wo/ loss
                'unpadded_length': torch.tensor([len(ps)], device=ps.device)     # [B]     w/ loss          
            }
            token_tensor = self.structure_tokenizer.batch_tokenize(**input).squeeze(0)
            token_str = "".join([
                self.structure_template[0].format(token_id=token_id)
                for token_id in token_tensor.cpu().tolist()
            ])
            structure_tokens.append(token_str if as_str else token_tensor)
        return structure_tokens
    
    @torch.no_grad()
    def decode(self, *args, **kwargs) -> Dict[str, Any]:
        doc = self.tokenizer.decode(*args, **kwargs)
        return self.multimodal_decode(doc, **kwargs)
    
    @torch.no_grad()
    def multimodal_decode(self, doc, **kwargs) -> Dict[str, Any]:
        seq_output, struct_output, entity_output, multimodal_output = [], [], [], []
        pattern = rf'({re.escape(self.tokenizer.bostruct_token)}.*?{re.escape(self.tokenizer.eostruct_token)})'
        chunks = re.split(pattern, doc)
        for i, c in enumerate(chunks):
            if len(c) == 0: continue
            if self.tokenizer.bostruct_token in c:
                # as structure
                dplm_decoder_input = {
                    'struct_tokens': torch.tensor(
                        [int(m) for m in re.findall(self.structure_template[1], c)], device=self.structure_tokenizer.device
                    ).unsqueeze(0),
                    'res_mask': None,
                }
                output = self.structure_tokenizer.batch_detokenize(**dplm_decoder_input)
                struct_output.append(c)
                entity_output.append(output)
                multimodal_output.append(output)
            else:
                # as text
                seq_output.append(c)
                multimodal_output.append(c)
        return {
            'multimodal':   multimodal_output,
            'seq':          seq_output,
            'struct':       struct_output,
            'entity':       entity_output
        }
    
    def constant_helper(self) -> Dict[str, int]:
        (
            pad_token,
            boseq_token,
            eoseq_token,
            bostruct_token,
            eostruct_token,
            bos_token,
            eos_token,
        ) = self.tokenizer.encode(''.join([
            self.tokenizer.pad_token,
            self.tokenizer.boseq_token,
            self.tokenizer.eoseq_token,
            self.tokenizer.bostruct_token,
            self.tokenizer.eostruct_token,
            self.tokenizer.bos_token,
            self.tokenizer.eos_token,
        ]))
        seq_tokens = self.tokenizer.encode("ABCDEFGHIKLMNOPQRSTUVWXYZ")
        struct_tokens = self.tokenizer.encode(''.join(
            [self.structure_template[0].format(token_id=i) for i in range(self.structure_vocab_size)])
        )
        
        return {
            'pad_token': pad_token,
            'boseq_token': boseq_token,
            'eoseq_token': eoseq_token,
            'bostruct_token': bostruct_token,
            'eostruct_token': eostruct_token,
            'bos_token': bos_token,
            'eos_token': eos_token,
            'seq_tokens': seq_tokens,
            'struct_tokens': struct_tokens
        }

    def compute_align(self, structure1: OpenfoldProtein, structure2: OpenfoldProtein) -> Tuple[float, float]:
        # HINT: structure1 is decoded thus requires metadata from structure2
        structure1.inherit(structure2)
        return structure1.align_with(structure2, chain_wise=True)
    
    def compute_acc(self, structure1: str, structure2: str) -> float:
        pred_token = self.tokenizer.encode(structure1, return_tensors='pt')
        true_token = self.tokenizer.encode(structure2, return_tensors='pt')
        return ((pred_token == true_token).sum() / pred_token.size(-1)).item() * 100
    