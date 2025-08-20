import transformers
from transformers import ProcessorMixin, PreTrainedTokenizerFast
from transformers.feature_extraction_utils import BatchFeature
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput

from utils.progen2_utils import progen2_merged_tokenizer
from utils.openfold_utils import OpenfoldEntity
from utils.protenix_utils import rmsd_globally_aligned
from .tokeniers import DPLMTokenizer

from typing import Any, Optional, List, Tuple
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
        **kwargs,
    ):
        self.structure_tokenizer = structure_tokenizer
        self.structure_template = structure_template
        self.structure_vocab_size = 8192
        self.tokenizer = tokenizer
        super().__init__(tokenizer)
    
    @torch.no_grad()
    def __call__(
        self,
        text: Optional[TextInput | PreTokenizedInput] = None,
        protein_text: Optional[TextInput | PreTokenizedInput] = None,
        protein_structure: Optional[OpenfoldEntity | List[OpenfoldEntity]] = None,
        task: str = 'default',
        **kwargs,
    ) -> Tuple[BatchFeature, BatchFeature]:
        
        if isinstance(protein_text, str): protein_text = [protein_text]
        if isinstance(protein_structure, OpenfoldEntity): protein_structure = [protein_structure]
        
        bsz = len(protein_text) if protein_text is not None else \
            (len(protein_structure) if protein_structure is not None else 0)
        
        if protein_structure is not None:
            protein_structure_text: List[str] = self.tokenize_structure(protein_structure, as_str=True)
        
        prompt_list = []
        prompt_generation_list = []
        for idx in range(bsz):
            prompt_protein_text = protein_text[idx] if protein_text is not None else ''
            prompt_protein_structure = protein_structure_text[idx] if protein_structure is not None else ''
            if task == 'default' or task == 'folding':
                prompt = self.tokenizer.bos_token + \
                        self.tokenizer.boseq_token + prompt_protein_text + self.tokenizer.eoseq_token + \
                        self.tokenizer.bostruct_token + prompt_protein_structure + self.tokenizer.eostruct_token + \
                        self.tokenizer.eos_token
                prompt_generation = self.tokenizer.bos_token + \
                        self.tokenizer.boseq_token + prompt_protein_text + self.tokenizer.eoseq_token + \
                        self.tokenizer.bostruct_token
            else:
                raise NotImplementedError()
            prompt_list.append(prompt)
            prompt_generation_list.append(prompt_generation)
        
        # HINT: padding, return tensor, return length, etc
        prompt_inputs = self.tokenizer(prompt_list, **kwargs)
        prompt_generation_inputs = self.tokenizer(prompt_generation_list, **kwargs)
        return BatchFeature(prompt_inputs), BatchFeature(prompt_generation_inputs)
        
    def tokenize_structure(self, protein_structure: List[OpenfoldEntity], as_str: bool = False) -> List[Any]:
        structure_tokens = []
        for ps in protein_structure:
            ps = ps.to(self.structure_tokenizer.device) 
            input = {
                'residue_atom37_coord': ps.feature['residue_atom37_coord'].unsqueeze(0),    # [B, L, 37]
                'residue_missing_mask': 1 - ps.feature['residue_mask'].unsqueeze(0),        # [B, L] wo/ loss
                'unpadded_length': torch.tensor([len(ps)], device=ps.device)                # [B]     w/ loss          
            }
            token_tensor = self.structure_tokenizer.batch_tokenize(**input).squeeze(0)
            token_str = "".join([
                self.structure_template[0].format(token_id=token_id)
                for token_id in token_tensor.cpu().tolist()
            ])
            structure_tokens.append(token_str if as_str else token_tensor)
        return structure_tokens
    
    @torch.no_grad()
    def decode(self, *args, **kwargs):
        doc = self.tokenizer.decode(*args, **kwargs)
        return self.multimodal_decode(doc, **kwargs)
    
    @torch.no_grad()
    def multimodal_decode(self, doc, **kwargs) -> Tuple[List[str], List[str], List[Tuple[str, OpenfoldEntity]]]:
        text_output, structure_output, multimodal_output = [], [], []
        pattern = rf'({re.escape(self.tokenizer.bostruct_token)}.*?{re.escape(self.tokenizer.eostruct_token)})'
        chunks = re.split(pattern, doc)
        for c in chunks:
            if len(c) == 0: continue
            if self.tokenizer.bostruct_token in c:
                # as structure
                input = {
                    'residue_structure_token': torch.tensor(
                        [int(m) for m in re.findall(self.structure_template[1], c)], device=self.structure_tokenizer.device
                    ).unsqueeze(0),
                    'residue_missing_mask': kwargs.get('residue_missing_mask')
                }
                output = self.structure_tokenizer.batch_detokenize(**input)
                structure_output.append((c, output))
                multimodal_output.append(output)
            else:
                # as text
                text_output.append(c)
                multimodal_output.append(c)
        return multimodal_output, text_output, structure_output
    
    def constant_helper(self):
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

    def compute_rmsd(self, structure1: OpenfoldEntity, structure2: OpenfoldEntity) -> float:
        pred_pose = structure1.backbone_coord
        true_pose = structure2.backbone_coord
        return rmsd_globally_aligned(
            pred_pose, true_pose, pred_pose.new_ones(pred_pose.shape[:-1], dtype=torch.bool)
        )[0].item()
        
    def compute_acc(self, structure1: str, structure2: str) -> float:
        pred_token = self.tokenizer.encode(structure1, return_tensors='pt')[:, 1:-1]
        true_token = self.tokenizer.encode(structure2, return_tensors='pt')[:, 1:-1]
        return ((pred_token == true_token).sum() / pred_token.size(-1)) * 100
        
        