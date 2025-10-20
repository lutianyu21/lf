
import chunk
from sys import version
import time
from typing import Any, Dict, Optional, List, Tuple

import re
from pathlib import Path

import numpy as np
import torch

import transformers
from transformers import ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature

from .text_tokenizer import TextTokenizer
from .protein_tokenizer import ProteinTokenizer
from ..openfold_utils.io import OpenfoldProtein



__all__ = ['ProteinProcessor']

class ProteinProcessor(ProcessorMixin):
    """ Organize components. """
    tokenizer: TextTokenizer
    struct_tokenizer: ProteinTokenizer
    struct_vsz: int
    struct_regex: str
    struct_template: str
    constant: Dict[str, int | List[int] | Any]
    
    attributes = ["tokenizer"]
    tokenizer_class = "PreTrainedTokenizerFast"
    
    def __init__(
        self,
        tokenizer: TextTokenizer,
        struct_tokenizer: ProteinTokenizer,
    ):
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
        # HINT tokenizer's vsz could be larger than actual vsz
        self.struct_tokenizer = struct_tokenizer
        self.struct_vsz = struct_tokenizer.vsz
        self.struct_regex = tokenizer.struct_regex
        self.struct_template = tokenizer.struct_template
        self.constant = self.constant_helper()

    @torch.no_grad()
    def __call__(
        self,
        seq_input: str | List[str],
        struct_input: OpenfoldProtein | List[OpenfoldProtein],
        **kwargs,
    ) -> Tuple[BatchFeature, BatchFeature]:
        
        if isinstance(struct_input, OpenfoldProtein): struct_input = [struct_input]
        if isinstance(seq_input, str): seq_input = [seq_input]
        struct_input = [s.to(self.struct_tokenizer.device) for s in struct_input]

        right_out = self.struct_tokenizer(struct_input)
        
        # convert to structure text
        seq_text: List[str] = seq_input
        struct_text: List[str] = []
        batch_token_ids: torch.Tensor = right_out['batch_token_ids']           # [B, L_seq]
        batch_padding_mask: torch.Tensor = (batch_token_ids == -100).long()    # [B, L_seq]
        for token_ids, padding_mask in zip(batch_token_ids, batch_padding_mask):
            token_ids = token_ids[~padding_mask.bool()]
            struct_text.append("".join([self.struct_template.format(token_id=i) for i in token_ids]))
        
        train_folding = lambda t, s: self.tokenizer.bos_token + self.tokenizer.boseq_token + t + self.tokenizer.eoseq_token \
                    + self.tokenizer.bostruct_token + s + self.tokenizer.eostruct_token + self.tokenizer.eos_token
        eval_folding = lambda t, s: self.tokenizer.bos_token + self.tokenizer.boseq_token + t + self.tokenizer.eoseq_token \
                    + self.tokenizer.bostruct_token

        train_inputs = self.tokenizer(list(map(train_folding, seq_text, struct_text)), **kwargs)
        eval_inputs = self.tokenizer(list(map(eval_folding, seq_text, struct_text)), **kwargs)
        train_inputs.pop('token_type_ids', None)    # not used
        eval_inputs.pop('token_type_ids', None)     # not used
        
        # copy k,v other than batch_token_ids, batch_padding_mask
        # these keys will be passed to mutimodal_decode()
        for k, v in right_out.items():
            if k not in ['batch_token_ids']:
                train_inputs[k] = v                 # [B, L_seq]
                eval_inputs[k] = v                  # [B, L_seq]
        
        if kwargs.get('return_tensors') != 'pt':
            raise NotImplementedError('Only support pt tensors') # TODO

        return BatchFeature(train_inputs), BatchFeature(eval_inputs) # type: ignore

    @torch.no_grad()
    def multimodal_decode(self, token_ids: torch.Tensor, **kwargs) -> Dict[str, Any]:
        
        # HINTL token_ids is left-padded, and kwargs might be right padded depending on tokenizer call
        # specifying additional kwargs for structure decoding e.g. residue_mask
        
        string = self.tokenizer.decode(token_ids)
        pattern = rf'({re.escape(self.tokenizer.bostruct_token)}.*?{re.escape(self.tokenizer.eostruct_token)})'
        chunks = re.split(pattern, string)
        seq_output, struct_output, entity_output = [], [], []
        
        for i, c in enumerate(chunks):
            if len(c) == 0: continue
            if self.tokenizer.bostruct_token in c:
                # as structure
                protein = self.struct_tokenizer.decode(
                    token_ids=torch.tensor(
                        [int(i) for i in re.findall(self.struct_regex, c)], device=self.device
                    ),
                    **kwargs
                )
                struct_output.append(c)
                entity_output.append(protein)
            else:
                # as text
                seq_output.append(c)
        return {
            'text':         string,
            'seq':          seq_output,
            'struct':       struct_output,
            'entity':       entity_output
        }
    
    def constant_helper(self) -> Dict[str, int | List[int] | Any]:
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
        
        seq_vocab_ids = self.tokenizer.seq_vocab_ids
        struct_vocab_ids = self.tokenizer.struct_vocab_ids
        return {
            'pad_token': pad_token,
            'boseq_token': boseq_token,
            'eoseq_token': eoseq_token,
            'bostruct_token': bostruct_token,
            'eostruct_token': eostruct_token,
            'bos_token': bos_token,
            'eos_token': eos_token,
            'seq_vocab_ids': seq_vocab_ids,
            'struct_vocab_ids': struct_vocab_ids
        }

    @staticmethod
    def compute_tm_align(structure1: OpenfoldProtein, structure2: OpenfoldProtein, ref: OpenfoldProtein | None) -> Tuple[float, float]:
        if ref is not None:
            structure1.inherit(ref)
            structure2.inherit(ref)
        return structure1.align_with(structure2, chain_wise=True)
    
    @staticmethod
    def compute_kbastch_align(structure1: OpenfoldProtein, structure2: OpenfoldProtein) -> Tuple[float, float]:
        raise NotImplementedError()
    
    
    def build_dataset(
        self,
        src: str,
        batch: List[dict],
        verbose: bool = True
    ) -> List[dict]:
        return []
        
        # TODO fix this
        
        proteins: List[OpenfoldProtein] = []
        results: List[dict] = []
        
            
        if src == 'af_tax':
            # In this case, multiple AF entries are stored in a tar file for each tax_id
            # e.g. for proteome-tax_id-1974607-0_v4.tar, we have
            # - AF-A0A2H0UIM4-F1-model_v4.cif.gz
            # - AF-A0A2H0UIM4-F1-confidence_v4.json.gz
            # - AF-A0A2H0UIM4-F1-predicted_aligned_error_v4.json.gz
            # so we need to extract each entry from the .tar file first
            # create a temporary directory to extract the files
            # then process each file and finally remove them
            import tarfile
            import tempfile
            
            
            # connect proteins into chunks to avoid OOM
            
            
            for row in batch:
                print('processing tax_id:', row['tax_id'])
                tax_id = row['tax_id']
                tar_path = Path(f"/GenSIvePFS/users/lutianyu/lf/data/afdb_proteome_cif_v4/proteome-tax_id-{tax_id}-0_v4.tar")
                with tarfile.open(tar_path, 'r') as tar:
                    for member in tar.getmembers():
                        if member.name.endswith('-model_v4.cif.gz'):
                            # a temporary file to store the extracted cif file
                            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                                tmp.write(tar.extractfile(member).read()) # type: ignore
                                f = tmp.name

                            if f is not None:
                                p = OpenfoldProtein.from_file(f, verbose=verbose)
                                # TODO remove constraint
                                if p.entry != 'empty':
                                    proteins += p.split()
        
        else:
            raise NotImplementedError()
        
        
        
        # optional1: given entry + split
        # if 'protein_path' not in batch[0].keys():
        #     tmpl = {
        #         "afdb_swissprot":   "/GenSIvePFS/users/lutianyu/lf/data/swissprot_cif_v4/{x}.cif.gz",
        #         "pdb":              "/GenSIvePFS/users/lutianyu/lf/data/rcsb_mmcif/{x}.cif",
        #         "cameo2022":       "/GenSIvePFS/users/lutianyu/lf/data/rcsb_mmcif/{x}.cif",
        #     }
        #     for row in batch:
        #         protein_path = Path(tmpl[row["split"]].format(x=row["pdb_name"]))
        #         p = OpenfoldProtein.from_file(protein_path, verbose=verbose)
        #         # TODO remove constraint
        #         if p.entry != 'empty':
        #             proteins += p.split()
        
        # # optional2: given protein_path
        # else:
        #     for row in batch:
        #         protein_path = Path(row["protein_path"])
        #         p = OpenfoldProtein.from_file(protein_path, verbose=verbose)
        #         # TODO remove constraint
        #         if p.entry != 'empty':
        #             proteins += p.split()
        
        if len(proteins) == 0: return results
        
        # batch tokenizeation (GPU)
        proteins = [p.to(self.struct_tokenizer.device) for p in proteins]
        out = self.struct_tokenizer(proteins)
        batch_token_ids = out['batch_token_ids'] # [B, L]
        batch_padding_mask = (batch_token_ids == -100).long()
        
        for protein, token_ids, padding_mask in zip(proteins, batch_token_ids, batch_padding_mask):
            seq_text = str(protein)
            seq_length = len(protein)
            token_ids = token_ids[~padding_mask.bool()]
            struct_text = "".join([self.struct_template.format(token_id=i) for i in token_ids])
            struct_length = len(token_ids)
            text = f"<|bos|><|boseq|>{seq_text}<|eoseq|><|bostruct|>{struct_text}<|eostruct|><|eos|>"
            prompt = f"<|bos|><|boseq|>{seq_text}<|eoseq|><|bostruct|>"  
            results.append({
                "pdb_name": protein.entry,
                "text": text,
                "prompt": prompt,
                "seq_length": seq_length,
                "struct_length": struct_length,
                "total_length": seq_length + struct_length + 6,
                "seq_text": seq_text,
                "struct_text": struct_text,
            })
        return results
    
    
    
    
    
    def _build_dataset_from_tax(
        self,
        batch: List[dict],
        verbose: bool = True,
        oom: int = 500
    ) -> List[dict]:
        # In this case, multiple AF entries are stored in a tar file for each tax_id
        # e.g. for proteome-tax_id-1974607-0_v4.tar, we have
        # - AF-A0A2H0UIM4-F1-model_v4.cif.gz
        # - AF-A0A2H0UIM4-F1-confidence_v4.json.gz
        # - AF-A0A2H0UIM4-F1-predicted_aligned_error_v4.json.gz
        # so we need to extract each entry from the .tar file first
        # create a temporary directory to extract the files
        # then process each file and finally remove them
        
        def submit2tokenizer(proteins: List[OpenfoldProtein]) -> List[dict]:
            if len(proteins) == 0:
                return []
            proteins = [p.to(self.struct_tokenizer.device) for p in proteins]
            out = self.struct_tokenizer(proteins)
            batch_token_ids = out['batch_token_ids'] # [B, L]
            batch_padding_mask = (batch_token_ids == -100).long()
            local_results: List[dict] = []
            for protein, token_ids, padding_mask in zip(proteins, batch_token_ids, batch_padding_mask):
                seq_text = str(protein)
                seq_length = len(protein)
                token_ids = token_ids[~padding_mask.bool()]
                struct_text = "".join([self.struct_template.format(token_id=i) for i in token_ids])
                struct_length = len(token_ids)
                text = f"<|bos|><|boseq|>{seq_text}<|eoseq|><|bostruct|>{struct_text}<|eostruct|><|eos|>"
                prompt = f"<|bos|><|boseq|>{seq_text}<|eoseq|><|bostruct|>"  
                local_results.append({
                    "pdb_name":     protein.entry,
                    "text":         text,
                    "seq_text":     seq_text,
                    "struct_text":  struct_text,
                    "prompt":       prompt,
                    "seq_length":   seq_length,
                    "struct_length": struct_length,
                    "total_length": seq_length + struct_length + 6,
                })
            return local_results
        
        import tarfile
        import tempfile
        # we strictly limit the num_proteins for tokenizer.forward() to avoid OOM
        proteins: List[OpenfoldProtein] = []
        results: List[dict] = []
        proteins_buffer: List[OpenfoldProtein] = []
        for row in batch:
            tax_id = row['tax_id']
            tar_path = Path(f"/GenSIvePFS/users/lutianyu/lf/data/afdb_proteome_cif_v4/proteome-tax_id-{tax_id}-0_v4.tar")
            with tarfile.open(tar_path, 'r') as tar:
                for member in tar:
                    if member.name.endswith('-model_v4.cif.gz'):
                        # a temporary *.cif.gz
                        # removed after processing
                        with tempfile.NamedTemporaryFile(delete=True) as tmp:
                            tmp.write(tar.extractfile(member).read()) # type: ignore
                            f = tmp.name
                            p = OpenfoldProtein.from_file(f, verbose=verbose)
                            if p.empty():
                                if verbose: raise RuntimeError(f"Failed to load protein from {f}")
                                continue
                            p_split = p.split()
                            if len(proteins_buffer) + len(p_split) > oom:
                                # immediately submit to tokenizer
                                results += submit2tokenizer(proteins_buffer)
                                proteins_buffer.clear()
                            proteins_buffer += p_split
                # almost end processing one tax.tar
        # almost end processing one batch
        if len(proteins_buffer) > 0:
            results += submit2tokenizer(proteins_buffer)
            proteins_buffer.clear()
        # end processing batch
        return results

    def to(self, device: str | torch.device):
        self.struct_tokenizer.to(device)
        return self
    
    @property
    def device(self) -> torch.device:
        return self.struct_tokenizer.device

