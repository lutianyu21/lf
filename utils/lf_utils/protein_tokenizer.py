import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, cast
from byprot.tasks.lm import dplm
from gemmi import Op
from omegaconf import OmegaConf
import torch
from torch import nn

from ..openfold_utils.io import OpenfoldBackbone, OpenfoldProtein

__all__ = [
    'ProteinTokenizer',
    'DPLMProteinTokenizer', 'dplm_protein_tokenizer',
]


class ProteinTokenizer(nn.Module):
    """ protein-in(wo/ mask), protein-out(w/ mask) """
    
    def __init__(self):
        super().__init__()
        pass
    
    def __call__(self, batch_proteins: List[OpenfoldProtein]) -> Dict[str, torch.Tensor]:
        # support batch encode, thus padding_mask is returned
        raise NotImplementedError()

    def batch_decode(self, batch_token_ids: torch.Tensor, **kwargs) -> List[OpenfoldProtein]:
        # support batch decode, thus padding_mask is expected(in kwargs)
        raise NotImplementedError()

    def encode(self, protein: OpenfoldProtein) -> torch.Tensor:
        raise NotImplementedError()
    
    def decode(self, token_ids: torch.Tensor, **kwargs) -> OpenfoldProtein:
        raise NotImplementedError()
    
    @property
    def device(self) -> torch.device:
        raise NotImplementedError()
    
    @property
    def vsz(self) -> int:
        raise NotImplementedError()
    
    
class DPLMProteinTokenizer(ProteinTokenizer):
    """ Wrapper implementation for DPLM2 tokenizer. """    
    def __init__(
        self,
        config_path: Path,
        ckpt_path: Optional[Path] = None,
        eval_mode: bool = True
    ):
        from ..dplm_utils import VQModel as DPLMVQModel
        super().__init__()
        config = OmegaConf.load(config_path)
        OmegaConf.resolve(config)
        tokenizer = DPLMVQModel(**config) # type: ignore
        if ckpt_path is not None:
            pretrained_state_dict = torch.load(ckpt_path, map_location="cpu",)
            missing_keys, unexpected_keys = tokenizer.load_state_dict(pretrained_state_dict, strict=True)
            tokenizer = tokenizer.requires_grad_(False)
            tokenizer = tokenizer.train(not eval_mode)
        self.tokenizer = tokenizer
    
    def __call__(self, batch_proteins: List[OpenfoldProtein]) -> Dict[str, torch.Tensor]:
        # support batch encode, thus batch_padding_mask is returned
        collect_residue_atom37_coord = [p.residue_atom37_coord for p in batch_proteins]
        collect_residue_mask = [p.residue_mask for p in batch_proteins]
        
        # organized as padded batch, with corresponding padding_mask
        max_length: int = max([len(p) for p in batch_proteins])
        batch_residue_atom37_coord = torch.stack([
            torch.nn.functional.pad(
                p, (0, 0, 0, 0, 0, max_length - len(p)), value=0.0
            ) for p in collect_residue_atom37_coord
        ], dim=0) # [l, 37, 3]... pad > stack > [B, L, 37, 3]
        
        # residue_mask includes both padding and missing residues
        batch_residue_mask = torch.stack([
            torch.nn.functional.pad(
                p, (0, max_length - len(p)), value=0.0
            ) for p in collect_residue_mask
        ], dim=0) # [l,]... pad > stack > [B, L]
        
        # TODO [B] representation is enough to generate right padding mask
        batch_lengths = torch.tensor(
            [len(p) for p in batch_proteins],
            dtype=torch.long, device=batch_residue_mask.device
        ) # [B]
        batch_padding_mask = 1 - (
            torch.arange(batch_residue_mask.shape[1], device=batch_residue_mask.device)[None, :]
            < batch_lengths[:, None]
        ).to(batch_residue_mask.dtype) # [B, L]
        
        # core implementation
        output = self.tokenizer.tokenize(
            atom_positions=batch_residue_atom37_coord,  # [B, L, 37, 3]
            res_mask=batch_residue_mask,                # [B, L]
            seq_length=batch_lengths                    # [B]
        )
        
        # convert to left-padding
        return {
            'batch_token_ids': output,                  # [B, L]
            'batch_padding_mask': batch_padding_mask,   # [B, L]
            'batch_residue_mask': batch_residue_mask    # [B, L]
        }
        
    def batch_decode(self, batch_token_ids: torch.Tensor, **kwargs) -> List[OpenfoldProtein]:
        # support batch decode, thus padding_mask is expected(in kwargs)
        if batch_token_ids.size(0) > 1:
            assert 'batch_lengths' in kwargs, "Expect 'batch_lengths' in kwargs for batch decode."
        batch_lengths = kwargs.get(
            'batch_lengths',
            torch.tensor([batch_token_ids.size(1)], device=batch_token_ids.device)
        )
        
        if 'batch_residue_mask' not in kwargs:
            warnings.warn("Expect 'batch_residue_mask' in kwargs for batch decode. Assuming no missing residues.")
        
        # NOTE while decoding, we ignore paddings by batch_residue_mask
        # and each sequence is decoded to a backbone structure(w/ missing resiudes)
        output = self.tokenizer.detokenize(
            struct_tokens=batch_token_ids,              # [B, L]
            res_mask=kwargs.get('batch_residue_mask')   # [B, L] or None
        )
        
        batch_proteins = []
        for i, l in enumerate(batch_lengths):
            residue_atom37_coord = output['atom37_positions'][i, :l, :] # [l, 37, 3]
            residue_atom37_mask = output['atom37_mask'][i, :l, :]       # [l, 37]
            if kwargs.get('batch_residue_mask') is not None: # inherit missing residues
                residue_atom37_mask *= kwargs['batch_residue_mask'][i, :l].unsqueeze(1)  # [l, 1]
            backbone = OpenfoldBackbone.from_dict(dict(
                residue_atom37_coord=residue_atom37_coord,
                residue_atom37_mask=residue_atom37_mask
            ))
            protein = OpenfoldProtein.from_backbone(backbone)
            batch_proteins.append(protein)
        return batch_proteins
    
    def encode(self, protein: OpenfoldProtein) -> torch.Tensor:
        # single encode, thus no padding_mask is returned
        output = self.__call__([protein])
        return output['batch_token_ids'].squeeze(0) # [L]
    
    def decode(self, token_ids: torch.Tensor, **kwargs) -> OpenfoldProtein:
        # batch_lengths is not necessary for single decode
        if 'residue_mask' not in kwargs:
            assert 'ref' in kwargs
            residue_mask = self([kwargs.pop('ref')])['batch_residue_mask'][0]
            kwargs['residue_mask'] = residue_mask
            # this 'leakage' function is to conveniently evaluate structure prediction
        
        residue_mask = kwargs.pop('residue_mask', None)
        return self.batch_decode(
            batch_token_ids=token_ids.unsqueeze(0),
            batch_residue_mask=residue_mask.unsqueeze(0) if residue_mask is not None else None
        )[0]
        
    @property
    def device(self) -> torch.device:
        return next(self.tokenizer.parameters()).device
    
    @property
    def vsz(self) -> int:
        raise NotImplementedError()

dplm_protein_tokenizer_path = Path(__file__).parent.parent/'dplm_utils/checkpoints/struct_tokenizer'
torch.hub.set_dir(dplm_protein_tokenizer_path)
dplm_protein_tokenizer = DPLMProteinTokenizer(
    config_path=dplm_protein_tokenizer_path/'config.yaml',
    ckpt_path=dplm_protein_tokenizer_path/'dplm2_struct_tokenizer.ckpt',
    eval_mode=True
)
