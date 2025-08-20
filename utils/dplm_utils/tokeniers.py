from pathlib import Path
from typing import Optional, Union
from omegaconf import OmegaConf
from pathlib import Path
from byprot.models.structok.structok_lfq import VQModel
import torch

from utils.openfold_utils.io import OpenfoldEntity

__all__ = ['DPLMTokenizer', 'dplm_tokenizer']

class DPLMTokenizer:
    
    def __init__(self, path: Path, eval_mode: bool = True):
        cfg = OmegaConf.load(path/'config.yaml')
        OmegaConf.resolve(cfg)
        tokenizer = VQModel(**cfg) # type: ignore
        pretrained_state_dict = torch.load(path/"dplm2_struct_tokenizer.ckpt", map_location="cpu",)
        missing_keys, unexpected_keys = tokenizer.load_state_dict(pretrained_state_dict, strict=True)
        tokenizer = tokenizer.requires_grad_(False)
        tokenizer = tokenizer.train(not eval_mode)
        self.tokenizer = tokenizer
    
    @torch.no_grad
    def batch_tokenize(self,
        residue_atom37_coord: torch.Tensor,     # [B, L, 37]
        residue_missing_mask: torch.Tensor,     # [B, L] wo/ loss
        unpadded_length: torch.Tensor           # [B]     w/ loss   
    ) -> torch.Tensor:                          # [B, L]
        # COMMENT: paddings share a random token, missings share a fixed token 0 (hard coded)
        return self.tokenizer.tokenize(residue_atom37_coord, 1 - residue_missing_mask, unpadded_length)
    
    @torch.no_grad
    def batch_detokenize(self,
        residue_structure_token: torch.Tensor,  # [B, L]
        residue_missing_mask: Optional[torch.Tensor], # [B, L]
    ) -> OpenfoldEntity:                          # [B, L, 37]
        output = self.tokenizer.detokenize(residue_structure_token, 1 - residue_missing_mask if residue_missing_mask is not None else None)
        return OpenfoldEntity.from_feature({
            'residue_atom37_coord': output['atom37_positions'].squeeze(0),
            'residue_atom37_mask': output['atom37_mask'].squeeze(0)
        })
    
    def to(self, device: Union[str, torch.device]):
        self.tokenizer.to(device)
        return self
    
    @property
    def device(self) -> torch.device:
        return next(self.tokenizer.parameters()).device

tokenizer_path = Path(__file__).parent/'checkpoints/struct_tokenizer'
torch.hub.set_dir(tokenizer_path)
dplm_tokenizer = DPLMTokenizer(tokenizer_path)
