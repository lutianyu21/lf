import torch.nn.functional as F
import torch
import math
from pathlib import Path
from typing import Optional, Union, Dict
from omegaconf import OmegaConf

from utils.dist_utils.vit_fsq import DiscreteTokenizer as Body
from utils.dist_utils.vit_structure_module import StructurePredictionModel as Head
from utils.openfold_utils.io import OpenfoldProtein, OpenfoldBackbone

__all__ = ['DistTokenizer', 'dist_tokenizer']


class DistTokenizer:
    
    def __init__(self, path: Path, eval_mode: bool = True):
        # TODO @zhanzghe, maybe merge decoder
        cfg = OmegaConf.load(path/'config.yaml')
        OmegaConf.resolve(cfg)
        
        # enc + q + dec
        body = Body(**cfg.body)
        state_dict_old: Dict[str, torch.Tensor] = torch.load(path/"body.ckpt", map_location="cpu",) # strip prefix 'model.'
        state_dict_new = {k.replace('model.', ''): v for k, v in state_dict_old.items()}
        missing_keys, unexpected_keys = body.load_state_dict(state_dict_new, strict=False)
        body = body.requires_grad_(False)
        body = body.train(not eval_mode)
        self.body = body
        
        # structure module
        head = Head(**cfg.head)
        state_dict_old = torch.load(path/"head.ckpt",  map_location="cpu",) # strip prefix 'model.'
        state_dict_new = {k.replace('model.', ''): v for k, v in state_dict_old.items()}
        missing_keys, unexpected_keys = head.load_state_dict(state_dict_new, strict=False)
        head = head.requires_grad_(False)
        head = head.train(not eval_mode)
        self.head = head
        
        self.scale = cfg.data.std
    
    @torch.no_grad()
    def batch_tokenize(self):
        raise NotImplementedError()
    
    @torch.no_grad()
    def batch_detokenize(self):
        raise NotImplementedError()
    
    # @zhangzhe, unified as OpenfoldProtein as input?
    @torch.no_grad()
    def tokenize(self, distance: torch.Tensor) -> torch.Tensor:
        distance = self.patchify(distance, 16) / self.scale
        assert distance.dim() == 3, 'Distance should be [L, L, 3]. Do you want to call batch_tokenize()?'
        batched_distance = distance.unsqueeze(0)
        quantized, vocab_ids, loss = self.body.encode(batched_distance)
        return vocab_ids.squeeze(0)

    @torch.no_grad()
    def detokenize(self, vocab_ids: torch.Tensor, **kwargs) -> OpenfoldProtein:
        assert vocab_ids.dim() == 1, f'Tokens should be [L] rather than {vocab_ids.shape}. Do you want to call batch_detokenize()?'
        batched_tokens = vocab_ids.unsqueeze(0)
        batched_codes = self.body.quantizer.indices_to_codes(batched_tokens) # flattened [B, HxW, z_channel]
        B, HxW, Z = batched_codes.shape
        H = W = int(math.sqrt(HxW))
        batched_codes = batched_codes.reshape(B, H, W, Z).permute(0, 3, 1, 2)
        batched_reconstructed = self.body.decode(batched_codes)         # [B, H, W, C]
        reconstructed = batched_reconstructed.squeeze(0)
        
        # Important: the output of body is padded, we need to remove the padding
        raw_length = kwargs.get('raw_length', None)
        if raw_length is not None:
            reconstructed = reconstructed[:raw_length, :raw_length, :]
        
        calpha: torch.Tensor = self.head(reconstructed) & self.scale    # [L, 3]
        # adapt to OpenFoldProtein input

        residue_atom37_coord = torch.zeros((calpha.shape[0], 37, 3), device=calpha.device)
        residue_atom37_coord[:, 1, :] = calpha
        residue_atom37_mask = torch.zeros((calpha.shape[0], 37), device=calpha.device)
        residue_atom37_mask[:, 1] = 1.0
        struct = OpenfoldProtein.from_backbone(OpenfoldBackbone.from_dict({
            'residue_atom37_coord': residue_atom37_coord,
            'residue_atom37_mask':  residue_atom37_mask
        }))
        return struct
    
    def patchify(self, tensor: torch.Tensor, patch_size: int) -> torch.Tensor:
        L, _, C = tensor.shape
        padded_L = math.ceil(L / patch_size) * patch_size
        padding_needed = padded_L - L
        if padding_needed > 0:
            # Pad the right and bottom edges
            padding = (0, 0, 0, padding_needed, 0, padding_needed)  # (left, right, top, bottom, front, back)
            tensor = F.pad(tensor, padding, mode='constant', value=0.0)
        return tensor
    
    def to(self, device: Union[str, torch.device]):
        self.body.to(device)
        self.head.to(device)
        return self
    
    @property
    def device(self) -> torch.device:
        return next(self.body.parameters()).device
    
    def get_vocab_size(self) -> int:
        return self.body.quantizer.n_embed
    
# provide instance
tokenizer_path = Path(__file__).parent/'checkpoints'
dist_tokenizer = DistTokenizer(tokenizer_path)