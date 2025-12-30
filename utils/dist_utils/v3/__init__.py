import os
from .model import DiscreteTokenizer
from .structure_diffusion.model import StructurePredictionModel, REFERENCE_FRAME
from utils.lf_utils.constant import LF_TOKENIZER_CKPT_ROOT

# Tokenizer checkpoint (can be overridden via environment variable)
# Available checkpoints:
#   - v4-epoch=46-val_loss=0.1712.ckpt (latest)
#   - v4-epoch=00-val_loss=0.0928.ckpt
#   - v4-ar-epoch=00-val_loss=0.1949.ckpt (with AR head)
_default_tokenizer = 'v4-epoch=46-val_loss=0.1712.ckpt'
_default_structure = 'v3-structure-epoch=04-val_rmsd=0.3359.ckpt'

version_discrete_tokenizer = os.environ.get(
    'LF_TOKENIZER_CKPT',
    str(LF_TOKENIZER_CKPT_ROOT / _default_tokenizer)
)
version_structure_head = os.environ.get(
    'LF_STRUCTURE_CKPT',
    str(LF_TOKENIZER_CKPT_ROOT / _default_structure)
)
