from .model import DiscreteTokenizer
from .structure_diffusion.model import StructurePredictionModel, REFERENCE_FRAME
from utils.lf_utils.constant import LF_TOKENIZER_CKPT_ROOT

# version_discrete_tokenizer = str(LF_TOKENIZER_CKPT_ROOT / 'v4-ar-epoch=00-val_loss=0.1949.ckpt')
version_discrete_tokenizer = str(LF_TOKENIZER_CKPT_ROOT / 'v4-epoch=00-val_loss=0.0928.ckpt')
version_structure_head = str(LF_TOKENIZER_CKPT_ROOT / 'v3-structure-epoch=04-val_rmsd=0.3359.ckpt')
