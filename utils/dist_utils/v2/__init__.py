from .model import DiscreteTokenizer
from .structure_diffusion.model import StructurePredictionModel
from utils.lf_utils.constant import LF_TOKENIZER_CKPT_ROOT

version_discrete_tokenizer = str(LF_TOKENIZER_CKPT_ROOT / 'v2-stable-epoch=86-val_loss=0.0027.ckpt')
version_structure_head = str(LF_TOKENIZER_CKPT_ROOT / 'v2-structure-backbone-epoch=34-val_loss=2e-5.ckpt')