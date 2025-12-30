from .model import DiscreteTokenizer
from .structure_diffusion.model import StructurePredictionModel
from ...common import GlobalConstants

version_discrete_tokenizer =  f'{GlobalConstants.working_dir}/utils/dist_utils/checkpoints/stable-epoch=86-val_loss=0.0027.ckpt'
version_structure_head = f'{GlobalConstants.working_dir}/utils/dist_utils/checkpoints/v2-structure-backbone-epoch=34-val_loss=2e-5.ckpt'