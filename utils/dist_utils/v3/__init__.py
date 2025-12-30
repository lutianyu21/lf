from .model import DiscreteTokenizer
from .structure_diffusion.model import StructurePredictionModel, REFERENCE_FRAME
from ...common import GlobalConstants

version_discrete_tokenizer = f'{GlobalConstants.working_dir}/utils/dist_utils/checkpoints/v4-ar-epoch=00-val_loss=0.1949.ckpt'
# version_discrete_tokenizer = f'{GlobalConstants.working_dir}/utils/dist_utils/checkpoints/v4-epoch=00-val_loss=0.0928.ckpt'
version_structure_head = f'{GlobalConstants.working_dir}/utils/dist_utils/checkpoints/v3-structure-epoch=04-val_rmsd=0.3359.ckpt'