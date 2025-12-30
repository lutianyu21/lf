import sys
from ..common.global_constants import GlobalConstants
sys.path.append(f"{GlobalConstants.working_dir}/utils/dplm_utils/dplm/src")
sys.path.append(f"{GlobalConstants.working_dir}/utils/dplm_utils/esm")
sys.path.append(f"{GlobalConstants.working_dir}/utils/dplm_utils/openfold")
from byprot.models.dplm2 import MultimodalDiffusionProteinLanguageModel as DPLM2
from byprot.models.structok.structok_lfq import VQModel
