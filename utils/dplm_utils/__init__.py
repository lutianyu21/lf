import sys
from utils.lf_utils.constant import LF_ROOT

# Use LF_ROOT + relative paths
sys.path.append(str(LF_ROOT / "utils/dplm_utils/dplm/src"))
sys.path.append(str(LF_ROOT / "utils/dplm_utils/esm"))
sys.path.append(str(LF_ROOT / "utils/dplm_utils/openfold"))
from byprot.models.dplm2 import MultimodalDiffusionProteinLanguageModel as DPLM2
from byprot.models.structok.structok_lfq import VQModel
