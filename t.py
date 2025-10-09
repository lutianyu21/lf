# test build_dataset_from_entry
from utils.lf_utils import protein_processor, dplm_protein_tokenizer, build_dataset
from pathlib import Path
import ray

ray.init()

# ---- example ----
build_dataset(
    csv_path=Path('/AIRvePFS/ai4science/users/tianyu/lf/data/cameo2022.csv'),
    jsonl_path=Path('/AIRvePFS/ai4science/users/tianyu/lf/data/cameo2022_dplm.jsonl'),
    batch_size=10,
    num_workers=1,
)

# build_dataset(
#     csv_path=Path('/AIRvePFS/ai4science/users/tianyu/lf/trash/test.csv'),
#     jsonl_path=Path('/AIRvePFS/ai4science/users/tianyu/lf/trash/test.jsonl'),
#     batch_size=80,
# )