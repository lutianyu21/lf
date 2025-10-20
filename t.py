# test build_dataset_from_entry
from utils.lf_utils import protein_processor, build_dataset
from pathlib import Path
import ray

ray.init(log_to_driver=True)


# ---- example ----
build_dataset(
    csv_path=Path('/GenSIvePFS/users/lutianyu/lf/data/cameo2022.csv'),
    jsonl_path=Path('/GenSIvePFS/users/lutianyu/lf/data/cameo2022_dplm_io2.jsonl'),
    batch_size=50,
    num_workers=2,
)

# build_dataset(
#     csv_path=Path('/GenSIvePFS/users/lutianyu/lf/data/dataset2.csv'),
#     jsonl_path=Path('/GenSIvePFS/users/lutianyu/lf/data/dataset2_dplm_new.jsonl'),
#     batch_size=100,
#     num_workers=2
# )