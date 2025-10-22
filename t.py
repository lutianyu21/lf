# test build_dataset_from_entry
from utils.lf_utils import protein_processor, main_pickle
from pathlib import Path
import ray

ray.init(log_to_driver=True)

main_pickle(
    src_dir="/GenSIvePFS/users/lutianyu/lf/data/swissprot_cif_v4",
    dst_dir="/GenSIvePFS/users/lutianyu/lf/data/pickle/swissprot_cif_v4",
    max_concurrent=1000,
)


# ---- example ----
# build_dataset(
#     csv_path=Path('/GenSIvePFS/users/lutianyu/lf/data/cameo2022.csv'),
#     jsonl_path=Path('/GenSIvePFS/users/lutianyu/lf/data/cameo2022_dplm_io2.jsonl'),
#     batch_size=50,
#     num_workers=2,
# )

# build_dataset(
#     csv_path=Path('/GenSIvePFS/users/lutianyu/lf/data/dataset2.csv'),
#     jsonl_path=Path('/GenSIvePFS/users/lutianyu/lf/data/dataset2_dplm_new.jsonl'),
#     batch_size=100,
#     num_workers=2
# )