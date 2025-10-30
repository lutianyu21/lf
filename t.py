# test build_dataset_from_entry
import os
from joblib import parallel_backend
from utils.lf_utils import protein_processor, step1_pickle, step2_parquet
from pathlib import Path
import ray
from ray.util.queue import Queue

from utils.lf_utils.dataset import step3_merge

ray.init(address=os.environ["RAY_ADDRESS"], log_to_driver=True)
# ray.init()

# HK version
step2_parquet(
    src_dir="/GenSIvePFS/users/lutianyu/data/AFDB/pickle/part-03",
    dst_dir="/GenSIvePFS/users/lutianyu/data/AFDB/parquet/part-03",
    tokenizer_name='dist',
    num_gpu_workers=8,
    batch_size=6000,
    chunk_size=12000,
)




# step3_merge(
#     src_dir="/GenSIvePFS/users/lutianyu/data/AFDB/parquet/part-02",
#     dst_dir="/GenSIvePFS/users/lutianyu/data/AFDB/parquet/part-02",
#     add_split='afdb',
# )



# step1_pickle(
#     src_dir="/GenSIvePFS/users/lutianyu/data/AFDB/part_01",
#     dst_dir="/GenSIvePFS/users/lutianyu/data/AFDB/pickle/part-01",
#     dataset_name="afdb",
#     clear=True, # for afdb only
#     max_concurrent=3000,
# )

# step2_parquet(
#     src_dir="/GenSIvePFS/users/lutianyu/lf/pytest/pickle/swissprot_cif_v4",
#     dst_dir="/GenSIvePFS/users/lutianyu/lf/pytest/parquet/swissprot_cif_v4",
#     tokenizer_name='dist',
#     num_gpu_workers=2,
#     batch_size=6000,
#     chunk_size=12000,
# )




