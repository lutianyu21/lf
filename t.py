# test build_dataset_from_entry
import os
from utils.lf_utils import protein_processor, step1_pickle, step2_parquet
from pathlib import Path
import ray
from ray.util.queue import Queue



# ray.init(address=os.environ.get("RAY_ADDRESS"), log_to_driver=True)
ray.init()

# HK version
# step2_parquet(
#     src_dir="/GenSIvePFS/users/lutianyu/data/AFDB/pickle/part-00",
#     dst_dir="/GenSIvePFS/users/lutianyu/data/AFDB/parquet/part-00",
#     tokenizer_name='dist',
#     num_gpu_workers=8,
#     batch_size=6000,
#     chunk_size=12000,
# )

# step2_parquet(
#     src_dir="/GenSIvePFS/users/lutianyu/data/AFDB/pickle/split_00",
#     dst_dir="/GenSIvePFS/users/lutianyu/data/AFDB/parquet/split_00",
#     tokenizer_name='dist',
#     num_cpu_workers=40,
#     num_gpu_workers=8,
#     batch_size=6000,
#     part_size=12000,
# )


# step3_merge(
#     src_dir="/GenSIvePFS/users/lutianyu/data/AFDB/parquet/part-02",
#     dst_dir="/GenSIvePFS/users/lutianyu/data/AFDB/parquet/part-02",
#     add_split='afdb',
# )


# step1_pickle(
#     src_dir="/GenSIvePFS/users/lutianyu/data/AFDB/part_00",
#     dst1_dir="/GenSIvePFS/users/lutianyu/data/AFDB/pickle/split_00",
#     dst2_dir="/GenSIvePFS/users/lutianyu/data/AFDB/pickle/split_01",
#     dataset_name="afdb",
#     clear=True, # for afdb only
#     max_concurrent=3000,
# )

step2_parquet(
    src_dir="/GenSIvePFS/users/lutianyu/lf/pytest/pickle/swissprot_cif_v4",
    dst_dir="/GenSIvePFS/users/lutianyu/lf/pytest/tmp/swissprot_cif_v4",
    tokenizer_name='dist',
    num_gpu_workers=2,
    batch_size=6000,
    part_size=12000,
)




