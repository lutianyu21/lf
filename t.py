# test build_dataset_from_entry
from utils.lf_utils import protein_processor, step1_pickle, step2_parquet
from pathlib import Path
import ray

ray.init(log_to_driver=True)


step1_pickle(
    src_dir="/GenSIvePFS/users/lutianyu/data/AFDB/part_00",
    dst_dir="/GenSIvePFS/users/lutianyu/data/AFDB/pickle/part-00",
    dataset_name="afdb",
    clear=True, # for afdb only
    max_concurrent=3000,
)

# step2_parquet(
#     src_dir="/GenSIvePFS/users/lutianyu/lf/data/pickle/cameo2022",
#     dst_dir="/GenSIvePFS/users/lutianyu/lf/data/parquet/cameo2022",
#     tokenizer_name='dist',
#     num_cpu_workers=10,
#     num_gpu_workers=2,
#     batch_size=1000,
#     part_size=10000,
# )


# HK version
# step2_parquet(
#     src_dir="/GenSIvePFS/users/lutianyu/data/AFDB/pickle/part-02",
#     dst_dir="/GenSIvePFS/users/lutianyu/data/AFDB/parquet/part-02",
#     tokenizer_name='dist',
#     num_cpu_workers=10,
#     num_gpu_workers=8,
#     batch_size=6000,
#     part_size=100000,
# )



