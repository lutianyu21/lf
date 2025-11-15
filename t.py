# test build_dataset_from_entry
from math import pi
import os
import pipe
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


def pipe_cameo2022():
    step1_pickle(
        dataset_name="cameo",
        src_dir="/GenSIvePFS/users/lutianyu/lf/data/raw/cameo2022",
        dst1_dir="/GenSIvePFS/users/lutianyu/lf/data/pickle/cameo2022",
        clear=False, # for afdb only
        max_concurrent=3000,
    )
    step2_parquet(
        dataset_name="cameo2022",
        src_dir="/GenSIvePFS/users/lutianyu/lf/data/pickle/cameo2022",
        dst_dir="/GenSIvePFS/users/lutianyu/lf/data/parquet/cameo2022",
        tokenizer_name='dist',
        num_cpu_workers=10,
        num_gpu_workers=2,
        batch_size=5000,
        part_size=100000,
    )
    
    
def pipe_rcsb_monomer():
    # step1_pickle(
    #     dataset_name="cameo",
    #     src_dir="/GenSIvePFS/users/lutianyu/lf/data/pickle/rcsb/chain",
    #     dst1_dir="/GenSIvePFS/users/lutianyu/lf/data/pickle/rcsb/chain",
    #     clear=False, # for afdb only
    #     max_concurrent=3000,
    # )
    step2_parquet(
        dataset_name="rcsb_monomer",
        src_dir="/GenSIvePFS/users/lutianyu/lf/data/pickle/rcsb/chain",
        dst_dir="/GenSIvePFS/users/lutianyu/lf/data/parquet/rcsb/chain",
        tokenizer_name='dist',
        num_cpu_workers=10,
        num_gpu_workers=2,
        batch_size=5000,
        part_size=100000,
    )
    
    

if __name__ == "__main__":
    # pipe_cameo2022()
    pipe_rcsb_monomer()



# step2_parquet(
#     src_dir="/GenSIvePFS/users/lutianyu/data/AFDB/pickle/split_00",
#     dst_dir="/GenSIvePFS/users/lutianyu/data/AFDB/parquet/split_00",
#     tokenizer_name='dist',
#     num_cpu_workers=40,
#     num_gpu_workers=8,
#     batch_size=5000,
#     part_size=10000,
# )


# step3_merge(
#     src_dir="/GenSIvePFS/users/lutianyu/data/AFDB/parquet/part-02",
#     dst_dir="/GenSIvePFS/users/lutianyu/data/AFDB/parquet/part-02",
#     add_split='afdb',
# )




# step2_parquet(
#     src_dir="/GenSIvePFS/users/lutianyu/lf/pytest/pickle/swissprot_cif_v4",
#     dst_dir="/GenSIvePFS/users/lutianyu/lf/pytest/tmp/swissprot_cif_v4",
#     tokenizer_name='dist',
#     num_gpu_workers=2,
#     batch_size=6000,
#     part_size=12000,
# )




