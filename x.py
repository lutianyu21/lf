import ray
ray.init(ignore_reinit_error=True)

from utils.lf_utils.data_engine import DataEngine
from pathlib import Path



# DataEngine().query_rcsb(
#     output_dir=Path("/GenSIvePFS/users/lutianyu/lf/data/unicluster_40"),
#     query_path=Path("/GenSIvePFS/users/lutianyu/lf/data/unicluster_40/clusters-by-entity-40.txt"),
#     max_concurrent=5000,
# )


# DataEngine().query_afdb(
#     output_dir=Path("/GenSIvePFS/users/lutianyu/lf/afdb"),
#     max_concurrent=2000,
#     query_path=Path("/GenSIvePFS/users/lutianyu/lf/clusters-by-entity-40.txt"),
#     shard_id=0,
# )


DataEngine().process_pickle2parquet(
    bq_path=Path("/GenSIvePFS/users/lutianyu/lf/data/unicluster40/bq.parquet"),
    pickle_dir=Path("/GenSIvePFS/users/lutianyu/lf/data/unicluster40/pickle"),
    parquet_dir=Path("/GenSIvePFS/users/lutianyu/lf/data"),
    bsz=100,
    num_consumers=2,
    num_producers=10,
    tokenizer_name="dplm",
    dataset_name="unicluster",
)

