import ray
ray.init(ignore_reinit_error=True)

from utils.lf_utils.data_engine import DataEngine
from pathlib import Path

# DataEngine()._query_afdb(
#     query_path=Path("/GenSIvePFS/users/lutianyu/lf/clusters-by-entity-40.txt"),
#     bq_path=Path("/GenSIvePFS/users/lutianyu/lf/bq.parquet"),
#     output_dir=Path("/GenSIvePFS/users/lutianyu/lf/afdb"),
#     max_concurrent=8000,
# )

DataEngine().query_rcsb(
    output_dir=Path("/GenSIvePFS/users/lutianyu/lf/data/unicluster_40"),
    query_path=Path("/GenSIvePFS/users/lutianyu/lf/data/unicluster_40/clusters-by-entity-40.txt"),
    max_concurrent=5000,
)