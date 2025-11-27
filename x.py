import ray
ray.init(ignore_reinit_error=True)

from utils.lf_utils.data_engine import DataEngine
from pathlib import Path

DataEngine()._query_hk_afdb(
    query_path=Path("/GenSIvePFS/users/lutianyu/lf/clusters-by-entity-40.txt"),
    output_dir=Path("/GenSIvePFS/users/lutianyu/lf/afdb"),
    max_concurrent=3000,
)