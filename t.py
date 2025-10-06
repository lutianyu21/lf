# test build_dataset_from_entry
from utils.lf_utils import protein_processor, lf_tokenizer, dplm_protein_tokenizer, build_dataset
from pathlib import Path
import ray

# processor = protein_processor.ProteinProcessor(tokenizer=lf_tokenizer, struct_tokenizer=dplm_protein_tokenizer.to('cuda:0'))
# processor.build_dataset_from_entry(
#     csv_path=Path('/AIRvePFS/ai4science/users/tianyu/lf/data/dataset1_metadata.csv'),
#     jsonl_path=Path('/AIRvePFS/ai4science/users/tianyu/lf/data/dataset1_metadata.jsonl')
# )

ray.init()

# ---- example ----
build_dataset(
    csv_path=Path('/AIRvePFS/ai4science/users/tianyu/lf/data/dataset1_metadata.csv'),
    jsonl_path=Path('/AIRvePFS/ai4science/users/tianyu/lf/data/dataset1.jsonl'),
    batch_size=80,
    num_workers=2,
)

# build_dataset(
#     csv_path=Path('/AIRvePFS/ai4science/users/tianyu/lf/trash/test.csv'),
#     jsonl_path=Path('/AIRvePFS/ai4science/users/tianyu/lf/trash/test.jsonl'),
#     batch_size=80,
# )