import ray
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import os
import torch

from .protein_processor import ProteinProcessor
from .protein_tokenizer import DPLMProteinTokenizer, dplm_protein_tokenizer
from .text_tokenizer import lf_tokenizer

# ---- GPU Worker ----
@ray.remote(num_gpus=1)
class GPUWorker:
    def __init__(self):
        from .protein_processor import ProteinProcessor
        from .protein_tokenizer import dplm_protein_tokenizer
        from .text_tokenizer import lf_tokenizer
        gpu_ids = ray.get_gpu_ids()
        if not gpu_ids:
            raise RuntimeError("No GPU assigned to this actor")
        self.device = torch.device(f"cuda:{gpu_ids[0]}")
        self.processor = ProteinProcessor(
            tokenizer=lf_tokenizer,
            struct_tokenizer=dplm_protein_tokenizer,
        ).to(self.device)

    def fn(self, batch):
        return self.processor.build_dataset(batch)

# ---- main ----
def build_dataset(csv_path: Path, jsonl_path: Path, batch_size=16, num_workers=1):
    df = pd.read_csv(csv_path)
    ds = ray.data.from_pandas(df)
    total = ds.count()
    workers = [GPUWorker.remote() for _ in range(num_workers)]
    results = []
    pending = []
    for i, batch in enumerate(ds.iter_batches(batch_size=batch_size, batch_format="pandas")):
        worker = workers[i % num_workers]               # round-robin
        batch_dict = batch.to_dict(orient="records")    # type: ignore
        pending.append(worker.fn.remote(batch_dict))    # type: ignore

    counter = 0
    with tqdm(total=total) as pbar:
        while pending:
            done, pending = ray.wait(pending, num_returns=1)
            r = ray.get(done[0])
            results.extend(r)
            pbar.update(len(r))
            counter += len(r)
            
            # frequently save results checkpoint
            if counter >= 1000:
                counter = 0
                pd.DataFrame(results).to_json(jsonl_path, orient="records", lines=True)
                
    pd.DataFrame(results).to_json(jsonl_path, orient="records", lines=True)
