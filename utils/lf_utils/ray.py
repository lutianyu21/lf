import ray
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import torch

from tokenizers import Tokenizer

from .protein_processor import ProteinProcessor
from .protein_tokenizer import DPLMProteinTokenizer
from .protein_tokenizer import DistMatrixTokenizer
from .text_tokenizer import TextTokenizer

# ---- GPU Worker ----
@ray.remote(num_gpus=1)
class GPUWorker:
    def __init__(self):
        from .protein_processor import ProteinProcessor
        gpu_ids = ray.get_gpu_ids()
        if not gpu_ids:
            raise RuntimeError("No GPU assigned to this actor")
        self.device = torch.device(f"cuda:0")
        struct_tokenizer=DPLMProteinTokenizer.get_instance()
        text_tokenizer = TextTokenizer(
            tokenizer_object=Tokenizer.from_file(str(Path(__file__).parent.parent/'progen2_utils/progen/progen2/tokenizer.json')),
            pad_token='<|pad|>',
            bos_token='<|bos|>',
            eos_token='<|eos|>',
            padding_side='left',
            struct_vsz=struct_tokenizer.vsz,
        )
        self.processor = ProteinProcessor(
            tokenizer=text_tokenizer,
            struct_tokenizer=struct_tokenizer,
        ).to(self.device)

    def fn(self, batch):
        return self.processor.build_dataset(batch, verbose=False)

# ---- main ----
def build_dataset(csv_path: Path, jsonl_path: Path, batch_size=16, num_workers=1):
    # seed everything
    import random
    import numpy as np
    import torch
    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    df = pd.read_csv(csv_path)
    # df = df.sort_values(by='protein_path')
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
            if r == []: continue
            results.extend(r)
            pbar.update(len(r))
            counter += len(r)
            # frequently save results checkpoint
            if counter >= 1000:
                counter = 0
                pd.DataFrame(results).to_json(jsonl_path, orient="records", lines=True, mode='a')
                results.clear()
                
    pd.DataFrame(results).to_json(jsonl_path, orient="records", lines=True, mode='a')
