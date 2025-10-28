from typing import Any, Iterator, List
from pathlib import Path
import tarfile
import tempfile
import logging
import colorlog
import pickle
import os
import shutil
import pyarrow
import pyarrow.parquet
import time
from tqdm import tqdm

import ray
from ray.util.actor_pool import ActorPool
from ray.util.queue import Queue
import random
import numpy as np
import torch

from tokenizers import Tokenizer
from ..openfold_utils import OpenfoldProtein
from .protein_processor import ProteinProcessor
from .protein_tokenizer import DPLMProteinTokenizer
from .protein_tokenizer import DistMatrixTokenizer
from .text_tokenizer import TextTokenizer

__all__ = [
    "step1_pickle",
    "step2_parquet",
]

logger = logging.getLogger(__name__)
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s" + "[%(asctime)s][%(levelname)s]" + " %(message)s",
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'bold_red',
    }
))
logger.handlers.clear()
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


queue = Queue(maxsize=200)
queue_ref = ray.put(queue)

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    

def _stream_iterate_afdb(tax_dir: str | Path) -> Iterator[Path]:
    if isinstance(tax_dir, str): tax_dir = Path(tax_dir)
    # e.g. for proteome-tax_id-1974607-0_v4.tar, we have
    # - AF-A0A2H0UIM4-F1-model_v4.cif.gz
    # - AF-A0A2H0UIM4-F1-confidence_v4.json.gz
    # - AF-A0A2H0UIM4-F1-predicted_aligned_error_v4.json.gz
    tmp_root = Path(tempfile.gettempdir()) / f"pid_{os.getpid()}"
    tmp_root.mkdir(parents=True, exist_ok=True)
    for tar_path in tax_dir.glob("proteome-tax_id-*_v4.tar"):
        try:
            with tarfile.open(tar_path, "r") as tf:
                for member in tf:
                    if member.name.endswith("-F1-model_v4.cif.gz"):
                        f = tf.extractfile(member)
                        if f is None: continue
                        tmp_path = tmp_root / Path(member.name).name
                        with open(tmp_path, "wb") as out_f:
                            shutil.copyfileobj(f, out_f)
                        yield tmp_path
        except Exception as e:
            logger.error(f"Failed to extract {tar_path}: {e}")


def _stream_iterate_swissprot(swiss_dir: str | Path) -> Iterator[Path]:
    if isinstance(swiss_dir, str): swiss_dir = Path(swiss_dir)
    # e.g. AF-A0A0A0MRZ7-F1-model_v4.cif.gz
    for cif_path in swiss_dir.glob("AF-*-F1-model_v4.cif.gz"):
        yield cif_path
        
def _stream_iterate_rcsb(rcsb_dir: str | Path) -> Iterator[Path]:
    if isinstance(rcsb_dir, str): rcsb_dir = Path(rcsb_dir)
    # if file_list.txt exists, read from it
    file_list_path = rcsb_dir / "file_list.txt"
    if file_list_path.exists():
        with open(file_list_path, 'r') as f:
            for line in f:
                cif_path = Path(line.strip())
                yield cif_path
    else:
        for cif_path in rcsb_dir.glob("*.cif"):
            yield cif_path


@ray.remote
def process_file(input_path: Path, output_dir: Path, clear: bool):
    """
        ("success", output_path) -> success
        ("skipped", output_path) -> skipped
        ("failed", input_path, str(error)) -> failed
    """
    output_path = output_dir / (input_path.name.strip('.gz').strip('.cif') + ".pkl")
    if output_path.exists():
        if clear: input_path.unlink()
        return ("skipped", str(output_path))
    try:
        protein = OpenfoldProtein.from_file(input_path)
        with output_path.open("wb") as f:
            pickle.dump(protein, f, protocol=pickle.HIGHEST_PROTOCOL)
        if clear: input_path.unlink()
        return ("success", str(output_path))
    except Exception as e:
        if clear: input_path.unlink()
        return ("failed", str(input_path), str(e))
    

def step1_pickle(
    src_dir: str | Path,
    dst_dir: str | Path,
    dataset_name: str = "afdb",
    max_concurrent: int = 8,
    clear: bool = False,
):
    if isinstance(src_dir, str): src_dir = Path(src_dir)
    if isinstance(dst_dir, str): dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(2025)
    
    results, failures, futures = [], [], []
    total_count = 0
    # TODO convert generator to support others
    stream_iterate = {
        "rcsb":             _stream_iterate_rcsb,
        "afdb":             _stream_iterate_afdb,
        "swissprot_v4":     _stream_iterate_swissprot,
    }[dataset_name]
    for p in stream_iterate(src_dir):
        futures.append(process_file.remote(p, dst_dir, clear))
        total_count += 1
        if len(futures) >= max_concurrent:
            done, futures = ray.wait(futures, num_returns=max_concurrent)
            done_results = ray.get(done)
            for res in done_results:
                status = res[0]
                if status == "success":
                    logger.info(f"[uid={total_count}] Processed: {res[1]}")
                elif status == "skipped":
                    logger.warning(f"[uid={total_count}] Skipped (exists): {res[1]}")
                elif status == "failed":
                    logger.error(f"[uid={total_count}] Failed: {res[1]} Error: {res[2]}")
                    failures.append(res[1])
            results.extend(done_results)
            
    while futures:
        done, futures = ray.wait(futures, num_returns=min(max_concurrent, len(futures)))
        done_results = ray.get(done)
        for res in done_results:
            status = res[0]
            if status == "success":
                logger.info(f"[uid={total_count}] Processed: {res[1]}")
            elif status == "skipped":
                logger.warning(f"[uid={total_count}] Skipped (exists): {res[1]}")
            elif status == "failed":
                logger.error(f"[uid={total_count}] Failed: {res[1]} Error: {res[2]}")
                failures.append(res[1])
        results.extend(done_results)
    
    failures_file = dst_dir / "failures.txt"
    if failures_file.exists(): failures_file.unlink()
    if failures:
        with open(dst_dir/'failures.txt', "w") as f:
            for item in failures:
                f.write(f"{item}\n")
        logger.info(f"Total failed items: {len(failures)}. Saved to {failures_file}")
    logger.info(f"All files processed. Total submitted: {total_count}")


@ray.remote(num_gpus=1)
class GPUWorker:
    def __init__(self, tokenizer_name: str):
        from .protein_processor import ProteinProcessor
        gpu_ids = ray.get_gpu_ids()
        if not gpu_ids:
            raise RuntimeError("No GPU assigned to this actor")
        self.device = torch.device(f"cuda:0")
        # lazy initilaize
        self.tokenizer_name = tokenizer_name
        self.processor = None

    def fn(self, batch: List[OpenfoldProtein]):
        if self.processor is None:
            struct_tokenizer = {
                "dplm": DPLMProteinTokenizer,
                "dist": DistMatrixTokenizer,
            }[self.tokenizer_name].get_instance()
            text_tokenizer = TextTokenizer(
                tokenizer_object=Tokenizer.from_file(str(Path(__file__).parent/'aatype_tokenizer.json')),
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
        return self.processor.preprocess_dataset(batch, verbose=False)


@ray.remote
class PickleWorker:
    def __init__(
        self,
        root: str,
        batch_size: int,
        group_size: int,
        group_id: int,
    ):
        self.root = Path(root)
        self.group_size = group_size
        self.group_id = group_id
        self.batch_size = batch_size

    def fn(self, out_queue: Queue):
        batch = []
        count = 0 
        with os.scandir(self.root) as it:
            for entry in it:
                if not entry.is_file() or not entry.name.endswith(".pkl"):
                    continue
                if count % self.group_size != self.group_id:
                    count += 1
                    continue
                count += 1
                try:
                    with open(entry.path, "rb") as f:
                        obj = pickle.load(f)
                    batch.append(obj)
                    logger.debug(f"Reading {entry.path}...")
                    if len(batch) >= self.batch_size:
                        out_queue.put(batch)
                        batch = []
                except Exception as e:
                    logger.warning(f"Failed to read {entry.path}: {e}")
        if batch:
            out_queue.put(batch)
        logger.info(f"[gid={self.group_id}] Finished reading files.")
        out_queue.put(None)


def step2_parquet(
    src_dir: str | Path,
    dst_dir: str | Path,
    tokenizer_name: str,
    queue: Queue,
    num_cpu_workers: int = 8,
    num_gpu_workers: int = 4,
    batch_size: int = 64,
    part_size: int = 100000,
):
    if isinstance(src_dir, str): src_dir = Path(src_dir)
    if isinstance(dst_dir, str): dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    seed_everything(2025)
    
    gpu_workers = [GPUWorker.remote(tokenizer_name) for _ in range(num_gpu_workers)]
    gpu_workers_max = num_gpu_workers * 2
    cpu_workers = [PickleWorker.remote(str(src_dir), batch_size, num_cpu_workers, i) for i in range(num_cpu_workers)]
    cpu_workers_done = 0
    for w in cpu_workers:
        w.fn.remote(queue)  # type: ignore
    
    bid = 0
    current_part = 0
    current_part_size = 0
    parquet_writer = None
    time_start = time.time()
    pending_refs = []
    while True:
        batch = queue.get()
        if batch is None:
            cpu_workers_done += 1
            if cpu_workers_done >= num_cpu_workers:
                break
            else:
                continue
            
        # submit batch to GPU worker
        gpu_worker= gpu_workers[bid % num_gpu_workers]
        ref = gpu_worker.fn.remote(batch) # type: ignore
        pending_refs.append(ref)
        bid += 1
            
        if len(pending_refs) >= gpu_workers_max:
            ready, pending_refs = ray.wait(pending_refs, num_returns=1)
            result = ray.get(ready[0])
            elapsed = time.time() - time_start
            time_start = time.time()
            logger.info(f"[{int(elapsed)}s] Processed batch of {len(result)} items (pending={len(pending_refs)})")

            table = pyarrow.Table.from_pylist(result)
            if parquet_writer is None:
                dst_file = dst_dir / f"dataset_part{current_part}.parquet"
                parquet_writer = pyarrow.parquet.ParquetWriter(str(dst_file), table.schema, compression="snappy")
            parquet_writer.write_table(table)
            current_part_size += len(result)
            
            if current_part_size >= part_size:
                parquet_writer.close()
                logger.info(f"✅ Part {current_part} finished with {current_part_size} entries.")
                current_part += 1
                current_part_size = 0
                parquet_writer = None
                
    logger.info("⌛ Waiting for remaining GPU tasks...")
    while pending_refs:
        ready, pending_refs = ray.wait(pending_refs, num_returns=1)
        result = ray.get(ready[0])
        elapsed = time.time() - time_start
        time_start = time.time()
        logger.info(f"[{int(elapsed)}s] Processed batch of {len(result)} items (pending={len(pending_refs)})")
        table = pyarrow.Table.from_pylist(result)
        if parquet_writer is None:
            dst_file = dst_dir / f"dataset_part{current_part}.parquet"
            parquet_writer = pyarrow.parquet.ParquetWriter(str(dst_file), table.schema, compression="snappy")
        parquet_writer.write_table(table)
        current_part_size += len(result)
        if current_part_size >= part_size:
            parquet_writer.close()
            logger.info(f"✅ Part {current_part} finished with {current_part_size} entries.")
            current_part += 1
            current_part_size = 0
            parquet_writer = None
    
    if parquet_writer is not None:
        parquet_writer.close()
    logger.info(f"🏁 All batches processed and saved to {dst_dir}")
    