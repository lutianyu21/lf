from typing import Any, Dict, Iterator, List
from pathlib import Path
import tarfile
import tempfile
import logging
import colorlog
import pickle
import os
import shutil
import pandas as pd
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
    "step3_merge",
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
        gpu_ids = ray.get_gpu_ids()
        if not gpu_ids:
            raise RuntimeError("No GPU assigned to this actor")
        self.device = torch.device(f"cuda:0")
        self.tokenizer_name = tokenizer_name
        self.processor = None

    def _init_processor(self):
        """Lazy initialize processor only once per actor."""
        if self.processor is not None: return
        struct_tokenizer = {
            "dplm": DPLMProteinTokenizer,
            "dist": DistMatrixTokenizer,
        }[self.tokenizer_name].get_instance()
        text_tokenizer = TextTokenizer(
            tokenizer_object=Tokenizer.from_file(
                str(Path(__file__).parent / "aatype_tokenizer.json")
            ),
            pad_token="<|pad|>",
            bos_token="<|bos|>",
            eos_token="<|eos|>",
            padding_side="left",
            struct_vsz=struct_tokenizer.vsz,
        )

        self.processor = ProteinProcessor(
            tokenizer=text_tokenizer,
            struct_tokenizer=struct_tokenizer,
        ).to(self.device)

    def process_bytes(self, batch: List[Any]) -> List[Dict[str, Any]]:
        self._init_processor()
        proteins = []
        for prot in batch:
            prot = OpenfoldProtein.from_dict(prot.as_py())
            proteins.append(prot)
        return self.processor.preprocess_dataset(proteins, verbose=False) # type: ignore


def step2_parquet(
    src_dir: str | Path,
    dst_dir: str | Path,
    tokenizer_name: str,
    num_gpu_workers: int = 4,
    batch_size: int = 64,
    chunk_size: int = 10000,  # ✅ 每次写入 parquet 的最小行数
):
    if isinstance(src_dir, str): src_dir = Path(src_dir)
    if isinstance(dst_dir, str): dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    workers = [GPUWorker.remote(tokenizer_name) for _ in range(num_gpu_workers)]
    
    def decode_bytes_batch(batch):
        byte_list = batch["bytes"].to_pylist()
        dict_list = [pickle.loads(b).to_dict() for b in byte_list]
        return {"proteins": dict_list}
    
    ds = ray.data.read_binary_files(str(src_dir), include_paths=False)
    ds = ds.map_batches(decode_bytes_batch, batch_size=batch_size, batch_format=None)
    
    inflight = []
    counter = 0
    part_idx = 0
    start_time = time.time()
    buffer = []  # ✅ 临时存储结果的缓冲区

    # --- 主循环 ---
    for batch in ds.iter_batches(batch_size=batch_size, batch_format=None):
        worker = workers[counter % num_gpu_workers]
        ref = worker.process_bytes.remote(batch['proteins'])  # type: ignore
        inflight.append(ref)
        counter += 1

        # 控制并发数量
        if len(inflight) >= 2 * num_gpu_workers:
            ready, inflight = ray.wait(inflight, num_returns=1, fetch_local=False)
            result = ray.get(ready[0])
            buffer.extend(result)  # ✅ 累积结果
            elapsed_time = time.time() - start_time
            start_time = time.time()
            logger.info(f"[✅ {int(elapsed_time)}s] Collected {len(result)} rows")

            # --- 当 buffer 达到指定大小，写出 parquet ---
            if len(buffer) >= chunk_size:
                df = pd.DataFrame(buffer)
                out_path = dst_dir / f"dataset_part{part_idx:04d}.parquet"
                df.to_parquet(out_path, index=False)
                buffer.clear()
                part_idx += 1

    # --- 等待所有 inflight 任务完成 ---
    while inflight:
        ready, inflight = ray.wait(inflight, num_returns=1, fetch_local=False)
        result = ray.get(ready[0])
        buffer.extend(result)
        elapsed_time = time.time() - start_time
        start_time = time.time()
        logger.info(f"[✅ {int(elapsed_time)}s] Collected {len(result)} rows")

        if len(buffer) >= chunk_size:
            df = pd.DataFrame(buffer)
            out_path = dst_dir / f"dataset_part{part_idx:04d}.parquet"
            df.to_parquet(out_path, index=False)
            buffer.clear()
            part_idx += 1

    # --- 写出剩余数据 ---
    if buffer:
        df = pd.DataFrame(buffer)
        out_path = dst_dir / f"dataset_part{part_idx:04d}.parquet"
        df.to_parquet(out_path, index=False)
        part_idx += 1

    logger.info("✅ All batches processed successfully.")


def step3_merge(
    src_dir: Path | str,
    dst_dir: Path | str,
    add_split: str | None,
):
    if isinstance(src_dir, str): src_dir = Path(src_dir)
    if isinstance(dst_dir, str): dst_dir = Path(dst_dir)
    # merge all parquet files into one(add split column if needed)
    all_tables = []
    for p in src_dir.glob("dataset_part*.parquet"):
        try:
            table = pyarrow.parquet.read_table(str(p))
        except Exception as e:
            logger.error(f"Failed to read {p}: {e}")
            continue
        if add_split is not None:
            split_array = pyarrow.array([add_split] * table.num_rows)
            table = table.append_column("split", split_array)
        all_tables.append(table)
        logger.info(f"Read {p} with {table.num_rows} rows.")
    merged_table = pyarrow.concat_tables(all_tables)
    
    # log merged table info
    logger.info(f"Merged table has {merged_table.num_rows} rows. Columns: {merged_table.column_names}")
  
    # save merged table
    dst_file = dst_dir / "dataset_merged.parquet"
    pyarrow.parquet.write_table(merged_table, str(dst_file), compression="snappy")
    logger.info(f"Saved merged parquet to {dst_file}")
