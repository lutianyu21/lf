# from tqdm import tqdm

# import pandas as pd
# import torch

# from tokenizers import Tokenizer

# from .protein_processor import ProteinProcessor
# from .protein_tokenizer import DPLMProteinTokenizer
# from .protein_tokenizer import DistMatrixTokenizer
# from .text_tokenizer import TextTokenizer

# # ---- GPU Worker ----
# @ray.remote(num_gpus=1)
# class GPUWorker:
#     def __init__(self):
#         from .protein_processor import ProteinProcessor
#         gpu_ids = ray.get_gpu_ids()
#         if not gpu_ids:
#             raise RuntimeError("No GPU assigned to this actor")
#         self.device = torch.device(f"cuda:0")
#         struct_tokenizer=DPLMProteinTokenizer.get_instance()
#         text_tokenizer = TextTokenizer(
#             tokenizer_object=Tokenizer.from_file(str(Path(__file__).parent.parent/'progen2_utils/progen/progen2/tokenizer.json')),
#             pad_token='<|pad|>',
#             bos_token='<|bos|>',
#             eos_token='<|eos|>',
#             padding_side='left',
#             struct_vsz=struct_tokenizer.vsz,
#         )
#         self.processor = ProteinProcessor(
#             tokenizer=text_tokenizer,
#             struct_tokenizer=struct_tokenizer,
#         ).to(self.device)

#     def fn(self, batch):
#         return self.processor.build_dataset(batch, verbose=False)
    

from typing import Iterator
from pathlib import Path
import tarfile
import tempfile
import logging
import colorlog
import pickle
import os
import shutil

import ray
import random
import numpy as np
from sympy import O
import torch

from ..openfold_utils import OpenfoldProtein


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
    

def iterate_afdb(tax_dir: str | Path) -> Iterator[Path]:
    if isinstance(tax_dir, str): tax_dir = Path(tax_dir)
    # e.g. for proteome-tax_id-1974607-0_v4.tar, we have
    # - AF-A0A2H0UIM4-F1-model_v4.cif.gz
    # - AF-A0A2H0UIM4-F1-confidence_v4.json.gz
    # - AF-A0A2H0UIM4-F1-predicted_aligned_error_v4.json.gz
    tmp_root = Path(tempfile.gettempdir()) / f"pid_{os.getpid()}"
    tmp_root.mkdir(parents=True, exist_ok=True)
    for tar_path in tax_dir.glob("proteome-tax_id-*_v4.tar"):
        with tarfile.open(tar_path, "r") as tf:
            for member in tf:
                if member.name.endswith("-F1-model_v4.cif.gz"):
                    f = tf.extractfile(member)
                    if f is None: continue
                    tmp_path = tmp_root / Path(member.name).name
                    with open(tmp_path, "wb") as out_f:
                        shutil.copyfileobj(f, out_f)
                    yield tmp_path


def iterate_swissprot(swiss_dir: str | Path) -> Iterator[Path]:
    if isinstance(swiss_dir, str): swiss_dir = Path(swiss_dir)
    # e.g. AF-A0A0A0MRZ7-F1-model_v4.cif.gz
    for cif_path in swiss_dir.glob("AF-*-F1-model_v4.cif.gz"):
        yield cif_path


@ray.remote
def process_file(input_path: Path, output_dir: Path):
    """
        ("success", output_path) -> success
        ("skipped", output_path) -> skipped
        ("failed", input_path, str(error)) -> failed
    """
    output_path = output_dir / (input_path.name.strip('.gz').strip('.cif') + ".pkl")
    if output_path.exists():
        input_path.unlink()
        return ("skipped", str(output_path))
    try:
        protein = OpenfoldProtein.from_file(input_path)
        with output_path.open("wb") as f:
            pickle.dump(protein, f, protocol=pickle.HIGHEST_PROTOCOL)
        input_path.unlink()
        return ("success", str(output_path))
    except Exception as e:
        input_path.unlink()
        return ("failed", str(input_path), str(e))


def main_pickle(
    src_dir: str | Path,
    dst_dir: str | Path,
    max_concurrent: int = 8,
):
    if isinstance(src_dir, str): src_dir = Path(src_dir)
    if isinstance(dst_dir, str): dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(2025)
    
    results, failures, futures = [], [], []
    total_count = 0
    # TODO convert generator to support others
    for p in iterate_afdb(src_dir):
        futures.append(process_file.remote(p, dst_dir))
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
   
    if failures:
        with open(dst_dir/'failures.txt', "w") as f:
            for item in failures:
                f.write(f"{item}\n")
        logger.info(f"Total failed items: {len(failures)}. Saved to {dst_dir/'failures.txt'}")
    logger.info(f"All files processed. Total submitted: {total_count}")
