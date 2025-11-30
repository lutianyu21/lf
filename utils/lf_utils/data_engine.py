import ray
import re
import os
from pathlib import Path
import tarfile
import shutil
import colorlog
import logging
import pandas as pd
from typing import Iterator, Optional, Tuple, Any


__all__ = ['DataEngine']


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


@ray.remote
def extract2target(iter: Any, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    tar_path, member_name = iter
    try:
        with tarfile.open(tar_path, "r") as tf:
            member = tf.getmember(member_name)
            f = tf.extractfile(member)
            if f is None:
                return ("failed", member_name, "Extracted file is None")
            target_path = output_dir / Path(member.name).name
            if target_path.exists():
                return ("skipped", member_name)
            with open(target_path, "wb") as out_f:
                shutil.copyfileobj(f, out_f)
            return ("success", member_name)
    except Exception as e:
        return ("failed", member_name, str(e))


# TODO: organization by sql ?
class DataEngine:
    
    ### Scanning Functions ###
    # scan AFDB / SwissProt / RCSB / CASP / CAMEO dataset and yield paths
    @classmethod
    def _scan_afdb(cls, tax_id: Optional[int] = None) -> Iterator[None | Tuple[Path, str]]:
        # e.g. for proteome-tax_id-1974607-0_v4.tar, we have
        # - AF-A0A2H0UIM4-F1-model_v4.cif.gz
        # - AF-A0A2H0UIM4-F1-confidence_v4.json.gz
        # - AF-A0A2H0UIM4-F1-predicted_aligned_error_v4.json.g[z
        # scanning will return 
        # - tmp_root/AF-A0A2H0UIM4-F1-model_v4.cif.gz
        
        # if tax_id is provided, filter by it proteome-tax_id-{tax_id}-{shard_id}_v4.tar
        if tax_id is not None:
            pattern = re.compile(fr"^proteome-tax_id-{tax_id}-\d+_v4\.tar$")
        else:
            pattern = re.compile(r"^proteome-tax_id-\d+-\d+_v4\.tar$")
            
        for split_dir in [
            Path("/GenSIvePFS/users/lutianyu/data/AFDB/part_00"),
            Path("/GenSIvePFS/users/lutianyu/data/AFDB/part_01"),
            Path("/GenSIvePFS/users/lutianyu/data/AFDB/part_02"),
            Path("/GenSIvePFS/users/lutianyu/data/AFDB/part_03"),
            Path("/GenSIvePFS/users/lutianyu/data/AFDB/part_04"),
            Path("/GenSIvePFS/users/lutianyu/data/AFDB/part_05_dest"),
            Path("/GenSIvePFS/users/lutianyu/data/AFDB/part_06"),
        ]:
            for tar_path in os.scandir(split_dir):
                if not pattern.match(tar_path.name): continue
                try:
                    with tarfile.open(tar_path, "r") as tf:
                        for member in tf:
                            if member.name.endswith("-F1-model_v4.cif.gz"):
                                # Workers will take care of tmp files
                                yield (Path(tar_path.path), member.name)
                except Exception as e:
                    logger.error(f"Failed to scan {tar_path.path}: {e}")
                    continue
    
    ### API ###
    def _query_afdb(
        self,
        output_dir: Path,
        max_concurrent: int,
        query_path: Path,           # a .txt file containing list of accession ids
        bq_path: Optional[Path],    # a .parquet file containing `taxId` `uniprotAccession`
    ):
        if not query_path.name.endswith('.txt'): raise NotImplementedError()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # load query set: AF_AFQ8UGE8F1_1 AF_AFQ8ZB83F1_1 ...
        query_set = set()
        with open(query_path, 'r') as f:
            for line in f:
                for item in line.strip().split():
                    if not item.startswith('AF'):
                        continue
                    else:
                        # FIX here: rcsb-style AF_AFA0A075B5T2F1_1 >> uniref-style AF-A0A075B5T2-F1-model_v4
                        item_fixed = f"AF-{item.split('_')[1][2:-2]}-F1-model_v4"
                        query_set.add(item_fixed)
                        
        if bq_path is None:
            logger.warning("No tax_path provided, querying all AFDB entries ...")
            # TODO a effective way to load all taxId and uniprotAccession from AFDB
            raise NotImplementedError()
        else:
            bq_df = pd.read_parquet(bq_path)
            bq_df['uniprotAccession'] = bq_df['uniprotAccession'].apply(lambda x: f"AF-{x}-F1-model_v4")
            
        grouped_queries = bq_df.groupby('taxId')['uniprotAccession'].apply(set).to_dict()
        hit_count, futures, failures = 0, [], []
        for tax_id, required_accessions in grouped_queries.items():
            
            logger.info(f"Scanning AFDB for Tax ID: {tax_id} (Targeting {len(required_accessions)} proteins)")
            # {tax-id : set(uniprotAccessions)}
            for it in self._scan_afdb(tax_id=tax_id):
                if isinstance(it, tuple):
                    tar_path, member_name = it
                    # remove.cif.gz，to get accession ID
                    p_name = Path(Path(member_name).name).stem
                    p_name = Path(p_name).stem 
                else:
                    raise NotImplementedError()

                # submit tasks only if in required_accessions
                if p_name in required_accessions:
                    futures.append(extract2target.remote(it, output_dir))
                
                # reduce memory pressure
                if len(futures) >= max_concurrent:
                    done, futures = ray.wait(futures, num_returns=1)
                    done_results = ray.get(done)
                    for res in done_results:
                        status = res[0]
                        if status == "success":
                            hit_count += 1
                            logger.info(f"[{hit_count}/{len(query_set)}] Processed: {res[1]}")
                        elif status == "skipped":
                            hit_count += 1
                            logger.warning(f"[{hit_count}/{len(query_set)}] Skipped: {res[1]}")
                        elif status == "failed":
                            failures.append(res[1])
                            logger.error(f"[{hit_count}/{len(query_set)}] Failed: {res[1]} reason: {res[2]}")
                            
        # collecting remaining futures
        while futures:
            done, futures = ray.wait(futures)
            done_results = ray.get(done)
            for res in done_results:
                status = res[0]
                if status == "success":
                    hit_count += 1
                    logger.info(f"[{hit_count}/{len(query_set)}] Processed: {res[1]}")
                elif status == "skipped":
                    hit_count += 1
                    logger.warning(f"[{hit_count}/{len(query_set)}] Skipped (exists): {res[1]}")
                elif status == "failed":
                    failures.append(res[1])
                    logger.error(f"[{hit_count}/{len(query_set)}] Failed: {res[1]} reason: {res[2]}")
                    
        # HINT: in current design, we can only ensure that all [existed/queried] items are processed
        failures_file = output_dir / "failures.txt"
        if failures_file.exists(): failures_file.unlink()
        if failures:
            with open(failures_file, "w") as f:
                for item in failures:
                    f.write(f"{item}\n")
            logger.info(f"Total failed items: {len(failures)}. Saved to {failures_file}")
        logger.info(f"All tasks completed. Total queried: {hit_count}/{len(query_set)}")
