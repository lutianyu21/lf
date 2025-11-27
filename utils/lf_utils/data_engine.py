from pyexpat import features
import ray
from pathlib import Path
import tarfile
import shutil
import colorlog
import logging
from typing import Iterator, Tuple, Any


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
    try:
        if isinstance(iter, tuple):
            tar_path, member_name = iter
            with tarfile.open(tar_path, "r") as tf:
                member = tf.getmember(member_name)
                f = tf.extractfile(member)
                if f is None:
                    return ("failed", str(iter), "Failed to extract member")
                
                target_path = output_dir / Path(member.name).name
                if target_path.exists():
                    return ("skipped", str(target_path))
                
                with open(target_path, "wb") as out_f:
                    shutil.copyfileobj(f, out_f)
                return ("success", str(target_path))
        else:
            raise NotImplementedError()
    
    except Exception as e:
        return ("failed", str(iter), str(e))


# TODO: organization by sql ?
class DataEngine:
    
    ### Scanning Functions ###
    # scan AFDB / SwissProt / RCSB / CASP / CAMEO dataset and yield paths
    @classmethod
    def _scan_hk_afdb(cls) -> Iterator[Tuple[Path, str]]:
        # e.g. for proteome-tax_id-1974607-0_v4.tar, we have
        # - AF-A0A2H0UIM4-F1-model_v4.cif.gz
        # - AF-A0A2H0UIM4-F1-confidence_v4.json.gz
        # - AF-A0A2H0UIM4-F1-predicted_aligned_error_v4.json.g[z
        # scanning will return 
        # - tmp_root/AF-A0A2H0UIM4-F1-model_v4.cif.gz
        for split_dir in [
            Path("/GenSIvePFS/users/lutianyu/data/AFDB/part_00"),
            Path("/GenSIvePFS/users/lutianyu/data/AFDB/part_01"),
            Path("/GenSIvePFS/users/lutianyu/data/AFDB/part_02"),
            Path("/GenSIvePFS/users/lutianyu/data/AFDB/part_03"),
            Path("/GenSIvePFS/users/lutianyu/data/AFDB/part_04"),
            Path("/GenSIvePFS/users/lutianyu/data/AFDB/part_05_dest"),
            Path("/GenSIvePFS/users/lutianyu/lutianyu/data/AFDB/part_06"),
        ]:
            for tar_path in split_dir.glob("proteome-tax_id-*_v4.tar"):
                try:
                    with tarfile.open(tar_path, "r") as tf:
                        for member in tf:
                            if member.name.endswith("-F1-model_v4.cif.gz"):
                                # Workers will take care of tmp files
                                yield (tar_path, member.name)
                except Exception as e:
                    logger.error(f"Failed to scan {tar_path}: {e}")
                    continue
                    
    
    ### API ###
    def _query_hk_afdb(
        self,
        query_path: Path,
        output_dir: Path,
        max_concurrent: int,
    ):
        if not query_path.name.endswith('.txt'): raise NotImplementedError()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # load query set: AF_AFQ8UGE8F1_1 AF_AFQ8ZB83F1_1 ...
        query_set = set()
        with open(query_path, 'r') as f:
            for line in f:
                for item in line.strip().split():
                    if not item.startswith('AF'):
                        logger.warning(f"Ignore query item: {item}")
                    else:
                        # FIX here: uniref-style AF_AFA0A075B5T2F1_1; HK-style AF-AFA0A075B5T2F1-F1-model_v4
                        item_fixed = f"AF-{item.split('_')[1]}-F1-model_v4"
                        query_set.add(item_fixed)
        
        # submit cpu jobs
        total_count = 0
        failures,futures = [], []
        for i, it in enumerate(self._scan_hk_afdb()):
            if isinstance(it, tuple):
                tar_path, member_name = it
                p_name = Path(member_name).name
                p_name = Path(p_name).stem      # remove .gz
                p_name = Path(p_name).stem      # remove .cif
            else:
                raise NotImplementedError()
            
            if p_name not in query_set:
                logger.warning(f"Skipping {p_name} ...")
                continue
            else:
                logger.info(f"Collecting {p_name} ...")
            
            # if match, submit a copy/parsing task
            futures.append(extract2target.remote(it, output_dir))
            if len(futures) >= max_concurrent:
                done, futures = ray.wait(futures, num_returns=1)
                done_results = ray.get(done)
                for res in done_results:
                    status = res[0]
                    if status == "success":
                        total_count += 1
                        logger.info(f"[uid={total_count}/{len(futures)}] Processed: {res[1]}")
                    elif status == "skipped":
                        logger.warning(f"[uid={total_count}/{len(futures)}] Skipped (exists): {res[1]}")
                    elif status == "failed":
                        logger.error(f"[uid={total_count}/{len(futures)}] Failed: {res[1]} Error: {res[2]}")
                        failures.append(res[1])

        # collecting remaining futures
        while futures:
            done, futures = ray.wait(futures)
            done_results = ray.get(done)
            for res in done_results:
                status = res[0]
                if status == "success":
                    total_count += 1
                    logger.info(f"[uid={total_count}] Processed: {res[1]}")
                elif status == "skipped":
                    logger.warning(f"[uid={total_count}] Skipped (exists): {res[1]}")
                elif status == "failed":
                    logger.error(f"[uid={total_count}] Failed: {res[1]} Error: {res[2]}")
                    failures.append(res[1])
                    
        failures_file = output_dir / "failures.txt"
        if failures_file.exists(): failures_file.unlink()
        if failures:
            with open(failures_file, "w") as f:
                for item in failures:
                    f.write(f"{item}\n")
            logger.info(f"Total failed items: {len(failures)}. Saved to {failures_file}")
        logger.info(f"All tasks completed. Total submitted: {total_count}")