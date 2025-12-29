__all__ = ['LF_ROOT', 'LF_DATA_ROOT', 'LF_MODEL_ROOT', 'LF_TOKENIZER_CKPT_ROOT', 'DATASET_SPLIT', 'DATASET_RAW_ROOT']

import os
from pathlib import Path

# Global project root: can be set via environment variable or auto-detected
# Priority: LF_ROOT env var > auto-detect from this file's location
_default_root = Path(__file__).parent.parent.parent.resolve()  # utils/lf_utils/constant.py -> project root
LF_ROOT = Path(os.environ.get("LF_ROOT", str(_default_root)))

# Data and model roots (can be overridden via environment variables)
LF_DATA_ROOT = Path(os.environ.get("LF_DATA_ROOT", str(LF_ROOT / "data")))
LF_MODEL_ROOT = Path(os.environ.get("LF_MODEL_ROOT", "/SPXvePFS/model"))
LF_TOKENIZER_CKPT_ROOT = Path(os.environ.get("LF_TOKENIZER_CKPT_ROOT", "/SPXvePFS/share/zzhang/LLMFolding_tokenizer/ckpt"))


DATASET_SPLIT = {
    'dev':              0,
    'cameo2022':        1,
    'casp15':           2,
    'casp16':           3,
    'rcsb':             4,
    'afdb_swissprot':   5,
}


DATASET_RAW_ROOT = {
    'dev':              (str(LF_DATA_ROOT / 'rcsb/raw'),                 '.cif'),
    'cameo2022':        (str(LF_DATA_ROOT / 'rcsb/raw'),                 '.cif'),
    'casp15':           (str(LF_DATA_ROOT / 'casp15/raw'),               '.pdb'),
    'casp16':           (str(LF_DATA_ROOT / 'casp16/raw'),               '.pdb'),
    'rcsb':             (str(LF_DATA_ROOT / 'rcsb/raw'),                 '.cif'),
    'afdb_swissprot':   (str(LF_DATA_ROOT / 'swissprot_v4/raw'),         '.cif.gz'),
}