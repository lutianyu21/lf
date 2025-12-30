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
    'p2s/dev':              0,
    'p2s/cameo2022':        1,
    'p2s/casp15':           2,
    'p2s/casp16':           3,
    'p2s/rcsb':             4,
    'p2s/afdb_swissprot':   5,
    'p2s/unicluster40':     6,
}


DATASET_RAW_ROOT = {
    'p2s/dev':              (str(LF_DATA_ROOT / 'rcsb/raw'),                 '.cif'),
    'p2s/cameo2022':        (str(LF_DATA_ROOT / 'rcsb/raw'),                 '.cif'),
    'p2s/casp15':           (str(LF_DATA_ROOT / 'casp15/raw'),               '.pdb'),
    'p2s/casp16':           (str(LF_DATA_ROOT / 'casp16/raw'),               '.pdb'),
    'p2s/rcsb':             (str(LF_DATA_ROOT / 'rcsb/raw'),                 '.cif'),
    'p2s/afdb_swissprot':   (str(LF_DATA_ROOT / 'swissprot_v4/raw'),         '.cif.gz'),
    'p2s/unicluster40':     (str(LF_DATA_ROOT / 'unicluster40/raw'),         '.cif.gz'),
}