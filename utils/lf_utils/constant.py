__all__ = ['DATASET_SPLIT', 'DATASET_RAW_ROOT']

import pdb


DATASET_SPLIT = {
    'dev':              0,
    'cameo2022':        1,
    'casp15':           2,
    'casp16':           3,
    'rcsb':             4,
    'afdb_swissprot':   5,
}


DATASET_RAW_ROOT = {
    'dev':              ('/GenSIvePFS/users/lutianyu/lf/data/rcsb/raw',                 '.cif'),
    'cameo2022':        ('/GenSIvePFS/users/lutianyu/lf/data/rcsb/raw',                 '.cif'),
    'casp15':           ('/GenSIvePFS/users/lutianyu/lf/data/casp15/raw',               '.pdb'),
    'casp16':           ('/GenSIvePFS/users/lutianyu/lf/data/casp16/raw',               '.pdb'),
    'rcsb':             ('/GenSIvePFS/users/lutianyu/lf/data/rcsb/raw',                 '.cif'),
    'afdb_swissprot':   ('/GenSIvePFS/users/lutianyu/lf/data/swissprot_v4/raw',         '.cif.gz'),
}