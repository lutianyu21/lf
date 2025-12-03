# create a huge meta-table for iteration
import pandas as pd
from pathlib import Path
from tqdm import  tqdm
import logging
import colorlog

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

df = {
    'unirefAccession':          [],
    'unirefAccessionExtended':  [],
    'picklePath':               [],
    'cifPath':                  [],
}

# for all files under /GenSIvePFS/users/lutianyu/lf/data/unicluster_40/pickle/*.pkl
input_dir = Path("/GenSIvePFS/users/lutianyu/lf/data/unicluster_40/pickle")
for i, pickle_file in enumerate(tqdm(input_dir.glob("*.pkl"))):
    logger.info(f"[{i}] Processing file: {pickle_file.name}")
    # 1ema%1.pkl
    uniref_accession_extended = pickle_file.stem
    df['unirefAccessionExtended'].append(uniref_accession_extended)

    uniref_accession = uniref_accession_extended.split('%')[0]
    df['unirefAccession'].append(uniref_accession)
    
    df['picklePath'].append(pickle_file.name)
    df['cifPath'].append(pickle_file.stem + ".cif")  # placeholder
    
# save to bq.parquet
meta_df = pd.DataFrame(df)
meta_df.to_parquet("/GenSIvePFS/users/lutianyu/lf/data/unicluster_40/bq.parquet", index=False)
print(f"Total items in meta table: {len(meta_df)}")
