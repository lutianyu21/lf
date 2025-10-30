# import pandas as pd
# from pathlib import Path
# from collections import defaultdict
# pdb_name_count = defaultdict(int)
# parquet_dir = Path("/GenSIvePFS/users/lutianyu//data/AFDB/parquet/part-02")
# cumu_set = set()
# for parquet_file in parquet_dir.glob("*.parquet"):
#     try:
#         df = pd.read_parquet(parquet_file)
#         new_set = set(df['pdb_name'].tolist())
#         intersect = cumu_set.intersection(new_set)
#         if len(intersect) > 0:
#             print(f"Found {len(intersect)} duplicate pdb_names in {parquet_file}: {intersect}")
#         else:
#             print(f"No duplicate pdb_names in {parquet_file}, total {len(cumu_set)}")
#         cumu_set.update(new_set)
#     except Exception as e:
#         print(f"Error reading {parquet_file}: {e}")


import os
import shutil
import random
from tqdm import tqdm

src_dir = "/GenSIvePFS/users/lutianyu/data/AFDB/pickle/part-00"
dst1 = "/GenSIvePFS/users/lutianyu/data/AFDB/pickle/part_00"
dst2 = "/GenSIvePFS/users/lutianyu/data/AFDB/pickle/part_01"

os.makedirs(dst1, exist_ok=True)
os.makedirs(dst2, exist_ok=True)
def iter_files(directory):
    with os.scandir(directory) as it:
        for entry in it:
            if entry.is_file():
                yield entry.name

files = list(iter_files(src_dir))
random.shuffle(files)
for i, f in tqdm(enumerate(files)):
    dst = dst1 if i % 2 == 0 else dst2
    shutil.move(os.path.join(src_dir, f), os.path.join(dst, f))