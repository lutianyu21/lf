import pandas as pd
from pathlib import Path
from collections import defaultdict
pdb_name_count = defaultdict(int)
parquet_dir = Path("/GenSIvePFS/users/lutianyu//data/AFDB/parquet/part-02")
cumu_set = set()
for parquet_file in parquet_dir.glob("*.parquet"):
    try:
        df = pd.read_parquet(parquet_file)
        new_set = set(df['pdb_name'].tolist())
        intersect = cumu_set.intersection(new_set)
        if len(intersect) > 0:
            print(f"Found {len(intersect)} duplicate pdb_names in {parquet_file}: {intersect}")
        else:
            print(f"No duplicate pdb_names in {parquet_file}, total {len(cumu_set)}")
        cumu_set.update(new_set)
    except Exception as e:
        print(f"Error reading {parquet_file}: {e}")