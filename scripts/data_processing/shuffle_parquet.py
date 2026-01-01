#!/usr/bin/env python
"""
Shuffle rows in parquet files.
Usage:
    python shuffle_parquet.py file1.parquet file2.parquet ...
    python shuffle_parquet.py --input file1.parquet file2.parquet --output_dir ./shuffled/
    python shuffle_parquet.py --input file1.parquet --seed 42
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm


def shuffle_parquet(input_path: str, output_path: str, seed: int = None):
    """Shuffle rows in a parquet file and save to output path."""
    print(f"Loading: {input_path}")
    df = pd.read_parquet(input_path)

    print(f"  Original shape: {df.shape}")

    # Shuffle
    if seed is not None:
        np.random.seed(seed)
    df = df.sample(frac=1).reset_index(drop=True)

    print(f"  Shuffled shape: {df.shape}")

    # Save
    df.to_parquet(output_path, index=False)
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Shuffle rows in parquet files')
    parser.add_argument('files', nargs='*', help='Parquet files to shuffle (positional)')
    parser.add_argument('--input', '-i', nargs='+', help='Parquet files to shuffle')
    parser.add_argument('--output_dir', '-o', type=str, default=None,
                        help='Output directory. If not specified, overwrites input files')
    parser.add_argument('--suffix', '-s', type=str, default='_shuffled',
                        help='Suffix to add to output filenames (default: _shuffled). '
                             'Use empty string to overwrite: --suffix ""')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # Collect input files
    input_files = []
    if args.files:
        input_files.extend(args.files)
    if args.input:
        input_files.extend(args.input)

    if not input_files:
        parser.error("No input files specified. Use positional arguments or --input")

    # Process each file
    for input_path in tqdm(input_files, desc="Shuffling files"):
        input_path = Path(input_path)

        if not input_path.exists():
            print(f"Warning: File not found: {input_path}")
            continue

        # Determine output path
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / input_path.name
        elif args.suffix:
            output_path = input_path.parent / f"{input_path.stem}{args.suffix}.parquet"
        else:
            # Overwrite input file
            output_path = input_path

        shuffle_parquet(str(input_path), str(output_path), args.seed)

    print("\nDone!")


if __name__ == "__main__":
    main()
