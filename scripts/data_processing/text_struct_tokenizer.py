#!/usr/bin/env python
"""
Multi-GPU structure tokenization script.
Reads text_with_struct_id.parquet, tokenizes structures, produces final parquet.
Output format: text_description + <struct>tokens</struct>
"""

import argparse
import time
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
import pyarrow
import pyarrow.parquet
import torch
import ray
from ray.util.queue import Queue
import logging
import colorlog

# Setup logger
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


@ray.remote(num_gpus=1)
class GPUWorker:
    """GPU worker for structure tokenization."""

    def __init__(
        self,
        tokenizer_ckpt_path: str,
        structure_ckpt_path: str,
        afdb_dir: str,
        pdb_dir: str,
    ):
        import os
        import sys
        # Change to the LLMFolding_tokenizer directory for relative path imports
        os.chdir('/SPXvePFS/share/zzhang/LLMFolding_tokenizer')
        sys.path.insert(0, '/SPXvePFS/share/zzhang/LLMFolding_tokenizer')

        from utils.lf_utils.dist_matrix_tokenizer import DistMatrixTokenizer
        from utils.openfold_utils.io import OpenfoldProtein

        gpu_ids = ray.get_gpu_ids()
        if not gpu_ids:
            raise RuntimeError("No GPU assigned to this actor")
        self.device = torch.device("cuda:0")

        self.tokenizer_ckpt_path = Path(tokenizer_ckpt_path)
        self.structure_ckpt_path = Path(structure_ckpt_path)
        self.afdb_dir = Path(afdb_dir)
        self.pdb_dir = Path(pdb_dir)

        # Lazy initialize
        self.tokenizer = None
        self.OpenfoldProtein = OpenfoldProtein
        self.struct_template = "<|s{token_id:0>4d}|>"

    def _init_tokenizer(self):
        if self.tokenizer is None:
            from utils.lf_utils.dist_matrix_tokenizer import DistMatrixTokenizer
            self.tokenizer = DistMatrixTokenizer(
                tokenizer_ckpt_path=self.tokenizer_ckpt_path,
                structure_ckpt_path=self.structure_ckpt_path,
                map_location=self.device,
            )
            # 显式移动模型到GPU（修复DistMatrixTokenizer的device property bug）
            self.tokenizer.model.to(self.device)
            self.tokenizer.structure_model.to(self.device)
            self.tokenizer._device = self.device

    def _load_protein(self, struct_id: str, struct_source: str):
        """Load protein structure from AFDB or PDB."""
        if struct_source == "afdb":
            # AlphaFold format: AF-{accession}-F1-model_v4
            cif_path = self.afdb_dir / f"{struct_id}.cif.gz"
            if cif_path.exists():
                return self.OpenfoldProtein.from_file(cif_path)
        elif struct_source == "pdb":
            # PDB format: {pdb_id} (4 chars) or {pdb_id}_{chain}
            if '_' in struct_id:
                pdb_id, chain = struct_id.split('_')
                cif_path = self.pdb_dir / f"{pdb_id}.cif"
                if cif_path.exists():
                    return self.OpenfoldProtein.from_file(cif_path, subchains=[chain])
            else:
                cif_path = self.pdb_dir / f"{struct_id}.cif"
                if cif_path.exists():
                    return self.OpenfoldProtein.from_file(cif_path)
        return None

    @torch.no_grad()
    def process_batch(self, batch: List[dict]) -> List[dict]:
        """Process a batch of records, tokenize structures."""
        self._init_tokenizer()

        results = []
        proteins = []
        valid_indices = []

        # Load proteins
        for i, row in enumerate(batch):
            struct_id = row.get("struct_id", "")
            struct_source = row.get("struct_source", "")

            if not struct_id or not struct_source:
                # No structure, keep text only
                results.append({
                    "split": row["split"],
                    "pdb_name": row["pdb_name"],
                    "plddt": row["plddt"],
                    "text": row["text"],
                    "seq_length": row["seq_length"],
                    "struct_length": 0,
                })
                continue

            try:
                protein = self._load_protein(struct_id, struct_source)
                if protein is not None:
                    proteins.append(protein)
                    valid_indices.append(i)
                else:
                    # Structure file not found
                    results.append({
                        "split": row["split"],
                        "pdb_name": row["pdb_name"],
                        "plddt": row["plddt"],
                        "text": row["text"],
                        "seq_length": row["seq_length"],
                        "struct_length": 0,
                    })
            except Exception as e:
                # Error loading structure
                results.append({
                    "split": row["split"],
                    "pdb_name": row["pdb_name"],
                    "plddt": row["plddt"],
                    "text": row["text"],
                    "seq_length": row["seq_length"],
                    "struct_length": 0,
                })

        # Batch tokenize proteins on GPU
        if proteins:
            proteins = [p.to(self.tokenizer.device) for p in proteins]
            out = self.tokenizer(proteins)
            batch_token_ids = out["batch_token_ids"]
            batch_padding_mask = out["batch_padding_mask"]

            for idx, (protein, token_ids, padding_mask) in enumerate(
                zip(proteins, batch_token_ids, batch_padding_mask)
            ):
                orig_idx = valid_indices[idx]
                row = batch[orig_idx]

                # Get valid tokens (remove padding)
                token_ids = token_ids[~padding_mask.bool()]
                struct_text = "".join([
                    self.struct_template.format(token_id=int(t)) for t in token_ids
                ])

                # Combine text with structure
                # Original text ends with </seq>, we append <struct>...</struct>
                original_text = row["text"]
                if original_text.endswith("</seq>"):
                    final_text = original_text + f"<struct>{struct_text}</struct>"
                else:
                    final_text = original_text + f" <struct>{struct_text}</struct>"

                results.append({
                    "split": row["split"],
                    "pdb_name": row["pdb_name"],
                    "plddt": protein.plddt,
                    "text": final_text,
                    "seq_length": row["seq_length"],
                    "struct_length": len(token_ids),
                })

        return results


@ray.remote
class DataProducer:
    """Producer that reads parquet and sends batches to queue."""

    def __init__(self, df: pd.DataFrame, batch_size: int):
        self.df = df
        self.batch_size = batch_size

    def run(self, queue: Queue):
        batch = []
        for _, row in self.df.iterrows():
            batch.append(row.to_dict())
            if len(batch) >= self.batch_size:
                queue.put(batch)
                batch = []

        if batch:
            queue.put(batch)
        queue.put(None)  # Signal done


def main():
    parser = argparse.ArgumentParser(description='Multi-GPU structure tokenization')
    parser.add_argument('--input', type=str, required=True,
                        help='Input parquet with text and struct_id')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for parquet files')
    parser.add_argument('--afdb-dir', type=str,
                        default='/SPXvePFS/share/zzhang/AFDB',
                        help='Directory containing AlphaFold CIF files')
    parser.add_argument('--pdb-dir', type=str,
                        default='/SPXvePFS/share/zzhang/PDB/mmcif',
                        help='Directory containing PDB CIF files')
    parser.add_argument('--tokenizer-ckpt', type=str,
                        default='/SPXvePFS/share/zzhang/LLMFolding_tokenizer/ckpt/v4-epoch=46-val_loss=0.1712.ckpt',
                        help='Tokenizer checkpoint path')
    parser.add_argument('--structure-ckpt', type=str,
                        default='/SPXvePFS/share/zzhang/LLMFolding_tokenizer/ckpt/v3-structure-epoch=04-val_rmsd=0.3359.ckpt',
                        help='Structure model checkpoint path')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--num-gpus', type=int, default=4,
                        help='Number of GPU workers')
    parser.add_argument('--num-producers', type=int, default=2,
                        help='Number of data producers')
    parser.add_argument('--split', type=str, default='text/swissprot',
                        help='Split name for dataset')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit entries for testing (0 for all)')
    parser.add_argument('--merge-shards', action='store_true',
                        help='Merge all shards into single file')
    args = parser.parse_args()

    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    parquet_dir = output_dir / 'parquet'
    parquet_dir.mkdir(parents=True, exist_ok=True)

    processing_start = time.time()

    # Load input parquet
    logger.info(f"[{int(time.time() - processing_start)}s] Loading input from {input_path}...")
    df = pd.read_parquet(input_path)

    if args.limit > 0:
        df = df.head(args.limit)

    # Filter to only entries with struct_id
    df_with_struct = df[df['struct_id'].notna() & (df['struct_id'] != '')]
    logger.info(f"[{int(time.time() - processing_start)}s] Found {len(df_with_struct)} entries with struct_id")

    # Pre-filter: check which structure files exist (faster than checking in producer)
    logger.info(f"[{int(time.time() - processing_start)}s] Pre-filtering entries with existing structure files...")
    afdb_dir = Path(args.afdb_dir)
    pdb_dir = Path(args.pdb_dir)

    def check_exists(row):
        struct_id = row['struct_id']
        struct_source = row['struct_source']
        if struct_source == 'afdb':
            return (afdb_dir / f"{struct_id}.cif.gz").exists()
        elif struct_source == 'pdb':
            pdb_id = struct_id.split('_')[0] if '_' in struct_id else struct_id
            return (pdb_dir / f"{pdb_id}.cif").exists()
        return False

    exist_mask = df_with_struct.apply(check_exists, axis=1)
    df_exists = df_with_struct[exist_mask]
    df_not_exists = df_with_struct[~exist_mask]

    # Save not_exist entries immediately
    if len(df_not_exists) > 0:
        not_exist_path = output_dir / "not_exist.parquet"
        df_not_exists.to_parquet(not_exist_path, index=False)
        logger.info(f"[{int(time.time() - processing_start)}s] Saved {len(df_not_exists)} entries without structure to {not_exist_path}")

    num_total = len(df_exists)
    logger.info(f"[{int(time.time() - processing_start)}s] Processing {num_total} entries with existing structure files")

    # Split data for producers
    num_per_producer = (num_total + args.num_producers - 1) // args.num_producers

    # Setup queue
    queue = Queue(1000)

    # Create producers
    producers = []
    for i in range(args.num_producers):
        start_idx = i * num_per_producer
        end_idx = min((i + 1) * num_per_producer, num_total)
        producer_df = df_exists.iloc[start_idx:end_idx]
        producer = DataProducer.remote(producer_df, args.batch_size)
        producers.append(producer)

    # Create GPU workers
    consumers = []
    for _ in range(args.num_gpus):
        worker = GPUWorker.remote(
            tokenizer_ckpt_path=args.tokenizer_ckpt,
            structure_ckpt_path=args.structure_ckpt,
            afdb_dir=args.afdb_dir,
            pdb_dir=args.pdb_dir,
        )
        consumers.append(worker)

    # Start producers
    logger.info(f"[{int(time.time() - processing_start)}s] Starting {args.num_producers} producers x {args.num_gpus} GPU workers...")
    for producer in producers:
        producer.run.remote(queue)

    # Processing loop
    num_producers_done = 0
    processed_count = 0
    pending_refs = []
    bid = 0

    writer_checkpoint_frequency = 100000
    writer_checkpoint_buffer = 0
    writer_checkpoint_cnt = 0
    parquet_writer = None

    time_start = time.time()

    while True:
        batch = queue.get()
        if batch is None:
            num_producers_done += 1
            if num_producers_done >= args.num_producers:
                break
            continue

        # Submit to consumer (round-robin)
        current_consumer = consumers[bid % args.num_gpus]
        ref = current_consumer.process_batch.remote(batch)
        pending_refs.append(ref)
        bid += 1

        # Process completed results
        if len(pending_refs) >= args.num_gpus:
            ready, pending_refs = ray.wait(pending_refs, num_returns=1)
            result = ray.get(ready[0])
            elapsed = time.time() - time_start
            time_start = time.time()
            processed_count += len(result)

            logger.info(
                f"[{int(time.time() - processing_start)}s][{processed_count}/{num_total}] "
                f"Processed batch of {len(result)} items in {elapsed:.1f}s (pending={len(pending_refs)})"
            )

            # Write to parquet
            table = pyarrow.Table.from_pylist(result)
            if parquet_writer is None:
                parquet_writer = pyarrow.parquet.ParquetWriter(
                    str(parquet_dir / f"shard_{writer_checkpoint_cnt}.parquet"),
                    table.schema,
                    compression="snappy"
                )
            parquet_writer.write_table(table)
            writer_checkpoint_buffer += len(result)

            # Save checkpoint
            if writer_checkpoint_buffer >= writer_checkpoint_frequency:
                parquet_writer.close()
                logger.info(
                    f"[{int(time.time() - processing_start)}s] "
                    f"[shard_{writer_checkpoint_cnt}] finished with {writer_checkpoint_buffer} entries."
                )
                writer_checkpoint_buffer = 0
                writer_checkpoint_cnt += 1
                parquet_writer = None

    # Process remaining pending tasks
    logger.info(f"[{int(time.time() - processing_start)}s] Waiting for remaining GPU tasks...")
    while pending_refs:
        ready, pending_refs = ray.wait(pending_refs, num_returns=1)
        result = ray.get(ready[0])
        elapsed = time.time() - time_start
        time_start = time.time()
        processed_count += len(result)

        logger.info(
            f"[{int(time.time() - processing_start)}s][{processed_count}/{num_total}] "
            f"Processed batch of {len(result)} items in {elapsed:.1f}s (pending={len(pending_refs)})"
        )

        table = pyarrow.Table.from_pylist(result)
        if parquet_writer is None:
            parquet_writer = pyarrow.parquet.ParquetWriter(
                str(parquet_dir / f"shard_{writer_checkpoint_cnt}.parquet"),
                table.schema,
                compression="snappy"
            )
        parquet_writer.write_table(table)
        writer_checkpoint_buffer += len(result)

        if writer_checkpoint_buffer >= writer_checkpoint_frequency:
            parquet_writer.close()
            logger.info(
                f"[{int(time.time() - processing_start)}s] "
                f"[shard_{writer_checkpoint_cnt}] finished with {writer_checkpoint_buffer} entries."
            )
            writer_checkpoint_buffer = 0
            writer_checkpoint_cnt += 1
            parquet_writer = None

    if parquet_writer is not None:
        parquet_writer.close()

    logger.info(
        f"[{int(time.time() - processing_start)}s] "
        f"All batches processed and saved to {parquet_dir}"
    )

    # Merge shards if requested
    if args.merge_shards:
        logger.info(f"[{int(time.time() - processing_start)}s] Merging all parquet shards...")
        all_tables = []
        for shard_path in sorted(parquet_dir.glob("shard_*.parquet")):
            table = pyarrow.parquet.read_table(shard_path)
            all_tables.append(table)

        merged_table = pyarrow.concat_tables(all_tables)
        merged_path = parquet_dir / "dataset.parquet"
        pyarrow.parquet.write_table(merged_table, merged_path, compression="snappy")
        logger.info(
            f"[{int(time.time() - processing_start)}s] "
            f"Merged parquet saved to {merged_path} with {merged_table.num_rows} entries."
        )

        # Shuffle and split
        logger.info(f"[{int(time.time() - processing_start)}s] Shuffling and splitting...")
        df_merged = merged_table.to_pandas()
        df_shuffled = df_merged.sample(frac=1.0, random_state=2025).reset_index(drop=True)
        num_eval = int(len(df_shuffled) * 0.04)
        df_eval = df_shuffled.iloc[:num_eval]
        df_train = df_shuffled.iloc[num_eval:]

        train_path = parquet_dir / "train.parquet"
        eval_path = parquet_dir / "eval.parquet"
        df_train.to_parquet(train_path, index=False)
        df_eval.to_parquet(eval_path, index=False)

        logger.info(f"Train: {train_path} ({len(df_train)} entries)")
        logger.info(f"Eval: {eval_path} ({len(df_eval)} entries)")

    logger.info(f"[{int(time.time() - processing_start)}s] Done!")


if __name__ == "__main__":
    main()
