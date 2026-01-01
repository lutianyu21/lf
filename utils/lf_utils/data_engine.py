import time
from typing import List
import pyarrow
import ray
from pathlib import Path
import colorlog
import logging
import pandas as pd
import torch
from transformers import AutoTokenizer, Qwen2TokenizerFast

from ray.util.queue import Queue

from utils.openfold_utils.io import OpenfoldProtein
from .protein_tokenizer import DPLMProteinTokenizer, DistMatrixTokenizerV2, DistMatrixTokenizerV3
from .protein_processor import ProteinProcessor


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



@ray.remote(num_gpus=1)
class GPUWorker:
    def __init__(self, dataset_name: str, tokenizer_name: str, tokenizer_ckpt: str, structure_ckpt: str, qwen_tokenizer_path: str):
        gpu_ids = ray.get_gpu_ids()
        if not gpu_ids:
            raise RuntimeError("No GPU assigned to this actor")
        self.device = torch.device(f"cuda:0")
        # lazy initilaize
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.tokenizer_ckpt = tokenizer_ckpt
        self.structure_ckpt = structure_ckpt
        self.qwen_tokenizer_path = qwen_tokenizer_path
        self.processor = None

    @torch.no_grad()
    def fn(self, batch: List[OpenfoldProtein]):
        if self.processor is None:
            if self.tokenizer_name == "dplm":
                protein_tokenizer = DPLMProteinTokenizer.get_instance()
            elif self.tokenizer_name == "dist2":
                protein_tokenizer = DistMatrixTokenizerV2(
                    tokenizer_ckpt_path=self.tokenizer_ckpt,
                    structure_ckpt_path=self.structure_ckpt,
                    map_location="cpu",
                )
            elif self.tokenizer_name == "dist3":
                protein_tokenizer = DistMatrixTokenizerV3(
                    tokenizer_ckpt_path=self.tokenizer_ckpt,
                    structure_ckpt_path=self.structure_ckpt,
                    map_location="cpu",
                )
            else:
                raise ValueError(f"Unknown tokenizer_name: {self.tokenizer_name}")
            
            qwen2_tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained(
                self.qwen_tokenizer_path
            )
            qwen2_tokenizer.padding_side = "right"
            qwen2_tokenizer.truncation_side = "right"
            qwen2_tokenizer.boseq_token = '<seq>'
            qwen2_tokenizer.eoseq_token = '</seq>'
            qwen2_tokenizer.bostruct_token = '<struct>'
            qwen2_tokenizer.eostruct_token = '</struct>'
            qwen2_tokenizer.struct_regex = r"<\|s(\d{4})\|>"
            qwen2_tokenizer.struct_template = "<|s{token_id:0>4d}|>"
            qwen2_tokenizer.struct_vsz = protein_tokenizer.vsz
            qwen2_tokenizer.add_special_tokens({
                'additional_special_tokens': \
                [qwen2_tokenizer.boseq_token, qwen2_tokenizer.eoseq_token, qwen2_tokenizer.bostruct_token, qwen2_tokenizer.eostruct_token] + \
                [qwen2_tokenizer.struct_template.format(token_id=i) for i in range(qwen2_tokenizer.struct_vsz)] # type: ignore
            }, replace_additional_special_tokens=False)
            qwen2_tokenizer.seq_vocab_ids = qwen2_tokenizer.convert_tokens_to_ids([
                ' A', ' B', ' C', ' D', ' E', ' F', ' G', ' H', ' I', ' K', ' L', ' M', ' N', ' O', ' P', ' Q', ' R', ' S', ' T', ' U', ' V', ' W', ' X', ' Y', ' Z'    
            ])
            qwen2_tokenizer.struct_vocab_ids = qwen2_tokenizer.convert_tokens_to_ids([
                qwen2_tokenizer.struct_template.format(token_id=i) for i in range(qwen2_tokenizer.struct_vsz)
            ])
            qwen2_tokenizer.boseq_token_id = qwen2_tokenizer.convert_tokens_to_ids(qwen2_tokenizer.boseq_token)
            qwen2_tokenizer.eoseq_token_id = qwen2_tokenizer.convert_tokens_to_ids(qwen2_tokenizer.eoseq_token)
            qwen2_tokenizer.bostruct_token_id = qwen2_tokenizer.convert_tokens_to_ids(qwen2_tokenizer.bostruct_token)
            qwen2_tokenizer.eostruct_token_id = qwen2_tokenizer.convert_tokens_to_ids(qwen2_tokenizer.eostruct_token)
            
            protein_processor = ProteinProcessor(qwen2_tokenizer, protein_tokenizer)
            self.processor = protein_processor.to(self.device)
        return self.processor.preprocess_dataset(self.dataset_name,batch, verbose=False)

@ray.remote
class RawFileWorker:
    def __init__(self, df: pd.DataFrame, structure_dir: Path, bsz: int, max_seq_length: int = 4096):
        self.bq = df
        self.structure_dir = Path(structure_dir)
        self.bsz = bsz
        self.max_seq_length = max_seq_length
        self.skipped_count = 0

    def _find_structure_file(self, accession: str) -> Path:
        """查找蛋白质结构文件，支持多种格式和命名规则

        支持的格式:
        - AF-xxx: AlphaFold 预测结构
        - 1kgf%1: PDB 实验结构 (pdb_id%entity_id)，%后面是 entity_id
        - 7v8t@A: 按 auth_asym_id 提取链

        注意: io.py 的 _gemmi_parser 会解析文件名中的 % 和 @ 符号来确定如何提取链
        - % 表示按 entity_id 提取
        - @ 表示按 auth_asym_id 提取
        """
        # 对于 PDB 格式 (1kgf%1)，提取 PDB ID 来查找文件
        # 但返回的路径需要包含 %entity_id 以便 io.py 正确解析
        if '%' in accession:
            pdb_id = accession.split('%')[0]
            entity_id = accession.split('%')[1]
            for suffix in ['.cif.gz', '.cif', '.pdb.gz', '.pdb']:
                path = self.structure_dir / f'{pdb_id}{suffix}'
                if path.exists():
                    # 返回带 %entity_id 的虚拟路径，io.py 会解析它
                    return self.structure_dir / f'{pdb_id}%{entity_id}{suffix}'
            raise FileNotFoundError(f'Cannot find structure file for: {accession} in {self.structure_dir}')

        # 对于其他格式 (AlphaFold, @auth_asym_id)
        accession_variants = [
            accession,
            accession.replace('@', '-'),
            accession.replace('@', '_'),
        ]

        for acc in accession_variants:
            for suffix in ['.cif.gz', '.cif', '.pdb.gz', '.pdb']:
                path = self.structure_dir / f'{acc}{suffix}'
                if path.exists():
                    return path
        raise FileNotFoundError(f'Cannot find structure file for: {accession} in {self.structure_dir}')

    def fn(self, queue: Queue):
        batch = []
        for _, row in self.bq.iterrows():
            accession = row['unirefAccession']
            try:
                path = self._find_structure_file(accession)
                # io.py 的 from_file 会根据路径中的 % 或 @ 自动提取对应的链/entity
                protein = OpenfoldProtein.from_file(path)

                # 检查序列长度，跳过过大的蛋白
                seq_len = len(protein.residue_aatype)
                if seq_len > self.max_seq_length:
                    self.skipped_count += 1
                    logger.warning(f"Skipped {accession} (seq_len={seq_len} > max={self.max_seq_length})")
                    continue
                # 使用原始 accession 作为 entry，保持格式一致性
                protein.entry = accession
                batch.append(protein)
            except Exception as e:
                self.skipped_count += 1
                logger.warning(f"Failed to load {accession}: {e}")
                continue
            if len(batch) >= self.bsz:
                queue.put(batch)
                batch = []
        if batch:
            queue.put(batch)
        if self.skipped_count > 0:
            logger.info(f"Total skipped proteins: {self.skipped_count}")
        queue.put(None)  # signal done


class DataEngine:
    
    ### process functions ###
    @classmethod
    def pipe(
        cls,
        dataset_dir:    Path,
        bq_path:        Path,
        structure_dir:  Path,                           # 蛋白质结构文件目录
        bsz:            int,
        num_consumers:  int,
        num_producers:  int,
        tokenizer_name: str = "dist3",                  # dplm / dist2 / dist3
        tokenizer_ckpt: str = "",                       # discrete tokenizer checkpoint path
        structure_ckpt: str = "",                       # structure head checkpoint path
        qwen_tokenizer_path: str = "/SPXvePFS/model/Qwen3-0.6B",  # qwen tokenizer path
        dataset_name:   str = "p2s/unicluster40",       # will be mapped to feature['split']
        ops:            List[str] = ['merge', 'shuffle', 'split'],
        max_seq_length: int = 4096,                     # 最大序列长度，超过的将被跳过
    ):
        # load bigtable
        time_start = time.time()
        logger.info(f"Loading bigtable from {bq_path} ...")
        bq_df = pd.read_parquet(bq_path)
        num_items_total = len(bq_df)
        num_items_producer = (num_items_total + num_producers - 1) // num_producers
        logger.info(f"{int(time.time() - time_start)}s Loaded bigtable with {num_items_total} items.")
        time_start = time.time()
        
        # n producer x m consumer system
        # load and split bq.parquet, each producer will handle a subset
        logger.info(f"Starting {num_producers} producers x {num_consumers} consumers ...")
        queue = Queue(1000)
        producers = [
            RawFileWorker.remote(bq_df[
                i * num_items_producer : min((i + 1) * num_items_producer, len(bq_df))
            ], structure_dir, bsz, max_seq_length) for i in range(num_producers)
        ]
        num_consumers_max = num_consumers - 1
        num_producers_done = 0
        consumers = [
            GPUWorker.remote(dataset_name, tokenizer_name, tokenizer_ckpt, structure_ckpt, qwen_tokenizer_path) for _ in range(num_consumers)
        ]
        
        # prepare directory
        tmp_dir = dataset_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        category, name = dataset_name.split('/')[0], dataset_name.split('/')[-1]
        final_dir = dataset_dir / {
            'p2s':          'folding',
            'psps':         'cfolding',
            's':            'structure',
            'p':            'sequence',
        }[category] / {
            'unicluster40': 'unicluster40',
            'uniref50':     'uniref50',
            'rcsb':         'rcsb',
            'cameo2022':    'benchmark',
            'casp15':       'benchmark',
            'casp16':       'benchmark',
        }[name]
        final_dir.mkdir(parents=True, exist_ok=True)
        
        # start producers first
        for w in producers: w.fn.remote(queue)  # type: ignore
        logger.info(f"[{int(time.time() - time_start)}s] All producers started.")
        time_start = time.time()
        
        # start consumption loop
        time_start = time.time()
        writer_checkpoint_frequency = 100000
        writer_checkpoint_buffer = 0
        writer_checkpoint_cnt = 0
        total_processed = 0  # 已处理总数
        parquet_writer, bid, pending_refs = None, 0, []
        # loop until all producers are done
        while True:
            batch = queue.get()
            if batch is None:
                num_producers_done += 1
                if num_producers_done >= num_producers: break
                continue

            # submit to a consumer, round-robin
            current_consumer = consumers[bid % num_consumers]
            ref = current_consumer.fn.remote(batch)  # type: ignore
            pending_refs.append(ref)
            bid += 1

            if len(pending_refs) >= num_consumers_max:
                ready, pending_refs = ray.wait(pending_refs, num_returns=1)
                result = ray.get(ready[0])
                total_processed += len(result)
                progress_pct = total_processed / num_items_total * 100
                logger.info(f"[{int(time.time() - time_start)}s] Processed {total_processed}/{num_items_total} ({progress_pct:.1f}%) (pending={len(pending_refs)})")
                time_start = time.time()

                # write to parquet
                table = pyarrow.Table.from_pylist(result)
                if parquet_writer is None:
                    parquet_writer = pyarrow.parquet.ParquetWriter( # type: ignore
                        str(tmp_dir / f"shard_{writer_checkpoint_cnt}.parquet"),
                        table.schema,
                        compression="snappy"
                    )
                parquet_writer.write_table(table)
                writer_checkpoint_buffer += len(result)
                
                # save checkpoint
                if writer_checkpoint_buffer >= writer_checkpoint_frequency:
                    parquet_writer.close()
                    logger.info(f"[shard_{writer_checkpoint_cnt}] finished with {writer_checkpoint_buffer} entries.")
                    writer_checkpoint_buffer = 0
                    writer_checkpoint_cnt += 1
                    parquet_writer = None
            
        logger.info("Waiting for remaining GPU tasks...")
        while pending_refs:
            ready, pending_refs = ray.wait(pending_refs, num_returns=1)
            result = ray.get(ready[0])
            total_processed += len(result)
            progress_pct = total_processed / num_items_total * 100
            logger.info(f"[{int(time.time() - time_start)}s] Processed {total_processed}/{num_items_total} ({progress_pct:.1f}%) (pending={len(pending_refs)})")
            time_start = time.time()
            
            # write to parquet
            table = pyarrow.Table.from_pylist(result)
            if parquet_writer is None:
                parquet_writer = pyarrow.parquet.ParquetWriter( # type: ignore
                    str(tmp_dir / f"shard_{writer_checkpoint_cnt}.parquet"),
                    table.schema,
                    compression="snappy"
                )
            parquet_writer.write_table(table)
            writer_checkpoint_buffer += len(result)
            
            if writer_checkpoint_buffer >= writer_checkpoint_frequency:
                parquet_writer.close()
                logger.info(f"[shard_{writer_checkpoint_cnt}] finished with {writer_checkpoint_buffer} entries.")
                writer_checkpoint_buffer = 0
                writer_checkpoint_cnt += 1
                parquet_writer = None
        
        if parquet_writer is not None:
            parquet_writer.close()
        logger.info(f"All batches are processed and saved to {tmp_dir}")
        
        # operation list:
        # ['merge', 'shuffle', 'split', 'extract', 'clean']
        if not 'merge' in ops: return
        logger.info("Merging parquet shards ...")
        all_tables = []
        for shard_path in sorted(tmp_dir.glob("shard_*.parquet")):
            table = pyarrow.parquet.read_table(shard_path) # type: ignore
            all_tables.append(table)
        merged_table = pyarrow.concat_tables(all_tables)
        merged_path = final_dir / f"{name}.parquet"
        pyarrow.parquet.write_table(merged_table, merged_path, compression="snappy") # type: ignore
        logger.info(f"[{int(time.time() - time_start)}s] Merged parquet saved to {merged_path} with {merged_table.num_rows} entries.")
        time_start = time.time()
        
        if not 'shuffle' in ops: return
        logger.info("Shuffling and splitting merged parquet ...")
        df_merged = merged_table.to_pandas()
        df_shuffled = df_merged.sample(frac=1.0, random_state=2025).reset_index(drop=True)
        df_shuffled.to_parquet(merged_path, index=False)
        logger.info(f"[{int(time.time() - time_start)}s] Shuffled parquet saved to {merged_path} with {len(df_shuffled)} entries.")
        time_start = time.time()
        
        if not 'split' in ops: return
        num_eval = int(len(df_shuffled) * 0.04)
        df_train = df_shuffled.iloc[num_eval:]
        df_eval = df_shuffled.iloc[0:num_eval]
        train_path = final_dir / "train.parquet"
        eval_path = final_dir / "eval.parquet"
        df_train.to_parquet(train_path, index=False)
        df_eval.to_parquet(eval_path, index=False)
        logger.info(f"[{int(time.time() - time_start)}s] Train parquet saved to {train_path} with {len(df_train)} entries.")
        logger.info(f"[{int(time.time() - time_start)}s] Eval parquet saved to {eval_path} with {len(df_eval)} entries.")
        time_start = time.time()  
