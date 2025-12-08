import re
import time
import hydra
import logging
import colorlog
from typing import Any, Dict, Optional, List, Text, Tuple, Union, cast
import os
from pathlib import Path
import warnings
import wandb
import pandas as pd

import numpy as np
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn

from omegaconf import OmegaConf, DictConfig
import datasets
from datasets import (
    Features,
    Value,
    Dataset,
    IterableDataset,
    load_dataset,
    interleave_datasets,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2TokenizerFast,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    PreTrainedModel,
    EvalPrediction,
    is_datasets_available,
)
from transformers.generation.configuration_utils import GenerationConfig
from trl import SFTTrainer, SFTConfig
from trl.trainer.utils import ConstantLengthDataset

from utils.dplm_utils.dplm.generate_dplm import generate
from utils.lf_utils.protein_tokenizer import DistMatrixTokenizer
from utils.openfold_utils import OpenfoldProtein
from utils.lf_utils import (
    DistMatrixTokenizer,
    DPLMProteinTokenizer,
    TextTokenizer,
    ProteinProcessor, 
    ItemwiseConstantLengthDataset,
    ExtraColumnCollator,
    UnbatchedModalityLogitsProcessorBase,
    DATASET_SPLIT, DATASET_RAW_ROOT,
    PackingFoldingTrainer,
    dataset,
)

# log color whenrank=0 & silent when rank>0
rank = int(os.environ.get("RANK", 0))
logger = logging.getLogger(__name__)
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s" + f"[rank{rank}]" + "[%(asctime)s][%(levelname)s]" + " %(message)s",
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


# Implementation of SFT trainer
@hydra.main(version_base=None, config_path="./config", config_name="config.yaml")
def sft(config: DictConfig):
    
    start_time = time.time()
    config_dataset, config_lm, config_trainer = config.dataset, config.lm, config.trainer
    config_trainer.output_dir = str(Path(__file__).parent/f'output/checkpoints/{config.name}')
    if (rank := int(os.environ.get("RANK", 0))) == 0:
        wandb.init(project="LLMFolding", name=config.name, config=OmegaConf.to_container(config, resolve=True)) # type: ignore
    elapsed = time.time() - start_time
    logger.info(f'[{int(elapsed)}s] Loaded config ...')
    
    # prepare dataset
    start_time = time.time()
    features = Features({
        "split":            Value("string"),
        "pdb_name":         Value("string"),
        "plddt":            Value("float32"),
        "text":             Value("string"),
        "seq_length":       Value("int64"),
        "struct_length":    Value("int64"),
    })
    
    def make_perpetual(ds):
        def gen():
            epoch = 0
            while True:
                ds.set_epoch(epoch)
                for ex in ds:
                    yield ex
                epoch += 1
        return IterableDataset.from_generator(gen, features=ds.features)
    
    # dataset_eval = load_dataset('parquet', streaming=False, split='train', data_files=config_dataset.eval)
    # for quick evaluation, resitrict to 1000 samples of each eval dataset
    dataset_eval_small = []
    for eval_ds in config_dataset.eval:
        ds = load_dataset('parquet', streaming=False, split='train', data_files=eval_ds, features=features)
        ds = ds.select(range(min(1000, len(ds)))) # type: ignore
        dataset_eval_small.append(ds)
    dataset_eval = datasets.concatenate_datasets(dataset_eval_small)
    
    dataset_train = interleave_datasets(
        datasets=[
            make_perpetual(
                load_dataset('parquet', streaming=True, split='train', data_files=fpath, features=features).shuffle(seed=2025)
            )
            for fpath in config_dataset.train
        ],    
        probabilities=config_dataset.weight,
        seed=2025,
    )
    elapsed = time.time() - start_time
    logger.info(f'[{int(elapsed)}s] Loaded dataset ...\n- {config_dataset.train}\n- {config_dataset.eval}')
    
    # prepare qwen3 tokenizer
    start_time = time.time()
    protein_tokenizer = {
        "dist": DistMatrixTokenizer,
        "dplm": DPLMProteinTokenizer,
    }[str(config_dataset.type)].get_instance()
    qwen2_tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained(config_lm.model_dir)
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
    elapsed = time.time() - start_time
    logger.info(f'[{int(elapsed)}s] Loaded and updated tokenizers ...')
    
    # prepare qwen3 model
    start_time = time.time()
    qwen3_model = AutoModelForCausalLM.from_pretrained(
        config_lm.model_dir,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    qwen3_model.resize_token_embeddings(len(qwen2_tokenizer))
    elapsed = time.time() - start_time
    logger.info(f'[{int(elapsed)}s] Loaded and updated model ...')
    
    
    # prepare trainer
    # HINT: for packing we have to use <|endoftext|> rather than <|im_end|>
    start_time = time.time()
    eod_token, eos_token = qwen2_tokenizer.pad_token, qwen2_tokenizer.eos_token
    qwen2_tokenizer.eos_token = eod_token
    qwen2_tokenizer.eos_token_id = qwen2_tokenizer.pad_token_id
    protein_processor = ProteinProcessor(qwen2_tokenizer, protein_tokenizer)
    sft_trainer = PackingFoldingTrainer(
        processor=protein_processor,
        model=qwen3_model,
        tokenizer=qwen2_tokenizer,
        args=SFTConfig(**config_trainer),
        train_dataset=dataset_train, # type: ignore
        eval_dataset=dataset_eval,   # type: ignore
        eval_packing=False,
        eval_collator=ExtraColumnCollator(),
        compute_metrics=PackingFoldingTrainer.compute_metrics,
    )
    sft_trainer.train() # type: ignore
    
    elapsed = time.time() - start_time
    logger.info(f'[{int(elapsed)}s] Finished SFT training ...')


if __name__ == "__main__":
    sft()
