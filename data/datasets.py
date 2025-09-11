from typing import Any, Dict, List
from huggingface_hub import Padding
import torch
import torch.utils
import torch.utils.data
import pandas as pd
from pathlib import Path

from utils.dplm_utils import DPLMProcessor, dplm_tokenizer
from utils.progen2_utils import progen2_merged_tokenizer
from utils.openfold_utils import OpenfoldEntity


__all__ = ["DPLMCollator", "TextCollator"]

















class DPLMCollator:
    """train format: <bos><boseq>xxx<eoseq><bostruct>xxx<eostruct><eos>
        eval format: <bos><boseq>xxx<eoseq><bostruct>
    """
    def __init__(self,
        processor: DPLMProcessor,
        mode: str = 'train',
        train_task: str = 'folding_lm',
        eval_task: str = 'folding_lm'
    ):
        self.mode = mode
        self.train_task = train_task
        self.eval_task = eval_task
        self.processor = processor
        self.device = processor.structure_tokenizer.device
    
    def __call__(self, batch: List[Dict[str, str]]):
        protein_structure_list = [OpenfoldEntity.from_file(Path(b['mmcif_path'])) for b in batch]
        protein_text_list = [str(p) for p in protein_structure_list]
        batch_train, batch_eval = self.processor(
            protein_text=protein_text_list,
            protein_struct=protein_structure_list,
            train_task=self.train_task,
            eval_task=self.eval_task,
            return_tensors='pt',
            padding=True
        )
        if self.mode == 'train':
            batch_train['labels'] = batch_train['input_ids'].clone() # [B, L]
            return batch_train
        elif self.mode == 'eval':
            batch_eval['labels'] = batch_train['input_ids'].clone() # [B, L] label's longer than input_ids
            return batch_eval



import json
import math
import os
import pickle as pkl
from typing import Iterable, Sequence, TypeVar, Union

import numpy as np
import torch
import torch.distributed as dist
from datasets import load_dataset

from transformers import EsmTokenizer








class DPLMCollater(object):
    """Wrapped for OA Collater to operate on ESM w/ ESM alphabet and batch
    converter/tokens."""

    def __init__(self, tokenizer_path=None):
        # by default we use the EsmTokenizer and the esm vocab.
        # if you want to use the different vocab,
        # please set the vocab path to the tokenizer_path
        if tokenizer_path is None:
            self.alphabet = EsmTokenizer.from_pretrained(
                "facebook/esm2_t30_150M_UR50D"
            )
        else:
            self.alphabet = EsmTokenizer.from_pretrained(tokenizer_path)

    def __call__(self, sequences):
        if len(list(zip(*sequences))) == 0:
            print("list idx error!")
            print(sequences)
        input_data = sequences
        batch = self.alphabet.batch_encode_plus(
            input_data,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        )

        batch = {
            "input_ids": batch["input_ids"],
            "input_mask": batch["attention_mask"].bool(),
            "targets": batch["input_ids"].clone(),
        }
        return batch






