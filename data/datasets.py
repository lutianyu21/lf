from typing import Dict, List
import torch
import torch.utils
import torch.utils.data
import pandas as pd
from pathlib import Path

from utils.dplm_utils import DPLMProcessor, dplm_tokenizer
from utils.progen2_utils import progen2_merged_tokenizer
from utils.openfold_utils import OpenfoldEntity


__all__ = ['DPLMCollator']

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
        