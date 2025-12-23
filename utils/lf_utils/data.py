import logging
import os
from typing import Any, Callable, Dict, Iterable, List

import math
import warnings
import random
import colorlog
import numpy as np
from sympy import re
import torch
import torch.utils
import torch.utils.data
import torch.distributed as dist
from transformers import Qwen2TokenizerFast
from trl.trainer.utils import ConstantLengthDataset

from .constant import DATASET_SPLIT
from .protein_processor import ProteinProcessor

__all__ = [
    'ExtraColumnCollator', 'ItemwiseConstantLengthDataset',
]


class ExtraColumnCollator:
    # handle extra columns in multi-modal dataset
    def __init__(self, tokenizer: Any, **kwargs):
        self._tokenizer = tokenizer
        self._seed = kwargs.get('seed', 42)
        self._cropping = kwargs.get('cropping', False)
        self._cropping_size = kwargs.get('cropping_size', 1024)
        self._concatenation = kwargs.get('concatenation', False)
        self._concatenation_size = kwargs.get('concatenation_size', 2)
        self._concatenation_ratio = kwargs.get('concatenation_ratio', 0.0)
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Any:
        if len(batch) > 1: 
            raise NotImplementedError("ExtraColumnCollator only accepts batch size of 1")
        
        organized_batch = dict()
        for k, v in batch[0].items():
            if isinstance(v, int) or isinstance(v, float):
                organized_batch[k] = torch.tensor([item[k] for item in batch])
            # TODO batching & padding is required to support batch-evaluation
            elif isinstance(v, list):
                organized_batch[k] = torch.tensor([item[k] for item in batch])
            elif isinstance(v, torch.Tensor):
                organized_batch[k] = torch.stack([item[k] for item in batch], dim=0)
            elif isinstance(v, str):
                organized_batch[k] = [item[k] for item in batch]
            else:
                raise NotImplementedError(f"Type {type(v)} of key {k} is not supported yet.")
        
        ## cropping fn ##
        if organized_batch['split'][0] in ['p/uniref50'] and self._cropping:
            # before: <seq>...........</seq>
            # after:  <seq>..</seq> (cropped)
            seq_start = int(torch.where(organized_batch['input_ids'][0] == self._tokenizer.boseq_token_id)[0][-1].item())
            seq_end   = int(torch.where(organized_batch['input_ids'][0] == self._tokenizer.eoseq_token_id)[0][-1].item())
            if (L := seq_end - seq_start + 1) > self._cropping_size + 2:
                rng = np.random.default_rng(self._seed)
                cropping_start = rng.integers(seq_start + 1, seq_end - self._cropping_size + 1) # [)
                organized_batch['input_ids'] = torch.cat([
                    organized_batch['input_ids'][0][:seq_start+1],
                    organized_batch['input_ids'][0][cropping_start : cropping_start + self._cropping_size],
                    organized_batch['input_ids'][0][seq_end:],
                ], dim=-1).unsqueeze(0)
                organized_batch['attention_mask'] = torch.ones_like(organized_batch['input_ids'])
                organized_batch['labels'] = organized_batch['input_ids'].clone()
                assert organized_batch['input_ids'][0].shape[0] == self._cropping_size + 2, \
                    f"{organized_batch['input_ids'][0].shape[0]} != {self._cropping_size + 2}, expcted: L' = 1 + min(L, cropping_size) + 1"
        
        ## masking fn ## (not required for evaluation, only for training)
        
        ## concatenation fn ##
        if organized_batch['split'][0] in ['p2s/unicluster40', 'p2s/afdb_swissprot', 'p2s/rcsb', 'cameo2022', 'casp15', 'casp16', 'dev'] and self._concatenation:
            # before: <seq>....</seq><struct>....</struct>
            # after:  <seq>....</seq><struct>....</struct><seq>....</seq><struct>....</struct>
            M, ratio, tmp = self._concatenation_size, self._concatenation_ratio, []
            for i in range(M - 1):
                # generate noise version <seq>....</seq><struct>....</struct>
                # replcae `ratio` structure tokens with random structure tokens
                copy_input_ids = organized_batch['input_ids'][0].clone()
                struct_start = int(torch.where(copy_input_ids == self._tokenizer.bostruct_token_id)[0][-1].int().item())
                struct_end   = int(torch.where(copy_input_ids == self._tokenizer.eostruct_token_id)[0][-1].int().item())
                num_struct_tokens = struct_end - struct_start + 1 - 2
                if (N := int(num_struct_tokens * ratio)) > 0:
                    rng = np.random.default_rng(seed=self._seed + i) # for reproducibility
                    replace_indices = rng.choice(np.arange(struct_start + 1, struct_end), size=N, replace=False)
                    replace_values = rng.choice(self._tokenizer.struct_vocab_ids, size=N, replace=True)
                    copy_input_ids[replace_indices] = torch.tensor(replace_values, dtype=copy_input_ids.dtype)
                tmp.append(copy_input_ids)
            tmp.append(organized_batch['input_ids'][0])
            organized_batch['input_ids'] = torch.cat(tmp, dim=0).unsqueeze(0)
            organized_batch['attention_mask'] = torch.ones_like(organized_batch['input_ids'])
            organized_batch['labels'] = organized_batch['input_ids'].clone()
            assert organized_batch['input_ids'][0].shape[0] == (M * tmp[0].shape[0]), \
                f"{organized_batch['input_ids'][0].shape[0]} != {M * tmp[0].shape[0]}, expcted: L' = M * L"
        return organized_batch


class ItemwiseConstantLengthDataset(ConstantLengthDataset):
    
    def __init__(self, *args, **kwargs):
        # pop extra arguments from kwargs
        self._cropping = kwargs.pop('cropping', False)
        self._cropping_size = kwargs.pop('cropping_size', 1024)
        self._masking = kwargs.pop('masking', False)
        self._concatenation = kwargs.pop('concatenation', False)
        self._concatenation_size = kwargs.pop('concatenation_size', 2)
        self._concatenation_ratio = kwargs.pop('concatenation_ratio', 0.0)
        super().__init__(*args, **kwargs)
    
    # unlike ConstantLengthDataset, this class will make sure that each yielded
    # example starts as a complete item from the original dataset(still packing multiple items)
    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    tmp = next(iterator)
                    buffer.append(self.formatting_func(tmp))
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        warnings.warn("The dataset reached end and the iterator is reset to the start.")
                    else:
                        more_examples = False
                        break
                
            # for items in buffer, apply cropping & masking if needed
            feature = {
                'input_ids':    [],
                'labels':       [],
            }
            for it in buffer:
                f = self.tokenizer(it, add_special_tokens=self.add_special_tokens, truncation=False, text_target=it)
                if self._cropping:
                    f = self._apply_cropping(f)
                if self._concatenation:
                    f = self._apply_concatenation(f)
                if self._masking:
                    f = self._apply_masking(f)
                feature['input_ids'].append(f['input_ids'])
                feature['labels'].append(f['labels'])
            
            # Original Implementation:
            # |-----d1-----|----d2----|---d3--
            # -d3|----d4----|-----d5-----|-d6-
            # all_token_ids = []
            # for tokenized_input in tokenized_inputs:
            #     if self.append_concat_token:
            #         tokenized_input = tokenized_input + [self.concat_token_id]
            #     all_token_ids.extend(tokenized_input)
            # examples = []
            # for i in range(0, len(all_token_ids), self.seq_length):
            #     input_ids = all_token_ids[i : i + self.seq_length]
            #     if len(input_ids) == self.seq_length:
            #         examples.append(input_ids)
            # if self.shuffle:
            #     random.shuffle(examples)
            # for example in examples:
            #     self.current_size += 1
            #     yield {
            #         "input_ids": torch.LongTensor(example),
            #         "labels": torch.LongTensor(example),
            #     }
            
            # Modified Implementation:
            # |-----d1-----|----d2----|---d3--
            # |----d4----|-----d5-----|-d6-|--
            examples = [] # stores (input_ids, labels)
            lines_input, lines_labels = [], []
            for tokenized_input, tokenized_labels in zip(feature['input_ids'], feature['labels']):
                if self.append_concat_token:
                    tokenized_input = tokenized_input + [self.concat_token_id]
                    tokenized_labels = tokenized_labels + [self.concat_token_id]
                lines_input.extend(tokenized_input)
                lines_labels.extend(tokenized_labels)
                if len(lines_input) >= self.seq_length:
                    examples.append((lines_input[:self.seq_length], lines_labels[:self.seq_length]))
                    lines_input = [] # truncate the remaining
                    lines_labels = []
            if self.shuffle:
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                input_ids, labels = example
                yield {
                    "input_ids": torch.LongTensor(input_ids),
                    "labels": torch.LongTensor(labels),
                }
                
                
    def _is_sequence_dataset(self, feature: Dict[str, List[int]]) -> bool:
        return  feature['input_ids'][0]  == self.tokenizer.boseq_token_id \
            and feature['input_ids'][-1] == self.tokenizer.eoseq_token_id
    
    def _is_structure_dataset(self, feature: Dict[str, List[int]]) -> bool:
        return  feature['input_ids'][0]  == self.tokenizer.bostruct_token_id \
            and feature['input_ids'][-1] == self.tokenizer.eostruct_token_id
            
    def _is_folding_dataset(self, feature: Dict[str, List[int]]) -> bool:
        return not self._is_sequence_dataset(feature) and not self._is_structure_dataset(feature)        
    
    
    # Reference:
    # - (dplm) https://arxiv.org/pdf/2402.18567
    def _apply_cropping(self, feature: Dict[str, List[int]]) -> Dict[str, List[int]]:
        # genetic sequences are usually longer than the model max_length
        if not self._is_sequence_dataset(feature):
            return feature
        
        if (L := len(feature['input_ids'])) <= 1 + self._cropping_size + 1:
            return feature
        cropping_start = random.randint(1, L - self._cropping_size - 1)
        feature['input_ids'] = feature['input_ids'][:1] \
                             + feature['input_ids'][cropping_start : cropping_start + self._cropping_size] \
                             + feature['input_ids'][-1:]
        feature['labels'] = feature['input_ids'].copy()
        return feature
    
    # Reference:
    # - (poet) 
    def _apply_concatenation(self, feature: Dict[str, List[int]]) -> Dict[str, List[int]]:
        if not self._is_folding_dataset(feature):
            return feature
        
        # repeat M times & replcae ratio(%) structure tokens with random structure tokens:
        # <seq>....</seq><struct>....</struct>
        # <seq>....</seq><struct>....</struct> | <seq>....</seq><struct>....</struct> | ...
        L, M, ratio, tmp = len(feature['input_ids']), self._concatenation_size, self._concatenation_ratio, []
        struct_start = L - 1 - feature['input_ids'][::-1].index(self.tokenizer.bostruct_token_id)
        struct_end   = L - 1 - feature['input_ids'][::-1].index(self.tokenizer.eostruct_token_id)
        for _ in range(M - 1):
            input_ids = feature['input_ids'].copy()
            # HINT second occurrence hacking can be implemented by setting ratio=0.0
            if (num := int((struct_end - struct_start + 1 - 2) * ratio)) > 0:
                rng = np.random.default_rng()
                replace_indices = rng.choice(np.arange(struct_start + 1, struct_end), size=num, replace=False)
                replace_values  = rng.choice(self.tokenizer.struct_vocab_ids, size=num, replace=True)
                # numpy indexing trick
                input_ids = np.array(input_ids)
                input_ids[replace_indices] = replace_values.tolist()
                input_ids = input_ids.tolist()
            tmp += input_ids
        feature['input_ids'] = tmp + feature['input_ids']
        feature['labels'] = feature['input_ids'].copy()
        return feature
    
    # Reference:
    # - (I2T, T2I)
    def _apply_masking(self, feature: Dict[str, List[int]]) -> Dict[str, List[int]]:
        if not self._is_folding_dataset(feature):
            return feature
        # do not compute loss on prompt region:
        # <seq>....</seq><struct>....</struct><seq>....</seq><struct>....</struct>
        # -100 .... -100 .... -100 .... -100 .... -100 ....  <struct>....</struct>
        L = len(feature['input_ids'])
        struct_start = L - 1 - feature['input_ids'][::-1].index(self.tokenizer.bostruct_token_id)
        struct_end   = L - 1 - feature['input_ids'][::-1].index(self.tokenizer.eostruct_token_id)
        # only modify labels
        tmp = [-100] * L
        tmp[struct_start : struct_end + 1] = feature['labels'][struct_start : struct_end + 1]
        feature['labels'] = tmp
        return feature
    

class TextCollator:
    # handle pure-text dataset
    def __init__(self, processor: ProteinProcessor, eval_mode: bool = False):
        self.processor = processor
        self.eval_mode = eval_mode
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Any:
        # extra feature
        feature = dict()
        for k in batch[0].keys():
            if not k in ['input_ids', 'attention_mask', 'labels', 'prompt']:
                feature[k] = list(map(lambda x: x[k], batch))
        
        # main feature
        text_feature: Dict[str, torch.Tensor] = self.processor.tokenizer(
            list(map(lambda x: x["text"], batch)),
            return_tensors='pt',
            padding=True
        ) # type: ignore
        input_ids = text_feature['input_ids']
        attention_mask = text_feature['attention_mask']
        labels = text_feature['input_ids'].clone()
        labels = torch.where(attention_mask, labels, -100)
        
        prompt_feature = None
        if self.eval_mode:
            prompt_feature = self.processor.tokenizer(
                list(map(lambda x: x["prompt"], batch)),
                return_tensors='pt',
                padding=True
            ) # type: ignore
            # overite input_ids and attention_mask
            input_ids = prompt_feature['input_ids']
            attention_mask = prompt_feature['attention_mask']
            
        return feature.update(dict(
            labels=labels,
            input_ids=input_ids,
            attention_mask=attention_mask,
        ))

class SortishSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        sequence_lengths: Iterable,
        bucket_size: int,
        num_replicas: int = 1,
        rank: int = 0,
    ):
        if dist.is_available():
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
        self.data = np.argsort(sequence_lengths) # type: ignore
        self.num_replicas = num_replicas
        self.num_samples = int(math.ceil(len(self.data) * 1.0 / self.num_replicas))
        self.bucket_size = bucket_size
        n_buckets = int(np.ceil(len(self.data) / self.bucket_size))
        self.data = [
            self.data[i * bucket_size : i * bucket_size + bucket_size] for i in range(n_buckets)
        ]
        self.rank = rank
        self.epoch = 0
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        np.random.seed(self.epoch)
        for bucket in self.data:
            np.random.shuffle(bucket)
        np.random.shuffle(self.data)
        indices = [item for sublist in self.data for item in sublist]
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        start = self.rank * self.num_samples
        end = start + self.num_samples
        indices = indices[start:end]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
 
class ApproxBatchSampler(torch.utils.data.BatchSampler):
    def __init__(
        self,
        sampler,
        max_tokens,
        max_square_tokens,
        max_batch,
        sample_lengths,
        max_len,
        drop_last=False,
        batch_size=None,
    ):
        super().__init__(sampler, max_batch, drop_last)
        self.longest_token = 0
        self.max_tokens = max_tokens
        self.max_batch = max_batch
        self.sampler = sampler
        self.sample_lengths = sample_lengths
        self.max_square_tokens = max_square_tokens
        self.max_len = max_len
        self.epoch = 0
        self.batches = self._build_batches()

    def _build_batches(self):
        batches = []
        length = 0
        ell_sq = 0
        batch = []
        for i, idx in enumerate(self.sampler):
            this_length = min(self.max_len, self.sample_lengths[idx])
            linear = (len(batch) + 1) * max(length, this_length)
            quadratic = (len(batch) + 1) * max(ell_sq, this_length**2)
            if (
                linear <= self.max_tokens
                and quadratic < self.max_square_tokens
            ):
                batch.append(idx)
                length = max(length, this_length)
                ell_sq = max(ell_sq, this_length**2)
                if len(batch) == self.max_batch:
                    batches.append(batch)
                    batch = []
                    length = 0
            else:
                if len(batch) == 0:
                    print("Current batch is empty! idx is ", idx)
                    continue
                batches.append(batch) # submit a batch
                batch = [idx]
                length = this_length
                ell_sq = this_length**2
        if len(batch) > 0:
            batches.append(batch)

        if self.sampler.num_replicas > 1:
            num_samples = torch.tensor(len(batches)).cuda()
            print(
                f"==============Local Rank {self.sampler.rank} Num Samples {num_samples}=============="
            )
            dist.all_reduce(num_samples, op=dist.ReduceOp.MAX)
            print(
                f"==============All Reduce Num Samples {num_samples}=============="
            )
            num_samples = num_samples.item()

            if len(batches) < num_samples:
                # padding_size = num_samples - len(batches)
                a = int(num_samples // len(batches))
                b = num_samples % len(batches)
                new_batches = batches * a
                new_batches += batches[:b]
                assert len(new_batches) == num_samples
                batches = new_batches
            print(
                f"==============After Reduce, Rank{self.sampler.rank}, Num Samples {num_samples}=============="
            )
        return batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.sampler.set_epoch(epoch)
        print(f"==============Building batches for epoch %{epoch}===============")
        self.batches = self._build_batches()
        np.random.default_rng(epoch).shuffle(self.batches)

class SortishApproxBatchDataloader(torch.utils.data.DataLoader):
    def __init__(
        self,
        ds: Any,
        collater: Callable,
        max_tokens: int = 6000,
        max_square_tokens: int = 1000000,
        bucket_size: int = 1000,
        max_batch_size: int = 800,
        num_workers: int = 8,
        rank: int = 0,
        world_size: int = 1,
        max_len: int = 512,
    ) -> None:
        # count special tokens
        lens = [x + y + 2 + 2 for x, y in zip(ds['seq_length'], ds['struct_length'])]
        train_sortish_sampler = SortishSampler(
            lens,
            bucket_size,
            num_replicas=world_size,
            rank=rank
        )
        train_sampler = ApproxBatchSampler(
            train_sortish_sampler,
            max_tokens,
            max_square_tokens,
            max_batch_size,
            lens,
            max_len=max_len,
        )
        self.apporx_batch_sampler = train_sampler
        super().__init__(
            dataset=ds,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=collater,
        )
    def set_epoch(self, epoch):
        self.apporx_batch_sampler.set_epoch(epoch)
