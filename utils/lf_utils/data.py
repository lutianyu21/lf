from typing import Any, Callable, Dict, Iterable, List

import math
import warnings
import random
import numpy as np
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
    def __init__(self):
        pass
        
    def __call__(self, batch: List[Dict[str, Any]]) -> Any:
        if len(batch) > 1: raise NotImplementedError("ExtraColumnCollator only accepts batch size of 1")
        return dict(
            input_ids=torch.tensor(batch[0]['input_ids']).unsqueeze(0),
            attention_mask=torch.tensor(batch[0]['attention_mask']).unsqueeze(0),
            labels=torch.tensor(batch[0]['labels']).unsqueeze(0),
            pdb_name=list(map(lambda x: x["pdb_name"], batch)),
            split=list(map(lambda x: x["split"], batch)),
            seq_length=torch.tensor(list(map(lambda x: x["seq_length"], batch))),
            struct_length=torch.tensor(list(map(lambda x: x["struct_length"], batch)))
        )
        
class ItemwiseConstantLengthDataset(ConstantLengthDataset):
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
                    # TODO apply cropping here if sequence is too long
                    # following dplm https://arxiv.org/pdf/2402.18567 a 1024 segment is cropped from the original sequence
                    if tmp['seq_length'] > self.seq_length and tmp['struct_length'] == 0:
                        pool = tmp['text'].split(' ')
                        boseq, eoseq = pool[0], pool[-1]
                        start_idx = random.randint(1, len(pool) - 1024 - 1)
                        end_idx = start_idx + 1024
                        tmp['text'] = ' '.join([boseq] + pool[start_idx:end_idx]) + eoseq
                        tmp['seq_length'] = 1024
                        warnings.warn(f"Sequence {tmp['pdb_name']} is too long, cropped to length 1024.")
                    
                    buffer.append(self.formatting_func(tmp))
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        warnings.warn("The dataset reached end and the iterator is reset to the start.")
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, add_special_tokens=self.add_special_tokens, truncation=False)[
                "input_ids"
            ]
            
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
            examples = []
            lines = []
            for tokenized_input in tokenized_inputs:
                if self.append_concat_token:
                    tokenized_input = tokenized_input + [self.concat_token_id]
                lines.extend(tokenized_input)
                if len(lines) >= self.seq_length:
                    examples.append(lines[:self.seq_length])
                    lines = [] # truncate the remaining
            if self.shuffle:
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }

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
