from functools import partial
import random
import time
from typing import Any, Dict, Optional, List, Text, Tuple, Union, Callable, cast
import os
from pathlib import Path
import warnings
import pandas as pd
import logging
import colorlog

import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler

import datasets
from datasets import Dataset
from transformers import (
    PreTrainedModel,
    EvalPrediction,
    Trainer,
)
import sacrebleu
from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils.import_utils import is_datasets_available
from transformers.trainer_utils import seed_worker
from trl import SFTTrainer, SFTConfig
from trl.trainer.utils import ConstantLengthDataset


from utils.dplm_utils.dplm.vendor.openfold.openfold.utils import loss
from utils.protenix_utils import rmsd

from ..common import GlobalConstants
from ..openfold_utils import OpenfoldProtein

from .protein_processor import ProteinProcessor
from .logits import UnbatchedModalityLogitsProcessorBase
from .data import ItemwiseConstantLengthDataset, ExtraColumnCollator





__all__ = [
    'PackingFoldingTrainer',
]


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


class PackingFoldingTrainer(SFTTrainer):
    # Implementation of packing trainer
    def __init__(
        self,
        processor: ProteinProcessor,
        *args,
        **kwargs
    ):
        self.processor = processor
        self.use_bitwise_cross_entropy = kwargs.pop('use_bitwise_cross_entropy', False)
        
        self._eval_collator = ExtraColumnCollator(**kwargs)
        self._seed = 42
        self._cropping = kwargs.pop('cropping', False)
        self._cropping_size = kwargs.pop('cropping_size', 1024)
        self._masking = kwargs.pop('masking', False)
        self._concatenation = kwargs.pop('concatenation', False)
        self._concatenation_size = kwargs.pop('concatenation_size', 2)
        self._concatenation_ratio = kwargs.pop('concatenation_ratio', 0.0)
        super().__init__(*args, **kwargs)
        
    # WARN: get_eval_dataloader() is just a adaptive interface
    # we override _get_dataloader() to control dataloader creation
    def _get_dataloader(
        self,
        dataset: Dataset,
        description: str,
        batch_size: int,
        sampler_fn: Optional[Callable[[Dataset], torch.utils.data.Sampler]] = None,
        is_training: bool = False,
        dataloader_key: Optional[str] = None,
    ) -> DataLoader:
        """Create a [`~torch.utils.data.DataLoader`] from the given dataset."""

        data_collator = self.data_collator
            
        # PLUGIN: customized data collator for evaluation
        if is_training:
            if is_datasets_available() and isinstance(dataset, datasets.Dataset):
                dataset = self._remove_unused_columns(dataset, description=description)
            else:
                data_collator = self._get_collator_with_removed_columns(self.data_collator, description=description)
        else:
            logger.warning("Using customized evaluation data collator ...")
            data_collator = self._eval_collator

        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(dataset, torch.utils.data.IterableDataset):
            if sampler_fn is not None:
                dataloader_params["sampler"] = sampler_fn(dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            if is_training:
                dataloader_params["worker_init_fn"] = partial(
                    seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
                )

        dataloader = self.accelerator.prepare(DataLoader(dataset, **dataloader_params)) # type: ignore

        # Store the prepared dataloader for subsequent evaluations if using persistent workers.
        if dataloader_key is not None and self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = dataloader
            else:
                self._eval_dataloaders = {dataloader_key: dataloader}

        return dataloader
    
    
    def _prepare_packed_dataloader(
        self,
        tokenizer,
        dataset,
        dataset_text_field,
        max_seq_length,
        num_of_sequences,
        chars_per_token,
        formatting_func=None,
        append_concat_token=True,
        add_special_tokens=True,
    ):
        if dataset_text_field is not None or formatting_func is not None:
            if tokenizer is None:
                raise ValueError("You need to pass a tokenizer when using `dataset_text_field` with `SFTTrainer`.")
            
            # WARN: we move complicated packing & formatting logic of 
            # processing training dataset to `ItemwiseConstantLengthDataset()`
            # processing evaluation dataset to `_prepare_packed_dataloader()`
            extra_kwrags = dict(
                cropping=self._cropping,
                cropping_size=self._cropping_size,
                masking=self._masking,
                concatenation=self._concatenation,
                concatenation_size=self._concatenation_size,
                concatenation_ratio=self._concatenation_ratio,
            )
            constant_length_iterator = ItemwiseConstantLengthDataset(
                tokenizer,
                dataset,
                dataset_text_field=dataset_text_field,
                formatting_func=formatting_func,
                seq_length=max_seq_length,
                infinite=False,
                num_of_sequences=num_of_sequences,
                chars_per_token=chars_per_token,
                eos_token_id=tokenizer.eos_token_id,
                append_concat_token=append_concat_token,
                add_special_tokens=add_special_tokens,
                **extra_kwrags,
            )

            if isinstance(dataset, datasets.IterableDataset):
                return constant_length_iterator

            def data_generator(constant_length_iterator):
                yield from constant_length_iterator

            try:
                packed_dataset = Dataset.from_generator(
                    data_generator, gen_kwargs={"constant_length_iterator": constant_length_iterator}
                )
            except (DatasetGenerationError, SchemaInferenceError) as exc: # type: ignore
                raise ValueError(
                    "Error occurred while packing the dataset. "
                    "Make sure that your dataset has enough samples to at least yield one packed sequence."
                ) from exc
            return packed_dataset
        else:
            raise ValueError(
                "You need to pass a `dataset_text_field` or `formatting_func` argument to the SFTTrainer if you want to use the `ConstantLengthDataset`."
            )
    
    def _prepare_non_packed_dataloader(
        self,
        tokenizer,
        dataset,
        dataset_text_field,
        max_seq_length,
        formatting_func: Any = None,
        add_special_tokens=True,
        remove_unused_columns=False,
    ):
        use_formatting_func = formatting_func is not None and dataset_text_field is None
        self._dataset_sanity_checked = False

        # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
        def tokenize(element):
            outputs = tokenizer(
                element["text"] if not use_formatting_func else formatting_func(element),
                add_special_tokens=add_special_tokens,
                truncation=False,
                padding=False,
                max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )
            if use_formatting_func and not self._dataset_sanity_checked:
                if not isinstance(formatting_func(element), list):
                    raise ValueError(
                        "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                    )
                else:
                    self._dataset_sanity_checked = True
            # keep any other columns besides signature columns +++
            outputs['labels'] = outputs["input_ids"]
            extra_keys = ['split', 'pdb_name', 'seq_length', 'struct_length']
            return {
                "input_ids":        outputs["input_ids"],
                "attention_mask":   outputs["attention_mask"],
                "labels":           outputs["labels"],
                **{k: v for k, v in element.items() if k in extra_keys}
            }
        # end tokenize fn

        signature_columns = ["input_ids", "labels", "attention_mask"]
        extra_columns = list(set(dataset.column_names) - set(signature_columns))
        if not remove_unused_columns and len(extra_columns) > 0:
            warnings.warn(
                "You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with the default collator and yield to errors. If you want to "
                f"inspect dataset other columns (in this case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you used the default collator and create your own data collator in order to inspect the unused dataset columns."
            )
        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names if remove_unused_columns else None,
            num_proc=self.dataset_num_proc,
            batch_size=self.dataset_batch_size,
        )
        return tokenized_dataset
    
    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):    
        # logger.warning(self.processor.tokenizer.decode(inputs['labels'][0].cpu().tolist()))
        
        # if self.use_bitwise_cross_entropy:
        #     # (dplm2.5) https://arxiv.org/abs/2504.11454: bit-wise cross-entropy loss
            
        #     constant_helper = self.processor.constant_helper
        #     structure_vocab_ids: List[int] = constant_helper['structure_token_ids']
        #     smin, smax = min(structure_vocab_ids), max(structure_vocab_ids) # assuming continuous ids
        #     bitwise_width = int(np.ceil(np.log2(smax - smin + 1)))
            
        #     mask_is_padding = inputs['labels'] == -100                                              # [B, L]
        #     mask_is_bitwise = smin <= inputs['labels'] <= smax                                      # [B, L]
            
        #     # obtain raw logits
        #     outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        #     logits: torch.Tensor = outputs.logits  # type: ignore # [B, L]
            
        #     # hybrid gradient
        #     # logits_bitwise = (inputs['input_ids'] - smin) * mask_is_bitwise.long().unsqueeze(-1)    # [B, L, V]
        #     # logits_idxwise = inputs['input_ids'] * (~mask_is_bitwise).long().unsqueeze(-1)          # [B, L, V]
        #     # logits_idxwise = logits_idxwise.unsqueeze(-2).repeat(1, 1, bitwise_width, 1)            # [B, L, W, V]
            
        #     probs = torch.softmax(logits, dim=-1)                                               # [B, L, V]
        #     probs_bitwise = probs * mask_is_bitwise.long().unsqueeze(-1)                        # [B, L, V]
        #     probs_idxwise = probs * (~mask_is_bitwise).long().unsqueeze(-1)                     # [B, L, V]
        #     probs_idxwise = probs_idxwise.unsqueeze(-2).repeat(1, 1, bitwise_width, 1)          # [B, L, W, V]
        #     probs_idxwise = probs_idxwise[:, :, :, smin : smax + 1]                             # [B, L, W, S <= 2^W]
            
        #     B, L, W, S = probs_idxwise.shape
        #     probs_idxwise = probs_idxwise.reshape(B, L * W, S)                                  # [B, L * W, S]
            
        #     # magic mask: 00001111 | 00110011 | 01010101 || 00001111 | 00110011 | 01010101
        #     # we use this mask to gather probs for bit-wise cross-entropy
        #     # magic extracts p(bit=1), ~magic extracts p(bit=0)
        #     magic = torch.arange(L * W - 1, -1, -1, device=probs_idxwise.device)                # [L * W]
        #     magic = magic.unsqueeze(0).repeat(B, 1)                                             # [B, L * W]
        #     magic = magic % W                                                                   # [B, L * W]
        #     magic = torch.arange(2**W, device=probs_idxwise.device).unsqueeze(0).unsqueeze(0).repeat(B, L * W, 1)\
        #         >> magic.unsqueeze(-1) & 1                                                      # [B, L * W, 2^W]
        #     magic = magic[:, :, :S]                                                             # [B, L * W, S]
        #     probs_bitwise_0 = torch.sum(probs_idxwise * (1 - magic).long(), dim=-1)             # [B, L * W]
        #     probs_bitwise_1 = torch.sum(probs_idxwise * magic.long(), dim=-1)                   # [B, L * W]
        
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
    
    # evaluation for protein sequence understanding: 
    # - eval/plm/sequence_loss
    # - eval/plm/sequence_acc
    @torch.no_grad()
    def _prediction_step_plm(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ):
        assert inputs['labels'].shape[0] == 1, "pLM evaluation only supports batch size = 1."
        assert inputs['split'][0] in ['p/uniref50'], f"pLM evaluation only supports uniref50 dataset, but got {inputs['split']}."
        start_time = time.time()
        model.eval()
        
        # modeling protein sequence understanding p(sequence)
        output = model(input_ids=inputs['labels'], labels=inputs['labels'], attention_mask=torch.ones_like(inputs['labels']))
        target_token_ids = inputs['labels'][0]              # <seq>....</seq>
        eps_token_ids = output.logits[0].argmax(dim=-1)     # ....</seq><endoftext>
        sequence_loss = output.loss
        sequence_acc = (eps_token_ids[:-1] == target_token_ids[1:]).float().mean().item() * 100.0
        
        # logging metrics and the first 5 tokens
        logger.info(f"""
////// Evaluated [{inputs['pdb_name'][0]}] from [{inputs['split'][0]}] in {time.time() - start_time:.2f}s:) //////
Sequence Loss/Acc:      {sequence_loss.item():.4f}/{sequence_acc:.4f}
""")
        metrics = dict(
            sequence_loss = sequence_loss.item(),
            sequence_acc = sequence_acc,
        )
        model.train()
        return (
            output.loss,
            {k:torch.tensor(v).to(model.device) for k, v in metrics.items()},
            inputs['input_ids']
        )
    
    
    # evaluation for protein structure understanding: 
    # - eval/slm/structure_loss
    # - eval/slm/structure_acc
    @torch.no_grad()
    def _prediction_step_slm(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ):
        assert inputs['labels'].shape[0] == 1, "sLM evaluation only supports batch size = 1."
        start_time = time.time()
        model.eval()
        
        # modeling protein structure understanding p(structure)
        output = model(input_ids=inputs['labels'], labels=inputs['labels'], attention_mask=torch.ones_like(inputs['labels']))
        target_token_ids = inputs['labels'][0]              # <struct>....</struct>
        eps_token_ids = output.logits[0].argmax(dim=-1)     # ....</struct><endoftext>
        structure_loss = output.loss
        structure_acc = (eps_token_ids[:-1] == target_token_ids[1:]).float().mean().item() * 100.0
        
        # logging metrics and the first 5 tokens
        logger.info(f"""
////// Evaluated [{inputs['pdb_name'][0]}] from [{inputs['split'][0]}] in {time.time() - start_time:.2f}s:) //////
Structure Loss/Acc:     {structure_loss.item():.4f}/{structure_acc:.4f}
""")
        metrics = dict(
            structure_loss = structure_loss.item(),
            structure_acc = structure_acc,
        )
        model.train()
        return (
            output.loss,
            {k:torch.tensor(v).to(model.device) for k, v in metrics.items()},
            inputs['input_ids']
        )
    
    
    # evaluation for protein sequence-to-structure:
    # - eval/p2s/struct_loss
    # - eval/p2s/struct_acc
    # - eval/p2s/folding_loss
    # - eval/p2s/folding_acc
    @torch.no_grad()
    def _prediction_step_p2s(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ):
        assert inputs['labels'].shape[0] == 1, "p2s evaluation only supports batch size = 1."
        start_time = time.time()
        model.eval()
        
        constant_helper = self.processor.constant_helper
        sequence_start = int(torch.where(
            inputs['labels'][0] == constant_helper['boseq_token_id'])[0][-1].int().item())
        structure_start = int(torch.where(
            inputs['labels'][0] == constant_helper['bostruct_token_id'])[0][-1].int().item())
        sequence_length = structure_start - sequence_start
        structure_length = inputs['labels'].shape[1] - structure_start
        
        def evalute_section(
            model: PreTrainedModel,
            inputs: Dict[str, Any],
            start: int,
        ) -> Tuple[torch.Tensor, float]:
            labels: torch.Tensor = inputs['labels'][:, start:]
            output = model(
                input_ids=labels,
                attention_mask=torch.ones_like(labels)
            )
            target_token_ids = labels[0][-structure_length:]
            eps_logits = output.logits[0][-structure_length:, :]
            eps_token_ids = eps_logits.argmax(dim=-1)
            loss = torch.nn.functional.nll_loss(
                input=torch.log_softmax(eps_logits, dim=-1)[:-1, :],
                target=target_token_ids[1:]
            )
            acc = (eps_token_ids[:-1] == target_token_ids[1:]).float().mean().item() * 100.0
            del output
            torch.cuda.empty_cache()
            return loss, acc
        
        # 1. forward <struct>....</struct>
        structure_loss, structure_acc = evalute_section(model, inputs, structure_start)
        # 2. forward <seq>....</seq><struct>....</struct>
        folding_loss, folding_acc = evalute_section(model, inputs, sequence_start)
        
        # logging metrics and the first 5 tokens
        logger.info(f"""
////// Evaluated [{inputs['pdb_name'][0]}] from [{inputs['split'][0]}] in {time.time() - start_time:.2f}s:) //////
Structure Loss/Acc:    {structure_loss.item():.4f}/{structure_acc:.4f}
Folding   Loss/Acc:    {folding_loss.item():.4f}/{folding_acc:.4f}
""")
        metrics = dict(
            structure_loss=structure_loss.item(),
            structure_acc=structure_acc,
            folding_loss=folding_loss.item(),
            folding_acc=folding_acc,
        )
        model.train()
        return (
            torch.tensor(folding_loss, device=model.device),
            {k:torch.tensor(v, device=model.device) for k, v in metrics.items()},
            inputs['input_ids']
        )
    
    
    # evaluation for protein sequence-to-structure with context:
    # - eval/p2s/struct_loss
    # - eval/p2s/struct_acc
    # - eval/p2s/folding_loss
    # - eval/p2s/folding_acc
    # - eval/p2s/cfolding_loss
    # - eval/p2s/cfolding_acc
    @torch.no_grad()
    def _prediction_step_psps(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ):
        assert inputs['labels'].shape[0] == 1, "p2s evaluation only supports batch size = 1."
        start_time = time.time()
        model.eval()
        
        constant_helper = self.processor.constant_helper
        context_start = 0
        sequence_start = int(torch.where(
            inputs['labels'][0] == constant_helper['boseq_token_id'])[0][-1].int().item())
        structure_start = int(torch.where(
            inputs['labels'][0] == constant_helper['bostruct_token_id'])[0][-1].int().item())
        context_length = sequence_start - context_start
        sequence_length = structure_start - sequence_start
        structure_length = inputs['labels'].shape[1] - structure_start
        
        def evalute_section(
            model: PreTrainedModel,
            inputs: Dict[str, Any],
            start: int,
        ) -> Tuple[torch.Tensor, float]:
            labels: torch.Tensor = inputs['labels'][:, start:]
            output = model(
                input_ids=labels,
                attention_mask=torch.ones_like(labels)
            )
            target_token_ids = labels[0][-structure_length:]
            eps_logits = output.logits[0][-structure_length:, :]
            eps_token_ids = eps_logits.argmax(dim=-1)
            loss = torch.nn.functional.nll_loss(
                input=torch.log_softmax(eps_logits, dim=-1)[:-1, :],
                target=target_token_ids[1:]
            )
            acc = (eps_token_ids[:-1] == target_token_ids[1:]).float().mean().item() * 100.0
            del output
            torch.cuda.empty_cache()
            return loss, acc
        
        # 1. forward <struct>....</struct>
        structure_loss, structure_acc = evalute_section(model, inputs, structure_start)
        # 2. forward <seq>....</seq><struct>....</struct>
        folding_loss, folding_acc = evalute_section(model, inputs, sequence_start)
        # 3. forward <seq>....</struct><seq>....</seq><struct>....</struct><seq>....</seq><struct>....</struct>
        cfolding_loss, cfolding_acc = evalute_section(model, inputs, context_start)
        
        # logging metrics and the first 5 tokens
        logger.info(f"""
////// Evaluated [{inputs['pdb_name'][0]}] from [{inputs['split'][0]}] in {time.time() - start_time:.2f}s:) //////
Structure Loss/Acc:    {structure_loss.item():.4f}/{structure_acc:.4f}
Folding   Loss/Acc:    {folding_loss.item():.4f}/{folding_acc:.4f}
CFolding  Loss/Acc:    {cfolding_loss.item():.4f}/{cfolding_acc:.4f}
""")
        metrics = dict(
            structure_loss=structure_loss.item(),
            structure_acc=structure_acc,
            folding_loss=folding_loss.item(),
            folding_acc=folding_acc,
            cfolding_loss=cfolding_loss.item(),
            cfolding_acc=cfolding_acc,
        )
        model.train()
        return (
            torch.tensor(cfolding_loss, device=model.device),
            {k:torch.tensor(v, device=model.device) for k, v in metrics.items()},
            inputs['input_ids']
        )
    
    
    # ! IMPORTANT !
    # correlation between eval-loss & folding metrics is not clear yet
    # in-domain loss >> out-of-domain loss >> ar generation loss >> ar generation bleu >> reconstruction
    # --------------OOD------------------exposure--------------argmax---------------tokenizer--------------
    @torch.no_grad()
    def _prediction_step_benchmark(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ):
        assert inputs['labels'].shape[0] == 1, "folding evaluation only supports batch size = 1."
        start_time = time.time()
        model.eval()
        
        # {prompt} | <seq>....</seq> | <struct>....</struct>
        constant_helper = self.processor.constant_helper
        struct_start = int(torch.where(inputs['labels'][0] == constant_helper['bostruct_token_id'])[0][-1].int().item())
        prompt_length = struct_start # wo/ <struct> token itself
        answer_length = inputs['labels'].shape[1] - prompt_length
        
        # exposure result can be obtained from p2s evaluation
        # we only need to run autoregressive generation here
        generation_config = GenerationConfig(
            use_cache=True,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            do_sample=False,
            max_new_tokens=12*1024,
            return_dict_in_generate=True,
            output_scores=False,
            output_logits=True,
        )
        logits_processor = UnbatchedModalityLogitsProcessorBase(
            **self.processor.constant_helper,
            processor=self.processor,
            templates=[('struct', inputs['struct_length'][0].item())]
        )
        ar = model.generate(
            input_ids=inputs["labels"][:, :prompt_length],
            attention_mask=torch.ones_like(inputs["labels"][:, :prompt_length]),
            generation_config=generation_config,
            logits_processor=[logits_processor],
        ) # type: ignore
        
        target_token_ids = inputs['labels'][0, prompt_length :]                         # <struct>....</struct>
        ar_token_ids = ar.sequences[0, prompt_length : -1]                              # <struct>....</struct>
        ar_acc = (ar_token_ids == target_token_ids).float().mean().item()
        ar_bleu = sacrebleu.corpus_bleu(
            [" ".join(map(str, ar_token_ids.cpu().tolist()))],
            [[" ".join(map(str, target_token_ids.cpu().tolist()))]]
        ).score
        
        split, pdb_name, device = inputs['split'][0], inputs['pdb_name'][0], inputs['input_ids'].device
        p_nature = OpenfoldProtein.from_file(GlobalConstants.auto_pathing(pdb_name)).to(device)
        
        # WARN processor's priority: move protein to processor's device
        # so we should either move processor to cuda here, or
        # move protein back to cuda later (to counter processor's device move)
        self.processor.to(device)
        p_vq = self.processor.multimodal_decode(target_token_ids, ref=p_nature)['entity'][0].to(device)
        p_ar = self.processor.multimodal_decode(ar_token_ids, ref=p_nature)['entity'][0].to(device)
        # ar ---- vq ---- nature, where the former is up to ar modeling, the latter is up to vq reconstruction
        tm_vq, rmsd_l_vq, rmsd_g_vq = self.processor.compute_tm_align(p_vq, p_nature, ref=p_nature)
        tm_ar, rmsd_l_ar, rmsd_g_ar = self.processor.compute_tm_align(p_ar, p_vq, ref=p_nature)
        tm_final, rmsd_l_final, rmsd_g_final = self.processor.compute_tm_align(p_ar, p_nature, ref=p_nature)
        metrics = dict(
            tm_vq = tm_vq,
            tm_ar = tm_ar,
            tm_final = tm_final,
            rmsd_vq = rmsd_l_vq,
            rmsd_ar = rmsd_l_ar,
            rmsd_final = rmsd_l_final,
        )
        
        logger.info(
f"""////// Evaluated [{inputs['pdb_name'][0]}] from [{inputs['split'][0]}] in {time.time() - start_time:.2f}s:) //////
Target  Structure:          {self.processor.tokenizer.decode(target_token_ids.cpu().tolist()[:50])}...
AR      Structure:          {self.processor.tokenizer.decode(ar_token_ids.cpu().tolist()[:50])}...
VQ v.s. Nature: TM-score =  {tm_vq:.4f}, RMSD_L = {rmsd_l_vq:.4f}, RMSD_G = {rmsd_g_vq:.4f}
AR v.s. VQ:     TM-score =  {tm_ar:.4f}, RMSD_L = {rmsd_l_ar:.4f}, RMSD_G = {rmsd_g_ar:.4f}
AR v.s. Nature: TM-score =  {tm_final:.4f}, RMSD_L = {rmsd_l_final:.4f}, RMSD_G = {rmsd_g_final:.4f}
""")
        model.train()
        return (
            torch.tensor(0.0, device=model.device),
            {k:torch.tensor(v).to(device) for k, v in metrics.items()},
            inputs['input_ids']
        )
    
    
    @torch.no_grad()
    def prediction_step(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ):
        split = inputs['split'][0]
        if split in ['p/uniref50']:
            return self._prediction_step_plm(model, inputs, prediction_loss_only, ignore_keys)
        elif split in ['s/unicluster40']:
            return self._prediction_step_slm(model, inputs, prediction_loss_only, ignore_keys)
        elif split in ['p2s/unicluster40']:
            return self._prediction_step_p2s(model, inputs, prediction_loss_only, ignore_keys)
        elif split in ['psps/unicluster40']:
            return self._prediction_step_psps(model, inputs, prediction_loss_only, ignore_keys)
        elif split in ['psps/cameo2022', 'psps/casp15', 'psps/casp16']:
            # TO test OOD, call exposure evaluation first
            loss1, metrics1, inputs1 = self._prediction_step_psps(model, inputs, prediction_loss_only, ignore_keys)
            torch.cuda.empty_cache()
            loss2, metrics2, inputs2 = self._prediction_step_benchmark(model, inputs, prediction_loss_only, ignore_keys)
            merged_metrics = metrics1 | metrics2
            return (loss2, merged_metrics, inputs2)
        elif split in ['p2s/cameo2022', 'p2s/casp15', 'p2s/casp16']:
            # TO test OOD, call exposure evaluation first
            loss1, metrics1, inputs1 = self._prediction_step_p2s(model, inputs, prediction_loss_only, ignore_keys)
            torch.cuda.empty_cache()
            loss2, metrics2, inputs2 = self._prediction_step_benchmark(model, inputs, prediction_loss_only, ignore_keys)
            merged_metrics = metrics1 | metrics2
            return (loss2, merged_metrics, inputs2)
    
    
    @classmethod
    def compute_metrics(cls, eval_pred: EvalPrediction):
        preds: Dict[str, np.ndarray] = eval_pred.predictions # type: ignore
        df = pd.DataFrame({k: v for k, v in preds.items()})
        metrics = {}

        # TRICK: decide dataset type by checking metrics keys
        is_benchmark = 'tm_ar' in df.columns
        is_psps = 'cfolding_loss' in df.columns and not is_benchmark
        is_p2s = 'folding_loss' in df.columns and not is_benchmark and not is_psps
        is_plm = 'sequence_loss' in df.columns and not is_benchmark and not is_psps and not is_p2s
        is_slm = 'structure_loss' in df.columns and not is_benchmark and not is_psps and not is_p2s
        
        if is_benchmark:
            logger.info(f"[{len(df)}rows] Computing metrics for protein folding benchmark tasks.")
            # for benchmarks, report mean & std
            metrics['ar_acc']       = df['ar_acc'].mean()
            metrics['ar_bleu']      = df['ar_bleu'].mean()
            metrics['tm_ar_mean']   = df['tm_ar'].mean()
            metrics['tm_ar_std']    = df['tm_ar'].std()
            metrics['tm_vq_mean']   = df['tm_vq'].mean()
            metrics['tm_vq_std']    = df['tm_vq'].std()
            metrics['tm_final_mean'] = df['tm_final'].mean()
            metrics['tm_final_std']  = df['tm_final'].std()
            metrics['rmsd_ar_mean'] = df['rmsd_ar'].mean()
            metrics['rmsd_ar_std']  = df['rmsd_ar'].std()
            metrics['rmsd_vq_mean'] = df['rmsd_vq'].mean()
            metrics['rmsd_vq_std']  = df['rmsd_vq'].std()
        elif is_psps:
            logger.info(f"[{len(df)}rows] Computing metrics for protein sequence-to-structure with context task.")
            metrics['p2s/structure_loss'] = df['structure_loss'].mean()
            metrics['p2s/structure_acc']  = df['structure_acc'].mean()
            metrics['p2s/folding_loss']   = df['folding_loss'].mean()
            metrics['p2s/folding_acc']    = df['folding_acc'].mean()             
            metrics['p2s/cfolding_loss']  = df['cfolding_loss'].mean()
            metrics['p2s/cfolding_acc']   = df['cfolding_acc'].mean()
        elif is_p2s:
            logger.info(f"[{len(df)}rows] Computing metrics for protein sequence-to-structure task.")
            metrics['p2s/structure_loss'] = df['structure_loss'].mean()
            metrics['p2s/structure_acc']  = df['structure_acc'].mean()
            metrics['p2s/folding_loss']   = df['folding_loss'].mean()
            metrics['p2s/folding_acc']    = df['folding_acc'].mean()
        elif is_plm:
            logger.info(f"[{len(df)}rows] Computing metrics for protein language modeling task.")
            metrics['plm/sequence_loss']  = df['sequence_loss'].mean()
            metrics['plm/sequence_acc']   = df['sequence_acc'].mean()
        elif is_slm:
            logger.info(f"[{len(df)}rows] Computing metrics for structure language modeling task.")
            metrics['slm/structure_loss'] = df['structure_loss'].mean()
            metrics['slm/structure_acc']  = df['structure_acc'].mean()
        else:
            raise NotImplementedError("Unknown evaluation prediction format found during metrics computation.")
        return metrics

    
    @classmethod
    def formatting_func_concatenate_templates(cls, example: Dict[str, List[Any]]) -> List[str] | str:
        if isinstance(example['text'], str):
            # single example mode
            text = example['text']
            templates = example['templates']
            if len(templates) > 0:
                template_str = random.choice(templates)
                return template_str + text
            else:
                return text
        else:
            # batch example mode
            formatted_text = []
            for text, templates in zip(example['text'], example['templates']):
                if len(templates) > 0:
                    template_str = random.choice(templates)
                    formatted_text.append(template_str + text)
                else:
                    formatted_text.append(text)
            return formatted_text
