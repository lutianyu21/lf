from ast import arg
from calendar import c
import random
from sys import prefix
import time
from typing import Any, Dict, Optional, List, Text, Tuple, Union, cast
import os
from pathlib import Path
import warnings
import pandas as pd
import logging
import colorlog

import numpy as np
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn

import datasets
from datasets import Dataset
from transformers import (
    PreTrainedModel,
    EvalPrediction,
    Trainer,
)
import sacrebleu
from transformers.generation.configuration_utils import GenerationConfig
from trl import SFTTrainer, SFTConfig
from trl.trainer.utils import ConstantLengthDataset

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
        
    
    def get_eval_dataloader(self, eval_dataset: Any = None) -> torch.utils.data.DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        if hasattr(self, "_eval_dataloader") and self.args.dataloader_persistent_workers:
            return self.accelerator.prepare(self._eval_dataloader)
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self._eval_collator
        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            self._eval_dataloader = eval_dataloader
        return self.accelerator.prepare(eval_dataloader)
    
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
    
    @property
    def dummy_metrics(self):
        return dict(
            # unique task id
            tid=-1.0,
            # plm evaluation
            sequence_loss=1e5,
            sequence_acc=0.0,
            sequence_bleu=0.0,
            # slm evaluation
            structure_loss=1e5,
            structure_acc=0.0,
            structure_bleu=0.0,
            # p2s evaluation
            folding_loss=1e5,
            folding_acc=0.0,
            folding_bleu=0.0,
            cfolding_loss=1e5,
            cfolding_acc=0.0,
            cfolding_bleu=0.0,
            # benchmark
            benchmark=-1.0,
            ar_loss=1e5,
            ar_acc=0.0,
            ar_bleu=0.0,
            tm_ar=0.0,
            tm_vq=0.0,
            rmsd_ar=1e5,
            rmsd_vq=1e5,        
        )
    
    
    # evaluation for protein sequence understanding: 
    # - eval/plm/sequence_loss
    # - eval/plm/sequence_acc
    # - eval/plm/sequence_bleu
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
        sequence_bleu = sacrebleu.corpus_bleu(
            [" ".join(map(str, eps_token_ids.cpu().tolist()))],
            [[" ".join(map(str, target_token_ids.cpu().tolist()))]]
        ).score
        
        # logging metrics and the first 5 tokens
        logger.info(f"""
////// Evaluated [{inputs['pdb_name'][0]}] from [{inputs['split'][0]}] in {time.time() - start_time:.2f}s:) //////
Sequence Loss/Acc/Bleu:     {sequence_loss.item():.4f}/{sequence_acc:.4f}/{sequence_bleu:.4f}
""")
        metrics = dict(
            tid=0.0,
            sequence_loss = sequence_loss.item(),
            sequence_acc = sequence_acc,
            sequence_bleu = sequence_bleu,
        )
        metrics = self.dummy_metrics | metrics
        model.train()
        return (output.loss, {k:torch.tensor(v).to(model.device) for k, v in metrics.items()}, inputs['input_ids'])
    
    
    # evaluation for protein structure understanding: 
    # - eval/slm/structure_loss
    # - eval/slm/structure_acc
    # - eval/slm/structure_bleu
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
        structure_bleu = sacrebleu.corpus_bleu(
            [" ".join(map(str, eps_token_ids.cpu().tolist()))],
            [[" ".join(map(str, target_token_ids.cpu().tolist()))]]
        ).score
        
        # logging metrics and the first 5 tokens
        logger.info(f"""
////// Evaluated [{inputs['pdb_name'][0]}] from [{inputs['split'][0]}] in {time.time() - start_time:.2f}s:) //////
Structure Loss/Acc/Bleu:    {structure_loss.item():.4f}/{structure_acc:.4f}/{structure_bleu:.4f}
""")
        metrics = dict(
            tid=1.0,
            structure_loss = structure_loss.item(),
            structure_acc = structure_acc,
            structure_bleu = structure_bleu,
        )
        metrics = self.dummy_metrics | metrics
        model.train()
        return (output.loss, {k:torch.tensor(v).to(model.device) for k, v in metrics.items()}, inputs['input_ids'])
    
    
    # evaluation for protein sequence-to-structure:
    # - eval/p2s/structure_loss
    # - eval/p2s/structure_acc
    # - eval/p2s/structure_bleu
    # - eval/p2s/folding_loss
    # - eval/p2s/folding_acc
    # - eval/p2s/folding_bleu
    # - eval/p2s/cfolding_loss
    # - eval/p2s/cfolding_acc
    # - eval/p2s/cfolding_bleu
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
        
        # for inputs icl | seq | struct, we measure: 
        # 1. whether seq is conditioned: model(struct) << model(seq | struct)               a.k.a structure metrics << folding metrics
        # 2. whether icl is conditioned: model(seq | struct) << model(icl | seq | struct)   a.k.a folding metrics << icl metrics
        constant_helper = self.processor.constant_helper
        icl_start = 0
        seq_start = int(torch.where(
            inputs['labels'][0] == constant_helper['boseq_token_id'])[0][-1].int().item())
        struct_start = int(torch.where(
            inputs['labels'][0] == constant_helper['bostruct_token_id'])[0][-1].int().item())
        icl_length = seq_start - icl_start
        seq_length = struct_start - seq_start
        struct_length = inputs['labels'].shape[1] - struct_start
        
        def evalute_section(
            model: PreTrainedModel,
            inputs: Dict[str, Any],
            start: int,
        ) -> Tuple[torch.Tensor, float, float]:
            labels: torch.Tensor = inputs['labels'][:, start:]
            output = model(
                input_ids=labels,
                attention_mask=torch.ones_like(labels)
            )
            target_token_ids = labels[0][-struct_length:]
            eps_logits = output.logits[0][-struct_length:, :]
            eps_token_ids = eps_logits.argmax(dim=-1)
            loss = torch.nn.functional.nll_loss(
                input=torch.log_softmax(eps_logits, dim=-1)[:-1, :],
                target=target_token_ids[1:]
            )
            acc = (eps_token_ids[:-1] == target_token_ids[1:]).float().mean().item() * 100.0
            bleu = sacrebleu.corpus_bleu(
                [" ".join(map(str, eps_token_ids[:-1].cpu().tolist()))],
                [[" ".join(map(str, target_token_ids[1:].cpu().tolist()))]]
            ).score
            del output
            torch.cuda.empty_cache()
            return loss, acc, bleu
        
        # 1. forward <struct>....</struct>
        structure_loss, structure_acc, structure_bleu = evalute_section(model, inputs, struct_start)
        # 2. forward <seq>....</seq><struct>....</struct>
        folding_loss, folding_acc, folding_bleu = evalute_section(model, inputs, seq_start)
        # 3. forward <seq>....</struct><seq>....</seq><struct>....</struct><seq>....</seq><struct>....</struct>
        cfolding_loss, cfolding_acc, cfolding_bleu = evalute_section(model, inputs, icl_start)
        
        # logging metrics and the first 5 tokens
        logger.info(f"""
////// Evaluated [{inputs['pdb_name'][0]}] from [{inputs['split'][0]}] in {time.time() - start_time:.2f}s:) //////
Structure Loss/Acc/Bleu:    {structure_loss.item():.4f}/{structure_acc:.4f}/{structure_bleu:.4f}
Folding   Loss/Acc/Bleu:    {folding_loss.item():.4f}/{folding_acc:.4f}/{folding_bleu:.4f}
CFolding  Loss/Acc/Bleu:    {cfolding_loss.item():.4f}/{cfolding_acc:.4f}/{cfolding_bleu:.4f}
""")
        metrics = dict(
            tid=2.0,
            folding_loss=folding_loss.item(),
            folding_acc=folding_acc,
            folding_bleu=folding_bleu,
            cfolding_loss=cfolding_loss.item(),
            cfolding_acc=cfolding_acc,
            cfolding_bleu=cfolding_bleu,
            structure_loss=structure_loss.item(),
            structure_acc=structure_acc,
            structure_bleu=structure_bleu,
        )
        metrics = self.dummy_metrics | metrics
        model.train()
        return (torch.tensor(cfolding_loss, device=model.device), {k:torch.tensor(v, device=model.device) for k, v in metrics.items()}, inputs['input_ids'])
    
    
    # ! IMPORTANT !
    # correlation between eval-loss & folding metrics is not clear yet
    # in-domain loss >> out-of-domain loss >> ar generation loss >> ar generation bleu >> reconstruction
    # --------------OOD------------------exposure--------------argmax---------------tokenizer--------------
    @torch.no_grad()
    def _prediction_step_folding(
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
        
        ar_logits: torch.Tensor = torch.stack(ar.logits).permute(1, 0, 2)[0]            # <struct>....</struct><endoftext>
        ar_loss = torch.nn.functional.nll_loss(
            input=torch.log_softmax(ar_logits[: -1, :], dim=-1),                        # <struct>....</struct>
            target=inputs['labels'][0, prompt_length :],                                # <struct>....</struct>
            reduction='mean'
        )
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
        # p_nature = p_nature.to(device)
        tm_vq, rmsd_l_vq, rmsd_g_vq = self.processor.compute_tm_align(p_vq, p_nature, ref=p_nature)
        tm_ar, rmsd_l_ar, rmsd_g_ar = self.processor.compute_tm_align(p_ar, p_nature, ref=p_nature)
        
        metrics = dict(
            tid=3.0,
            benchmark=GlobalConstants.auto_numeric(split),
            ar_loss=ar_loss.item(),
            ar_acc=ar_acc,
            ar_bleu=ar_bleu,
            tm_ar=tm_ar,
            tm_vq=tm_vq,
            rmsd_ar=rmsd_l_ar,
            rmsd_vq=rmsd_l_vq,
        )
        metrics = self.dummy_metrics | metrics
        
        logger.info(
f"""////// Evaluated [{inputs['pdb_name'][0]}] from [{inputs['split'][0]}] in {time.time() - start_time:.2f}s:) //////
Target  Structure:          {self.processor.tokenizer.decode(target_token_ids.cpu().tolist()[:4])}...
AR      Structure:          {self.processor.tokenizer.decode(ar_token_ids.cpu().tolist()[:4])}...
AR Loss/Acc/Bleu:           {ar_loss.item():.4f}/{ar_acc:.4f}/{ar_bleu:.4f}
VQ v.s. Nature: TM-score =  {tm_vq:.4f}, RMSD_L = {rmsd_l_vq:.4f}, RMSD_G = {rmsd_g_vq:.4f}
AR v.s. Nature: TM-score =  {tm_ar:.4f}, RMSD_L = {rmsd_l_ar:.4f}, RMSD_G = {rmsd_g_ar:.4f}
""")
        model.train()
        return (ar_loss, {k:torch.tensor(v).to(device) for k, v in metrics.items()}, inputs['input_ids'])
    
    
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
        else:
            return self._prediction_step_folding(model, inputs, prediction_loss_only, ignore_keys)
            # benchmarking, combine p2s + folding evaluation
            # loss1, metrics1, inputs1 = self._prediction_step_p2s(model, inputs, prediction_loss_only, ignore_keys)
            # torch.cuda.empty_cache()
            # loss2, metrics2, inputs2 = self._prediction_step_folding(model, inputs, prediction_loss_only, ignore_keys)
            # # merge metrics
            # metrics1_keys = [
            #     'structure_loss', 'structure_acc', 'structure_bleu',
            #     'folding_loss', 'folding_acc', 'folding_bleu',
            #     'cfolding_loss', 'cfolding_acc', 'cfolding_bleu'
            # ]
            # metrics2_keys = [
            #     'tid', 'benchmark',
            #     'ar_loss', 'ar_acc', 'ar_bleu',
            #     'tm_ar', 'tm_vq',
            #     'rmsd_ar', 'rmsd_vq'
            # ]
            # # ! WARN ! should ensure key ordering
            # merged_metrics = {k:torch.tensor(v, device=model.device) for k, v in self.dummy_metrics.items()}
            # merged_metrics = merged_metrics | {k: metrics1[k] for k in metrics1_keys} | {k: metrics2[k] for k in metrics2_keys}
            # return (loss1, merged_metrics, inputs1)
    

    @classmethod
    def compute_metrics(cls, eval_pred: EvalPrediction):
        preds: Dict[str, np.ndarray] = eval_pred.predictions # type: ignore
        df = pd.DataFrame({k: v for k, v in preds.items()})
        df['tid'] = df['tid'].astype(int)
        metrics = {}
        # group dataframe by tid
        for tid, group in df.groupby('tid'):
            if tid == 0:
                # pLM metrics
                metrics['plm/sequence_loss']  = group['sequence_loss'].mean()
                metrics['plm/sequence_acc']   = group['sequence_acc'].mean()
                metrics['plm/sequence_bleu']  = group['sequence_bleu'].mean()
            elif tid == 1:
                # sLM metrics
                metrics['slm/structure_loss'] = group['structure_loss'].mean()
                metrics['slm/structure_acc']  = group['structure_acc'].mean()
                metrics['slm/structure_bleu'] = group['structure_bleu'].mean()
            elif tid == 2:
                # p2s metrics
                metrics['p2s/structure_loss'] = group['structure_loss'].mean()
                metrics['p2s/structure_acc']  = group['structure_acc'].mean()
                metrics['p2s/structure_bleu'] = group['structure_bleu'].mean()
                metrics['p2s/folding_loss']   = group['folding_loss'].mean()
                metrics['p2s/folding_acc']    = group['folding_acc'].mean()
                metrics['p2s/folding_bleu']   = group['folding_bleu'].mean()                
                metrics['p2s/cfolding_loss']  = group['cfolding_loss'].mean()
                metrics['p2s/cfolding_acc']   = group['cfolding_acc'].mean()
                metrics['p2s/cfolding_bleu']  = group['cfolding_bleu'].mean()
            elif tid == 3:
                # another group by `split` field
                for split, split_group in group.groupby('benchmark'):
                    prefix = GlobalConstants.auto_string(int(split)) + '/' # type: ignore
                    # from ar setting
                    metrics[prefix + 'tm_ar']     = split_group['tm_ar'].mean()
                    metrics[prefix + 'tm_vq']     = split_group['tm_vq'].mean()
                    metrics[prefix + 'rmsd_ar']   = split_group['rmsd_ar'].mean()
                    metrics[prefix + 'rmsd_vq']   = split_group['rmsd_vq'].mean()
                    metrics[prefix + 'ar_loss']   = split_group['ar_loss'].mean()
                    metrics[prefix + 'ar_acc']    = split_group['ar_acc'].mean()
                    metrics[prefix + 'ar_bleu']   = split_group['ar_bleu'].mean()
            else:
                raise NotImplementedError(f"Unknown task id {type(tid)} {tid} found during metrics computation.")
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
