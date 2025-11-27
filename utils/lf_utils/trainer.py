from cProfile import label
import time
from typing import Any, Dict, Optional, List, Text, Tuple, Union, cast
import os
from pathlib import Path
import warnings
import pandas as pd
import logging
import colorlog

import numpy as np
from sympy import sequence, use
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn

import datasets
from datasets import Dataset, IterableDataset, load_dataset
from transformers import (
    PreTrainedModel,
    EvalPrediction,
    Trainer,
)
import sacrebleu
from transformers.generation.configuration_utils import GenerationConfig
from trl import SFTTrainer, SFTConfig
from trl.trainer.utils import ConstantLengthDataset

from utils.lf_utils.protein_tokenizer import DistMatrixTokenizer
from utils.openfold_utils import OpenfoldProtein
from utils.lf_utils import (
    ProteinProcessor, 
    ItemwiseConstantLengthDataset,
    ExtraColumnCollator,
    UnbatchedModalityLogitsProcessorBase,
    DATASET_SPLIT, DATASET_RAW_ROOT,
)



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
        eval_collator: ExtraColumnCollator,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.eval_collator = eval_collator
        
    def get_eval_dataloader(self, eval_dataset: Any = None) -> torch.utils.data.DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        if hasattr(self, "_eval_dataloader") and self.args.dataloader_persistent_workers:
            return self.accelerator.prepare(self._eval_dataloader)
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.eval_collator
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
                truncation='prompt' not in element.keys(), # True for training; False for eval
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
            if 'prompt' in element.keys():
                outputs_prompt = tokenizer(
                    element["prompt"] if not use_formatting_func else formatting_func(element),
                    add_special_tokens=add_special_tokens,
                    truncation=False,
                    padding=False,
                    max_length=max_seq_length,
                    return_overflowing_tokens=False,
                    return_length=False,
                )
                outputs['labels'] = outputs["input_ids"]
                outputs['input_ids'] = outputs_prompt["input_ids"] # truncated input_ids
            else:
                outputs['labels'] = outputs["input_ids"]
                
            return {
                "input_ids":        outputs["input_ids"],
                "attention_mask":   outputs["attention_mask"],
                "labels":           outputs["labels"],
                **{k: element[k] for k in element.keys() if k not in outputs.keys()}
            }

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
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
    
    @property
    def dummy_metrics(self):
        return dict(
            tid=-1,
            sequence_loss=0.0,
            sequence_acc=0.0,
        )
    
    @torch.no_grad()
    def _prediction_step_plm(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ):
        assert inputs['labels'].shape[0] == 1, "sLM evaluation only supports batch size = 1."
        assert inputs['split'] in ['uniref50'], "pLM evaluation only supports uniref50 dataset."
        
        # evaluation for protein sequence understanding: 
        # eval/plm/sequence_loss + eval/plm/sequence_acc
        start_time = time.time()
        model.eval()
        
        output = model(
            input_ids=inputs['labels'],
            labels=inputs['labels'],
            attention_mask=torch.ones_like(inputs['labels']),
        )
        
        target_token_ids = inputs['labels'][0]              # <seq>....</seq>
        eps_token_ids = output.logits[0].argmax(dim=-1)     #      ....</seq><endoftext>
        sequence_loss = output.loss
        sequence_acc = (eps_token_ids[:-1] == target_token_ids[1:]).float().mean().item() * 100.0
        
        # logging metrics and the first 50 tokens
        logger.info(f"""
Evaluated [{inputs['pdb_name'][0]}] from [{inputs['split'][0]}] in {time.time() - start_time:.2f}s:)
//////
Target Results:   {self.processor.tokenizer.decode(target_token_ids.cpu().tolist()[:100])}...
Exposure Results: {self.processor.tokenizer.decode(eps_token_ids.cpu().tolist()[:100])}...
//////
Sequence Loss:    {sequence_loss.item():.4f}
Sequence Acc:     {sequence_acc:.4f}
//////
""")
        metrics = dict(
            tid=0,
            sequence_loss = sequence_loss.item(),
            sequence_acc = sequence_acc,
        )
        metrics = self.dummy_metrics | metrics
        model.train()
        return (sequence_loss, {k:torch.tensor(v).to(model.device) for k, v in metrics.items()}, inputs['input_ids'])
    
    @torch.no_grad()
    def _prediction_step_slm(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ):
        assert inputs['labels'].shape[0] == 1, "sLM evaluation only supports batch size = 1."
        assert inputs['split'] in ['dplm'], "sLM evaluation only supports dplm dataset."
        
        # evaluation for protein structure understanding: 
        # eval/slm/structure_loss + eval/slm/structure_acc
        start_time = time.time()
        model.eval()
        
        output = model(
            input_ids=inputs['labels'],
            labels=inputs['labels'],
            attention_mask=torch.ones_like(inputs['labels']),
        )
        
        target_token_ids = inputs['labels'][0]              # <struct>....</struct>
        eps_token_ids = output.logits[0].argmax(dim=-1)     #      ....</struct><endoftext>
        structure_loss = output.loss
        structure_acc = (eps_token_ids[:-1] == target_token_ids[1:]).float().mean().item() * 100.0
        
        # logging metrics and the first 50 tokens
        logger.info(f"""
Evaluated [{inputs['pdb_name'][0]}] from [{inputs['split'][0]}] in {time.time() - start_time:.2f}s:)
//////
Target Results:   {self.processor.tokenizer.decode(target_token_ids.cpu().tolist()[:100])}...
Exposure Results: {self.processor.tokenizer.decode(eps_token_ids.cpu().tolist()[:100])}...
//////
Structure Loss:   {structure_loss.item():.4f}
Structure Acc:    {structure_acc:.4f}
//////
""")
        metrics = dict(
            tid=1,
            structure_loss = structure_loss.item(),
            structure_acc = structure_acc,
        )
        metrics = self.dummy_metrics | metrics
        model.train()
        return (structure_loss, {k:torch.tensor(v).to(model.device) for k, v in metrics.items()}, inputs['input_ids'])
        
    @torch.no_grad()
    def _prediction_step_p2s(
        
    ):
        pass 
    
        
        
        
        

        
        
        
        
        
        
        
    
    
    
    
    
    
#     @torch.no_grad()
#     def _prediction_step_mllm(
#         self,
#         model: PreTrainedModel,
#         inputs: Dict[str, Any],
#         prediction_loss_only: bool,
#         ignore_keys: Optional[List[str]] = None
#     ):
#         # inputs_ids:   <seq>...</seq>
#         # labels:       <seq>...</seq><struct>....</struct>
        
#         # CORE: we want to find the correlation among following metrics:
#         # {sequence, structure} x {loss, acc, bleu} x {generation, exposure}
#         # for [cameo2022, casp15, casp16, eval, overfit] respectively
#         model.eval()
        
#         # alias
#         (
#             tokenizer,
#             split,
#             pdb_name,
#             prompt_length,
#             answer_length,
#             device,
#             root,
#             format,
#         ) = (
#             self.processor.tokenizer,
#             inputs['split'][0],
#             inputs['pdb_name'][0],
#             len(inputs['input_ids'][0]),
#             len(inputs['labels'][0]) - len(inputs['input_ids'][0]),
#             inputs['input_ids'].device,
#             DATASET_RAW_ROOT[inputs['split'][0]][0],
#             DATASET_RAW_ROOT[inputs['split'][0]][1],
#         )
        
#         lambda_acc = lambda x, y: (x == y).float().mean().item() * 100.0
#         lambda_bleu = lambda x, y: sacrebleu.corpus_bleu(
#             [" ".join(map(str, x.cpu().tolist()))],
#             [[" ".join(map(str, y.cpu().tolist()))]]
#         ).score
        
#         # generation pipeline, sample 10 times
#         num_return_sequences = 10
#         generation_token_ids: torch.Tensor = model.generate(
#             input_ids=inputs["input_ids"],
#             attention_mask=inputs["attention_mask"],
#             min_new_tokens=answer_length,
#             max_new_tokens=answer_length,
#             num_return_sequences=num_return_sequences,
#         ) # type: ignore
#         generation_token_ids[:, -1] = tokenizer.eostruct_token_id
#         # reset last token to </struct>, so here <seq>...</seq><struct>....</struct>
        
#         # split labels
#         structure_target_token_ids = inputs['labels'][0, prompt_length :]           # <struct>....</struct>
#         sequence_target_token_ids  = inputs['labels'][0, : prompt_length]           # <seq>...</seq>
        
#         # split generation
#         structure_generation_token_ids = generation_token_ids[:, prompt_length :]   # <struct>....</struct>
#         sequence_generation_token_ids  = generation_token_ids[:, : prompt_length]   # <seq>...</seq>
        
#         assert structure_generation_token_ids.shape[1] == structure_target_token_ids.shape[0], "Structure generation length mismatch."
#         assert sequence_generation_token_ids.shape[1] == sequence_target_token_ids.shape[0], "Sequence generation length mismatch."
        
#         # split expsoure
#         exposure = model(
#             input_ids=inputs['labels'],
#             attention_mask=torch.ones_like(inputs['labels']),
#             labels=inputs['labels']
#         )
#         logits = exposure.logits[0] # type: ignore    
#         exposure_token_ids = exposure.logits[0].argmax(dim=-1)                      # ...</seq><struct>....</struct><endoftext>
#         # shift and left-append a <seq>
#         exposure_token_ids = torch.cat((
#             torch.tensor([tokenizer.boseq_token_id], device=device),
#             exposure_token_ids[:-1],
#         ), dim=0)
#         structure_exposure_token_ids = exposure_token_ids[prompt_length :]          # <struct>....</struct>
#         sequence_exposure_token_ids  = exposure_token_ids[: prompt_length]          # <seq>...</seq>
        
#         # manually compute token-wise exposure loss
#         structure_loss_eps = torch.nn.functional.nll_loss(
#             input=torch.log_softmax(logits[prompt_length : -1, :], dim=-1),
#             target=structure_target_token_ids[1:],
#             reduction='mean'
#         )
#         sequence_loss_eps = torch.nn.functional.nll_loss(
#             input=torch.log_softmax(logits[0 : prompt_length - 1, :], dim=-1),
#             target=sequence_target_token_ids[1:],
#             reduction='mean'
#         )
                     
#         metrics = dict(
#             # meta
#             split=DATASET_SPLIT[split],
#             # structure
#             struct_acc_gen = np.mean([lambda_acc(structure_generation_token_ids[i], structure_target_token_ids) for i in range(num_return_sequences)]),
#             struct_bleu_gen = np.mean([lambda_bleu(structure_generation_token_ids[i], structure_target_token_ids) for i in range(num_return_sequences)]),
#             struct_acc_eps = lambda_acc(structure_exposure_token_ids, structure_target_token_ids),
#             struct_bleu_eps = lambda_bleu(structure_exposure_token_ids, structure_target_token_ids),
#             struct_loss_eps = structure_loss_eps.item(),
#             # sequence
#             seq_acc_gen = np.mean([lambda_acc(sequence_generation_token_ids[i], sequence_target_token_ids) for i in range(num_return_sequences)]),
#             seq_bleu_gen = np.mean([lambda_bleu(sequence_generation_token_ids[i], sequence_target_token_ids) for i in range(num_return_sequences)]),
#             seq_acc_eps = lambda_acc(sequence_exposure_token_ids, sequence_target_token_ids),
#             seq_bleu_eps = lambda_bleu(sequence_exposure_token_ids, sequence_target_token_ids),
#             seq_loss_eps = sequence_loss_eps.item(),
#         )
        
#         logger.info(f"""Evaluated [{pdb_name}] from [{split}]:
# //////
# Sequence Exposure Loss:         {metrics['seq_loss_eps']:.4f}
# Sequence Exposure Acc/Bleu:     {metrics['seq_acc_eps']:.4f}/{metrics['seq_bleu_eps']:.4f}
# Sequence Generation Acc/Bleu:   {metrics['seq_acc_gen']:.4f}/{metrics['seq_bleu_gen']:.4f}
# //////
# Structure Exposure Loss:        {metrics['struct_loss_eps']:.4f}
# Structure Exposure Acc/Bleu:    {metrics['struct_acc_eps']:.4f}/{metrics['struct_bleu_eps']:.4f}
# Structure Generation Acc/Bleu:  {metrics['struct_acc_gen']:.4f}/{metrics['struct_bleu_gen']:.4f}
# //////
# """)
#         model.train()
#         return (exposure.loss, {k:torch.tensor(v).to(device) for k, v in metrics.items()}, inputs['input_ids'])
    
#     @torch.no_grad()
#     def prediction_step(
#         self,
#         model: PreTrainedModel,
#         inputs: Dict[str, Any],
#         prediction_loss_only: bool,
#         ignore_keys: Optional[List[str]] = None
#     ):
#         return self._prediction_step_mllm(model, inputs, prediction_loss_only, ignore_keys)
    
#     @classmethod
#     def compute_metrics(cls, eval_pred: EvalPrediction):
#         preds: Dict[str, np.ndarray] = eval_pred.predictions # type: ignore
#         df = pd.DataFrame({k: v for k, v in preds.items()})
#         metrics = {}
#         for i, group in df.groupby('split'):
#             prefix = list(DATASET_SPLIT.keys())[int(i)] # type: ignore
#             metrics[prefix] = {
#                 'structure_loss_eps':   group['struct_loss_eps'].mean(),
#                 'structure_acc_eps':    group['struct_acc_eps'].mean(),
#                 'structure_bleu_eps':   group['struct_bleu_eps'].mean(),
#                 'structure_acc_gen':    group['struct_acc_gen'].mean(),
#                 'structure_bleu_gen':   group['struct_bleu_gen'].mean(),
#                 'sequence_loss_eps':    group['seq_loss_eps'].mean(),
#                 'sequence_acc_eps':     group['seq_acc_eps'].mean(),
#                 'sequence_bleu_eps':    group['seq_bleu_eps'].mean(),
#                 'sequence_acc_gen':     group['seq_acc_gen'].mean(),
#                 'sequence_bleu_gen':    group['seq_bleu_gen'].mean(),
#             }
#         return metrics
    
        
        
        
        


    @torch.no_grad()
    def prediction_step(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ):
        # CORE: we want to find the correlation among following metrics:
        # exposure-loss ～ exposure-acc ～ generation-acc ～ tm-score
        # for [cameo2022, casp15, casp16, eval, overfit] respectively
        model.eval()
        
        # alias
        (
            tokenizer,
            split,
            pdb_name,
            prompt_length,
            device,
            root,
            format,
        ) = (
            self.processor.tokenizer,
            inputs['split'][0],
            inputs['pdb_name'][0],
            len(inputs['input_ids'][0]),
            inputs['input_ids'].device,
            DATASET_RAW_ROOT[inputs['split'][0]][0],
            DATASET_RAW_ROOT[inputs['split'][0]][1],
        )
        
        # generation pipeline
        generation_config = GenerationConfig(
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            max_new_tokens=12*1024,
        )
        logits_processor = UnbatchedModalityLogitsProcessorBase(
            **self.processor.constant_helper(),
            processor=self.processor,
            templates=[('struct', inputs['struct_length'][0].item())]
        )
        generation_token_ids: torch.Tensor = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            generation_config=generation_config,
            logits_processor=[logits_processor],
        ) # type: ignore
        
        target_token_ids = inputs['labels'][0, prompt_length :]             # <struct>....</struct>
        generation_token_ids = generation_token_ids[0, prompt_length : -1]  # <struct>....</struct>
        generation_acc = (generation_token_ids == target_token_ids).float().mean().item()
        generation_bleu = sacrebleu.corpus_bleu(
            [" ".join(map(str, generation_token_ids.cpu().tolist()))],
            [[" ".join(map(str, target_token_ids.cpu().tolist()))]]
        ).score
        p_nature = OpenfoldProtein.from_file(Path(root)/f"{pdb_name}{format}").to(device)
        p_vq = self.processor.multimodal_decode(target_token_ids, ref=p_nature)['entity'][0].to(device)
        p_ar = self.processor.multimodal_decode(generation_token_ids, ref=p_nature)['entity'][0].to(device)
        tm_vq, rmsd_l_vq, rmsd_g_vq = self.processor.compute_tm_align(p_vq, p_nature, ref=p_nature)
        tm_ar, rmsd_l_ar, rmsd_g_ar = self.processor.compute_tm_align(p_ar, p_nature, ref=p_nature)
        
        # exposure pipeline
        exposure = model(
            input_ids=inputs['labels'],                     # <seq>....</seq><struct>....</struct>
            attention_mask=inputs['attention_mask'],       
        )                                                   # ....</seq><struct>....</struct><endoftext>
        exposure_logits: torch.Tensor = exposure.logits[0, prompt_length - 1 : -1, :] # type: ignore
        exposure_token_ids = exposure_logits.argmax(dim=-1)                 # <struct>....</struct>
        exposure_bleu = sacrebleu.corpus_bleu(
            [" ".join(map(str, exposure_token_ids.cpu().tolist()))],
            [[" ".join(map(str, target_token_ids.cpu().tolist()))]]
        ).score
        exposure_loss = torch.nn.functional.nll_loss(
            input=torch.log_softmax(exposure_logits, dim=-1),
            target=target_token_ids,
            reduction='mean'
        )
        exposure_acc = (exposure_token_ids == target_token_ids).float().mean().item()
            
        metrics = dict(
            # meta
            split=DATASET_SPLIT[split],
            # generation pipeline
            acc_gen=generation_acc,
            bleu_gen=generation_bleu,
            tm_vq=tm_vq,
            rmsd_l_vq=rmsd_l_vq,
            rmsd_g_vq=rmsd_g_vq,
            tm_ar=tm_ar,
            rmsd_l_ar=rmsd_l_ar,
            rmsd_g_ar=rmsd_g_ar,
            # exposure pipeline
            acc_eps=exposure_acc,
            bleu_eps=exposure_bleu,
            loss_eps=exposure_loss,
        )
        
        # format a logging message here
        logger.info(f"""Evaluated [{pdb_name}] from [{split}]:
Target Results:     {self.processor.tokenizer.decode(target_token_ids.cpu().tolist())}
Exposure Results:   {self.processor.tokenizer.decode(exposure_token_ids.cpu().tolist())}
Generation Results: {self.processor.tokenizer.decode(generation_token_ids.cpu().tolist())}
Exposure Loss:  {metrics['loss_eps']:.4f}
Exposure Acc:   {metrics['acc_eps']:.4f}, BLEU: {metrics['bleu_eps']:.4f}
Generation Acc: {metrics['acc_gen']:.4f}, BLEU: {metrics['bleu_gen']:.4f}
VQ v.s. Nature: TM-score = {metrics['tm_vq']:.4f}, RMSD_L = {metrics['rmsd_l_vq']:.4f}, RMSD_G = {metrics['rmsd_g_vq']:.4f}
AR v.s. Nature: TM-score = {metrics['tm_ar']:.4f}, RMSD_L = {metrics['rmsd_l_ar']:.4f}, RMSD_G = {metrics['rmsd_g_ar']:.4f}
""")    
        model.train()
        return (exposure_loss, {k:torch.tensor(v).to(device) for k, v in metrics.items()}, inputs['input_ids'])
    
    @classmethod
    def compute_metrics(cls, eval_pred: EvalPrediction):
        preds: Dict[str, np.ndarray] = eval_pred.predictions # type: ignore
        df = pd.DataFrame({k: v for k, v in preds.items()})
        metrics = {}
        for i, group in df.groupby('split'):
            prefix = list(DATASET_SPLIT.keys())[int(i)] # type: ignore
            metrics[prefix] = {
                'count':        len(group),
                'acc_gen':      group['acc_gen'].mean(),
                'bleu_gen':     group['bleu_gen'].mean(),
                'tm_vq':        group['tm_vq'].mean(),
                'rmsd_l_vq':    group['rmsd_l_vq'].mean(),
                'rmsd_g_vq':    group['rmsd_g_vq'].mean(),
                'tm_ar':        group['tm_ar'].mean(),
                'rmsd_l_ar':    group['rmsd_l_ar'].mean(),
                'rmsd_g_ar':    group['rmsd_g_ar'].mean(),
                'acc_eps':      group['acc_eps'].mean(),
                'bleu_eps':     group['bleu_eps'].mean(),
                'loss_eps':     group['loss_eps'].mean(),
            }
        return metrics
