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
    constant,
    logits,
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
            
            # bascially we do not truncate inputs during evaluation
            # special handling for 'uniref50' long sequences
            is_long_sequence = element['split'] in ['uniref50']
            outputs = tokenizer(
                element["text"] if not use_formatting_func else formatting_func(element),
                add_special_tokens=add_special_tokens,
                truncation=is_long_sequence,
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
            # unique task id
            tid=-1,
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
            # benchmark
            benchmark=-1,
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
Target   Sequence:          {self.processor.tokenizer.decode(target_token_ids.cpu().tolist()[:5])}...
Exposure Sequence:          <seq>{self.processor.tokenizer.decode(eps_token_ids.cpu().tolist()[:4])}...
Sequence Loss/Acc/Bleu:     {sequence_loss.item():.4f}/{sequence_acc:.4f}/{sequence_bleu:.4f}
""")
        metrics = dict(
            tid=0,
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
Target   Structure:         {self.processor.tokenizer.decode(target_token_ids.cpu().tolist()[:5])}...
Exposure Structure:         <struct>{self.processor.tokenizer.decode(eps_token_ids.cpu().tolist()[:4])}...
Structure Loss/Acc/Bleu:    {structure_loss.item():.4f}/{structure_acc:.4f}/{structure_bleu:.4f}
""")
        metrics = dict(
            tid=1,
            structure_loss = structure_loss.item(),
            structure_acc = structure_acc,
            structure_bleu = structure_bleu,
        )
        metrics = self.dummy_metrics | metrics
        model.train()
        return (output.loss, {k:torch.tensor(v).to(model.device) for k, v in metrics.items()}, inputs['input_ids'])
    
    
    # evaluation for protein sequence-to-structure:
    # - eval/p2s/sequence_loss
    # - eval/p2s/sequence_acc
    # - eval/p2s/sequence_bleu
    # - eval/p2s/folding_loss
    # - eval/p2s/folding_acc
    # - eval/p2s/folding_bleu
    # - eval/p2s/structure_loss
    # - eval/p2s/structure_acc
    # - eval/p2s/structure_bleu
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
        
        # modeling protein sequence-to-structure p(structure | sequence)
        # we measures: 
        # 1. whether condition is fully understood p(sequence)
        # 2. whether structure is correctly predicted p(structure | sequence)
        # 3. whether structure is fully conditioned on sequence p(structure | sequence) != p(structure)
        
        output = model(input_ids=inputs['labels'], attention_mask=inputs['attention_mask'], labels=inputs['labels'])
        copy_output_loss = output.loss
        
        # find the first </seq> token, split into sequence / structure part
        constant_helper = self.processor.constant_helper()
        prompt_length = torch.sum(
            (inputs['labels'][0] == constant_helper['eoseq_token_id']).cumsum(dim=0) == 0
        ).int().item() + 1 # include both <seq> & </seq> token
        
        # for seq2seq task, we manually compute token-wise exposure loss
        labels: torch.Tensor = inputs['labels'][0]                  # <seq>....</seq><struct>....</struct>
        logits: torch.Tensor = output.logits[0] # type: ignore      # ....</seq><struct>....</struct>
        log_softmax_logits = torch.log_softmax(logits, dim=-1)      # ....</seq><struct>....</struct><endoftext>
        eps_token_ids = logits.argmax(dim=-1)                       # ....</seq><struct>....</struct><endoftext>
        copy_labels = labels
        copy_token_ids = eps_token_ids
        
        # sequence condition
        sequence_loss = torch.nn.functional.nll_loss(input=log_softmax_logits[0 : prompt_length - 1, :], target=labels[1 : prompt_length])
        sequence_acc = (eps_token_ids[0 : prompt_length - 1] == labels[1 : prompt_length]).float().mean().item() * 100.0
        sequence_bleu = sacrebleu.corpus_bleu(
            [" ".join(map(str, eps_token_ids[0 : prompt_length - 1].cpu().tolist()))],
            [[" ".join(map(str, labels[1 : prompt_length].cpu().tolist()))]]
        ).score
        
        # structure w/ sequence condition
        folding_loss = torch.nn.functional.nll_loss(input=log_softmax_logits[prompt_length : -1, :], target=labels[prompt_length + 1 :])
        folding_acc = (eps_token_ids[prompt_length : -1] == labels[prompt_length + 1 :]).float().mean().item() * 100.0
        folding_bleu = sacrebleu.corpus_bleu(
            [" ".join(map(str, eps_token_ids[prompt_length : -1].cpu().tolist()))],
            [[" ".join(map(str, labels[prompt_length + 1 :].cpu().tolist()))]]
        ).score
        
        # structure wo/ sequence condition
        labels = inputs['labels'][:, prompt_length :] 
        output = model(input_ids=labels, attention_mask=torch.ones_like(labels))
        target_token_ids = labels[0]                                        # <struct>....</struct>
        structure_loss = torch.nn.functional.nll_loss(
            input=torch.log_softmax(output.logits[0], dim=-1)[0 : -1, :],   # ....</struct>
            target=target_token_ids[1 :]                                    # ....</struct>
        )
        eps_token_ids = output.logits[0, :-1].argmax(dim=-1)                # ....</struct><endoftext>
        structure_acc = (eps_token_ids == target_token_ids[1 :]).float().mean().item() * 100.0
        structure_bleu = sacrebleu.corpus_bleu(
            [" ".join(map(str, eps_token_ids.cpu().tolist()))],
            [[" ".join(map(str, target_token_ids[1 :].cpu().tolist()))]]
        ).score
        
        
        # logging metrics and the first 5 tokens
        logger.info(f"""
////// Evaluated [{inputs['pdb_name'][0]}] from [{inputs['split'][0]}] in {time.time() - start_time:.2f}s:) //////
Target    Sequence:         {self.processor.tokenizer.decode(copy_labels[:prompt_length].cpu().tolist()[:5])}...
Exposure  Sequence:         <seq>{self.processor.tokenizer.decode(copy_token_ids[:prompt_length - 1].cpu().tolist()[:4])}...
Target    Structure:        {self.processor.tokenizer.decode(copy_labels[prompt_length:].cpu().tolist()[:5])}...
Exposure  Structure:        <struct>{self.processor.tokenizer.decode(copy_token_ids[prompt_length:].cpu().tolist()[:4])}...
Sequence  Loss/Acc/Bleu:    {sequence_loss.item():.4f}/{sequence_acc:.4f}/{sequence_bleu:.4f}
Folding   Loss/Acc/Bleu:    {folding_loss.item():.4f}/{folding_acc:.4f}/{folding_bleu:.4f}
Structure Loss/Acc/Bleu:    {structure_loss.item():.4f}/{structure_acc:.4f}/{structure_bleu:.4f}
""")
        metrics = dict(
            tid=2,
            sequence_loss = sequence_loss.item(),
            sequence_acc = sequence_acc,
            sequence_bleu = sequence_bleu,
            folding_loss = folding_loss.item(),
            folding_acc = folding_acc,
            folding_bleu = folding_bleu,
            structure_loss = structure_loss.item(),
            structure_acc = structure_acc,
            structure_bleu = structure_bleu,
        )
        metrics = self.dummy_metrics | metrics
        model.train()
        return (copy_output_loss, {k:torch.tensor(v).to(model.device) for k, v in metrics.items()}, inputs['input_ids'])
    
    
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
        
        # find the first </seq> token, split into sequence / structure part
        constant_helper = self.processor.constant_helper()
        prompt_length = torch.sum((inputs['labels'][0] == constant_helper['eoseq_token_id']).cumsum(dim=0) == 0).int().item() + 1 # include both <seq> & </seq> token itself
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
            **self.processor.constant_helper(),
            processor=self.processor,
            templates=[('struct', inputs['struct_length'][0].item())]
        )
        ar = model.generate(
            input_ids=inputs["labels"][:, :prompt_length],
            attention_mask=inputs["attention_mask"],
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
        root, format = DATASET_RAW_ROOT[split][0], DATASET_RAW_ROOT[split][1]
        p_nature = OpenfoldProtein.from_file(Path(root)/f"{pdb_name}{format}").to(device)
        p_vq = self.processor.multimodal_decode(target_token_ids, ref=p_nature)['entity'][0].to(device)
        p_ar = self.processor.multimodal_decode(ar_token_ids, ref=p_nature)['entity'][0].to(device)
        tm_vq, rmsd_l_vq, rmsd_g_vq = self.processor.compute_tm_align(p_vq, p_nature, ref=p_nature)
        tm_ar, rmsd_l_ar, rmsd_g_ar = self.processor.compute_tm_align(p_ar, p_nature, ref=p_nature)
        
        metrics = dict(
            tid=3,
            benchmark=DATASET_SPLIT[split],
            ar_loss=ar_loss.item(),
            ar_acc=ar_acc,
            ar_bleu=ar_bleu,
            tm_ar=tm_ar,
            tm_vq=tm_vq,
            rmsd_ar_l=rmsd_l_ar,
            rmsd_vq=rmsd_l_vq,
        )
        metrics = self.dummy_metrics | metrics
        
        logger.info(
f"""////// Evaluated [{inputs['pdb_name'][0]}] from [{inputs['split'][0]}] in {time.time() - start_time:.2f}s:) //////
Target  Structure:          {self.processor.tokenizer.decode(target_token_ids.cpu().tolist()[:4])}...
AR      Structure:          {self.processor.tokenizer.decode(ar_token_ids.cpu().tolist()[:4])}...
AR Loss/Acc/Bleu:           {ar_loss.item():.4f}/{ar_acc:.4f}/{ar_bleu:.4f}
VQ v.s. Nature: TM-score = {tm_vq:.4f}, RMSD_L = {rmsd_l_vq:.4f}, RMSD_G = {rmsd_g_vq:.4f}
AR v.s. Nature: TM-score = {tm_ar:.4f}, RMSD_L = {rmsd_l_ar:.4f}, RMSD_G = {rmsd_g_ar:.4f}
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
        logger.warning(split)
        if split in ['p/uniref50']:
            return self._prediction_step_plm(model, inputs, prediction_loss_only, ignore_keys)
        elif split in ['s/unicluster40']:
            return self._prediction_step_slm(model, inputs, prediction_loss_only, ignore_keys)
        elif split in ['p2s/unicluster40']:
            return self._prediction_step_p2s(model, inputs, prediction_loss_only, ignore_keys)
        else:
            # benchmarking, combine p2s + folding evaluation
            loss1, metrics1, inputs1 = self._prediction_step_p2s(model, inputs, prediction_loss_only, ignore_keys)
            loss2, metrics2, inputs2 = self._prediction_step_folding(model, inputs, prediction_loss_only, ignore_keys)
            # merge metrics
            metrics1_keys = ['sequence_loss', 'sequence_acc', 'sequence_bleu',
                             'folding_loss', 'folding_acc', 'folding_bleu',
                             'structure_loss', 'structure_acc', 'structure_bleu']
            metrics2_keys = ['tid', 'benchmark',
                             'ar_loss', 'ar_acc', 'ar_bleu',
                             'tm_ar', 'tm_vq',
                             'rmsd_ar', 'rmsd_vq']
            merged_metrics = {k: metrics1[k] for k in metrics1_keys}
            merged_metrics.update({k: metrics2[k] for k in metrics2_keys})
            return (loss1, merged_metrics, inputs1)
    

    @classmethod
    def compute_metrics(cls, eval_pred: EvalPrediction):
        preds: Dict[str, np.ndarray] = eval_pred.predictions # type: ignore
        df = pd.DataFrame({k: v for k, v in preds.items()})
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
                metrics['p2s/sequence_loss']  = group['sequence_loss'].mean()
                metrics['p2s/sequence_acc']   = group['sequence_acc'].mean()
                metrics['p2s/sequence_bleu']  = group['sequence_bleu'].mean()
                metrics['p2s/folding_loss']   = group['folding_loss'].mean()
                metrics['p2s/folding_acc']    = group['folding_acc'].mean()
                metrics['p2s/folding_bleu']   = group['folding_bleu'].mean()                
                metrics['p2s/structure_loss'] = group['structure_loss'].mean()
                metrics['p2s/structure_acc']  = group['structure_acc'].mean()
                metrics['p2s/structure_bleu'] = group['structure_bleu'].mean()
            elif tid == 3:
                # another group by `split` field
                for split, split_group in group.groupby('benchmark'):
                    split_name = list(DATASET_SPLIT.keys())[int(split)] # type: ignore
                    prefix = f"folding/{split_name}/"
                    # from ar setting
                    metrics[prefix + 'tm_ar']     = split_group['tm_ar'].mean()
                    metrics[prefix + 'tm_vq']     = split_group['tm_vq'].mean()
                    metrics[prefix + 'rmsd_ar']   = split_group['rmsd_ar'].mean()
                    metrics[prefix + 'rmsd_vq']   = split_group['rmsd_vq'].mean()
                    metrics[prefix + 'ar_loss']   = split_group['ar_loss'].mean()
                    metrics[prefix + 'ar_acc']    = split_group['ar_acc'].mean()
                    metrics[prefix + 'ar_bleu']   = split_group['ar_bleu'].mean()
                    # from exposure setting
                    metrics[prefix + 'sequence_loss']   = split_group['sequence_loss'].mean()
                    metrics[prefix + 'sequence_acc']    = split_group['sequence_acc'].mean()
                    metrics[prefix + 'sequence_bleu']   = split_group['sequence_bleu'].mean()
                    metrics[prefix + 'folding_loss']    = split_group['folding_loss'].mean()
                    metrics[prefix + 'folding_acc']     = split_group['folding_acc'].mean()
                    metrics[prefix + 'folding_bleu']    = split_group['folding_bleu'].mean()
                    metrics[prefix + 'structure_loss']  = split_group['structure_loss'].mean()
                    metrics[prefix + 'structure_acc']   = split_group['structure_acc'].mean()
                    metrics[prefix + 'structure_bleu']  = split_group['structure_bleu'].mean()
            else:
                raise NotImplementedError()
        return metrics
    
