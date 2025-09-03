import functools
import warnings
from multiprocessing import process
from typing import Any, Dict, Optional, List, Tuple
import hydra
import torch
import os
import pandas as pd
import wandb
import torch
import torch.utils
import torch.utils.data
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from transformers import Trainer, TrainingArguments, TrainerCallback, is_datasets_available
from transformers.generation.configuration_utils import GenerationConfig
import datasets
from datasets import Dataset
from data.datasets import DPLMCollator
from utils.transformers_utils import DynamicMultimodalLogitsProcessor
from utils.dplm_utils import DPLMProcessor, dplm_tokenizer
from utils.openfold_utils.io import OpenfoldEntity
from utils.progen2_utils import ProGenForCausalLM, progen2_merged_tokenizer


class TrainerWithCustomEval(Trainer):
    
    def __init__(self, *args, generation_config, eval_collator, processor, **kwargs):
        super().__init__(*args, **kwargs)
        self.generation_config = generation_config
        self.eval_collator = eval_collator
        self.processor = processor
        
    def get_eval_dataloader(self, eval_dataset: Any = None) -> torch.utils.data.DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        if hasattr(self, "_eval_dataloader") and self.args.dataloader_persistent_workers:
            return self.accelerator.prepare(self._eval_dataloader)
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.eval_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

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

    @torch.no_grad()
    def prediction_step(
        self,
        model,
        inputs: Dict[str, Any],
        prediction_loss_only,
        ignore_keys=None,
    ):
        if prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        labels, lengths = inputs["labels"], inputs["lengths"]
        model.eval()    
        logits_processor = DynamicMultimodalLogitsProcessor(**self.processor.constant_helper(), batch_length=lengths.tolist())
        generated_tokens = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            logits_processor=[logits_processor],
            generation_config=self.generation_config,
        )
        model.train()
        
        # to keep padding consistent
        constant = self.processor.constant_helper()
        generated_tokens = torch.where(generated_tokens == constant['pad_token'], -100, generated_tokens)
        
        return (None, generated_tokens, labels)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False
    ):
        outputs = model(**inputs)
        log_dict = {}
        if (seq_loss := outputs.seq_loss) is not None:
            log_dict["seq_loss"] = seq_loss.detach().cpu().item()
        if (struct_loss := outputs.struct_loss) is not None:
            log_dict["struct_loss"] = struct_loss.detach().cpu().item()
        if log_dict:
            self.log(log_dict)
        return (outputs.loss, outputs) if return_outputs else outputs.loss



def compute_generation_metrics(eval_preds, dplm_processor: DPLMProcessor):
    
    rmsd, acc, length = [], [], []
    gen_ids:   torch.Tensor = eval_preds.predictions
    label_ids: torch.Tensor = eval_preds.label_ids
    assert label_ids.shape == gen_ids.shape
    
    for i, (token, label) in enumerate(zip(gen_ids, label_ids)):
        
        gen_wo_pad, label_wo_pad = token[token != -100], label[label != -100]
        _, _, gen_structure_list = dplm_processor.decode(gen_wo_pad.tolist())
        _, _, label_structure_list = dplm_processor.decode(label_wo_pad.tolist())
        if len(gen_structure_list) > 1:
            warnings.warn('prediction contains multiple structures, only the first one will be used')
            
        gen_structure:   Tuple[str, OpenfoldEntity] = gen_structure_list[0]
        label_structure: Tuple[str, OpenfoldEntity] = label_structure_list[0]
        
        length.append(len(str(gen_structure[1])))
        acc.append(dplm_processor.compute_acc(gen_structure[0], label_structure[0]))
        rmsd.append(dplm_processor.compute_rmsd(gen_structure[1], label_structure[1]))
        # HINT: this rmsd ignores reconstruction error of structure-tokenizer
            
    return {
        "rmsd":     np.mean(rmsd),
        "acc":      np.mean(acc),
        "length":   np.mean(length)
    }


@hydra.main(version_base=None, config_path="./config", config_name="config.yaml")
def main(cfg: DictConfig):
    wandb.init(project="LLMFolding", name=cfg.name, config=OmegaConf.to_container(cfg, resolve=True)) # type: ignore
    cfg_dataset, cfg_lm, cfg_trainer = cfg.dataset, cfg.lm, cfg.trainer    
    
    # HINT: ProGen2 didn't implement `get_output_embeddings()` and therefore `model.tie_weights()`
    # inside/outside `from_pretrained()` is actually dummy!
    model: ProGenForCausalLM = ProGenForCausalLM.from_pretrained(Path(cfg_lm.pretrained_dir), torch_dtype=torch.float32) # type: ignore
    model.tie_weights() # ensurement
    model.resize_token_embeddings(cfg_lm.new_vocab_size)
    model.train()
    
    # TODO update files 
    csv = pd.read_csv(cfg_dataset.data_dir)
    csv = csv[csv.oligomeric_detail == 'monomeric']
    dataset = Dataset.from_dict({"mmcif_path": [str(Path(cfg_dataset.data_dir).parent/'rcsb_mmcif'/f'{p}.cif') for p in csv.pdb_name]})
    split_dataset = dataset.train_test_split(test_size=0.001, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    print(f"train_dataset size: {len(train_dataset)}, eval_dataset size: {len(eval_dataset)}")
    
    # TODO update processor
    structure_tokenizer = dplm_tokenizer.to(torch.device(int(os.environ["LOCAL_RANK"])))
    processor = DPLMProcessor(
        tokenizer=progen2_merged_tokenizer,
        structure_tokenizer=structure_tokenizer
    )

    training_args = TrainingArguments(**cfg_trainer, remove_unused_columns=False,)
    GENERATION_CONFIG = GenerationConfig(
        use_cache=True,
        eos_token_id=progen2_merged_tokenizer.eos_token_id,
        bos_token_id=progen2_merged_tokenizer.bos_token_id,
        pad_token_id=progen2_merged_tokenizer.pad_token_id,
        do_sample=True,
        top_k=2048,
        temperature=0.7,
        top_p=0.4,
        max_new_tokens=512,
    )
    trainer = TrainerWithCustomEval(
        generation_config=GENERATION_CONFIG,
        data_collator=DPLMCollator(processor, mode='train', train_task=cfg_dataset.train_task),
        eval_collator=DPLMCollator(processor, mode='eval', eval_task=cfg_dataset.eval_task),
        processor=processor,
        compute_metrics=functools.partial(compute_generation_metrics, dplm_processor=processor),
        model=model,
        args=training_args,
        train_dataset=eval_dataset,     # type: ignore
        eval_dataset=eval_dataset,      # type: ignore
    )
    trainer.train()

if __name__ == "__main__":
    main()
