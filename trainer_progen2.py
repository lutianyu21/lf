from typing import Any, Dict, Optional, List, Text, Tuple
import hydra
import torch
import os
import pandas as pd
import wandb
import torch
import torch.utils
import torch.utils.data
import torch.distributed as dist
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

import datasets
from datasets import Dataset, load_dataset
from transformers import Trainer, TrainingArguments, TrainerCallback, is_datasets_available
from transformers.generation.configuration_utils import GenerationConfig

from data.utils import SortishApproxBatchDataloader, TextCollator
from utils.progen2_utils import ProGenForCausalLM, ProGenConfig, progen2_merged_tokenizer



class TrainerWithCustomLoss(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        aux_log = {}        
        outputs = model(**inputs)
        if (pstruct_loss := outputs.struct_loss) is not None:   aux_log["pstruct_loss"] = pstruct_loss.detach().cpu().item()
        if (pseq_loss := outputs.seq_loss) is not None:         aux_log["pseq_loss"] = pseq_loss.detach().cpu().item()
        if (length := inputs['input_lengths']) is not None:            
            aux_log['input_lengths'] = length.float().mean().cpu().item()
            aux_log['input_lengths_raw'] = inputs['input_lengths_raw'].float().mean().cpu().item()
            aux_log['input_lengths_batch'] = inputs['labels'].shape[-1]
            aux_log['bsz'] = length.shape[0]
        if self.is_in_train:
            self.log(aux_log)
        outputs.aux_log = aux_log
        return (outputs.loss, outputs) if return_outputs else outputs.loss
    
    def get_train_dataloader(self):
        return SortishApproxBatchDataloader(
            ds=self.train_dataset,
            collater=TextCollator(progen2_merged_tokenizer),
            bucket_size=1000,
            max_batch_size=100,
            max_tokens=10000,
            max_square_tokens=3000000,
            max_len=2048,
        )


@hydra.main(version_base=None, config_path="./config", config_name="config.yaml")
def main(cfg: DictConfig):
    
    cfg_dataset, cfg_lm, cfg_trainer = cfg.dataset, cfg.lm, cfg.trainer    
    cfg.name = "M{}_D{}_B{}x{}x{}".format(
        cfg_lm.get('hf_model_type', 'dummy'),
        cfg_dataset.get('hf_data_type', 'dummy'),
        int(os.environ["WORLD_SIZE"]),
        'dyn',
        cfg_trainer.get('gradient_accumulation_steps', 1)
    )
    cfg_trainer.output_dir = f'/AIRvePFS/ai4science/users/tianyu/lf/output/checkpoints/{cfg.name}'
    
    # facilitate wandb
    if (rank := int(os.environ.get("RANK", 0))) == 0:
        wandb.init(project="LLMFolding", name=cfg.name, config=OmegaConf.to_container(cfg, resolve=True)) # type: ignore
    
    # HINT: ProGen2 didn't implement `get_output_embeddings()` and therefore 
    # `model.tie_weights()` inside/outside `from_pretrained()` is actually dummy!
    hf_config: ProGenConfig = ProGenConfig.from_pretrained(Path(cfg_lm.hf_checkpoint_dir))                                      # type: ignore
    # hf_model: ProGenForCausalLM = ProGenForCausalLM.from_pretrained(Path(cfg_lm.pretrained_dir), torch_dtype=torch.float32) # type: ignore
    hf_model: ProGenForCausalLM = ProGenForCausalLM(hf_config)
    # hf_model.tie_weights() # ensurement
    hf_model.resize_token_embeddings(cfg_lm.new_vocab_size)
    hf_model.train()
    
    # monomeric dataset
    full_dataset = load_dataset("json", data_files=cfg_dataset.hf_data_dir, split="train")
    length_dataset: Any = full_dataset.filter(lambda item: cfg_dataset.get("min_len", 0) <= item['length'] <= cfg_dataset.get("max_len", 2048))
    dataset = length_dataset
    split = dataset.train_test_split(test_size=0.1, seed=2025)
    train_dataset, eval_dataset = split['train'], split['test']
    print(f"train: {len(train_dataset)} items, eval: {len(eval_dataset)} items")
    
    # hf-style trainer
    training_args = TrainingArguments(**cfg_trainer, remove_unused_columns=False)
    tokenizer = progen2_merged_tokenizer
    trainer = TrainerWithCustomLoss(
        args=training_args,
        model=hf_model,
        data_collator=TextCollator(tokenizer),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()

if __name__ == "__main__":
    main()
