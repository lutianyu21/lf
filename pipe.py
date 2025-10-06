from multiprocessing import reduction
import token
from typing import Any, Dict, Optional, List, Text, Tuple, cast
import hydra
from openfold.np.protein import Protein
import torch
import os
import pandas as pd
import wandb
import torch
import torch.utils
import torch.utils.data
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

import datasets
from datasets import Dataset, load_dataset, Features, Value, ClassLabel, Sequence
from transformers import Trainer, TrainingArguments, TrainerCallback, is_datasets_available
from transformers.generation.configuration_utils import GenerationConfig
from transformers import PreTrainedModel
from transformers import EvalPrediction

from utils.dplm_utils.dplm import train
from utils.openfold_utils.io import OpenfoldProtein
from utils.progen2_utils import ProGenForCausalLM, ProGenConfig

from utils.lf_utils import (
    lf_tokenizer,
    dplm_protein_tokenizer,
    ProteinProcessor, 
    SortishApproxBatchDataloader,
    TextCollator,
    DynamicMultimodalLogitsProcessor
)


class LFTrainer(Trainer):
    
    def __init__(
        self,
        processor: ProteinProcessor,
        train_collator: Any,
        eval_collator: Any,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.train_collator = train_collator
        self.eval_collator = eval_collator
        self.eval_config = GenerationConfig(
            use_cache=True,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            bos_token_id=self.processor.tokenizer.bos_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            do_sample=False,
            max_new_tokens=2048,
        )
    
    def compute_loss(self, model, inputs: Dict[str, torch.Tensor], return_outputs=False):
        
        # ! we now no-longer relies on model forward to compute auxiliary loss !
        outputs = model(**inputs)
        
        aux_log = {}
        input_ids = cast(torch.Tensor, inputs['input_ids']) # [B, L], 
        labels = cast(torch.Tensor, inputs['labels'])       # [B, L], w/ left-pading(-100), wo/ shift
        logits = cast(torch.Tensor, outputs.logits)         # [B, L, V]

        # match any <boseq>...<eoseq> segments & <bostruct>...<eostruct> segments
        # calculate loss on these segments
        boseq_token_id = self.processor.tokenizer.boseq_token_id
        eoseq_token_id = self.processor.tokenizer.eoseq_token_id
        bostruct_token_id = self.processor.tokenizer.bostruct_token_id
        eostruct_token_id = self.processor.tokenizer.eostruct_token_id
        seq_mask = (
            (labels == boseq_token_id).cumsum(dim=1) - \
            (labels == eoseq_token_id).cumsum(dim=1) + \
            (labels == eoseq_token_id).float()
        ) 
        struct_mask = (
            (labels == bostruct_token_id).cumsum(dim=1) - \
            (labels == eostruct_token_id).cumsum(dim=1) + \
            (labels == eostruct_token_id).float()
        )
        shift_logits = logits[..., :-1, :].contiguous()     # [B, L-1, V]
        shift_labels = labels[..., 1:].contiguous()         # [B, L-1]
        shift_seq_mask = seq_mask[..., 1:].contiguous()     # [B, L-1]
        shift_struct_mask = struct_mask[..., 1:].contiguous() # [B, L-1]
        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
        loss: torch.Tensor = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ) # [B*(L-1)]
        seq_loss = loss[shift_seq_mask.view(-1).bool()].mean()
        struct_loss = loss[shift_struct_mask.view(-1).bool()].mean()
        aux_log['seq_loss'] = seq_loss.detach().cpu().item()
        aux_log['struct_loss'] = struct_loss.detach().cpu().item()
        
        # supervise GPU memory usage
        token_padded = labels.size(0) * labels.size(1)      # [BxL]
        token2_padded = labels.size(0) * labels.size(1)**2  # [BxLxL]
        token_nonpad = token_padded - (labels == -100).sum().item()
        aux_log['token_padded'] = token_padded
        aux_log['token2_padded'] = token2_padded
        aux_log['token_nonpad'] = token_nonpad
        aux_log['bsz'] = input_ids.size(0)
        
        aux_log['seq_length'] = inputs['seq_length'].float().mean().cpu().item()
        aux_log['struct_length'] = inputs['struct_length'].float().mean().cpu().item()
        if self.is_in_train:
            self.log(aux_log)
        outputs.aux_log = aux_log
        
        return (outputs.loss, outputs) if return_outputs else outputs.loss
        
    # during training: dynamic length-batching sampler
    # DEV: adjust these parameters based on architecture
    def get_train_dataloader(self):
        return SortishApproxBatchDataloader(
            ds=self.train_dataset,
            collater=self.train_collator,
            bucket_size=1000,
            max_batch_size=100,
            max_tokens=10000,
            max_square_tokens=3000000,
            max_len=2048,
        )
    
    # during evaluation: default padding sampler
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
        model: PreTrainedModel,
        inputs: Dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ):
        model.eval()
        # all batched
        device = inputs['input_ids'].device
        self.processor.to(device)

        labels = inputs["labels"]
        # exposure evaluation (training pipeline)
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=labels,
        )
        exposure_loss: torch.Tensor = outputs.loss # []
        
        # generation evaluation
        # step1: generate tokens
        logits_processor = DynamicMultimodalLogitsProcessor(
            **(self.processor.constant_helper()), # type: ignore
            seq_length=inputs['seq_length'],
            struct_length=inputs['struct_length']
        )
        generated_token_ids: torch.Tensor = model.generate(
            input_ids=inputs["eval_input_ids"],
            attention_mask=inputs["eval_attention_mask"],
            generation_config=self.eval_config,
            logits_processor=[logits_processor], # type: ignore
        ) # type: ignore [B, L]
        
        # <|pad|><|pad|><|bos|><|boseq|>...<|eoseq|><|bostruct|>...<|eostruct|><|eos|>
        # <|bos|><|boseq|>...<|eoseq|><|bostruct|>...<|eostruct|><|eos|><|pad|>
        total_length = inputs['total_length'] # [B]
        generated_token_ids = torch.where(generated_token_ids == self.processor.tokenizer.pad_token_id, -100, generated_token_ids)
        
        
        # step2: collect proteins 
        pdb_name, split = inputs["pdb_name"], inputs["split"]
        protein_collect = [
            OpenfoldProtein.from_file(Path(f"/AIRvePFS/ai4science/users/tianyu/lf/data/swissprot_cif_v4/{x}.cif.gz")).to(device)
            if y == "afdb_swissprot"
            else OpenfoldProtein.from_file(Path(f"/AIRvePFS/ai4science/users/tianyu/lf/data/rcsb_mmcif/{x}.cif")).to(device)
            for x, y in zip(pdb_name, split)
        ]
        
        tmp = {
            'tm_rec':       [],
            'rmsd_rec':     [],
            'tm_dec':       [],
            'rmsd_dec':     [],
            'tm_gen':       [],
            'rmsd_gen':     [],
            'token_acc':    [],
        }
        preds = {}

        for b in range(labels.size(0)):
            x1: torch.Tensor = generated_token_ids[b][generated_token_ids[b] != -100]   # strip both left & right padding, [l]
            x2: torch.Tensor = labels[b][labels[b] != -100]                             # strip both left padding, [l]
            assert x1.size(0) == x2.size(0)
            
            o1 = self.processor.multimodal_decode(x1, ref=protein_collect[b])
            o2 = self.processor.multimodal_decode(x2, ref=protein_collect[b])
            
            p1: OpenfoldProtein = o1['entity'][0]
            p2: OpenfoldProtein = o2['entity'][0]
            p3: OpenfoldProtein = protein_collect[b]
            
            tm_rec, rmsd_rec = self.processor.compute_tm_align(p2, p3, ref=p3)
            tm_dec, rmsd_dec = self.processor.compute_tm_align(p1, p2, ref=p3)
            tm_gen, rmsd_gen = self.processor.compute_tm_align(p1, p3, ref=p3)
            tmp['tm_rec'].append(tm_rec)
            tmp['rmsd_rec'].append(rmsd_rec)
            tmp['tm_dec'].append(tm_dec)
            tmp['rmsd_dec'].append(rmsd_dec)
            tmp['tm_gen'].append(tm_gen)
            tmp['rmsd_gen'].append(rmsd_gen)
            
            # token-accuracy
            token_acc = x1.eq(x2).float().mean()
            tmp['token_acc'].append(token_acc.cpu().item())
            
        # convert preds to tensor
        for k in tmp.keys():
            preds[k] = torch.tensor(tmp[k], device=device) # all batched [B]
        
        model.train()
        return (exposure_loss, preds, labels)
        

def lf_metrics(eval_pred: EvalPrediction):
    # simply collect and calculate me
    metrics = {
        k: float(v.mean()) for k, v in eval_pred.predictions.items() # type: ignore
    }
    return metrics


@hydra.main(version_base=None, config_path="./config", config_name="config.yaml")
def main(config: DictConfig):
    
    test = Path('/AIRvePFS/ai4science/users/tianyu/lf/data/rcsb_mmcif/7k68.cif')
    # test whether flie exists
    print("================")
    print(test.exists())
    print(OpenfoldProtein.from_file(test))
    
    
    return
    
    config_dataset, config_lm, config_trainer = config.dataset, config.lm, config.trainer
    config.name = "M{}_D{}_B{}x{}x{}".format(
        config_lm.get('hf_model_type', 'dummy'),
        config_dataset.get('hf_data_type', 'dummy'),
        int(os.environ["WORLD_SIZE"]),
        'dyn',
        config_trainer.get('gradient_accumulation_steps', 1)
    )
    config_trainer.output_dir = f'/AIRvePFS/ai4science/users/tianyu/lf/output/checkpoints/{config.name}'

    if (rank := int(os.environ.get("RANK", 0))) == 0:
        wandb.init(project="LLMFolding", name=config.name, config=OmegaConf.to_container(config, resolve=True)) # type: ignore
    
    # HINT: ProGen2 didn't implement `get_output_embeddings()` and therefore 
    # `model.tie_weights()` inside/outside `from_pretrained()` is actually dummy!
    hf_config: ProGenConfig = ProGenConfig.from_pretrained(Path(config_lm.hf_checkpoint_dir))                                      # type: ignore
    # hf_model: ProGenForCausalLM = ProGenForCausalLM.from_pretrained(Path(config_lm.pretrained_dir), torch_dtype=torch.float32) # type: ignore
    hf_model: ProGenForCausalLM = ProGenForCausalLM(hf_config)
    # hf_model.tie_weights() # ensurement
    hf_model.resize_token_embeddings(config_lm.new_vsz)
    hf_model.train()
    
    # monomeric dataset
    features = Features({
        'pdb_name': Value('string'),
        'split': Value('string'),
        'text': Value('string'),
        'prompt': Value('string'),
        'seq_text': Value('string'),
        'struct_text': Value('string'),
        'seq_length': Value('int32'),
        'struct_length': Value('int32'),
        'total_length': Value('int32'),
        'oligomeric_count': Value('int32'),
        'oligomeric_detail': Value('string'),
        'coil_percent': Value('float32'),
        'helix_percent': Value('float32'),
        'strand_percent': Value('float32'),
        'radius_gyration': Value('float32'),
        'avg_plddt': Value('float32'),
    })
    ds = load_dataset("json", data_files=config_dataset.hf_data_dir, split="train", features=features) # type: ignore
    # currently, our context length <= 2048
    ds = ds.filter(lambda x: x['total_length'] <= 2048)         # type: ignore
    split = ds.train_test_split(test_size=100, seed=2025)      # type: ignore
    train_dataset, eval_dataset = split['train'], split['test']
    print(f"train: {len(train_dataset)} items, eval: {len(eval_dataset)} items")
    
    # hf-style trainer
    training_args = TrainingArguments(**config_trainer, remove_unused_columns=False)
    processor = ProteinProcessor(tokenizer=lf_tokenizer, struct_tokenizer=dplm_protein_tokenizer)
    trainer = LFTrainer(
        processor=processor,
        model=hf_model,
        train_collator=TextCollator(processor, eval_mode=False),
        eval_collator=TextCollator(processor, eval_mode=True),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=lf_metrics,
    )
    trainer.train()

if __name__ == "__main__":
    main()
