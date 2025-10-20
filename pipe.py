from typing import Any, Dict, Optional, List, Text, Tuple, cast
import os
from pathlib import Path
import wandb
import pandas as pd

import numpy as np
import torch
import torch.utils
import torch.utils.data
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

import hydra
import logging
import colorlog
from omegaconf import OmegaConf, DictConfig
import datasets
from datasets import Dataset, load_dataset, Features, Value, ClassLabel, Sequence
from transformers import AutoConfig, Trainer, TrainingArguments, TrainerCallback, is_datasets_available
from transformers.generation.configuration_utils import GenerationConfig
from transformers import PreTrainedModel
from transformers import EvalPrediction
from tokenizers import Tokenizer

from utils.dplm_utils.dplm import train
from utils.lf_utils.protein_tokenizer import DistMatrixTokenizer
from utils.openfold_utils import OpenfoldProtein
from utils.lf_utils import (
    DistMatrixTokenizer,
    DPLMProteinTokenizer,
    TextTokenizer,
    ProteinProcessor, 
    SortishApproxBatchDataloader,
    TextCollator,
    DynamicMultimodalLogitsProcessor
)

# fix bug
# log color whenrank=0 & silent when rank>0
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
            max_new_tokens=5000,
        )

    def compute_loss(self, model, inputs: Dict[str, torch.Tensor], return_outputs=False, num_items_in_batch=None):
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
            max_batch_size=256,
            max_tokens=256000,
            max_square_tokens=256000000,
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
        template = {
            "afdb_swissprot":   "data/swissprot_cif_v4/{x}.cif.gz",
            "pdb":              "data/rcsb_mmcif/{x}.cif",
            "cameo2022":        "data/rcsb_mmcif/{x}.cif"
        }
        protein_collect = [
            OpenfoldProtein.from_file(Path(__file__).parent/template[y].format(x=x)).to(device)
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
        preds['dev'] = inputs['dev'] # [B]
        
        model.train()
        return (exposure_loss, preds, labels)
        

def lf_metrics(eval_pred: EvalPrediction):
    preds: Dict[str, np.ndarray] = eval_pred.predictions # type: ignore
    # group average by `dev` field in `preds`
    # add prefix `overfit`(when `dev`=1) or `test`(when `dev`=2)
    df = pd.DataFrame({k: v for k, v in preds.items()})
    metrics = {}
    for dev, group in df.groupby('dev'):
        if dev == 3: prefix = 'cameo2022'
        elif dev == 2: prefix = 'evalx100'
        elif dev == 1: prefix = 'overfitx100'
        else: prefix = 'train'
        metrics[f'{prefix}_tm_rec'] = group['tm_rec'].mean()
        metrics[f'{prefix}_tm_dec'] = group['tm_dec'].mean()
        metrics[f'{prefix}_tm_gen'] = group['tm_gen'].mean()
        metrics[f'{prefix}_rmsd_rec'] = group['rmsd_rec'].mean()
        metrics[f'{prefix}_rmsd_dec'] = group['rmsd_dec'].mean()
        metrics[f'{prefix}_rmsd_gen'] = group['rmsd_gen'].mean()
        metrics[f'{prefix}_token_acc'] = group['token_acc'].mean()
    return metrics


@hydra.main(version_base=None, config_path="./config", config_name="config.yaml")
def main(config: DictConfig):
    
    config_dataset, config_lm, config_trainer = config.dataset, config.lm, config.trainer
    config.name = "M{}_D{}_B{}x{}x{}".format(
        config_lm.get('model_type', 'dummy'),
        config_dataset.get('dataset_type', 'dummy'),
        int(os.environ["WORLD_SIZE"]), 'dyn', config_trainer.get('gradient_accumulation_steps', 1)
    )
    config_trainer.output_dir = str(Path(__file__).parent/f'output/checkpoints/{config.name}')
    if (rank := int(os.environ.get("RANK", 0))) == 0:
        wandb.init(project="LLMFolding", name=config.name, config=OmegaConf.to_container(config, resolve=True)) # type: ignore
    
    # exp1: dplm tokenizer + progen2 lm + dplm dataset
    # exp2: dist tokenizer + progen2 lm + dist dataset
    protein_tokenizer = {
        "dist": DistMatrixTokenizer,
        "dplm": DPLMProteinTokenizer,
    }[str(config_dataset.dataset_type).split('_')[-1]].get_instance()
    text_tokenizer = TextTokenizer(
        tokenizer_object=Tokenizer.from_file(str(Path(__file__).parent/'utils/progen2_utils/progen/progen2/tokenizer.json')),
        pad_token='<|pad|>',
        bos_token='<|bos|>',
        eos_token='<|eos|>',
        padding_side='left',
        struct_vsz=protein_tokenizer.vsz,
    )
    processor = ProteinProcessor(
        tokenizer=text_tokenizer,
        struct_tokenizer=protein_tokenizer
    )
    
    # find any file ends with `.pt` `.pth` `.bin` under the model_dir
    model_dir = Path(config_lm.model_dir)
    model_files = list(model_dir.rglob("*.pt")) + list(model_dir.rglob("*.pth")) + list(model_dir.rglob("*.bin")) + list(model_dir.rglob("*.safetensors"))
    
    # https://github.com/enijkamp/progen2
    if config_lm.model_type.startswith('progen'):
        from utils.progen2_utils import ProGenForCausalLM, ProGenConfig
        if not model_files:
            logger.info('Training ProGen2 from scratch ...')
            model = ProGenForCausalLM(config=ProGenConfig.from_pretrained(model_dir))
        else:
            logger.info(f'Loading ProGen2 from {model_dir} ...')
            model = ProGenForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32) # type: ignore
        
        if config_trainer.get('gradient_checkpointing', True):
            logger.warning(
                'Progen2 transformers gradient_checkpointing API not implemented yet, manually enable gradient_checkpoint config instead'
            )
            OmegaConf.set_struct(config_trainer, False)
            config_trainer.pop("gradient_checkpointing", False)
            OmegaConf.set_struct(config_trainer, True)
            model.transformer.config.gradient_checkpointing = True
        model.resize_token_embeddings(text_tokenizer.vocab_size)

    # https://github.com/QwenLM/Qwen3
    elif config_lm.model_type.startswith('qwen'):
        from transformers import AutoConfig, AutoModelForCausalLM
        if not model_files:
            raise NotImplementedError('Something wrong with memeory')
            model_config = AutoConfig.from_pretrained(model_dir)
            model_config.bos_token_id = text_tokenizer.bos_token_id
            model_config.eos_token_id = text_tokenizer.eos_token_id
            model_config.pad_token_id = text_tokenizer.pad_token_id
            model_config.vocab_size = text_tokenizer.vocab_size
            model = AutoModelForCausalLM.from_config(
                config=model_config,
                attn_implementation="eager"
            )
        else:
            logger.info(f'[Info] Loading Qwen3 from {model_dir} ...')
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2"
            )
            model.config.bos_token_id = text_tokenizer.bos_token_id
            model.config.eos_token_id = text_tokenizer.eos_token_id
            model.config.pad_token_id = text_tokenizer.pad_token_id
            model.resize_token_embeddings(text_tokenizer.vocab_size)
            
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
    
    ds = load_dataset("json", data_files=config_dataset.dataset_train, split="train", features=features) # type: ignore
    ds = ds.filter(lambda x: x['seq_length'] <= 512)            # type: ignore
    
    # here we construct a hybrid evaluation dataset
    # including 100 items from split['test] and 100 items from split['train']
    split = ds.train_test_split(test_size=100, seed=2025)      # type: ignore
    train_dataset, eval_dataset = split['train'], split['test']
    subsplit = train_dataset.train_test_split(test_size=100, seed=2025) # type: ignore
    overfit_dataset = subsplit['test']
    cameo_dataset: Any = load_dataset("json", data_files=config_dataset.dataset_test, split="train", features=features) # type: ignore
    
    # however we need to add a new field 'dev' to distinguish them
    # for train 'dev' = 0, for overfit 'dev' = 1, for eval 'dev' = 2, for test 'dev' = 3
    train_dataset = train_dataset.add_column("dev", [0] * len(train_dataset))    # type: ignore
    overfit_dataset = overfit_dataset.add_column("dev", [1] * len(overfit_dataset)) # type: ignore
    eval_dataset = eval_dataset.add_column("dev", [2] * len(eval_dataset))       # type: ignore
    test_dataset = cameo_dataset.add_column("dev", [3] * len(cameo_dataset))     # type: ignore
    logger.info(
        f"""Datasets include: train items x{len(train_dataset)}; overfit items x{len(overfit_dataset)}; eval items x{len(eval_dataset)}; cameo2022 items x{len(test_dataset)};"""
    )
    final_train_dataset = train_dataset
    final_eval_dataset = datasets.concatenate_datasets([overfit_dataset, eval_dataset, test_dataset]) # type: ignore
    
    # training process
    model.train()
    model._dynamic_tied_weights_keys = {'lm_head.weight', 'transformer.wte.weight'}
    training_args = TrainingArguments(**config_trainer, remove_unused_columns=False)
    trainer = LFTrainer(
        processor=processor,
        model=model,
        train_collator=TextCollator(processor, eval_mode=False),
        eval_collator=TextCollator(processor, eval_mode=True),
        args=training_args,
        train_dataset=final_train_dataset,
        eval_dataset=final_eval_dataset,
        compute_metrics=lf_metrics,
    )
    trainer.train()


if __name__ == "__main__":
    main()
