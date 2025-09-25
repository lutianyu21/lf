from typing import Any, Dict, Tuple

from openfold import data
from utils.dplm_utils.dplm import train
from utils.progen2_utils import ProGenForCausalLM, ProGenConfig, progen2_merged_tokenizer
from pathlib import Path
from datasets import load_dataset
import torch
from utils.transformers_utils import DynamicMultimodalLogitsProcessor
from utils.dplm_utils import DPLMProcessor, dplm_tokenizer
from utils.openfold_utils.io import OpenfoldProtein
from transformers.generation.configuration_utils import GenerationConfig
from tqdm import tqdm
import pandas as pd



# hard-coded cases
train_monomer_homomerx25 = ['2xjl', '1i92', '3cs4', '7xf7', '2xqz', '3v6p', '3d2i', '6hkm', '2r3l', '7r1k', '8fp3', '4ewd', '6x8w', '1e36', '1e25', '4roa', '5b0t', '1maz', '7d3a', '1qgv', '3bfh', '6urx', '1zbf', '7fz5', '1wvh']
train_monomer_heteromerx25 = ['1uu4', '6ntu', '3ghn', '4m4k', '1hkj', '2j7m', '4gwj', '1jj0', '4ij4', '3crb', '5t7s', '1oa7', '7r85', '1uwy', '4j2s', '2wsv', '6kbq', '1iud', '1qgi', '5lgc', '4jc1', '4e7n', '6vfv', '3x2p', '2zhn']
train_multimer_homomerx25 = ['3qp1', '2okz', '4i44', '4rd7', '5mro', '4asn', '6d2j', '3ejc', '4l7w', '4w8s', '7rv0', '1uec', '6gol', '3cy6', '3cru', '8h1e', '1qh9', '8bfa', '5ugi', '1ori', '6mqz', '2nto', '4b2h', '5kki', '4cv4']
train_multimer_heteromerx25 = ['4cz5', '1a0a', '5goj', '3wan', '4hwf', '3w2a', '6t0c', '8onp', '5usp', '6y7r', '2qqg', '4cp3', '2yzi', '2bn3', '8q9p', '5fru', '6oxt', '5ie9', '6bie', '3nwx', '7xv8', '3uvw', '5t1j', '1x8z', '5xns']
eval_monomer_homomerx25 = ['7v15', '7g07', '6sey', '5jts', '4as0', '5cwc', '2f0s', '7l3b', '3ry5', '3uqi', '4ft0', '2fnn', '5o7d', '1j1g', '5gyn', '5s9c', '5eps', '3co4', '4cms', '6hid', '2bb7', '4ete', '4p5y', '6b6b', '2w1y']
proxy_cases = dict(
    train_monomer_homomerx25 = train_monomer_homomerx25,
    train_monomer_heteromerx25 = train_monomer_heteromerx25,
    train_multimer_homomerx25 = train_multimer_homomerx25,
    train_multimer_heteromerx25 = train_multimer_heteromerx25,
    eval_monomer_homomerx25 = eval_monomer_homomerx25,
)

device = torch.device('cuda:0')
job = Path('/AIRvePFS/ai4science/users/tianyu/lf/output/checkpoints/Mprogen_B1xdynamic_lr2e-05/checkpoint-3188')
model: ProGenForCausalLM = ProGenForCausalLM.from_pretrained(job) # type: ignore
model.to(device) # type: ignore
model.eval()
tokenizer = progen2_merged_tokenizer
processor = DPLMProcessor(tokenizer, dplm_tokenizer.to(device))

GENERATION_CONFIG = GenerationConfig(
    use_cache=True,
    eos_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    do_sample=False,
    max_new_tokens=2048,
)

results = {
    "group":        [],
    "entry":        [],
    "path":         [],
    "nll":          [],
    "acc":          [],
    "rmsd":         [], 
    "tm":           [],
    "rmsd_pdb":     [],
    "gt_text":      [],
    "argmax_text":  [],
    "gen_text":     [],
}

full_dataset: Any = load_dataset("json", data_files='/AIRvePFS/ai4science/users/tianyu/lf/data/dplm_pdb.jsonl', split="train")
for k, v in proxy_cases.items():
    dataset: Any = full_dataset.filter(lambda item: item['entry'] in v)

    for item in tqdm(dataset):
        results['group'].append(k)
        results['entry'].append(item['entry'])
        results['path'].append(item['mmcif_path'])
        train_inputs = tokenizer([item['text']], return_tensors='pt', padding=True).to(device)
        train_labels = train_inputs['input_ids'].clone() # type: ignore
        train_labels[train_labels == tokenizer.pad_token_id] = -100
        loss = model(**train_inputs, labels=train_labels).loss
        results['nll'].append(loss.detach().cpu().item())
        
        gt_text= item['text'] # <|bos|>~<|eos|>        
        argmax_text = '<|bos|>' + tokenizer.decode(torch.argmax(model(**train_inputs).logits[:, :-1], dim=-1)[0]) # <|bos|>~<|eos|>
        results['gt_text'].append(gt_text)
        results['argmax_text'].append(argmax_text)
        
        eval_inputs: Any = tokenizer(['<|bos|><|boseq|>' + item['protein_text'] + '<|eoseq|><|bostruct|>'], return_tensors='pt', padding=True).to(device) 
        logits_processor = DynamicMultimodalLogitsProcessor(**processor.constant_helper(), batch_length=[item['lengths_struct']]) # type: ignore
        gen_tokens = model.generate(
            input_ids=eval_inputs["input_ids"],
            token_type_ids=torch.zeros_like(eval_inputs["input_ids"]).to(device), # remove for later ckpt
            attention_mask=eval_inputs["attention_mask"],
            logits_processor=[logits_processor], # type: ignore
            generation_config=GENERATION_CONFIG,
        )
        gen_text = tokenizer.decode(gen_tokens[0]) # <|bos|>~<|eos|>
        results['gen_text'].append(gen_text)
        
        # calculate metrics
        acc = processor.compute_acc(gen_text, gt_text)
        results['acc'].append(acc)
        
        pdb_entity = OpenfoldProtein.from_file(item['mmcif_path']).to(device)
        gt_entity: OpenfoldProtein = processor.decode(tokenizer.encode(gt_text))['entity'][0]
        gt_entity.inherit(pdb_entity)
        gen_entity: OpenfoldProtein = processor.decode(tokenizer.encode(gen_text))['entity'][0]
        gen_entity.inherit(pdb_entity)
        tm, rmsd = gt_entity.align_with(gen_entity, chain_wise=True)
        results['rmsd'].append(rmsd)
        results['tm'].append(tm)
        
        _, rmsd_pdb = pdb_entity.align_with(gen_entity, chain_wise=True)
        results['rmsd_pdb'].append(rmsd_pdb)
        
        # print to terminal as one line, metrics-only
        print(f"===== Group =====: {k}", f"===== Entry =====: {item['entry']}", f"===== NLL loss =====: {loss.detach().cpu().item()}", f"===== ACC ===== {acc}", f"===== RMSD ===== {rmsd}", f"===== TM ===== {tm}", f"===== RMSD_PDB ===== {rmsd_pdb}", sep='\n')
        
# save as one csv
results_df = pd.DataFrame(results)
results_df.to_csv('proxy.csv', index=False)
