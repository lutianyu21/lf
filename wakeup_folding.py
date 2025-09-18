from typing import Any, Tuple
from utils.progen2_utils import ProGenForCausalLM, ProGenConfig, progen2_merged_tokenizer
from pathlib import Path
from datasets import load_dataset
import torch
from utils.transformers_utils import DynamicMultimodalLogitsProcessor
from utils.dplm_utils import DPLMProcessor, dplm_tokenizer
from utils.openfold_utils.io import OpenfoldEntity
from transformers.generation.configuration_utils import GenerationConfig

full_dataset = load_dataset("json", data_files='/AIRvePFS/ai4science/users/tianyu/lf/data/dplm_pdb.jsonl', split="train")
filtered_dataset = full_dataset.filter(lambda item: 200 <= item['length'] <= 1200)
dataset: Any = filtered_dataset
split = dataset.train_test_split(test_size=0.1, seed=2025)
train_dataset, eval_dataset = split['train'], split['test']

ckpt_dir = Path('/AIRvePFS/ai4science/users/tianyu/lf/output/checkpoints/Mprogen_B8xdynamic_lr2e-05/checkpoint-65438')
hf_model: ProGenForCausalLM = ProGenForCausalLM.from_pretrained(ckpt_dir) # type: ignore
hf_model.to('cuda:0')
hf_model.eval()
hf_tokenizer = progen2_merged_tokenizer

processor = DPLMProcessor(structure_tokenizer=dplm_tokenizer.to('cuda:0'), tokenizer=progen2_merged_tokenizer)
GENERATION_CONFIG = GenerationConfig(
    use_cache=True,
    eos_token_id=progen2_merged_tokenizer.eos_token_id,
    bos_token_id=progen2_merged_tokenizer.bos_token_id,
    pad_token_id=progen2_merged_tokenizer.pad_token_id,
    do_sample=True,
    top_k=2048,
    temperature=0.7,
    top_p=0.4,
    max_new_tokens=1024,
)

for i in range(20):
    inputs = hf_tokenizer([train_dataset[i]['text']], return_tensors='pt', padding=True)
    labels = inputs['input_ids'].clone() # type: ignore
    labels[labels == hf_tokenizer.pad_token_id] = -100
    loss = hf_model(**inputs, labels=labels).loss
    print(f'===== NLL loss =====: {loss.detach().cpu().item()}')
    
    inputs = hf_tokenizer(['<|bos|><|boseq|>' + train_dataset[i]['protein_text'] + '<|eoseq|><|bostruct|>'], return_tensors='pt', padding=True)
    print("===== Gt Text =====", train_dataset[i]['text'])
    logits_processor = DynamicMultimodalLogitsProcessor(**processor.constant_helper(), batch_length=[len(train_dataset[i]['protein_structure'])]) # type: ignore
    generated_tokens = hf_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        logits_processor=[logits_processor],
        generation_config=GENERATION_CONFIG,
    )
    print("===== Gen Text =====", hf_tokenizer.decode(generated_tokens[0]))
    _, _, gt_structure_list = processor.decode(hf_tokenizer.encode(train_dataset[i]['text']))
    _, _, gen_structure_list = processor.decode(generated_tokens[0])
    gt_structure:   Tuple[str, OpenfoldEntity] = gt_structure_list[0]
    gen_structure:  Tuple[str, OpenfoldEntity] = gen_structure_list[0]
    print("===== ACC =====", processor.compute_acc(gen_structure[0], gt_structure[0]))
    print("===== RMSD =====", processor.compute_rmsd(gen_structure[1], gt_structure[1]))
