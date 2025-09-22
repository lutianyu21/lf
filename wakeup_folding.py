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
dataset: Any = full_dataset
split = dataset.train_test_split(test_size=0.1, seed=2025)
train_dataset, eval_dataset = split['train'], split['test']
device = 'cuda:0'

ckpt_dir = Path('/AIRvePFS/ai4science/users/tianyu/lf/output/checkpoints/Mprogen_B1xdynamic_lr2e-05/checkpoint-3188')
hf_model: ProGenForCausalLM = ProGenForCausalLM.from_pretrained(ckpt_dir) # type: ignore
hf_model.to(device)
hf_model.eval()
hf_tokenizer = progen2_merged_tokenizer

processor = DPLMProcessor(structure_tokenizer=dplm_tokenizer.to('cuda:0'), tokenizer=progen2_merged_tokenizer)
GENERATION_CONFIG = GenerationConfig(
    use_cache=True,
    eos_token_id=progen2_merged_tokenizer.eos_token_id,
    bos_token_id=progen2_merged_tokenizer.bos_token_id,
    pad_token_id=progen2_merged_tokenizer.pad_token_id,
    do_sample=False,
    max_new_tokens=2048,
)

# random sample from tarinset and eval

for i in range(20):
    inputs = hf_tokenizer([train_dataset[i]['text']], return_tensors='pt', padding=True).to(device)
    labels = inputs['input_ids'].clone() # type: ignore
    labels[labels == hf_tokenizer.pad_token_id] = -100
    loss = hf_model(**inputs, labels=labels).loss
    print(f'===== NLL loss =====: {loss.detach().cpu().item()}')
    print("===== Gt Text =====", train_dataset[i]['text'])
    print("===== Argmax Text =====", hf_tokenizer.decode(torch.argmax(hf_model(**inputs).logits, dim=-1)[0]))
    
    prompt = '<|bos|><|boseq|>' + train_dataset[i]['protein_text'] + '<|eoseq|><|bostruct|>'
    inputs = hf_tokenizer([prompt], return_tensors='pt', padding=True).to('cuda:0')
    logits_processor = DynamicMultimodalLogitsProcessor(**processor.constant_helper(), batch_length=[train_dataset[i]['lengths_struct']]) # type: ignore
    # HINT: generate config with keep consistent with model forwarding
    # here input_token_type is not used in generation
    generated_tokens = hf_model.generate(
        input_ids=inputs["input_ids"],
        token_type_ids=torch.zeros_like(inputs["input_ids"]).to(device),
        attention_mask=inputs["attention_mask"],
        logits_processor=[logits_processor],
        generation_config=GENERATION_CONFIG,
    )
    generation_text = hf_tokenizer.decode(generated_tokens[0])
    print(f"===== Gen Text =====", generation_text)
    _, _, gt_structure_list = processor.decode(hf_tokenizer.encode(train_dataset[i]['text']))
    _, _, gen_structure_list = processor.decode(generated_tokens[0])
    gt_structure:   Tuple[str, OpenfoldEntity] = gt_structure_list[0]
    gen_structure:  Tuple[str, OpenfoldEntity] = gen_structure_list[0]
    print("===== ACC =====", processor.compute_acc(gen_structure[0], gt_structure[0]).item())
    print("===== RMSD =====", processor.compute_rmsd(gen_structure[1], gt_structure[1]))
