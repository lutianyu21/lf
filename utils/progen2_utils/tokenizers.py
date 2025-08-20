from tokenizers import Tokenizer
from pathlib import Path
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

__all__ = [
    'progen2_merged_tokenizer'
]

progen2_tokenizer = Tokenizer.from_file(str(Path(__file__).parent/'progen/progen2/tokenizer.json'))
progen2_merged_tokenizer = Tokenizer.from_file(str(Path(__file__).parent/'progen/progen2/merged_tokenizer.json'))


# TODO organized by a new json
progen2_merged_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=progen2_merged_tokenizer,
    pad_token='<|pad|>',
    bos_token='<|bos|>',
    eos_token='<|eos|>',
    padding_side='left',
)
progen2_merged_tokenizer.add_special_tokens({
    'additional_special_tokens': ['<|boseq|>', '<|eoseq|>', '<|bostruct|>', '<|eostruct|>'] # type: ignore
})
setattr(progen2_merged_tokenizer, 'boseq_token', '<|boseq|>')
setattr(progen2_merged_tokenizer, 'eoseq_token', '<|eoseq|>')
setattr(progen2_merged_tokenizer, 'bostruct_token', '<|bostruct|>')
setattr(progen2_merged_tokenizer, 'eostruct_token', '<|eostruct|>')

