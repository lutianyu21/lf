from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from typing import List, cast


__all__ = ['TextTokenizer']

class TextTokenizer(PreTrainedTokenizerFast):
    
    boseq_token: str
    eoseq_token: str
    bostruct_token: str
    eostruct_token: str
    struct_regex: str
    struct_template: str
    struct_vsz: int
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        self.boseq_token = '<|boseq|>'
        self.eoseq_token = '<|eoseq|>'
        self.bostruct_token = '<|bostruct|>'
        self.eostruct_token = '<|eostruct|>'
        self.struct_regex = r"<\|struct(\d{4})\|>"
        self.struct_template = "<|struct{token_id:0>4d}|>"
        self.struct_vsz = kwargs.get("struct_vsz", 0)
        self.add_special_tokens({
            'additional_special_tokens': \
            ['<|boseq|>', '<|eoseq|>', '<|bostruct|>', '<|eostruct|>'] + \
            [self.struct_template.format(token_id=i) for i in range(self.struct_vsz)] # type: ignore
        })

    @property
    def boseq_token_id(self) -> int:
        return cast(int, self.convert_tokens_to_ids(self.boseq_token))
    
    @property
    def eoseq_token_id(self) -> int:
        return cast(int, self.convert_tokens_to_ids(self.eoseq_token))
    
    @property
    def bostruct_token_id(self) -> int:
        return cast(int, self.convert_tokens_to_ids(self.bostruct_token))

    @property
    def eostruct_token_id(self) -> int:
        return cast(int, self.convert_tokens_to_ids(self.eostruct_token))
    
    @property
    def seq_vocab_ids(self) -> List[int]:
        return self.encode("ABCDEFGHIKLMNOPQRSTUVWXYZ")
    
    @property
    def struct_vocab_ids(self) -> List[int]:
        return self.encode(''.join([self.struct_template.format(token_id=i) for i in range(self.struct_vsz)]))
    
    @property
    def vocab_size(self) -> int:
        return len(self.get_vocab())
