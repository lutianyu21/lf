from typing import Any, Dict, Optional, Tuple
import torch
from transformers import LogitsProcessor

from utils.progen2_utils.tokenizers import progen2_merged_tokenizer

__all__ = ['DynamicMultimodalLogitsProcessor']



class DynamicMultimodalLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        pad_token: int,
        bos_token: int,
        eos_token: int,
        seq_tokens: list[int],
        boseq_token: int,
        eoseq_token: int,
        struct_tokens: list[int],
        bostruct_token: int,
        eostruct_token: int,
        batch_length: list[int],
        **kwargs
    ):
        self.pad_token = pad_token          # text sequence
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.seq_tokens = seq_tokens        # protein sequence
        self.boseq_token = boseq_token
        self.eoseq_token = eoseq_token
        self.struct_tokens = struct_tokens  # protein structure
        self.bostruct_token = bostruct_token
        self.eostruct_token = eostruct_token
        
        self.states = {
            'INITIAL': 0,
            'TEXT': 1,
            'SEQ': 2,
            'STRUCTURE': 3
        }
        self.states_reverse = {v: k for k, v in self.states.items()}
        # Expansion
        self.batch_length = batch_length
        self.batch_current_state = [self.states['INITIAL'] for _ in range(len(batch_length))]
        self.batch_current_cnt = [{'SEQ_CNT': 0, 'STRUCTURE_CNT': 0} for _ in range(len(batch_length))]
        

    def _constraint_step(self, batch_id: int, last_token: int) -> Optional[Tuple[int, ...]]:
        
        # current_token \in f(current_state, last_token)
        current_state: int = self.batch_current_state[batch_id]
        current_cnt: Dict[str, int] = self.batch_current_cnt[batch_id]
        length: int = self.batch_length[batch_id]
        
        if current_state == self.states['INITIAL']:
            constraint = (self.bos_token,)
        elif current_state == self.states['TEXT']:
            # TODO remove after leveraging text ?
            if last_token == self.eostruct_token:
                constraint = (self.eos_token,)
            else:   
                constraint = None
        elif current_state == self.states['SEQ']:
            if current_cnt['SEQ_CNT'] == length:
                constraint = (self.eoseq_token,)
            else:
                constraint = tuple(self.seq_tokens)
        elif current_state == self.states['STRUCTURE']:
            if current_cnt['STRUCTURE_CNT'] == length:
                constraint = (self.eostruct_token,)
            else:
                constraint = tuple(self.struct_tokens)
        else:
            raise ValueError()
        return constraint

    def _fsm_step(self, batch_id: int, current_token: int) -> int:
        
        # next_state = f(current_state, current_token)
        current_state: int = self.batch_current_state[batch_id]
        current_cnt: Dict[str, int] = self.batch_current_cnt[batch_id]
        length: int = self.batch_length[batch_id]
        
        if current_state == self.states['INITIAL']:
            if current_token == self.pad_token:
                next_state = self.states['INITIAL']
            elif current_token == self.bos_token:
                next_state = self.states['TEXT']
            else:
                raise ValueError()
        elif current_state == self.states['TEXT']:
            if current_token == self.boseq_token:
                next_state = self.states['SEQ']
            elif current_token == self.bostruct_token:
                next_state = self.states['STRUCTURE']
            elif current_token == self.eos_token:
                next_state = self.states['INITIAL']
            else:
                # Might include tokens from seq or struct
                next_state = self.states['TEXT']    
        elif current_state == self.states['SEQ']:
            if current_token in self.seq_tokens:
                current_cnt['SEQ_CNT'] += 1
                if current_cnt['SEQ_CNT'] <= length:
                    next_state = self.states['SEQ']
                else:
                    raise ValueError()
            elif current_token == self.eoseq_token:
                # TODO consider checking prompts here
                next_state = self.states['TEXT']
                current_cnt['SEQ_CNT'] = 0
            else:
                raise ValueError()
        elif current_state == self.states['STRUCTURE']:
            if current_token in self.struct_tokens:
                current_cnt['STRUCTURE_CNT'] += 1
                if current_cnt['STRUCTURE_CNT'] <= length:
                    next_state = self.states['STRUCTURE']
                else:
                    raise ValueError()
            elif current_token == self.eostruct_token:
                next_state = self.states['TEXT']
                current_cnt['STRUCTURE_CNT'] = 0
            else:
                raise ValueError()
        else:
            raise ValueError()
        return next_state
    
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        B, V = scores.shape
        scores_mask = torch.full_like(scores, float('-inf'))
        
        # HINT param name: last_token -> current_state -> current_token -> next_state
        for batch_id in range(B):
            # step1 simulating all input ids: 
            if (current_state := self.batch_current_state[batch_id]) == self.states['INITIAL']:
                for i in range(L - 1):
                    current_token = int(input_ids[batch_id, i].item())
                    next_state = self._fsm_step(batch_id, current_token)
                    self.batch_current_state[batch_id] = next_state
            
            current_state = self.batch_current_state[batch_id]
            current_token = int(input_ids[batch_id, -1].item())
            next_state = self._fsm_step(batch_id, current_token)
            self.batch_current_state[batch_id] = next_state
            
            # step2: apply constraints
            constraint = self._constraint_step(batch_id, current_token)
            if constraint is None:
                scores_mask[batch_id, :] = 0
            else:
                scores_mask[batch_id, list(constraint)] = 0
                
        return scores + scores_mask
    