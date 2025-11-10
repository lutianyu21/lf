from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import LogitsProcessor

__all__ = [
    'UnbatchedModalityLogitsProcessorBase',
]


TEXT_GENERATION = 0
SEQUENCE_GENERATION = 1
STRUCTURE_GENERATION = 2



class UnbatchedModalityLogitsProcessorBase(LogitsProcessor):
    """
    Enforce strict generation constraints for unbatched generation.
    - <seq> ... </seq> : sequence generation under given length and vocab constraints
    - <struct> ... </struct> : structure generation under given length and vocab constraints
    - text generation is free-form until <eos> token is generated.
    
    Cases(basically, task description > prompt > reasoning > result):     
    - structure analysis & generation
        - structure analysis                xxx <struct>...</struct>    || xxx <eos>
        - forward-folding                   xxx <seq>...</seq>          || xxx <struct>...</struct> <eos>
        - multi-conformation                xxx <seq>...</seq>          || xxx <struct>...</struct> xxx <struct>...</struct> <eos>
        - motif-scaffolding                 xxx <struct>...</struct>    || xxx <struct>...</struct> <eos>

    - sequence analysis & generation
        - sequence analysis                 xxx <seq>...</seq>          || xxx <eos>
        - inv-folding                       xxx <struct>...</struct>    || xxx <seq>...</seq> <eos>
        - motif-scaffolding                 xxx <seq>...</seq>          || xxx <seq>...</seq> <eos>
    """

    def __init__(
        self,
        processor: Any,
        eos_token_id: int,
        boseq_token_id: int,
        eoseq_token_id: int,
        seq_vocab_ids: List[int],
        bostruct_token_id: int,
        eostruct_token_id: int,
        struct_vocab_ids: List[int],
        templates: List[Tuple[str, int]], # (key, length)
        **kwargs: Any,
    ):
        self.templates = templates
        self.templates.reverse()  # for easy pop()
        self.processor = processor
        self.eos_token_id = eos_token_id
        self.boseq_token_id = boseq_token_id
        self.eoseq_token_id = eoseq_token_id
        self.seq_vocab_ids = seq_vocab_ids
        self.bostruct_token_id = bostruct_token_id
        self.eostruct_token_id = eostruct_token_id
        self.struct_vocab_ids = struct_vocab_ids
        
        # initialize fsm state
        self.current_state = TEXT_GENERATION
        self.counter_statck = -1
        self.dummy = True
    
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # Update current state after a token is actually sampled.
        # There's a dummy jump since there's no bos token)
        
        if self.dummy:
            self.dummy = False
        else:
            token_id = int(input_ids[:, -1].item())
            self.current_state = self._fsm_step(self.current_state, token_id)
        # Apply constraints based on the current state
        scores_mask = torch.full_like(scores, float('-inf'))
        constraint = self._constraint_step(self.current_state)
        if constraint is not None:
            scores_mask[:, list(constraint)] = 0
        return scores + scores_mask   
    
    def _query_template(self, key: str) -> Tuple[int, int, List[int]]:
        return {
            'seq':      (self.boseq_token_id, self.eoseq_token_id, self.seq_vocab_ids),
            'struct':   (self.bostruct_token_id, self.eostruct_token_id, self.struct_vocab_ids),
        }[key]

    def _fsm_step(self, state1: int, token1_id: int) -> int:
        if state1 == TEXT_GENERATION:
            key, length = self.templates[-1]
            # Fill a stack(implemented as a counter here);
            if token1_id == self.boseq_token_id:
                self.counter_statck = length
                return SEQUENCE_GENERATION
            elif token1_id == self.bostruct_token_id:
                self.counter_statck = length
                return STRUCTURE_GENERATION
            else:
                return TEXT_GENERATION
            
        elif state1 in [SEQUENCE_GENERATION, STRUCTURE_GENERATION]:
            key, length = self.templates[-1]
            start_token_id, end_token_id, vocab_ids = self._query_template(key)
            # Keep popping a symbol
            if token1_id != end_token_id:
                self.counter_statck -= 1
                return state1
            # Commit when the satck emptys
            else:
                self.templates.pop()
                return TEXT_GENERATION
        
        raise NotImplementedError()
    
    def _constraint_step(self, state1: int) -> Optional[Tuple[int, ...]]:
        if state1 == TEXT_GENERATION:
            # Here we need to preview the next template. For example, 
            # if the next template is sequence generation, we need the constraint text + boseq
            # if the next template is structure generation, we need the constraint text + bostruct
            if len(self.templates) == 0:
                # TODO natural language is not implemented yet
                return (self.eos_token_id,)
            else:
                key, length = self.templates[-1]
                start_token_id, end_token_id, vocab_ids = self._query_template(key)
                return (start_token_id,)
                
        elif state1 in [SEQUENCE_GENERATION, STRUCTURE_GENERATION]:
            key, length = self.templates[-1]
            start_token_id, end_token_id, vocab_ids = self._query_template(key)
            if self.counter_statck == 0:
                return (end_token_id,)
            else:
                return tuple(vocab_ids)
            
        raise NotImplementedError()
