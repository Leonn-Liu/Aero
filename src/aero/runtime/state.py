from dataclasses import dataclass, field
from typing import List, Optional
import torch

@dataclass
class SpeculativeStats:
    total_drafted: int = 0
    total_accepted: int = 0
    num_rounds: int = 0
    acceptance_rate: float = 0.0

@dataclass
class SpeculativeSession:
    prompt_tokens: List[int]
    generated_tokens: List[int] = field(default_factory=list)
    draft_pkv: Optional[object] = None
    target_pkv: Optional[object] = None
    curr_target_logits: Optional[torch.Tensor] = None
    curr_draft_logits: Optional[torch.Tensor] = None
    stats: SpeculativeStats = field(default_factory=SpeculativeStats)

    def record_round(self, drafted: int, accepted: int):
        self.stats.total_drafted += drafted
        self.stats.total_accepted += accepted
        self.stats.num_rounds += 1
        if self.stats.total_drafted > 0:
            self.stats.acceptance_rate = self.stats.total_accepted / self.stats.total_drafted
