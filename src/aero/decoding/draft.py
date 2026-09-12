from typing import Tuple, List
import torch
from aero.models.hf_runner import HuggingFaceRunner
from aero.ops.sampling import AeroSampler

class DraftGenerator:
    def __init__(
        self,
        runner: HuggingFaceRunner,
        sampler: AeroSampler
    ):
        self.runner = runner
        self.sampler = sampler

    @torch.inference_mode()
    def generate_candidates(
        self,
        start_logits: torch.Tensor,
        past_key_values: object,
        gamma: int = 4,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> Tuple[torch.Tensor, List[torch.Tensor], object]:
        candidate_tokens_list: List[torch.Tensor] = []
        candidate_probs_list: List[torch.Tensor] = []
        curr_logits = start_logits
        curr_pkv = past_key_values

        for _ in range(gamma):
            next_token, next_prob = self.sampler.sample(
                logits=curr_logits,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                return_probs=True
            )
            if next_token.dim() == 1:
                next_token = next_token.unsqueeze(-1)

            candidate_tokens_list.append(next_token)
            candidate_probs_list.append(next_prob)

            outputs_logits, curr_pkv = self.runner.forward(
                input_ids=next_token,
                past_key_values=curr_pkv
            )
            curr_logits = outputs_logits[:, -1, :]

        candidate_tokens = torch.cat(candidate_tokens_list, dim=1)
        return candidate_tokens, candidate_probs_list, curr_pkv
