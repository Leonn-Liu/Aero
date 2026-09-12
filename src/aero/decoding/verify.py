from typing import Optional, Tuple, List
import torch
from aero.models.hf_runner import HuggingFaceRunner
from aero.ops.sampling import AeroSampler

class SpeculativeVerifier:
    def __init__(
        self,
        runner: HuggingFaceRunner,
        sampler: AeroSampler
    ):
        self.runner = runner
        self.sampler = sampler

    @torch.inference_mode()
    def verify(
        self,
        candidate_tokens: torch.Tensor,
        candidate_probs: List[torch.Tensor],
        target_start_logits: torch.Tensor,
        target_past_key_values: object,
        draft_past_key_values: object,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> Tuple[List[int], int, object, object, int]:
        candidate_tokens = candidate_tokens.to(self.runner.device)
        gamma = candidate_tokens.shape[1]

        outputs_logits, target_past_key_values = self.runner.forward(
            input_ids=candidate_tokens,
            past_key_values=target_past_key_values
        )

        target_logits_list = [target_start_logits]
        for i in range(gamma):
            target_logits_list.append(outputs_logits[:, i, :])

        target_vocab_size = self.runner.vocab_size
        accepted_tokens: List[int] = []
        accepted_candidates_count = 0
        all_accepted = True
        next_start_token: Optional[int] = None

        for i in range(gamma):
            cand_id = candidate_tokens[0, i].item()
            curr_target_logits = target_logits_list[i]

            if temperature == 0.0:
                target_greedy_id = torch.argmax(curr_target_logits, dim=-1).item()
                if cand_id == target_greedy_id:
                    accepted_tokens.append(cand_id)
                    accepted_candidates_count += 1
                else:
                    all_accepted = False
                    accepted_tokens.append(target_greedy_id)
                    next_start_token = target_greedy_id
                    break
            else:
                p = self.sampler.compute_distribution(
                    logits=curr_target_logits,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature
                )
                q = candidate_probs[i]
                if q.size(-1) < target_vocab_size:
                    q = torch.nn.functional.pad(q, (0, target_vocab_size - q.size(-1)), value=0.0)
                elif q.size(-1) > target_vocab_size:
                    q = q[..., :target_vocab_size]

                p_val = p[0, cand_id].item() if cand_id < target_vocab_size else 0.0
                q_val = q[0, cand_id].item() if cand_id < q.size(-1) else 0.0

                accept_ratio = 1.0 if q_val == 0.0 else min(1.0, p_val / q_val)
                rand_val = torch.rand(1).item()

                if rand_val <= accept_ratio:
                    accepted_tokens.append(cand_id)
                    accepted_candidates_count += 1
                else:
                    all_accepted = False
                    diff = torch.clamp(p - q, min=0.0)
                    diff_sum = diff.sum(dim=-1, keepdim=True)
                    if diff_sum.item() > 0:
                        corrected_probs = diff / diff_sum
                    else:
                        corrected_probs = p
                    resampled_token = self.sampler.sample_from_probs(corrected_probs)
                    resample_id = resampled_token.squeeze().item()
                    accepted_tokens.append(resample_id)
                    next_start_token = resample_id
                    break

        if all_accepted:
            bonus_logits = target_logits_list[gamma]
            bonus_token, _ = self.sampler.sample(
                logits=bonus_logits,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                return_probs=False
            )
            bonus_id = bonus_token.squeeze().item()
            accepted_tokens.append(bonus_id)
            next_start_token = bonus_id

        excess = gamma - accepted_candidates_count
        if excess > 0:
            if hasattr(target_past_key_values, "crop"):
                target_past_key_values.crop(-excess)
            if hasattr(draft_past_key_values, "crop"):
                draft_past_key_values.crop(-excess)

        return accepted_tokens, next_start_token, target_past_key_values, draft_past_key_values, accepted_candidates_count
