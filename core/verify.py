from typing import Optional, Tuple, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class TargetVerifier:
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=self.dtype,
            device_map=device
        )
        self.model.eval()

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, object]:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)
        outputs = self.model(input_ids=input_ids, use_cache=True)
        return outputs.logits[:, -1, :], outputs.past_key_values

    @torch.inference_mode()
    def step(self, token_id: int, past_key_values: object) -> Tuple[torch.Tensor, object]:
        inp = torch.tensor([[token_id]], device=self.device)
        outputs = self.model(input_ids=inp, past_key_values=past_key_values, use_cache=True)
        return outputs.logits[:, -1, :], outputs.past_key_values

    @torch.inference_mode()
    def verify(
        self,
        candidate_tokens: torch.Tensor,
        candidate_probs: List[torch.Tensor],
        target_start_logits: torch.Tensor,
        target_past_key_values: object,
        draft_past_key_values: object,
        temperature: float = 1.0
    ) -> Tuple[List[int], int, object, object, int]:
        candidate_tokens = candidate_tokens.to(self.device)
        gamma = candidate_tokens.shape[1]

        outputs = self.model(
            input_ids=candidate_tokens,
            past_key_values=target_past_key_values,
            use_cache=True
        )
        target_past_key_values = outputs.past_key_values

        target_logits_list = [target_start_logits]
        for i in range(gamma):
            target_logits_list.append(outputs.logits[:, i, :])

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
                p = torch.softmax(curr_target_logits / temperature, dim=-1)
                q = candidate_probs[i]
                p_val = p[0, cand_id].item()
                q_val = q[0, cand_id].item()

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
                    resampled_token = torch.multinomial(corrected_probs, num_samples=1).item()
                    accepted_tokens.append(resampled_token)
                    next_start_token = resampled_token
                    break

        if all_accepted:
            bonus_logits = target_logits_list[gamma]
            if temperature == 0.0:
                bonus_token = torch.argmax(bonus_logits, dim=-1).item()
            else:
                bonus_probs = torch.softmax(bonus_logits / temperature, dim=-1)
                bonus_token = torch.multinomial(bonus_probs, num_samples=1).item()
            accepted_tokens.append(bonus_token)
            next_start_token = bonus_token

        excess = gamma - accepted_candidates_count
        if excess > 0:
            target_past_key_values.crop(-excess)
            draft_past_key_values.crop(-excess)

        return accepted_tokens, next_start_token, target_past_key_values, draft_past_key_values, accepted_candidates_count
