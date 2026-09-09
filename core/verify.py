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
        last_logits = outputs.logits[:, -1, :]
        probs = torch.softmax(last_logits, dim=-1)
        return probs, outputs.past_key_values

    @torch.inference_mode()
    def verify(
        self,
        candidate_tokens: torch.Tensor,
        candidate_probs: torch.Tensor,
        last_target_prob: torch.Tensor,
        target_past_key_values: object,
        draft_past_key_values: object,
        temperature: float = 1.0
    ) -> Tuple[List[int], object, object, torch.Tensor]:
        candidate_tokens = candidate_tokens.to(self.device)
        gamma = candidate_tokens.shape[1]

        outputs = self.model(
            input_ids=candidate_tokens,
            past_key_values=target_past_key_values,
            use_cache=True
        )
        target_past_key_values = outputs.past_key_values
        new_logits = outputs.logits

        if temperature == 0.0:
            target_probs_steps = torch.softmax(new_logits, dim=-1)
        else:
            target_probs_steps = torch.softmax(new_logits / temperature, dim=-1)

        all_target_probs = [last_target_prob]
        for i in range(gamma):
            all_target_probs.append(target_probs_steps[:, i, :])

        accepted_tokens: List[int] = []
        accepted_count = 0
        all_accepted = True
        next_target_prob = None

        for i in range(gamma):
            cand_token_id = candidate_tokens[0, i].item()
            p = all_target_probs[i][0, cand_token_id].item()
            q = candidate_probs[0, i, cand_token_id].item()

            accept_ratio = 1.0 if q == 0.0 else min(1.0, p / q)
            random_val = torch.rand(1).item()

            if random_val <= accept_ratio:
                accepted_tokens.append(cand_token_id)
                accepted_count += 1
            else:
                all_accepted = False
                diff = torch.clamp(all_target_probs[i] - candidate_probs[:, i, :], min=0.0)
                diff_sum = diff.sum(dim=-1, keepdim=True)
                if diff_sum.item() > 0:
                    corrected_probs = diff / diff_sum
                else:
                    corrected_probs = all_target_probs[i]
                resampled_token = torch.multinomial(corrected_probs, num_samples=1).item()
                accepted_tokens.append(resampled_token)
                break

        if all_accepted:
            bonus_prob = all_target_probs[gamma]
            if temperature == 0.0:
                bonus_token = torch.argmax(bonus_prob, dim=-1).item()
            else:
                bonus_token = torch.multinomial(bonus_prob, num_samples=1).item()
            accepted_tokens.append(bonus_token)

        excess_tokens = gamma - accepted_count
        if excess_tokens > 0:
            target_past_key_values.crop(-excess_tokens)
            draft_past_key_values.crop(-excess_tokens)

        last_emitted_token = torch.tensor([[accepted_tokens[-1]]], device=self.device)
        out_next = self.model(
            input_ids=last_emitted_token,
            past_key_values=target_past_key_values,
            use_cache=True
        )
        target_past_key_values = out_next.past_key_values
        next_logits = out_next.logits[:, -1, :]
        if temperature == 0.0:
            next_target_prob = torch.softmax(next_logits, dim=-1)
        else:
            next_target_prob = torch.softmax(next_logits / temperature, dim=-1)

        return accepted_tokens, target_past_key_values, draft_past_key_values, next_target_prob
