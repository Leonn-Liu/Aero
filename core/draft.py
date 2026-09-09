from typing import Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class DraftModel:
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
    def generate_candidates(
        self,
        input_ids: torch.Tensor,
        gamma: int = 4,
        past_key_values: Optional[object] = None,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, object]:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)

        candidate_token_list = []
        candidate_prob_list = []
        curr_input_ids = input_ids
        curr_past_key_values = past_key_values

        for _ in range(gamma):
            if curr_past_key_values is None:
                outputs = self.model(input_ids=curr_input_ids, use_cache=True)
            else:
                outputs = self.model(
                    input_ids=curr_input_ids,
                    past_key_values=curr_past_key_values,
                    use_cache=True
                )
            curr_past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            if temperature == 0.0:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                scaled_logits = logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            candidate_token_list.append(next_token)
            candidate_prob_list.append(probs)
            curr_input_ids = next_token

        candidate_tokens = torch.cat(candidate_token_list, dim=1)
        candidate_probs = torch.stack(candidate_prob_list, dim=1)

        return candidate_tokens, candidate_probs, curr_past_key_values
