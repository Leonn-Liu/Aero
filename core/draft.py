from typing import Optional, Tuple, List
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
    def generate_candidates(
        self,
        start_logits: torch.Tensor,
        past_key_values: object,
        gamma: int = 4,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, List[torch.Tensor], object]:
        candidate_tokens_list = []
        candidate_probs_list = []
        curr_logits = start_logits
        curr_pkv = past_key_values

        for _ in range(gamma):
            if temperature == 0.0:
                probs = torch.softmax(curr_logits, dim=-1)
                next_token = torch.argmax(curr_logits, dim=-1, keepdim=True)
            else:
                scaled_logits = curr_logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            candidate_tokens_list.append(next_token)
            candidate_probs_list.append(probs)

            outputs = self.model(
                input_ids=next_token,
                past_key_values=curr_pkv,
                use_cache=True
            )
            curr_pkv = outputs.past_key_values
            curr_logits = outputs.logits[:, -1, :]

        candidate_tokens = torch.cat(candidate_tokens_list, dim=1)
        return candidate_tokens, candidate_probs_list, curr_pkv
