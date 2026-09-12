from typing import Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class HuggingFaceRunner:
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=self.dtype,
            device_map=device
        )
        self.model.eval()

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, object]:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)
        outputs = self.model(input_ids=input_ids, use_cache=True)
        return outputs.logits[:, -1, :], outputs.past_key_values

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: object
    ) -> Tuple[torch.Tensor, object]:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True
        )
        return outputs.logits, outputs.past_key_values
