from typing import List, Tuple
import torch
from core.draft import DraftModel
from core.verify import TargetVerifier

class SpeculativeEngine:
    def __init__(
        self,
        draft_model: DraftModel,
        target_verifier: TargetVerifier
    ):
        self.draft_model = draft_model
        self.target_verifier = target_verifier
        self.tokenizer = target_verifier.tokenizer

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 30,
        gamma: int = 4,
        temperature: float = 1.0
    ) -> str:
        text, _, _, _, _ = self.generate_with_stats(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            gamma=gamma,
            temperature=temperature
        )
        return text

    def generate_pure_target(
        self,
        prompt: str,
        max_new_tokens: int = 30,
        temperature: float = 1.0
    ) -> Tuple[str, List[int]]:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        curr_input_ids = input_ids.to(self.target_verifier.device)

        logits, pkv = self.target_verifier.prefill(curr_input_ids)
        generated_tokens: List[int] = []
        curr_logits = logits

        for _ in range(max_new_tokens):
            if temperature == 0.0:
                next_token_id = torch.argmax(curr_logits, dim=-1).item()
            else:
                probs = torch.softmax(curr_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()

            generated_tokens.append(next_token_id)
            if next_token_id == self.tokenizer.eos_token_id:
                break

            curr_logits, pkv = self.target_verifier.step(next_token_id, pkv)

        full_ids = torch.tensor([curr_input_ids[0].tolist() + generated_tokens])
        text = self.tokenizer.decode(full_ids[0], skip_special_tokens=True)
        return text, generated_tokens

    def generate_with_stats(
        self,
        prompt: str,
        max_new_tokens: int = 30,
        gamma: int = 4,
        temperature: float = 1.0
    ) -> Tuple[str, List[int], float, int, int]:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        curr_input_ids = input_ids.to(self.target_verifier.device)

        target_logits, target_pkv = self.target_verifier.prefill(curr_input_ids)
        draft_logits, draft_pkv = self.draft_model.prefill(curr_input_ids.to(self.draft_model.device))

        generated_tokens: List[int] = []
        curr_target_logits = target_logits
        curr_draft_logits = draft_logits

        total_drafted = 0
        total_accepted = 0

        while len(generated_tokens) < max_new_tokens:
            cand_tokens, cand_probs, draft_pkv = self.draft_model.generate_candidates(
                start_logits=curr_draft_logits,
                past_key_values=draft_pkv,
                gamma=gamma,
                temperature=temperature
            )

            accepted, next_token_id, target_pkv, draft_pkv, acc_count = self.target_verifier.verify(
                candidate_tokens=cand_tokens,
                candidate_probs=cand_probs,
                target_start_logits=curr_target_logits,
                target_past_key_values=target_pkv,
                draft_past_key_values=draft_pkv,
                temperature=temperature
            )

            total_drafted += gamma
            total_accepted += acc_count

            stopped = False
            for token_id in accepted:
                generated_tokens.append(token_id)
                if token_id == self.tokenizer.eos_token_id or len(generated_tokens) >= max_new_tokens:
                    stopped = True
                    break

            if stopped or len(generated_tokens) >= max_new_tokens:
                break

            curr_target_logits, target_pkv = self.target_verifier.step(next_token_id, target_pkv)
            curr_draft_logits, draft_pkv = self.draft_model.step(next_token_id, draft_pkv)

        acceptance_rate = total_accepted / max(1, total_drafted)
        full_ids = torch.tensor([curr_input_ids[0].tolist() + generated_tokens])
        text = self.tokenizer.decode(full_ids[0], skip_special_tokens=True)
        return text, generated_tokens, acceptance_rate, total_drafted, total_accepted
