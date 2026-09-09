from typing import List
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
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        curr_input_ids = input_ids.to(self.target_verifier.device)

        target_prob, target_pkv = self.target_verifier.prefill(curr_input_ids)
        draft_out = self.draft_model.model(
            input_ids=curr_input_ids.to(self.draft_model.device),
            use_cache=True
        )
        draft_pkv = draft_out.past_key_values

        generated_ids: List[int] = curr_input_ids[0].tolist()
        num_emitted = 0

        while num_emitted < max_new_tokens:
            last_token = torch.tensor([[generated_ids[-1]]], device=self.draft_model.device)
            candidates, cand_probs, draft_pkv = self.draft_model.generate_candidates(
                input_ids=last_token,
                gamma=gamma,
                past_key_values=draft_pkv,
                temperature=temperature
            )

            accepted_tokens, target_pkv, draft_pkv, target_prob = self.target_verifier.verify(
                candidate_tokens=candidates,
                candidate_probs=cand_probs,
                last_target_prob=target_prob,
                target_past_key_values=target_pkv,
                draft_past_key_values=draft_pkv,
                temperature=temperature
            )

            for token_id in accepted_tokens:
                generated_ids.append(token_id)
                num_emitted += 1
                if token_id == self.tokenizer.eos_token_id or num_emitted >= max_new_tokens:
                    break

            if generated_ids[-1] == self.tokenizer.eos_token_id:
                break

        full_output_ids = torch.tensor([generated_ids])
        return self.tokenizer.decode(full_output_ids[0], skip_special_tokens=True)
