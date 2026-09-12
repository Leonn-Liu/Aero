from typing import List, Tuple, Optional
import torch
from aero.models.hf_runner import HuggingFaceRunner
from aero.ops.sampling import AeroSampler
from aero.decoding.controller import AdaptiveGammaController
from aero.decoding.draft import DraftGenerator
from aero.decoding.verify import SpeculativeVerifier
from aero.runtime.state import SpeculativeSession

class SpeculativeEngine:
    def __init__(
        self,
        draft_runner: HuggingFaceRunner,
        target_runner: HuggingFaceRunner,
        sampler: Optional[AeroSampler] = None,
        adaptive_gamma: bool = False,
        initial_gamma: int = 4
    ):
        self.draft_runner = draft_runner
        self.target_runner = target_runner
        self.sampler = sampler or AeroSampler(backend="auto")
        self.adaptive_gamma = adaptive_gamma
        self.gamma_controller = AdaptiveGammaController(initial_gamma=initial_gamma)
        self.draft_gen = DraftGenerator(runner=self.draft_runner, sampler=self.sampler)
        self.verifier = SpeculativeVerifier(runner=self.target_runner, sampler=self.sampler)

    def generate_pure_target(
        self,
        prompt: str,
        max_new_tokens: int = 30,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> Tuple[str, List[int]]:
        tokenizer = self.target_runner.tokenizer
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        curr_input_ids = input_ids.to(self.target_runner.device)

        logits, pkv = self.target_runner.prefill(curr_input_ids)
        generated_tokens: List[int] = []
        curr_logits = logits

        for _ in range(max_new_tokens):
            next_token, _ = self.sampler.sample(
                logits=curr_logits,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                return_probs=False
            )
            next_token_id = next_token.squeeze().item()
            generated_tokens.append(next_token_id)
            if next_token_id == self.target_runner.eos_token_id:
                break
            step_in = torch.tensor([[next_token_id]], device=self.target_runner.device)
            out_logits, pkv = self.target_runner.forward(step_in, pkv)
            curr_logits = out_logits[:, -1, :]

        full_ids = torch.tensor([curr_input_ids[0].tolist() + generated_tokens])
        text = tokenizer.decode(full_ids[0], skip_special_tokens=True)
        return text, generated_tokens

    def generate_with_stats(
        self,
        prompt: str,
        max_new_tokens: int = 30,
        gamma: int = 4,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> Tuple[str, List[int], float, int, int]:
        tokenizer = self.target_runner.tokenizer
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        curr_input_ids = input_ids.to(self.target_runner.device)

        target_logits, target_pkv = self.target_runner.prefill(curr_input_ids)
        draft_logits, draft_pkv = self.draft_runner.prefill(curr_input_ids.to(self.draft_runner.device))

        generated_tokens: List[int] = []
        curr_target_logits = target_logits
        curr_draft_logits = draft_logits

        total_drafted = 0
        total_accepted = 0

        if self.adaptive_gamma:
            self.gamma_controller.reset()

        while len(generated_tokens) < max_new_tokens:
            curr_gamma = self.gamma_controller.get_gamma() if self.adaptive_gamma else gamma

            cand_tokens, cand_probs, draft_pkv = self.draft_gen.generate_candidates(
                start_logits=curr_draft_logits,
                past_key_values=draft_pkv,
                gamma=curr_gamma,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            accepted, next_start_token, target_pkv, draft_pkv, acc_count = self.verifier.verify(
                candidate_tokens=cand_tokens,
                candidate_probs=cand_probs,
                target_start_logits=curr_target_logits,
                target_past_key_values=target_pkv,
                draft_past_key_values=draft_pkv,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            total_drafted += curr_gamma
            total_accepted += acc_count

            if self.adaptive_gamma:
                self.gamma_controller.update(acc_count, curr_gamma)

            stopped = False
            for token_id in accepted:
                generated_tokens.append(token_id)
                if token_id == self.target_runner.eos_token_id or len(generated_tokens) >= max_new_tokens:
                    stopped = True
                    break

            if stopped or len(generated_tokens) >= max_new_tokens:
                break

            step_in_t = torch.tensor([[next_start_token]], device=self.target_runner.device)
            target_out_logits, target_pkv = self.target_runner.forward(step_in_t, target_pkv)
            curr_target_logits = target_out_logits[:, -1, :]

            step_in_d = torch.tensor([[next_start_token]], device=self.draft_runner.device)
            draft_out_logits, draft_pkv = self.draft_runner.forward(step_in_d, draft_pkv)
            curr_draft_logits = draft_out_logits[:, -1, :]

        acceptance_rate = total_accepted / max(1, total_drafted)
        full_ids = torch.tensor([curr_input_ids[0].tolist() + generated_tokens])
        text = tokenizer.decode(full_ids[0], skip_special_tokens=True)
        return text, generated_tokens, acceptance_rate, total_drafted, total_accepted

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 30,
        gamma: int = 4,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> str:
        text, _, _, _, _ = self.generate_with_stats(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            gamma=gamma,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        return text
