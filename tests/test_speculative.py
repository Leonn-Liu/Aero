import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import DraftModel, TargetVerifier, SpeculativeEngine

def test_speculative_decoding():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    draft_model = DraftModel(
        model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        device=device,
        dtype=dtype
    )

    target_verifier = TargetVerifier(
        model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        device=device,
        dtype=dtype
    )

    prompt = "System design and speculative decoding"
    input_ids = target_verifier.tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    gamma = 4
    candidates, cand_probs, draft_pkv = draft_model.generate_candidates(
        input_ids=input_ids,
        gamma=gamma,
        temperature=0.7
    )

    assert candidates.shape == (1, gamma)
    assert cand_probs.shape[1] == gamma

    target_prob, target_pkv = target_verifier.prefill(input_ids)
    accepted_tokens, target_pkv, draft_pkv, next_target_prob = target_verifier.verify(
        candidate_tokens=candidates,
        candidate_probs=cand_probs,
        last_target_prob=target_prob,
        target_past_key_values=target_pkv,
        draft_past_key_values=draft_pkv,
        temperature=0.7
    )

    assert len(accepted_tokens) >= 1
    assert len(accepted_tokens) <= gamma + 1

    engine = SpeculativeEngine(
        draft_model=draft_model,
        target_verifier=target_verifier
    )

    generated_text = engine.generate(
        prompt=prompt,
        max_new_tokens=20,
        gamma=gamma,
        temperature=0.7
    )

    print("Candidates Shape:", candidates.shape)
    print("Candidate Tokens:", candidates.tolist())
    print("Accepted Tokens:", accepted_tokens)
    print("Generated Text:\n", generated_text)

if __name__ == "__main__":
    test_speculative_decoding()
