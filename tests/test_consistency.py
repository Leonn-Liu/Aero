import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import DraftModel, TargetVerifier, SpeculativeEngine

def test_consistency():
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

    engine = SpeculativeEngine(
        draft_model=draft_model,
        target_verifier=target_verifier
    )

    prompt = "Artificial intelligence is revolutionizing modern technology"
    max_new_tokens = 25
    gamma = 4

    pure_text, pure_tokens = engine.generate_pure_target(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0
    )

    spec_text, spec_tokens, acc_rate, drafted, accepted = engine.generate_with_stats(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        gamma=gamma,
        temperature=0.0
    )

    print("Prompt:", prompt)
    print("Pure Target Tokens:", pure_tokens)
    print("Speculative Tokens:", spec_tokens)
    print("Exact Match:", pure_tokens == spec_tokens)
    print(f"Acceptance Rate: {acc_rate:.2%} ({accepted}/{drafted})")
    print("Output Text:\n", spec_text)

    assert pure_tokens == spec_tokens

if __name__ == "__main__":
    test_consistency()
