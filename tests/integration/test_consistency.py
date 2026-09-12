import torch
from aero.models.hf_runner import HuggingFaceRunner
from aero.runtime.engine import SpeculativeEngine

def test_lossless_consistency():
    if not torch.cuda.is_available():
        return

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    draft_runner = HuggingFaceRunner(model_name_or_path=model_name)
    target_runner = HuggingFaceRunner(model_name_or_path=model_name)

    engine = SpeculativeEngine(
        draft_runner=draft_runner,
        target_runner=target_runner,
        adaptive_gamma=False,
        initial_gamma=4
    )

    prompt = "Artificial intelligence is revolutionizing modern technology"
    max_new_tokens = 25

    pure_text, pure_tokens = engine.generate_pure_target(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0
    )

    spec_text, spec_tokens, acc_rate, drafted, accepted = engine.generate_with_stats(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        gamma=4,
        temperature=0.0
    )

    print(f"Pure tokens: {pure_tokens}")
    print(f"Spec tokens: {spec_tokens}")
    print(f"Acceptance Rate: {acc_rate * 100:.2f}% ({accepted}/{drafted})")

    assert pure_tokens == spec_tokens
    print("test_lossless_consistency PASSED (100% Exact Match)")

if __name__ == "__main__":
    test_lossless_consistency()
