import torch
from aero.models.hf_runner import HuggingFaceRunner
from aero.runtime.engine import SpeculativeEngine
from aero.ops.sampling import AeroSampler
from aero.decoding.verify import SpeculativeVerifier

def test_speculative_stochastic():
    if not torch.cuda.is_available():
        return

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    draft_runner = HuggingFaceRunner(model_name_or_path=model_name)
    target_runner = HuggingFaceRunner(model_name_or_path=model_name)

    engine = SpeculativeEngine(
        draft_runner=draft_runner,
        target_runner=target_runner,
        adaptive_gamma=True,
        initial_gamma=4
    )

    prompt = "Deep learning frameworks optimize"
    text, tokens, acc_rate, drafted, accepted = engine.generate_with_stats(
        prompt=prompt,
        max_new_tokens=20,
        temperature=0.8,
        top_k=50,
        top_p=0.9
    )

    assert len(tokens) == 20
    assert drafted > 0
    print(f"Generated text: {text}")
    print(f"Drafted: {drafted}, Accepted: {accepted}, Rate: {acc_rate * 100:.2f}%")
    print(f"Final Gamma: {engine.gamma_controller.get_gamma()}")
    print("test_speculative_stochastic PASSED")

def test_vocab_alignment_in_verify():
    if not torch.cuda.is_available():
        return

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    target_runner = HuggingFaceRunner(model_name_or_path=model_name)
    sampler = AeroSampler(backend="auto")
    verifier = SpeculativeVerifier(runner=target_runner, sampler=sampler)

    cand_tokens = torch.tensor([[100, 200, 300, 400]], device=target_runner.device)
    draft_vocab_size = 151936
    cand_probs = [
        torch.zeros(1, draft_vocab_size, device=target_runner.device) for _ in range(4)
    ]
    for i, tok in enumerate([100, 200, 300, 400]):
        cand_probs[i][0, tok] = 1.0

    target_start_logits = torch.randn(1, target_runner.vocab_size, device=target_runner.device)
    _, pkv = target_runner.prefill(torch.tensor([[15]], device=target_runner.device))

    accepted, next_tok, target_pkv, _, acc_count = verifier.verify(
        candidate_tokens=cand_tokens,
        candidate_probs=cand_probs,
        target_start_logits=target_start_logits,
        target_past_key_values=pkv,
        draft_past_key_values=pkv,
        temperature=0.8,
        top_k=50,
        top_p=0.9
    )
    assert len(accepted) >= 1
    print("test_vocab_alignment_in_verify PASSED")

if __name__ == "__main__":
    test_vocab_alignment_in_verify()
    test_speculative_stochastic()
