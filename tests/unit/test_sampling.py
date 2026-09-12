import torch
from aero.ops.sampling import AeroSampler

def test_sampler_greedy():
    sampler = AeroSampler(backend="auto")
    logits = torch.tensor([[1.0, 5.0, 2.0, 0.5]])
    token, _ = sampler.sample(logits, temperature=0.0)
    assert token.item() == 1

def test_sampler_cuda_backend():
    if not torch.cuda.is_available():
        return
    sampler = AeroSampler(backend="cuda")
    assert sampler.is_cuda
    logits = torch.randn(1, 151936, device="cuda", dtype=torch.float32)
    token, probs = sampler.sample(logits, temperature=0.8, top_k=50, top_p=0.9, return_probs=True)
    assert token.dim() == 1
    assert 0 <= token.item() < 151936
    assert probs is not None
    assert probs.shape == logits.shape
    assert torch.isclose(probs.sum(), torch.tensor(1.0, device="cuda"), atol=1e-3)

def test_sampler_distribution():
    sampler = AeroSampler(backend="pytorch")
    logits = torch.tensor([[2.0, 1.0, 0.5, -1.0]])
    probs = sampler.compute_distribution(logits, top_k=2, top_p=1.0, temperature=1.0)
    assert probs.shape == logits.shape
    assert probs[0, 2].item() == 0.0
    assert probs[0, 3].item() == 0.0
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-4)

if __name__ == "__main__":
    test_sampler_greedy()
    test_sampler_distribution()
    test_sampler_cuda_backend()
    print("test_sampling PASSED")
