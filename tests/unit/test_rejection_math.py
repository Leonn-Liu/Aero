import torch
from aero.ops.sampling import AeroSampler

def test_rejection_marginal_distribution():
    sampler = AeroSampler(backend="pytorch")

    p = torch.tensor([0.5, 0.3, 0.2])
    q = torch.tensor([0.2, 0.5, 0.3])

    accept_ratios = torch.clamp(p / q, max=1.0)
    alpha = (q * accept_ratios).sum().item()

    diff = torch.clamp(p - q, min=0.0)
    diff_sum = diff.sum()
    r = diff / diff_sum

    marginal = q * accept_ratios + (1.0 - alpha) * r
    assert torch.allclose(marginal, p, atol=1e-6)

    num_samples = 40000
    counts = torch.zeros(3)

    cand_indices = torch.multinomial(q.expand(num_samples, -1), num_samples=1).squeeze(-1)
    rand_vals = torch.rand(num_samples)

    for i in range(num_samples):
        cand = cand_indices[i].item()
        ratio = min(1.0, (p[cand] / q[cand]).item())
        if rand_vals[i].item() <= ratio:
            counts[cand] += 1
        else:
            resampled = sampler.sample_from_probs(r)
            counts[resampled.item()] += 1

    empirical_freq = counts / num_samples
    assert torch.allclose(empirical_freq, p, atol=0.015)
    print(f"Target: {p.tolist()}, Empirical: {empirical_freq.tolist()}")
    print("test_rejection_marginal_distribution PASSED")

if __name__ == "__main__":
    test_rejection_marginal_distribution()
