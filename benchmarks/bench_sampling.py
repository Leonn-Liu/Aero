import torch
from aero.ops.sampling import AeroSampler

def run_sampling_benchmark():
    if not torch.cuda.is_available():
        print("CUDA unavailable")
        return

    device = torch.device("cuda:0")
    batch_size = 1
    vocab_size = 151936
    top_k = 50
    top_p = 0.9
    temperature = 0.8
    num_warmup = 50
    num_iters = 1000

    sampler_cuda = AeroSampler(backend="cuda")
    sampler_torch = AeroSampler(backend="pytorch")

    logits = torch.randn(batch_size, vocab_size, device=device, dtype=torch.float32)

    for _ in range(num_warmup):
        _ = sampler_torch.sample(logits, top_k=top_k, top_p=top_p, temperature=temperature)
        _ = sampler_cuda.sample(logits, top_k=top_k, top_p=top_p, temperature=temperature)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        _ = sampler_torch.sample(logits, top_k=top_k, top_p=top_p, temperature=temperature)
    end_event.record()
    torch.cuda.synchronize()
    torch_time = start_event.elapsed_time(end_event) / num_iters

    start_event.record()
    for _ in range(num_iters):
        _ = sampler_cuda.sample(logits, top_k=top_k, top_p=top_p, temperature=temperature)
    end_event.record()
    torch.cuda.synchronize()
    cuda_time = start_event.elapsed_time(end_event) / num_iters

    speedup = torch_time / cuda_time if cuda_time > 0 else 0.0
    latency_reduction = (torch_time - cuda_time) / torch_time * 100.0 if torch_time > 0 else 0.0

    print("=" * 68)
    print("AERO OPERATOR BENCHMARK: SAMPLING (Vocab: 151,936, Iters: 1000)")
    print("=" * 68)
    print(f"PyTorch Native (topk+softmax+multinomial): {torch_time:.4f} ms")
    print(f"Aero Fused CUDA Kernel:                   {cuda_time:.4f} ms")
    print(f"Speedup:                                  {speedup:.2f}x")
    print(f"Latency Reduction:                        {latency_reduction:.1f}%")
    print("=" * 68)

if __name__ == "__main__":
    run_sampling_benchmark()
