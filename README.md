<div align="center">

# Aero

**Draft-model speculative decoding with a fused CUDA sampling kernel, and a timing model of where the time goes.**

[Overview](#overview) · [Architecture](#architecture) · [Results](#results) · [Limitations](#known-limitations-and-next-steps) · [Usage](#usage)

![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat)
![CUDA](https://img.shields.io/badge/cuda-12.0%2B-green?style=flat)

</div>

---

## Overview

Aero is a research prototype of draft-model speculative decoding on a single consumer GPU (RTX 4090 D). Qwen2.5-0.5B-Instruct drafts tokens and Qwen2.5-7B-Instruct verifies them.

* **Fused sampling kernel (CUDA).** A two-stage top-k / top-p sampler: each block pre-selects candidates from its shard of the 151,936-token vocabulary, then a refinement stage picks the final token. Intermediate results stay in shared memory and registers (warp-shuffle reductions) instead of being written back to global memory as separate tensors.
* **Lossless verification.** Standard speculative rejection sampling: accept a draft token with probability min(1, p/q), otherwise resample from the normalized residual max(0, p − q). The output distribution equals the target model's; with greedy decoding the output matches the target model token for token.
* **KV-cache rollback.** Rejected draft tokens are dropped by cropping the cache (`excess = γ − accepted`) without reallocating buffers.
* **Adaptive draft length.** γ is adjusted from a 5-round sliding window of acceptance rates with two thresholds (0.75 / 0.40).
* **Timing model.** A per-round latency model fitted to the measurements, used below to explain the end-to-end results.

---

## Architecture

<div align="center">
  <img src="assets/aero_architecture.svg" alt="Aero speculative decoding architecture" width="100%" />
</div>

---

## Results

**Setup.** One RTX 4090 D (24 GB, sm_89), driver 580.76, CUDA 13.0, PyTorch 2.13, Transformers 5.17. Target Qwen2.5-7B-Instruct and draft Qwen2.5-0.5B-Instruct, both FP16.

### Sampling kernel

Measured with `benchmarks/bench_sampling.py`: batch 1, fp32 logits over 151,936 tokens, top-k 50, top-p 0.9.

| Implementation | Latency per sampling call | Intermediate tensor traffic |
|---|---|---|
| PyTorch sampling ops (top-k, softmax, multinomial, ...) | 2364.9 µs | 6.37 MB per token |
| Aero fused kernel | 229.9 µs (10.3× faster) | 0.10 MB per token |

The traffic column counts intermediate tensors only; the logits themselves (about 0.6 MB per row) are still read once. Reading 0.6 MB at about 1 TB/s takes under 1 µs, so the kernel is still far from the memory-bandwidth bound (see limitations).

### End-to-end decoding

Workload: 6 hand-picked prompts (2 code, 1 math, 1 reasoning, 1 chat, 1 technical; 18 to 32 prompt tokens), 128 new tokens each, top-k 50, top-p 0.9, 2 warm-up runs and 3 measured runs per configuration (run `aero_bench_20260912_093120`). Temperature 0:

| Configuration | Throughput (tok/s) | vs. target only | Acceptance rate | Mean accepted length τ |
|---|---|---|---|---|
| Target only (7B) | 54.61 | 1.00× | – | 1.00 |
| γ = 2 | 36.54 | 0.67× | 71.8% | 2.41 |
| γ = 3 | 38.87 | 0.71× | 66.7% | 2.95 |
| γ = 4 | 38.83 | 0.71× | 58.9% | 3.32 |
| γ = 5 | 38.10 | 0.70× | 54.2% | 3.62 |
| γ = 6 | 37.45 | 0.69× | 51.4% | 4.00 |
| γ = 8 | 35.58 | 0.65× | 44.5% | 4.37 |
| Adaptive γ | 40.33 | 0.74× | 61.6% | 3.45 |

Temperature 0.8 gives the same picture: 0.65× to 0.70× for fixed γ, 0.69× adaptive. **In its current form, speculative decoding is slower than running the target model alone.**

### Why it is slower

Fitting the 12 fixed-γ configurations (both temperatures) gives

```
T_round ≈ 2 · T_target + (γ + 1) · T_draft        mean error 0.8%
T_target ≈ 18.3 ms,  T_draft ≈ 9.8 ms
```

Two causes follow from it:

1. **One extra target forward per round.** Verification feeds only the γ draft tokens to the target model; the logits for the first position come from a separate target forward pass. Feeding `[next_token, d1 … dγ]` in one verification pass removes it.
2. **The draft model is overhead-bound.** A 0.5B FP16 model reads about 1 GB of weights per step, roughly 1 ms at 1 TB/s, yet a draft step takes 9.8 ms, more than half a 7B step. The time goes to Python dispatch, kernel launches and per-token `.item()` synchronization. A static KV cache plus CUDA Graphs for the draft model targets this.

Estimates from the model for γ = 4 (not yet measured): fixing (1) alone gives about 0.9×; fixing (2) alone with a 2.5 ms draft step gives about 1.24×; fixing both gives about 2×.

The same numbers explain why recent methods replace the separate draft model with EAGLE-style feature drafting or built-in MTP heads: once drafting costs half a target step, little speedup is left.

---

## Known limitations and next steps

* Fix the two end-to-end bottlenecks above and re-measure.
* Profile the sampling kernel with Nsight Compute and compare it with FlashInfer's sorting-free sampling kernels.
* The adaptive controller only looks at acceptance rate. In batched serving the right γ also depends on batch size and latency targets: once decoding becomes compute-bound, extra verified tokens are no longer free.
* The evaluation is small (6 prompts, single request, batch size 1) and uses Qwen2.5. Newer models with EAGLE-3 or MTP drafting are the relevant comparison.

---

## Usage

```bash
git clone https://github.com/Leonn-Liu/Aero.git && cd Aero
pip install -e .
```

```python
from aero.models import HuggingFaceRunner
from aero.runtime import SpeculativeEngine

draft = HuggingFaceRunner("Qwen/Qwen2.5-0.5B-Instruct")
target = HuggingFaceRunner("Qwen/Qwen2.5-7B-Instruct")
engine = SpeculativeEngine(draft, target, adaptive_gamma=True)
print(engine.generate("Explain quantum computing in one sentence:", max_new_tokens=64))
```

End-to-end benchmark (writes raw JSON, CSV and a Markdown report):

```bash
python benchmarks/run_cloud_suite.py \
  --target Qwen/Qwen2.5-7B-Instruct --draft Qwen/Qwen2.5-0.5B-Instruct \
  --gamma_list 2 3 4 5 6 8 --max_tokens 128 --repeats 3
```
