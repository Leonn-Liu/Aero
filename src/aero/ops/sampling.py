import os
from pathlib import Path
from typing import Tuple, Optional
import torch

class AeroSampler:
    def __init__(self, backend: str = "auto"):
        self.requested_backend = backend
        self.backend = "pytorch"
        self._ext = None
        self._init_backend()

    def _init_backend(self):
        if self.requested_backend in ("auto", "cuda") and torch.cuda.is_available():
            try:
                import ninja
                os.environ["PATH"] = f"{ninja.BIN_DIR}:/usr/local/cuda/bin:{os.environ.get('PATH', '')}"
                os.environ["CUDA_HOME"] = "/usr/local/cuda"
                os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
                from torch.utils.cpp_extension import load
                csrc_dir = Path(__file__).resolve().parent.parent.parent.parent / "csrc"
                cpp_file = str(csrc_dir / "bindings.cpp")
                cu_file = str(csrc_dir / "kernels" / "sampling.cu")
                self._ext = load(
                    name="aero_fused_sampling_ext",
                    sources=[cpp_file, cu_file],
                    extra_cuda_cflags=["-O3", "--use_fast_math"],
                    verbose=False
                )
                self.backend = "cuda"
            except Exception as e:
                if self.requested_backend == "cuda":
                    raise RuntimeError(f"CUDA backend requested but failed to load: {e}")
                self.backend = "pytorch"
        else:
            if self.requested_backend == "cuda":
                raise RuntimeError("CUDA backend requested but CUDA is unavailable")
            self.backend = "pytorch"

    @property
    def is_cuda(self) -> bool:
        return self.backend == "cuda" and self._ext is not None

    def compute_distribution(
        self,
        logits: torch.Tensor,
        top_k: int = 50,
        top_p: float = 0.9,
        temperature: float = 1.0
    ) -> torch.Tensor:
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if temperature == 0.0:
            probs = torch.zeros_like(logits)
            max_idx = torch.argmax(logits, dim=-1, keepdim=True)
            probs.scatter_(-1, max_idx, 1.0)
            return probs

        scaled = logits / max(1e-5, temperature)
        if top_k > 0:
            k = min(top_k, scaled.size(-1))
            val, _ = torch.topk(scaled, k, dim=-1)
            scaled = scaled.masked_fill(scaled < val[..., -1:], -float("inf"))

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = torch.zeros_like(scaled, dtype=torch.bool).scatter_(
                -1, sorted_indices, sorted_indices_to_remove
            )
            scaled = scaled.masked_fill(indices_to_remove, -float("inf"))

        return torch.softmax(scaled, dim=-1)

    def sample(
        self,
        logits: torch.Tensor,
        top_k: int = 50,
        top_p: float = 0.9,
        temperature: float = 1.0,
        return_probs: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        probs: Optional[torch.Tensor] = None
        if return_probs:
            probs = self.compute_distribution(
                logits=logits,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature
            )

        if temperature == 0.0:
            tokens = torch.argmax(logits, dim=-1)
            return tokens, probs

        if self.is_cuda and logits.is_cuda and 1 <= top_k <= 64 and 0.0 < top_p <= 1.0:
            tokens = self._ext.fused_sampling(logits, top_k, top_p, temperature)
            return tokens, probs

        if probs is None:
            probs = self.compute_distribution(
                logits=logits,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature
            )
        tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return tokens, probs

    def sample_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
