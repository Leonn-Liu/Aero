#include <torch/extension.h>

torch::Tensor fused_sampling_cuda(
    torch::Tensor logits,
    int top_k,
    float top_p,
    float temperature
);

torch::Tensor fused_sampling(
    torch::Tensor logits,
    int top_k,
    float top_p,
    float temperature
) {
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D tensor [batch_size, vocab_size]");
    TORCH_CHECK(top_k > 0 && top_k <= 64, "top_k must be in range [1, 64] for fused kernel");
    TORCH_CHECK(top_p > 0.0f && top_p <= 1.0f, "top_p must be in range (0.0, 1.0]");
    TORCH_CHECK(temperature > 0.0f, "temperature must be positive");
    return fused_sampling_cuda(logits, top_k, top_p, temperature);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_sampling", &fused_sampling, "Fused Top-K Top-P Sampling");
}
