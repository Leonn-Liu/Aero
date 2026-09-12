#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void rejection_decision_kernel(
    const float* __restrict__ p_probs,
    const float* __restrict__ q_probs,
    const int64_t* __restrict__ cand_tokens,
    const float* __restrict__ rand_vals,
    int* __restrict__ accept_flags,
    int gamma,
    int vocab_size
) {
    int i = threadIdx.x;
    if (i < gamma) {
        int token_id = cand_tokens[i];
        float p_val = (token_id < vocab_size) ? p_probs[i * vocab_size + token_id] : 0.0f;
        float q_val = (token_id < vocab_size) ? q_probs[i * vocab_size + token_id] : 0.0f;
        float ratio = (q_val > 0.0f) ? fminf(1.0f, p_val / q_val) : 1.0f;
        accept_flags[i] = (rand_vals[i] <= ratio) ? 1 : 0;
    }
}
