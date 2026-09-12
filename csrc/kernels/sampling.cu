#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cmath>

struct Candidate {
    float val;
    int pos;
};

__device__ __forceinline__ Candidate warp_reduce_candidate(Candidate c) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, c.val, offset);
        int other_pos = __shfl_down_sync(0xffffffff, c.pos, offset);
        if (other_val > c.val) {
            c.val = other_val;
            c.pos = other_pos;
        }
    }
    return c;
}

__global__ void stage1_topk_kernel(
    const float* __restrict__ logits,
    float* __restrict__ intermediate_vals,
    int* __restrict__ intermediate_inds,
    int vocab_size,
    int slice_size,
    int k
) {
    int block_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;

    extern __shared__ float s_vals[];
    __shared__ float s_warp_vals[8];
    __shared__ int s_warp_pos[8];
    __shared__ int s_best_pos;

    int block_start = block_idx * slice_size;
    const float* b_logits = logits + batch_idx * vocab_size;

    for (int i = tid; i < slice_size; i += blockDim.x) {
        int g_idx = block_start + i;
        s_vals[i] = (g_idx < vocab_size) ? b_logits[g_idx] : -1e30f;
    }
    __syncthreads();

    int lane = tid % 32;
    int wid = tid / 32;
    int out_offset = (batch_idx * gridDim.x + block_idx) * k;

    for (int step = 0; step < k; ++step) {
        Candidate my_cand;
        my_cand.val = -1e30f;
        my_cand.pos = -1;

        for (int i = tid; i < slice_size; i += blockDim.x) {
            float v = s_vals[i];
            if (v > my_cand.val) {
                my_cand.val = v;
                my_cand.pos = i;
            }
        }

        my_cand = warp_reduce_candidate(my_cand);

        if (lane == 0) {
            s_warp_vals[wid] = my_cand.val;
            s_warp_pos[wid] = my_cand.pos;
        }
        __syncthreads();

        if (wid == 0) {
            Candidate block_cand;
            block_cand.val = (tid < 8) ? s_warp_vals[tid] : -1e30f;
            block_cand.pos = (tid < 8) ? s_warp_pos[tid] : -1;
            for (int offset = 4; offset > 0; offset /= 2) {
                float ov = __shfl_down_sync(0xffffffff, block_cand.val, offset);
                int op = __shfl_down_sync(0xffffffff, block_cand.pos, offset);
                if (ov > block_cand.val) {
                    block_cand.val = ov;
                    block_cand.pos = op;
                }
            }
            if (tid == 0) {
                s_best_pos = block_cand.pos;
                intermediate_vals[out_offset + step] = block_cand.val;
                intermediate_inds[out_offset + step] = block_start + block_cand.pos;
            }
        }
        __syncthreads();

        if (tid == 0 && s_best_pos >= 0 && s_best_pos < slice_size) {
            s_vals[s_best_pos] = -1e30f;
        }
        __syncthreads();
    }
}

__global__ void stage2_sampling_kernel(
    const float* __restrict__ intermediate_vals,
    const int* __restrict__ intermediate_inds,
    const float* __restrict__ rand_vals,
    int64_t* __restrict__ output_tokens,
    int total_candidates,
    int k,
    float top_p,
    float temperature
) {
    int b = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float s_cand_vals[4096];
    __shared__ int s_cand_inds[4096];
    __shared__ float s_warp_vals[8];
    __shared__ int s_warp_pos[8];
    __shared__ int s_best_pos;
    __shared__ float s_final_vals[64];
    __shared__ int s_final_inds[64];

    const float* in_vals = intermediate_vals + b * total_candidates;
    const int* in_inds = intermediate_inds + b * total_candidates;

    for (int i = tid; i < total_candidates; i += blockDim.x) {
        s_cand_vals[i] = in_vals[i];
        s_cand_inds[i] = in_inds[i];
    }
    for (int i = total_candidates + tid; i < 4096; i += blockDim.x) {
        s_cand_vals[i] = -1e30f;
        s_cand_inds[i] = 0;
    }
    __syncthreads();

    int lane = tid % 32;
    int wid = tid / 32;

    for (int step = 0; step < k; ++step) {
        Candidate my_cand;
        my_cand.val = -1e30f;
        my_cand.pos = -1;

        for (int i = tid; i < total_candidates; i += blockDim.x) {
            float v = s_cand_vals[i];
            if (v > my_cand.val) {
                my_cand.val = v;
                my_cand.pos = i;
            }
        }

        my_cand = warp_reduce_candidate(my_cand);

        if (lane == 0) {
            s_warp_vals[wid] = my_cand.val;
            s_warp_pos[wid] = my_cand.pos;
        }
        __syncthreads();

        if (wid == 0) {
            Candidate block_cand;
            block_cand.val = (tid < 8) ? s_warp_vals[tid] : -1e30f;
            block_cand.pos = (tid < 8) ? s_warp_pos[tid] : -1;
            for (int offset = 4; offset > 0; offset /= 2) {
                float ov = __shfl_down_sync(0xffffffff, block_cand.val, offset);
                int op = __shfl_down_sync(0xffffffff, block_cand.pos, offset);
                if (ov > block_cand.val) {
                    block_cand.val = ov;
                    block_cand.pos = op;
                }
            }
            if (tid == 0) {
                s_best_pos = block_cand.pos;
                s_final_vals[step] = block_cand.val;
                s_final_inds[step] = (block_cand.pos >= 0) ? s_cand_inds[block_cand.pos] : 0;
            }
        }
        __syncthreads();

        if (tid == 0 && s_best_pos >= 0 && s_best_pos < 4096) {
            s_cand_vals[s_best_pos] = -1e30f;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float max_logit = s_final_vals[0];
        float inv_temp = (temperature > 1e-5f) ? (1.0f / temperature) : 1.0f;
        float sum_exp = 0.0f;
        for (int j = 0; j < k; ++j) {
            float exp_val = expf((s_final_vals[j] - max_logit) * inv_temp);
            s_final_vals[j] = exp_val;
            sum_exp += exp_val;
        }

        float inv_sum = 1.0f / sum_exp;
        for (int j = 0; j < k; ++j) {
            s_final_vals[j] *= inv_sum;
        }

        float cumsum = 0.0f;
        int cutoff = k - 1;
        for (int j = 0; j < k; ++j) {
            cumsum += s_final_vals[j];
            if (cumsum >= top_p) {
                cutoff = j;
                break;
            }
        }

        float truncated_sum = 0.0f;
        for (int j = 0; j <= cutoff; ++j) {
            truncated_sum += s_final_vals[j];
        }

        float r = rand_vals[b] * truncated_sum;
        float running = 0.0f;
        int chosen_idx = 0;
        for (int j = 0; j <= cutoff; ++j) {
            running += s_final_vals[j];
            if (running >= r) {
                chosen_idx = j;
                break;
            }
        }
        output_tokens[b] = s_final_inds[chosen_idx];
    }
}

torch::Tensor fused_sampling_cuda(
    torch::Tensor logits,
    int top_k,
    float top_p,
    float temperature
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(logits));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto float_logits = logits.to(torch::kFloat32).contiguous();
    int batch_size = float_logits.size(0);
    int vocab_size = float_logits.size(1);

    int k = top_k;
    int num_blocks = 64;
    int slice_size = (vocab_size + num_blocks - 1) / num_blocks;
    int total_candidates = num_blocks * k;

    auto intermediate_vals = torch::empty({batch_size, total_candidates}, torch::TensorOptions().dtype(torch::kFloat32).device(float_logits.device()));
    auto intermediate_inds = torch::empty({batch_size, total_candidates}, torch::TensorOptions().dtype(torch::kInt32).device(float_logits.device()));
    auto rand_vals = torch::rand({batch_size}, torch::TensorOptions().dtype(torch::kFloat32).device(float_logits.device()));
    auto output_tokens = torch::empty({batch_size}, torch::TensorOptions().dtype(torch::kInt64).device(float_logits.device()));

    dim3 grid1(num_blocks, batch_size);
    int threads = 256;
    size_t shared_mem_stage1 = slice_size * sizeof(float);

    stage1_topk_kernel<<<grid1, threads, shared_mem_stage1, stream>>>(
        float_logits.data_ptr<float>(),
        intermediate_vals.data_ptr<float>(),
        intermediate_inds.data_ptr<int>(),
        vocab_size,
        slice_size,
        k
    );

    stage2_sampling_kernel<<<batch_size, threads, 0, stream>>>(
        intermediate_vals.data_ptr<float>(),
        intermediate_inds.data_ptr<int>(),
        rand_vals.data_ptr<float>(),
        output_tokens.data_ptr<int64_t>(),
        total_candidates,
        k,
        top_p,
        temperature
    );

    return output_tokens;
}
