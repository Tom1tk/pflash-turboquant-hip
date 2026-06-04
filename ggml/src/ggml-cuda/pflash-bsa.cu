#ifdef GGML_USE_HIP
#include "common.cuh"
#include <hip/hip_runtime.h>
#include <math.h>

#define BSA_BLOCK_SIZE 128

// Single-query BSA attention: O[q][h][:] = softmax(Q[q][h] @ K[h]^T) @ V[h]
// across selected KV blocks. Grid: dim3(n_q, n_heads).

template <int D>
static __global__ void pflash_bsa_single_query(
    const float * __restrict__ Q,
    const float * __restrict__ K,
    const float * __restrict__ V,
    const int * __restrict__ block_mask,
    int n_selected,
    float * __restrict__ O,
    float scale,
    int n_kv,
    int n_heads,
    int n_heads_kv,
    int q_stride,
    int q_head_stride,
    int o_stride,
    int o_head_stride,
    int kv_stride,
    int kv_head_stride,
    int head_dim)
{
    const int q_pos = blockIdx.x;
    const int h     = blockIdx.y;
    const int tid   = threadIdx.x;

    const int h_kv = h * n_heads_kv / n_heads;

    const float * Q_head = Q + (size_t)q_pos * q_stride + (size_t)h * q_head_stride;

    __shared__ float sdata[256];

    float O_reg[D / 128 + 1] = {0.0f};

    float m_i = -INFINITY;
    float l_i = 0.0f;

    for (int bi = 0; bi < n_selected; bi++) {
        int kv_block = block_mask[bi];
        int k_start  = kv_block * BSA_BLOCK_SIZE;
        int k_end    = min(k_start + BSA_BLOCK_SIZE, n_kv);

        for (int ki = k_start; ki < k_end; ki++) {
            if (ki > q_pos) break;

            const float * k_row = K + (size_t)ki * kv_stride + (size_t)h_kv * kv_head_stride;

            float partial = 0.0f;
            for (int d = tid; d < D; d += blockDim.x) {
                partial += Q_head[d] * k_row[d];
            }

            sdata[tid] = partial;
            __syncthreads();

            for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
                if (tid < offset) sdata[tid] += sdata[tid + offset];
                __syncthreads();
            }
            float s = sdata[0] * scale;

            float m_old = m_i;
            float m_new = fmaxf(m_old, s);
            float exp_scale = expf(m_old - m_new);

            for (int d = tid; d < D; d += blockDim.x) {
                O_reg[d / blockDim.x] *= exp_scale;
            }

            float p = expf(s - m_new);
            l_i = l_i * exp_scale + p;
            m_i = m_new;

            const float * v_row = V + (size_t)ki * kv_stride + (size_t)h_kv * kv_head_stride;
            for (int d = tid; d < D; d += blockDim.x) {
                O_reg[d / blockDim.x] += p * v_row[d];
            }
        }
    }

    float inv_l = 1.0f / fmaxf(l_i, 1e-12f);
    float * o_row = O + (size_t)q_pos * o_stride + (size_t)h * o_head_stride;
    for (int d = tid; d < D; d += blockDim.x) {
        o_row[d] = O_reg[d / blockDim.x] * inv_l;
    }
}

extern "C" {

int32_t pflash_bsa_forward(
    const float * d_Q,
    const float * d_K,
    const float * d_V,
    const int * d_block_mask,
    int n_selected,
    float * d_O,
    float scale,
    int n_heads,
    int n_heads_kv,
    int n_q,
    int n_kv,
    int head_dim,
    int q_stride,
    int q_head_stride,
    int o_stride,
    int o_head_stride,
    int kv_stride,
    int kv_head_stride)
{
    if (!d_Q || !d_K || !d_V || !d_block_mask || !d_O) return -1;
    if (n_selected <= 0 || n_heads <= 0 || n_kv <= 0 || head_dim <= 0 || n_q <= 0) return -1;

    int threads = min(256, head_dim);
    if (threads < 32) threads = 32;

    int threads_p2 = 1;
    while (threads_p2 < threads) threads_p2 *= 2;
    if (threads_p2 > 256) threads_p2 = 256;

    switch (head_dim) {
        case 64:
            hipLaunchKernelGGL((pflash_bsa_single_query<64>),
                dim3(n_q, n_heads), dim3(threads_p2), 0, 0,
                d_Q, d_K, d_V, d_block_mask, n_selected, d_O, scale,
                n_kv, n_heads, n_heads_kv, q_stride, q_head_stride,
                o_stride, o_head_stride, kv_stride, kv_head_stride, head_dim);
            break;
        case 128:
            hipLaunchKernelGGL((pflash_bsa_single_query<128>),
                dim3(n_q, n_heads), dim3(threads_p2), 0, 0,
                d_Q, d_K, d_V, d_block_mask, n_selected, d_O, scale,
                n_kv, n_heads, n_heads_kv, q_stride, q_head_stride,
                o_stride, o_head_stride, kv_stride, kv_head_stride, head_dim);
            break;
        case 256:
            hipLaunchKernelGGL((pflash_bsa_single_query<256>),
                dim3(n_q, n_heads), dim3(threads_p2), 0, 0,
                d_Q, d_K, d_V, d_block_mask, n_selected, d_O, scale,
                n_kv, n_heads, n_heads_kv, q_stride, q_head_stride,
                o_stride, o_head_stride, kv_stride, kv_head_stride, head_dim);
            break;
        default:
            return -1;
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) return -1;

    return 0;
}

} // extern "C"

// ============================================================================
// ggml_cuda wrapper — calls single-query kernel with dim3(n_q, n_heads)
// ============================================================================

void ggml_cuda_pflash_bsa_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q          = dst->src[0];
    const ggml_tensor * k          = dst->src[1];
    const ggml_tensor * v          = dst->src[2];
    const ggml_tensor * block_mask = dst->src[3];

    GGML_UNUSED(ctx);

    const int n_heads    = q->ne[2];
    const int n_heads_kv = k->ne[2];
    const int n_q        = q->ne[1];
    const int n_kv       = k->ne[1];
    const int head_dim   = q->ne[0];

    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    int n_selected = (int)ggml_get_op_params_f32(dst, 1);

    const int q_stride       = q->nb[1] / sizeof(float);
    const int q_head_stride  = q->nb[2] / sizeof(float);
    const int o_stride       = dst->nb[2] / sizeof(float);
    const int o_head_stride  = dst->nb[1] / sizeof(float);
    const int kv_stride      = k->nb[1] / sizeof(float);
    const int kv_head_stride = k->nb[2] / sizeof(float);

    int32_t ret = pflash_bsa_forward(
        (const float*)q->data, (const float*)k->data, (const float*)v->data,
        (const int*)block_mask->data, n_selected,
        (float*)dst->data, scale, n_heads, n_heads_kv, n_q, n_kv, head_dim,
        q_stride, q_head_stride, o_stride, o_head_stride, kv_stride, kv_head_stride);

    GGML_ASSERT(ret == 0 && "PFLASH_BSA_ATTN: kernel launch failed (unsupported head_dim?)");
    GGML_ASSERT(hipGetLastError() == hipSuccess);
}

bool ggml_cuda_pflash_bsa_attn_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
    return dst->src[0] != nullptr &&
           dst->src[0]->type == GGML_TYPE_F32 &&
           dst->src[1]->type == GGML_TYPE_F32 &&
           dst->src[2]->type == GGML_TYPE_F32 &&
           dst->src[3]->type == GGML_TYPE_I32;
}
#endif // GGML_USE_HIP
