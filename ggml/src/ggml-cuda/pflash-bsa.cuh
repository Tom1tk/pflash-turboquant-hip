#pragma once

#ifdef GGML_USE_HIP
#include "common.cuh"

void ggml_cuda_pflash_bsa_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_pflash_bsa_attn_supported(int device, const ggml_tensor * dst);
#endif // GGML_USE_HIP
