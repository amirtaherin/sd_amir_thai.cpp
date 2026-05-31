// =====================================================================================
// thai_vu (exp015) — per-call Q8_0 -> INT8 + cuBLAS INT8 GEMM + dequant.
//
// No caching of weights: every matmul re-converts the Q8_0 weights to INT8 and
// the activations to INT8, runs cuBLAS i8*i8->i32, then dequantizes to FP32.
// =====================================================================================

#include "common.cuh"
#include "int8_helpers.cuh"

void custom_kernels_mmq_thai_vu(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i,
    const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src0_dd_i  != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i   != nullptr);

    const int64_t ne00 = src0->ne[0]; // K
    const int64_t ne0  = dst->ne[0];  // M
    const int64_t K = ne00;
    const int64_t N = src1_ncols;
    const int64_t row_diff = row_high - row_low;

    int id = ggml_cuda_get_device();
    int64_t ldc = id == ctx.device ? ne0 : row_diff;

    // Step 1: Q8_0 -> INT8 weights + row scales (every call)
    ggml_cuda_pool_alloc<int8_t> src0_as_i8(ctx.pool(id));
    ggml_cuda_pool_alloc<float>  src0_scales(ctx.pool(id));
    src0_as_i8.alloc(row_diff * K);
    src0_scales.alloc(row_diff);
    convert_q8_0_to_int8_row_wise_cuda(
        src0_dd_i, src0_as_i8.get(), src0_scales.get(), row_diff, K, stream);

    // Step 2: activations FP32 -> INT8 + column scales
    ggml_cuda_pool_alloc<int8_t> src1_as_i8(ctx.pool(id));
    ggml_cuda_pool_alloc<float>  src1_scales(ctx.pool(id));
    src1_as_i8.alloc(N * K);
    src1_scales.alloc(N);
    quantize_fp32_to_int8_row_wise_cuda(
        src1_ddf_i, src1_as_i8.get(), src1_scales.get(), N, K, stream);

    // Step 3: cuBLAS i8 * i8 -> i32
    ggml_cuda_pool_alloc<int32_t> dst_i32(ctx.pool(id));
    dst_i32.alloc(ldc * N);

    const int32_t alpha_i32 = 1;
    const int32_t beta_i32  = 0;
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
    CUBLAS_CHECK(cublasGemmEx(
        ctx.cublas_handle(id),
        CUBLAS_OP_T, CUBLAS_OP_N,
        row_diff, N, K,
        &alpha_i32,
        src0_as_i8.get(), CUDA_R_8I,  K,
        src1_as_i8.get(), CUDA_R_8I,  K,
        &beta_i32,
        dst_i32.get(),    CUDA_R_32I, ldc,
        CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));

    // Step 4: i32 -> f32 (original uncoalesced dequant)
    dequantize_i32_to_f32_cuda(
        dst_i32.get(), dst_dd_i,
        src0_scales.get(), src1_scales.get(),
        row_diff, N, ldc, stream);

    GGML_UNUSED_VARS(dst, src1_ddq_i, src1_padded_row_size);
}
