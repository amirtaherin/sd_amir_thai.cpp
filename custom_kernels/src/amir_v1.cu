// =====================================================================================
// amir_v1 (exp016) — cached Q8_0 -> INT8 weight conversion.
//
// Same INT8 cuBLAS path as thai_vu, but the Q8_0 -> INT8 (+ row scales)
// conversion of the WEIGHTS (src0) is computed once per weight tensor and
// cached, instead of being recomputed on every matmul call. Weights are
// constant across sampling steps. Activations (src1) still change every call
// and are quantized per call.
//
// Cache key: the weight's device pointer (src0_dd_i), stable because ggml loads
// the weights once into a fixed backend buffer. Cached buffers are raw
// cudaMalloc'd and freed at process exit (acceptable for an experiment).
// =====================================================================================

#include "common.cuh"
#include "int8_helpers.cuh"

void custom_kernels_mmq_amir_v1(
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

    // Step 1: cached Q8_0 -> INT8 weight conversion (once per weight tensor)
    auto it = g_amir_int8_weight_cache.find((const void *) src0_dd_i);
    if (it == g_amir_int8_weight_cache.end()) {
        cached_int8_weight cw;
        CUDA_CHECK(cudaMalloc((void **) &cw.d_i8,     sizeof(int8_t) * row_diff * K));
        CUDA_CHECK(cudaMalloc((void **) &cw.d_scales, sizeof(float)  * row_diff));
        convert_q8_0_to_int8_row_wise_cuda(
            src0_dd_i, cw.d_i8, cw.d_scales, row_diff, K, stream);
        it = g_amir_int8_weight_cache.emplace((const void *) src0_dd_i, cw).first;
    }
    const cached_int8_weight & cw = it->second;

    // Step 2: activations FP32 -> INT8 (per call)
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
        cw.d_i8,          CUDA_R_8I,  K,
        src1_as_i8.get(), CUDA_R_8I,  K,
        &beta_i32,
        dst_i32.get(),    CUDA_R_32I, ldc,
        CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));

    // Step 4: i32 -> f32 (original uncoalesced dequant)
    dequantize_i32_to_f32_cuda(
        dst_i32.get(), dst_dd_i,
        cw.d_scales, src1_scales.get(),
        row_diff, N, ldc, stream);

    GGML_UNUSED_VARS(dst, src1_ddq_i, src1_padded_row_size);
}
