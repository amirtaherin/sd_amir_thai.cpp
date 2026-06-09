// =====================================================================================
// amir_v4 — Fused CUTLASS INT8 GEMM + per-row × per-col dequant.
//
// Implementation by Thai Vu. Replaces amir_v2/v3's cuBLAS-i8-GEMM + standalone
// coalesced dequant pair with a single CUTLASS kernel whose epilogue applies
// row_scale[i] · col_scale[j] and writes FP32 directly — no INT32 intermediate
// buffer, no separate dequant kernel.
//
// Build requirement: CUTLASS must be on the include path
// (-DCUTLASS_DIR=/path/to/cutlass — see custom_kernels/CMakeLists.txt and
// the README). When CUTLASS is wired in, the build system defines
// CUSTOM_KERNEL_HAVE_CUTLASS; without it this dispatch aborts at runtime to
// fail loudly (CUSTOM_KERNEL_VERSION ∈ {1..4} continue to work without CUTLASS).
//
// See custom_kernels/src/amir_v4_implementation.md for the technical write-up
// (EVT epilogue tree, tile shapes, validation notes, integration TODOs).
// See custom_kernels/src/cutlass_improvement.md for the original design doc.
// =====================================================================================

#include "common.cuh"
#include "int8_helpers.cuh"

void custom_kernels_mmq_amir_v4(
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

    // Step 1: cached Q8_0 -> INT8 weight conversion (shared with amir_v1/v2/v3).
    // If amir_v3's load-time preload populated the cache, this is a fast lookup.
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

    // Step 2: activations FP32 -> INT8 (per call). Same as amir_v2/v3.
    ggml_cuda_pool_alloc<int8_t> src1_as_i8(ctx.pool(id));
    ggml_cuda_pool_alloc<float>  src1_scales(ctx.pool(id));
    src1_as_i8.alloc(N * K);
    src1_scales.alloc(N);
    quantize_fp32_to_int8_row_wise_cuda(
        src1_ddf_i, src1_as_i8.get(), src1_scales.get(), N, K, stream);

    // Step 3: fused CUTLASS i8·i8→fp32 with per-row × per-col scaling in the
    // epilogue. No INT32 intermediate buffer; no standalone dequant.
#if defined(CUSTOM_KERNEL_HAVE_CUTLASS)
    bool matmul_success = matmul_w8a8_cutlass_cuda(
        cw.d_i8, src1_as_i8.get(), cw.d_scales, src1_scales.get(),
        dst_dd_i, row_diff, N, K, ldc, ctx.pool(id), stream);

    if (!matmul_success) {
        fprintf(stderr,
                "[amir_v4] CUTLASS matmul_w8a8_cutlass_cuda returned false\n");
    }
#else
    fprintf(stderr,
            "[amir_v4] CUSTOM_KERNEL_VERSION=5 selected but the build was made "
            "without CUTLASS. Reconfigure with -DCUTLASS_DIR=/path/to/cutlass "
            "(see custom_kernels/README.md).\n");
    abort();
#endif

    GGML_UNUSED_VARS(dst, src1_ddq_i, src1_padded_row_size);
}
