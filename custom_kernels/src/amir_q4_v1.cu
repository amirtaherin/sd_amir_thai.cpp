// =====================================================================================
// amir_q4_v1 — cached Q4_0 weight conversion.
//
// This is the Q4 analog of amir_v3 for Q8. thai_vu_q4 does, per matmul call,
// for both the INT4 path and the INT8 fallback path:
//
//   - dequantize Q4_0 -> FP32
//   - FWHT-rotate the FP32 weight in place
//   - quantize rotated FP32 -> INT4 + row scales
//     (or, for the fallback path, convert Q4_0 -> INT8 + row scales)
//
// Weights are constant across sampling steps, so all of this is pure waste
// after step 1. amir_q4_v1 caches *both* derivatives once at model load (via
// custom_kernels::preload_q4_0_weights, called from src/stable-diffusion.cpp)
// and reuses them for every subsequent matmul. Runtime dispatch (score vs
// threshold) is unchanged.
//
// Activations still get rotated + INT4-quantized per call (they change every
// step). Only the weight side is cached.
//
// The math and the runtime dispatch are identical to thai_vu_q4; the only
// change is when + where the weight transforms happen.
// =====================================================================================

#include "common.cuh"
#include "int8_helpers.cuh"          // matmul_w8a8_cutlass_cuda + dequantize_i32_to_f32_cuda
#include "ggml-cuda-int4.cuh"        // Q4 helpers + cached_q4_weight + shared dispatch helpers

// Cache definition (declared extern in ggml-cuda-int4.cuh, so amir_q4_v1.cu and
// preload_q4.cu see the same map).
std::unordered_map<const void *, cached_q4_weight> g_amir_q4_weight_cache;


// -----------------------------------------------------------------------------
// INT8 fallback path with cached weights.
// Mirrors custom_ggml_q4_weight_q8_compute_kernel in int4_library.cu, but skips
// the per-call Q4_0 -> INT8 conversion of the WEIGHT (it's already cached).
// The activation side is unchanged.
// -----------------------------------------------------------------------------
static void custom_ggml_q4_weight_q8_compute_kernel_cached(
    ggml_backend_cuda_context & ctx,
    const cached_q4_weight & cw,
    const ggml_tensor * src1,
    const float * src1_ddf_i,
    float * dst_dd_i,
    int64_t row_diff,
    int64_t N,
    int64_t K,
    int64_t ldc,
    cudaStream_t stream
) {
    int id = ggml_cuda_get_device();

    // Activation FP32 -> INT8 + column scales (per call, unchanged).
    ggml_cuda_pool_alloc<int8_t> src1_as_i8(ctx.pool(id));
    ggml_cuda_pool_alloc<float>  src1_scales(ctx.pool(id));
    src1_as_i8.alloc(N * K);
    src1_scales.alloc(N);
    quantize_fp32_to_int8_row_wise_cuda(
        src1_ddf_i, src1_as_i8.get(), src1_scales.get(), N, K, stream);

    // Fused INT8 GEMM + row/col dequant -> FP32.
    bool matmul_dequant = matmul_w8a8_cutlass_cuda(
        cw.d_i8,             // A: [row_diff, K] cached INT8 weights
        src1_as_i8.get(),    // B: [N, K]
        cw.d_scales_i8,      // alphaRow: [row_diff]
        src1_scales.get(),   // alphaCol: [N]
        dst_dd_i,            // D: fp32, col-major D[m + n * ldc]
        row_diff, N, K, ldc,
        ctx.pool(id),
        stream);
    GGML_ASSERT(matmul_dequant);

    GGML_UNUSED(src1);
}


// -----------------------------------------------------------------------------
// INT4 + SpinQuant path with cached rotated weights.
// The weight side (dequant + rotate + INT4 quant) is done at load time and
// stored in cw.d_i4_rot + cw.d_scales_rot. Per call we still:
//   - copy src1 F32 -> FP32 temp
//   - apply the same FWHT rotation to src1
//   - quantize rotated src1 -> INT4 + col scales
//   - CUTLASS INT4 GEMM
//   - INT32 -> FP32 dequant (coalesced; not fused)
// -----------------------------------------------------------------------------
static void custom_ggml_q4_kernel_spin_cached_int4(
    ggml_backend_cuda_context & ctx,
    const cached_q4_weight & cw,
    const float * src1_ddf_i,
    float * dst_dd_i,
    int64_t row_diff,
    int64_t N,
    int64_t K,
    int64_t ldc,
    cudaStream_t stream
) {
    int id = ggml_cuda_get_device();

    // Step 1: apply FWHT rotation to activations (out-of-place from src1_ddf_i
    // to a scratch buffer; we must not modify the activation tensor in place).
    ggml_cuda_pool_alloc<float> src1_f32(ctx.pool(id));
    src1_f32.alloc(N * K);
    bool rotate_src1 = block_fwht_rotate_rows_cuda(
        src1_ddf_i, src1_f32.get(), N, K, stream);
    GGML_ASSERT(rotate_src1);

    // Step 2: quantize rotated FP32 activations -> packed signed INT4 + col scales.
    const int64_t packed_B_bytes = N * (K / 2);
    ggml_cuda_pool_alloc<uint8_t> src1_as_i4(ctx.pool(id));
    ggml_cuda_pool_alloc<float>   src1_scales(ctx.pool(id));
    src1_as_i4.alloc(packed_B_bytes);
    src1_scales.alloc(N);
    quantize_f32_to_int4_row_wise_cuda(
        src1_f32.get(), src1_as_i4.get(), src1_scales.get(), N, K, stream);

    // Step 3: CUTLASS INT4 GEMM using cached rotated weights.
    ggml_cuda_pool_alloc<int32_t> dst_i32(ctx.pool(id));
    dst_i32.alloc(ldc * N);
    bool matmul_success = int4_matmul_cutlass_cuda(
        cw.d_i4_rot,          // A: [row_diff, K/2] cached rotated INT4 weights
        src1_as_i4.get(),     // B: [N,      K/2] rotated INT4 activations
        dst_i32.get(),        // C: [row_diff, N] col-major INT32 output
        (int) row_diff, (int) N, (int) K, (int) ldc,
        stream);
    GGML_ASSERT(matmul_success);

    // Step 4: dequantize INT32 -> FP32 with per-row (weight) and per-col (act) scales.
    dequantize_i32_to_f32_cuda(
        dst_i32.get(), dst_dd_i,
        cw.d_scales_rot, src1_scales.get(),
        row_diff, N, ldc, stream);
}


// =====================================================================================
// Top-level dispatch: amir_q4_v1 entry called from the ggml-cuda dispatch hook
// when CUSTOM_Q4_KERNEL_VERSION == 2.
// =====================================================================================
void custom_ggml_q4_kernel_amir_v1(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i,
    const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src0_dd_i  != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i   != nullptr);

    const int64_t ne00 = src0->ne[0];  // K
    const int64_t ne0  = dst->ne[0];   // M
    const int64_t K = ne00;
    const int64_t N = src1_ncols;
    const int64_t row_diff = row_high - row_low;

    int id = ggml_cuda_get_device();
    int64_t ldc = id == ctx.device ? ne0 : row_diff;

    GGML_ASSERT(row_diff > 0);
    GGML_ASSERT(K % QK4_0 == 0);
    GGML_ASSERT(K % 32 == 0);
    GGML_ASSERT(K % 2 == 0);
    GGML_ASSERT(N > 0);

    // Cache lookup. If preload didn't fire (missing / mismatched pointer),
    // fall back to thai_vu_q4's non-cached path — safer than asserting.
    auto it = g_amir_q4_weight_cache.find((const void *) src0_dd_i);
    if (it == g_amir_q4_weight_cache.end()) {
        // No cache entry: forward to the original (non-cached) kernel.
        extern void custom_ggml_q4_kernel_spin(
            ggml_backend_cuda_context & ctx,
            const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
            const char * src0_dd_i, const float * src1_ddf_i,
            const char * src1_ddq_i, float * dst_dd_i,
            const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
            const int64_t src1_padded_row_size, cudaStream_t stream);
        custom_ggml_q4_kernel_spin(
            ctx, src0, src1, dst,
            src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
            row_low, row_high, src1_ncols, src1_padded_row_size, stream);
        return;
    }
    const cached_q4_weight & cw = it->second;

    // Runtime dispatch (identical to thai_vu_q4).
    float score = compute_incoherence_score_wrapper(
        ctx, id, src1_ddf_i, N, K, stream);
    float threshold_q4_score = get_quantization_incoherent_threshold();

    if (score > threshold_q4_score) {
        // INT8 fallback with cached INT8 weights.
        custom_ggml_q4_weight_q8_compute_kernel_cached(
            ctx, cw, src1, src1_ddf_i, dst_dd_i,
            row_diff, N, K, ldc, stream);
    } else {
        // SpinQuant + INT4 path with cached rotated INT4 weights.
        const int SPIN_BLOCK_H = 256;
        if (K % SPIN_BLOCK_H != 0) {
            // K not divisible by SpinQuant block — fall back to INT8.
            custom_ggml_q4_weight_q8_compute_kernel_cached(
                ctx, cw, src1, src1_ddf_i, dst_dd_i,
                row_diff, N, K, ldc, stream);
        } else {
            custom_ggml_q4_kernel_spin_cached_int4(
                ctx, cw, src1_ddf_i, dst_dd_i,
                row_diff, N, K, ldc, stream);
        }
    }

    GGML_UNUSED_VARS(dst, src1_ddq_i, src1_padded_row_size);
}
