// =====================================================================================
// amir_v4 — CUTLASS-fused epilogue (SKELETON)
//
// Replaces amir_v3's GEMM-plus-dequant pair with a single CUTLASS INT8 GEMM whose
// epilogue applies the per-row × per-col scale and writes FP32 directly. Removes
// the standalone dequant kernel (~7 % of GPU time in exp018) and the M×N INT32
// intermediate buffer entirely.
//
// CURRENT STATE: this file is a SKELETON. The body inside
// `#if defined(AMIR_V4_USE_CUTLASS)` is left as TODO for the colleague writing
// the CUTLASS kernel. With the macro undefined (default), the function falls
// back to amir_v2's pipeline: cuBLAS INT8 GEMM + coalesced INT32→FP32 dequant.
// That means selecting `CUSTOM_KERNEL_VERSION=5` today produces output
// identical to `CUSTOM_KERNEL_VERSION=4` (amir_v3) — same kernel, same cache,
// same preload. It only becomes faster once the CUTLASS path is implemented.
//
// HOW TO WORK ON IT:
//   1. Build with `-DCUSTOM_KERNEL_VERSION=5`. Verify amir_v4 ≡ amir_v3 (both
//      run the fallback). This validates the dispatch and the surrounding
//      cache + activation-quant code.
//   2. Add CUTLASS to the build (git submodule under
//      experiments/custom_kernels/thirdparty/cutlass/, or system package).
//      Make sure `#include <cutlass/cutlass.h>` resolves from this TU's
//      include path.
//   3. Define `AMIR_V4_USE_CUTLASS` (e.g. via target_compile_definitions in
//      custom_kernels/CMakeLists.txt) and incrementally fill in the GEMM
//      template + RowColScale epilogue inside the `#if` block.
//   4. Validate numerics against amir_v3 (should be bit-identical at the
//      matmul level — same INT32 accumulator math, same FP32 scaling).
//
// See `cutlass_improvement.md` (this directory) for the design discussion,
// expected performance impact, and pointers to relevant CUTLASS examples.
// =====================================================================================

#include "common.cuh"
#include "int8_helpers.cuh"

// Toggle once the CUTLASS path is implemented. Until then, amir_v4 falls back
// to amir_v2's pipeline (cuBLAS INT8 GEMM + coalesced dequant).
// #define AMIR_V4_USE_CUTLASS

#if defined(AMIR_V4_USE_CUTLASS)
// -------------------------------------------------------------------------------------
// TODO[colleague]: CUTLASS includes
// -------------------------------------------------------------------------------------
// #include <cutlass/cutlass.h>
// #include <cutlass/gemm/device/gemm_universal.h>
// #include <cutlass/epilogue/thread/linear_combination.h>
// #include <cutlass/util/host_tensor.h>
// ... (epilogue + broadcast headers — see cutlass_improvement.md)
#endif


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

    // -------------------------------------------------------------------------------------
    // Step 1: cached Q8_0 → INT8 weights (shared with amir_v1/v2/v3 — DO NOT re-implement).
    // amir_v3's preload populates this cache at model load; if we get a miss here
    // it means amir_v3's preload didn't fire (e.g. CUSTOM_KERNEL_VERSION < 4 path),
    // and we lazy-fill like amir_v1/v2.
    // -------------------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------------------
    // Step 2: activations FP32 → INT8 (per call). Same as amir_v2/v3 — unchanged.
    // -------------------------------------------------------------------------------------
    ggml_cuda_pool_alloc<int8_t> src1_as_i8(ctx.pool(id));
    ggml_cuda_pool_alloc<float>  src1_scales(ctx.pool(id));
    src1_as_i8.alloc(N * K);
    src1_scales.alloc(N);
    quantize_fp32_to_int8_row_wise_cuda(
        src1_ddf_i, src1_as_i8.get(), src1_scales.get(), N, K, stream);

    // -------------------------------------------------------------------------------------
    // Step 3: fused INT8 GEMM + per-row × per-col dequant → FP32 output.
    //
    // Inputs to the CUTLASS GEMM:
    //   A           = cw.d_i8            (INT8, [row_diff × K], row-major, lda = K)
    //   B           = src1_as_i8.get()   (INT8, [K × N],        col-major, ldb = K)
    //   row_scales  = cw.d_scales        (FP32, length row_diff)  — broadcast along N
    //   col_scales  = src1_scales.get()  (FP32, length N)          — broadcast along M
    //
    // Output of the CUTLASS GEMM (epilogue writes this directly, no INT32 buffer):
    //   D           = dst_dd_i           (FP32, [row_diff × N], col-major, ldc = ldc)
    //
    // Epilogue arithmetic, per output element:
    //   D[i, j] = float(acc[i, j]) * row_scales[i] * col_scales[j]
    //
    // This is identical math to amir_v3 (cuBLAS i8·i8→s32 + coalesced dequant),
    // just fused into a single kernel with no INT32 intermediate.
    // -------------------------------------------------------------------------------------

#if defined(AMIR_V4_USE_CUTLASS)

    // =====================================================================
    // TODO[colleague]: CUTLASS implementation
    // =====================================================================
    //
    //   3.1  Pick template parameters tuned for sm_110 (Blackwell). Start from
    //        the closest sm_90 / sm_100 preset and sweep tile shapes once
    //        functional correctness is established. Candidate starting point:
    //
    //          using ElementA       = int8_t;
    //          using ElementB       = int8_t;
    //          using ElementAcc     = int32_t;
    //          using ElementD       = float;
    //          using ElementCompute = float;
    //          using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
    //          using WarpShape        = cutlass::gemm::GemmShape< 64,  64, 64>;
    //          using InstructionShape = cutlass::gemm::GemmShape< 16,   8, 32>;
    //
    //   3.2  Implement a "RowColScale" epilogue that takes two scale vectors
    //        (length M = row_diff, length N = src1_ncols) and applies them in
    //        the GEMM's epilogue stage. Options:
    //          (a) CUTLASS 3.x collective epilogue with broadcast vectors.
    //              See cutlass/examples/52_* for patterns.
    //          (b) CUTLASS 2.x with a custom ThreadEpilogueOp / GemmWithBroadcast.
    //              See cutlass/examples/35_* and 45_*.
    //        Either path: D[i,j] = float(acc[i,j]) * row_scales[i] * col_scales[j].
    //
    //   3.3  Instantiate Gemm and run:
    //
    //          typename Gemm::Arguments args{
    //              {/*M=*/ (int) row_diff, /*N=*/ (int) N, /*K=*/ (int) K},
    //              /* A    */ {cw.d_i8,          K},
    //              /* B    */ {src1_as_i8.get(), K},
    //              /* D    */ {dst_dd_i,         ldc},
    //              /* row_scale ptr (length M) */ cw.d_scales,
    //              /* col_scale ptr (length N) */ src1_scales.get(),
    //              /* split-k, batch, etc. */
    //          };
    //          // Workspace: allocate from ctx.pool(id) via ggml_cuda_pool_alloc<char>.
    //          size_t workspace_size = Gemm::get_workspace_size(args);
    //          ggml_cuda_pool_alloc<char> workspace(ctx.pool(id));
    //          workspace.alloc(workspace_size);
    //
    //          Gemm gemm_op;
    //          cutlass::Status status = gemm_op(args, workspace.get(), stream);
    //          if (status != cutlass::Status::kSuccess) {
    //              fprintf(stderr,
    //                  "amir_v4: CUTLASS GEMM failed: %d\n", (int) status);
    //              /* TODO: fall back to amir_v2 path or abort */
    //          }
    //
    //   3.4  Validate against amir_v3 output for a few weight tensors before
    //        integration (see cutlass_improvement.md "What can go wrong").
    // =====================================================================

    fprintf(stderr,
        "amir_v4: AMIR_V4_USE_CUTLASS is defined but the CUTLASS body is not "
        "implemented yet. See custom_kernels/src/cutlass_improvement.md.\n");
    abort();

#else
    // -------------------------------------------------------------------------------------
    // FALLBACK: amir_v2 pipeline (cuBLAS INT8 GEMM + coalesced dequant).
    // Identical math; not fused. amir_v4 in this mode behaves exactly like amir_v3
    // (assuming the preload populated the cache).
    // -------------------------------------------------------------------------------------
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

    dequantize_i32_to_f32_coalesced_cuda(
        dst_i32.get(), dst_dd_i,
        cw.d_scales, src1_scales.get(),
        row_diff, N, ldc, stream);
#endif

    GGML_UNUSED_VARS(dst, src1_ddq_i, src1_padded_row_size);
}
