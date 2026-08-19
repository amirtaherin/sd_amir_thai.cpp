/*
thai_vu_v6.cu: Quantization + Rotation

- src0: Q8_0 -> FP32 -> Rotation -> int8 + scales
- src1: FP32 -> Rotation -> int8 + scales
- dst: CUTLASS int8 x int8 -> FP32

NO Caching yet.
*/

#include "common.cuh"
#include "int8_helpers.cuh"
#include "ggml-cuda-rotation.cuh"

void custom_kernels_q8_rotation(
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
    const int64_t ne10 = src1->ne[0]; // K, must match src0's K
    const int64_t ne0 = dst->ne[0]; // M

    const int64_t K = ne00;
    const int64_t M = ne0;
    const int64_t N = src1_ncols;

    const int64_t row_diff = row_high - row_low;

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    int64_t ldc = id == ctx.device ? ne0 : row_diff;

    const int cc = ggml_cuda_info().devices[id].cc;

    /*
        ggml matmul view:
        src0->ne = [K, M]
        src1->ne = [K, N]
        dst ->ne = [M, N]

        Logical math:
        src0: [M, K]
        src1: [N, K]

        Equivalent: dst = src0 @ src1^T
    */

    // // ------------------------------------------------------------
    // // Step 1: convert src0 Q8_0 -> FP32 -> Rotation -> int8 + scales
    // // ------------------------------------------------------------
    ggml_cuda_pool_alloc<float> src0_rotated(ctx.pool(id));
    ggml_cuda_pool_alloc<int8_t> src0_as_i8(ctx.pool(id));
    ggml_cuda_pool_alloc<float> src0_scales(ctx.pool(id));

    src0_rotated.alloc(row_diff * K);
    src0_as_i8.alloc(row_diff * K);
    src0_scales.alloc(row_diff);

    fusion_dequantize_q8_and_hadamard_cuda(
        src0_dd_i,
        src0_rotated.get(),
        row_diff,
        K,
        stream
    );  

    quantize_fp32_to_int8_row_wise_cuda(
        src0_rotated.get(),
        src0_as_i8.get(),
        src0_scales.get(),
        row_diff,
        K,
        stream
    );

    // ------------------------------------------------------------
    // Step 2: quantize src1 fp32 -> Rotation -> int8 + scales
    // ------------------------------------------------------------
    ggml_cuda_pool_alloc<int8_t> src1_as_i8(ctx.pool(id));
    ggml_cuda_pool_alloc<float> src1_scales(ctx.pool(id));

    src1_as_i8.alloc(N * K);
    src1_scales.alloc(N);

    fusion_hadamard_quantize_fp32_to_int8_cuda(
        src1_ddf_i,
        src1_as_i8.get(),
        src1_scales.get(),
        N,
        K,
        stream
    );

    // ------------------------------------------------------------
    // Step 3: CUTLASS int8 x int8 -> Fp32
    // ------------------------------------------------------------
#if defined(CUSTOM_KERNEL_HAVE_CUTLASS)
    bool matmul_success = matmul_w8a8_cutlass_cuda(
        src0_as_i8.get(),    // A: [row_diff, K]
        src1_as_i8.get(),    // B: [N, K]
        src0_scales.get(),   // alphaRow: [row_diff]
        src1_scales.get(),   // alphaCol: [N]
        dst_dd_i,            // D: fp32, column-major D[m + n * ldc]
        row_diff,
        N,
        K,
        ldc,
        stream
    );

    if (!matmul_success) {
        fprintf(stderr,
                "[thai_vu_v6] CUTLASS matmul_w8a8_cutlass_cuda returned false\n");
    }
#else
    fprintf(stderr,
            "[thai_vu_v6] The CUTLASS library is not available."
            "Reconfigure with -DCUTLASS_DIR=/path/to/cutlass "
            "(see custom_kernels/README.md).\n");
    abort();
#endif

    GGML_UNUSED_VARS(dst, src1_ddq_i, src1_padded_row_size);
}