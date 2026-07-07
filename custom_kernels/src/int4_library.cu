#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-cuda/common.cuh"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <unordered_set>

#include "ggml-cuda-int8.cuh"
#include "ggml-cuda-int4.cuh"

/*
The kernel custom_ggml_q4_kernel_naive is a naive implementation of INT4 matmul, where:
- src0 is converted to row-wise INT4 and scale.
- src1 is quantized to row-wise INT4 and scale.
- The INT4 matmul is performed using CUTLASS.
- The result is dequantized back to FP32 using the scales from src0 and src1.

This kernel is for demonstration purposes. The latency and accuracy may not be optimal. 
*/
static void custom_ggml_q4_kernel_naive(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst,
    const char * src0_dd_i,
    const float * src1_ddf_i,
    const char * src1_ddq_i,
    float * dst_dd_i,
    const int64_t row_low,
    const int64_t row_high,
    const int64_t src1_ncols,
    const int64_t src1_padded_row_size,
    cudaStream_t stream
) {
    GGML_ASSERT(src0->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    GGML_ASSERT(src0_dd_i  != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i   != nullptr);

    const int64_t ne00 = src0->ne[0]; // K
    const int64_t ne10 = src1->ne[0]; // K
    const int64_t ne0  = dst->ne[0];  // M 

    GGML_ASSERT(ne00 == ne10);

    const int64_t K = ne00;
    const int64_t M = ne0;
    const int64_t N = src1_ncols;

    const int64_t row_diff = row_high - row_low;

    GGML_ASSERT(K % QK4_0 == 0);
    GGML_ASSERT(K % 32 == 0);

    int id = ggml_cuda_get_device();

    int64_t ldc = id == ctx.device ? ne0 : row_diff;

    /*
        GGML matmul view:

        src0->ne = [K, M]
        src1->ne = [K, N]
        dst ->ne = [M, N]

        Logical math:
            src0: [M, K]
            src1: [N, K]

        Output:
            dst[m, n] = sum_k src0[m, k] * src1[n, k]
    */

    // ------------------------------------------------------------
    // Step 1: Convert GGML Q4_0 src0 -> row-wise packed signed INT4
    // ------------------------------------------------------------
    const int64_t packed_A_bytes = row_diff * (K / 2);

    ggml_cuda_pool_alloc<uint8_t> src0_as_i4(ctx.pool(id));
    ggml_cuda_pool_alloc<float>   src0_scales(ctx.pool(id));

    src0_as_i4.alloc(packed_A_bytes);
    src0_scales.alloc(row_diff);

    convert_q4_0_to_int4_row_wise_cuda(
        src0_dd_i,
        src0_as_i4.get(),
        src0_scales.get(),
        row_diff,
        K,
        stream
    );

    // ------------------------------------------------------------
    // Step 2: Quantize FP32 src1 -> row-wise packed signed INT4
    // ------------------------------------------------------------
    const int64_t packed_B_bytes = N * (K / 2);

    ggml_cuda_pool_alloc<uint8_t> src1_as_i4(ctx.pool(id));
    ggml_cuda_pool_alloc<float>   src1_scales(ctx.pool(id));

    src1_as_i4.alloc(packed_B_bytes);
    src1_scales.alloc(N);

    quantize_fp32_to_int4_row_wise_cuda(
        src1_ddf_i,
        src1_as_i4.get(),
        src1_scales.get(),
        N,
        K,
        stream
    );

    // ------------------------------------------------------------
    // Step 3: CUTLASS INT4 GEMM
    // ------------------------------------------------------------
    ggml_cuda_pool_alloc<int32_t> dst_i32(ctx.pool(id));
    dst_i32.alloc(ldc * N);

    bool matmul_success = int4_matmul_cutlass_cuda(
        src0_as_i4.get(),  // A: [row_diff, K], packed signed INT4 row-major
        src1_as_i4.get(),  // B: [N, K], packed signed INT4 physical row-major
        dst_i32.get(),     // C: [row_diff, N], column-major physical
        row_diff,
        N,
        K,
        ldc,
        stream
    );

    GGML_ASSERT(matmul_success);

    // ------------------------------------------------------------
    // Step 4: Dequantize INT32 -> FP32
    // ------------------------------------------------------------
    dequantize_i32_to_f32_cuda(
        dst_i32.get(),
        dst_dd_i,
        src0_scales.get(),
        src1_scales.get(),
        row_diff,
        N,
        ldc,
        stream
    );

    GGML_UNUSED_VARS(dst, src1_ddq_i, src1_padded_row_size, M);
}

/*
The kernel custom_ggml_q4_weight_q8_compute_kernel take input (src0 and src1) as Q4_0 and F32.
- Convert src0 Q4_0 -> INT8 + row scales
- Quantize src1 F32 -> INT8 + row scales
- INT8 GEMM + dequantization to FP32.
*/
static void custom_ggml_q4_weight_q8_compute_kernel(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst,
    const char * src0_dd_i,
    const float * src1_ddf_i,
    const char * src1_ddq_i,
    float * dst_dd_i,
    const int64_t row_low,
    const int64_t row_high,
    const int64_t src1_ncols,
    const int64_t src1_padded_row_size,
    cudaStream_t stream
) {
    GGML_ASSERT(src0->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    GGML_ASSERT(src0_dd_i  != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i   != nullptr);

    const int64_t ne00 = src0->ne[0]; // K
    const int64_t ne10 = src1->ne[0]; // K
    const int64_t ne0  = dst->ne[0];  // same convention as working Q8 kernel

    GGML_ASSERT(ne00 == ne10);

    const int64_t K = ne00;
    const int64_t M = ne0;
    const int64_t N = src1_ncols;

    const int64_t row_diff = row_high - row_low;

    GGML_ASSERT(row_diff > 0);
    GGML_ASSERT(K % QK4_0 == 0);
    GGML_ASSERT(N > 0);

    int id = ggml_cuda_get_device();

    // dst_dd_i is treated as column-major D[m + n * ldc].
    int64_t ldc = id == ctx.device ? ne0 : row_diff;

    /*
        GGML matmul view:
            src0->ne = [K, M]
            src1->ne = [K, N]
            dst ->ne = [M, N]

        Logical math:
            src0: [M, K]
            src1: [N, K]

        Equivalent:
            dst = src0 @ src1^T
    */

    // ------------------------------------------------------------
    // Step 1: convert src0 Q4_0 -> INT8 + row scales
    // ------------------------------------------------------------
    ggml_cuda_pool_alloc<int8_t> src0_as_i8(ctx.pool(id));
    ggml_cuda_pool_alloc<float>  src0_scales(ctx.pool(id));

    src0_as_i8.alloc(row_diff * K);
    src0_scales.alloc(row_diff);

    convert_q4_0_to_int8_row_wise_cuda(
        src0_dd_i,
        src0_as_i8.get(),
        src0_scales.get(),
        row_diff,
        K,
        stream
    );

    // ------------------------------------------------------------
    // Step 2: quantize src1 F32 -> INT8 + row scales
    // ------------------------------------------------------------
    ggml_cuda_pool_alloc<int8_t> src1_as_i8(ctx.pool(id));
    ggml_cuda_pool_alloc<float>  src1_scales(ctx.pool(id));

    src1_as_i8.alloc(N * K);
    src1_scales.alloc(N);

    quantize_fp32_to_int8_row_wise_cuda(
        src1_ddf_i,
        src1_as_i8.get(),
        src1_scales.get(),
        N, // rows = src1_ncols
        K, // cols = src1->ne[0]
        stream
    );

    // ------------------------------------------------------------
    // Step 3 + 4: INT8 GEMM with fused dequantization to FP32
    // ------------------------------------------------------------
    bool matmul_dequant = matmul_w8a8_cutlass_cuda(
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

    GGML_ASSERT(matmul_dequant);

    GGML_UNUSED_VARS(dst, src1_ddq_i, src1_padded_row_size, M);
}


static float compute_incoherence_score_wrapper(
    ggml_backend_cuda_context & ctx,
    int id,
    const float * src1_ddf_i,
    int64_t N,
    int64_t K,
    cudaStream_t stream
) {
    const int64_t numel = N * K;

    // ------------------------------------------------------------
    // Allocate output score on device
    // ------------------------------------------------------------
    ggml_cuda_pool_alloc<float> score_device(ctx.pool(id));
    score_device.alloc(1);

    // ------------------------------------------------------------
    // Allocate temporary partial buffers
    // ------------------------------------------------------------
    int num_blocks = get_incoherence_num_blocks(numel);

    ggml_cuda_pool_alloc<float> partial_max_abs(ctx.pool(id));
    ggml_cuda_pool_alloc<float> partial_sum_sq(ctx.pool(id));

    partial_max_abs.alloc(num_blocks);
    partial_sum_sq.alloc(num_blocks);

    // ------------------------------------------------------------
    // Compute score on GPU
    // ------------------------------------------------------------
    compute_incoherence_score_cuda_fast(
        src1_ddf_i,
        score_device.get(),
        partial_max_abs.get(),
        partial_sum_sq.get(),
        N,
        K,
        num_blocks,
        stream
    );

    // ------------------------------------------------------------
    // Copy score back to host
    // ------------------------------------------------------------
    float score = 0.0f;

    CUDA_CHECK(cudaMemcpyAsync(
        &score,
        score_device.get(),
        sizeof(float),
        cudaMemcpyDeviceToHost,
        stream
    ));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    return score;
}


/*
custom_ggml_q4_kernel_spin is a custom Q4_0-weight matmul path.

It first computes an incoherence score for src1.

1. If incoherence score is larger than threshold, falls back to 
    the Q8 compute kernel (function custom_ggml_q4_weight_q8_compute_kernel)

2. If K is incompatible with the SpinQuant block size (K % 32 != 0), 
   it also falls back to the Q8 compute kernel (function custom_ggml_q4_weight_q8_compute_kernel).

2. Otherwise, it uses an INT4 + rotation path:
   - Dequantize Q4_0 src0 to FP32.
   - Copy src1 FP32 into a temporary FP32 buffer.
   - Apply the same block FWHT rotation to src0 and src1.
   - Quantize both rotated FP32 matrices to row-wise INT4 with scales.
   - Perform INT4 GEMM
   - Dequantize the INT32 accumulator output back to FP32 using the src0/src1 scales.

    The basic math is as follows:
    (src0 R) @ (src1 R)^T = src0 R R^T src1^T = src0 @ src1^T
*/
void custom_ggml_q4_kernel_spin(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst,
    const char * src0_dd_i,
    const float * src1_ddf_i,
    const char * src1_ddq_i,
    float * dst_dd_i,
    const int64_t row_low,
    const int64_t row_high,
    const int64_t src1_ncols,
    const int64_t src1_padded_row_size,
    cudaStream_t stream
) {
    // printf("[DEBUG]: My Custom int4 Kernel\n");
    // print_ggml_tensor_info(src0, "src0");
    // print_ggml_tensor_info(src1, "src1");
    // print_ggml_tensor_info(dst, "dst");

    GGML_ASSERT(src0->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    GGML_ASSERT(src0_dd_i  != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i   != nullptr);
    int id = ggml_cuda_get_device();

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne0  = dst->ne[0];

    GGML_ASSERT(ne00 == ne10);

    const int64_t K = ne00;
    const int64_t M = ne0;
    const int64_t N = src1_ncols;

    const int64_t row_diff = row_high - row_low;
    int64_t ldc = id == ctx.device ? ne0 : row_diff;

    GGML_ASSERT(row_diff > 0);
    GGML_ASSERT(K % QK4_0 == 0);
    GGML_ASSERT(K % 32 == 0);
    GGML_ASSERT(K % 2 == 0);
    GGML_ASSERT(N > 0);

    /*
    Branch 1: compute the incoherence score of src1. 
    If the score is above a threshold, we will use the INT8 computation kernel.
    Otherwise, we will use the INT4 + Rotation kernel.
    */
    float score = compute_incoherence_score_wrapper(
        ctx,
        id,
        src1_ddf_i,
        N,
        K,
        stream
    );

    float threshold_q4_score = get_quantization_incoherent_threshold();
    if (score > threshold_q4_score) {
        // Fallback to the INT8 kernel if the incoherence score is larger than threshold
        printf("[DEBUG]: Incoherence score %.6f > threshold %.6f, using INT8 kernel\n", score, threshold_q4_score);
        custom_ggml_q4_weight_q8_compute_kernel(
            ctx,
            src0,
            src1,
            dst,
            src0_dd_i,
            src1_ddf_i,
            src1_ddq_i,
            dst_dd_i,
            row_low,
            row_high,
            src1_ncols,
            src1_padded_row_size,
            stream
        );
        return;
    }

    const int SPIN_BLOCK_H = 256;
    if (K % SPIN_BLOCK_H != 0) {
        // Fallback to the INT8 kernel if K is not divisible by SPIN_BLOCK_H
        printf("[DEBUG]: K %% SPIN_BLOCK_H != 0, using INT8 kernel\n");
        custom_ggml_q4_weight_q8_compute_kernel(
            ctx,
            src0,
            src1,
            dst,
            src0_dd_i,
            src1_ddf_i,
            src1_ddq_i,
            dst_dd_i,
            row_low,
            row_high,
            src1_ncols,
            src1_padded_row_size,
            stream
        );
        return;
    }
    printf("[DEBUG] Using INT4 + Rotation kernel. ");

    /*
        SpinQuant-style rotation for this layout:

            src0_rot = src0 @ R
            src1_rot = src1 @ R

        Since R is orthogonal:

            src0_rot @ src1_rot^T
            =
            src0 R R^T src1^T
            =
            src0 @ src1^T
    */

    // ------------------------------------------------------------
    // Step 1: dequantize Q4_0 src0 -> FP32 temp
    // ------------------------------------------------------------
    ggml_cuda_pool_alloc<float> src0_f32(ctx.pool(id));
    src0_f32.alloc(row_diff * K);

    dequantize_q4_0_to_f32_cuda(
        src0_dd_i,
        src0_f32.get(),
        row_diff,
        K,
        stream
    );

    // ------------------------------------------------------------
    // Step 2: copy src1 F32 -> FP32 temp
    // ------------------------------------------------------------
    ggml_cuda_pool_alloc<float> src1_f32(ctx.pool(id));
    src1_f32.alloc(N * K);

    // ------------------------------------------------------------
    // Step 3: apply same orthogonal rotation to both FP32 matrices
    // ------------------------------------------------------------
    bool rotate_src0 = block_fwht_rotate_rows_inplace_cuda(
        src0_f32.get(),
        row_diff,
        K,
        stream
    );

    bool rotate_src1 = block_fwht_rotate_rows_cuda(
        src1_ddf_i,
        src1_f32.get(),
        N,
        K,
        stream
    );

    GGML_ASSERT(rotate_src0);
    GGML_ASSERT(rotate_src1);

    // ------------------------------------------------------------
    // Step 4: quantize rotated FP32 matrices -> packed signed INT4
    // ------------------------------------------------------------
    const int64_t packed_A_bytes = row_diff * (K / 2);
    const int64_t packed_B_bytes = N * (K / 2);

    ggml_cuda_pool_alloc<uint8_t> src0_as_i4(ctx.pool(id));
    ggml_cuda_pool_alloc<uint8_t> src1_as_i4(ctx.pool(id));

    ggml_cuda_pool_alloc<float> src0_scales(ctx.pool(id));
    ggml_cuda_pool_alloc<float> src1_scales(ctx.pool(id));

    src0_as_i4.alloc(packed_A_bytes);
    src1_as_i4.alloc(packed_B_bytes);

    src0_scales.alloc(row_diff);
    src1_scales.alloc(N);

    quantize_f32_to_int4_row_wise_cuda(
        src0_f32.get(),
        src0_as_i4.get(),
        src0_scales.get(),
        row_diff,
        K,
        stream
    );

    quantize_f32_to_int4_row_wise_cuda(
        src1_f32.get(),
        src1_as_i4.get(),
        src1_scales.get(),
        N,
        K,
        stream
    );

    // ------------------------------------------------------------
    // Step 5: CUTLASS INT4 GEMM
    // ------------------------------------------------------------
    ggml_cuda_pool_alloc<int32_t> dst_i32(ctx.pool(id));
    dst_i32.alloc(ldc * N);

    bool matmul_success = int4_matmul_cutlass_cuda(
        src0_as_i4.get(),
        src1_as_i4.get(),
        dst_i32.get(),
        row_diff,
        N,
        K,
        ldc,
        stream
    );

    GGML_ASSERT(matmul_success);

    // ------------------------------------------------------------
    // Step 6: dequantize INT32 -> FP32
    // ------------------------------------------------------------
    dequantize_i32_to_f32_cuda(
        dst_i32.get(),
        dst_dd_i,
        src0_scales.get(),
        src1_scales.get(),
        row_diff,
        N,
        ldc,
        stream
    );

    GGML_UNUSED_VARS(dst, src1_ddq_i, src1_padded_row_size, M);
}