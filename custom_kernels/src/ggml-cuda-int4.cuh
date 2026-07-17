#pragma once

#include <stdint.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>


void convert_q4_0_to_int4_row_wise_cuda(
    const char * src_q4,
    uint8_t * dst_i4_packed,
    float * row_scales,
    int rows,
    int K,
    cudaStream_t stream
);

void quantize_fp32_to_int4_row_wise_cuda(
    const float * src,
    uint8_t * dst_i4_packed,
    float * row_scales,
    int rows,
    int K,
    cudaStream_t stream
);

bool int4_matmul_cutlass_cuda(
    const uint8_t * A_i4_packed,
    const uint8_t * B_i4_packed,
    int32_t * C_i32,
    int M,
    int N,
    int K,
    int ldc,
    cudaStream_t stream
);

void convert_q4_0_to_int8_row_wise_cuda(
    const char * src_q4,
    int8_t * dst_i8,
    float * row_scales,
    int rows,
    int K,
    cudaStream_t stream
);

void dequantize_q4_0_to_f32_cuda(
    const char * src_q4,
    float * dst_f32,
    int rows,
    int K,
    cudaStream_t stream
);

void quantize_f32_to_int4_row_wise_cuda(
    const float * src,
    uint8_t * dst_i4,
    float * row_scales,
    int rows,
    int K,
    cudaStream_t stream
);

bool block_fwht_rotate_rows_inplace_cuda(
    float * x,
    int rows,
    int K,
    cudaStream_t stream
);

bool block_fwht_rotate_rows_cuda(
    const float * x_in,
    float * x_out,
    int rows,
    int K,
    cudaStream_t stream
);

void compute_incoherence_score_cuda(
    const float * src1_ddf_i,
    float * score_device,
    int64_t N,
    int64_t K,
    cudaStream_t stream
);

float get_quantization_incoherent_threshold();


static inline int get_incoherence_num_blocks(int64_t numel) {
    constexpr int BLOCK_SIZE = 256;

    int num_blocks = (int)((numel + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Avoid launching too many blocks.
    num_blocks = std::min(num_blocks, 4096);

    // At least one block.
    num_blocks = std::max(num_blocks, 1);

    return num_blocks;
};

void compute_incoherence_score_cuda_fast(
    const float * src1_ddf_i,
    float * score_device,
    float * partial_max_abs,
    float * partial_sum_sq,
    int64_t N,
    int64_t K,
    int num_blocks,
    cudaStream_t stream
) ;

// ============================================================================
// Reusable dispatch helpers (defined in int4_library.cu).
// amir_q4_v1 (custom_kernels_mmq_amir_q4_v1) forwards to these to share the
// runtime incoherence-score computation and the INT8 fallback path.
// The full types (ggml_backend_cuda_context, ggml_tensor) come from common.cuh
// / ggml.h which the .cu callers already include before this header.
// ============================================================================
struct ggml_backend_cuda_context;
struct ggml_tensor;

float compute_incoherence_score_wrapper(
    ggml_backend_cuda_context & ctx,
    int id,
    const float * src1_ddf_i,
    int64_t N,
    int64_t K,
    cudaStream_t stream
);

void custom_ggml_q4_weight_q8_compute_kernel(
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
);

// ============================================================================
// amir_q4_v1 — cached Q4_0 weight conversion.
//
// For each Q4_0 weight tensor we precompute two derivatives once (at model
// load, via custom_kernels::preload_q4_0_weights) and cache them:
//   d_i4_rot     : rotated INT4 packed weights (for the INT4 + SpinQuant path)
//   d_scales_rot : per-row scales for the rotated INT4 weights
//   d_i8         : plain INT8 weights (for the INT8 fallback path)
//   d_scales_i8  : per-row scales for the INT8 weights
//
// Both derivatives are cached because the runtime dispatch (score > threshold)
// can pick either at any moment. Keyed by src0's device pointer, which is
// stable (ggml loads weights once into a fixed backend buffer).
// ============================================================================
#include <unordered_map>

struct cached_q4_weight {
    // Rotated INT4 (for INT4 + SpinQuant path)
    uint8_t * d_i4_rot     = nullptr;
    float   * d_scales_rot = nullptr;

    // Plain INT8 (for INT8 fallback path — mirrors thai_vu_q4's INT8 conversion)
    int8_t  * d_i8         = nullptr;
    float   * d_scales_i8  = nullptr;

    int32_t   M = 0;   // = row_diff at cache time; recorded for sanity checks
    int32_t   K = 0;
};

extern std::unordered_map<const void *, cached_q4_weight> g_amir_q4_weight_cache;
