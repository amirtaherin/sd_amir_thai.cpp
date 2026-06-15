#pragma once

// Custom INT8 helper kernels for ggml experiments.
// Shared infrastructure used by every kernel variant (thai_vu, amir_v1, amir_v2, ...).
// Compiled into the ggml-cuda library via custom_kernels/CMakeLists.txt.

#include "ggml.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <unordered_map>

// Debug helper
void print_ggml_tensor_info(const struct ggml_tensor * t, const char * name);

// Row-wise FP32 -> INT8 quantization
//   input:  [rows, cols], row-major
//   output: [rows, cols], row-major
//   scales: [rows]
void quantize_fp32_to_int8_row_wise_cuda(
    const float * input, int8_t * output, float * scales,
    int rows, int cols, cudaStream_t stream);

// Column-wise FP32 -> INT8 quantization (operands stored column-major).
//   scales: [cols]
void quantize_fp32_to_int8_col_wise_cuda(
    const float * input, int8_t * output, float * scales,
    int rows, int cols, cudaStream_t stream);

// INT32 -> FP32 dequantization (out = i32 * row_scale[i] * col_scale[j]).
// Original kernel: (16x16) tile, threadIdx.x -> column -> uncoalesced.
void dequantize_i32_to_f32_cuda(
    const int32_t * input_i32, float * output_f32,
    const float * row_scales, const float * col_scales,
    int rows, int cols, int ldc, cudaStream_t stream);

// amir_v2: memory-coalesced variant. Identical math; 1D grid-stride loop
// over column-major contiguous elements -> coalesced loads/stores.
void dequantize_i32_to_f32_coalesced_cuda(
    const int32_t * input_i32, float * output_f32,
    const float * row_scales, const float * col_scales,
    int rows, int cols, int ldc, cudaStream_t stream);

// Convert ggml Q8_0 blocks to plain INT8 + a single row-wise scale.
// Used by all variants to derive the INT8 weight matrix from Q8_0 weights.
void convert_q8_0_to_int8_row_wise_cuda(
    const void * input_q8_0, int8_t * output_i8, float * output_scales,
    int rows, int cols, cudaStream_t stream);

// Cached-weight infrastructure (shared by amir_v1, amir_v2, future variants).
// Each cached weight stores its INT8 representation + per-row scale.
struct cached_int8_weight {
    int8_t * d_i8     = nullptr;
    float  * d_scales = nullptr;
};

// Process-global cache, keyed by the weight tensor's device pointer
// (stable because ggml loads weights once into a fixed backend buffer).
extern std::unordered_map<const void *, cached_int8_weight> g_amir_int8_weight_cache;

/*
Fused INT8 GEMM + per-row × per-col dequantization → FP32 output (CUTLASS).
Implementation by Thai Vu — used by amir_v4 (CUSTOM_KERNEL_VERSION=5).
Only declared when the build was made with CUTLASS (-DCUTLASS_DIR=...).

- A: [M, K], row-major INT8
- B: [N, K], row-major INT8
- alphaRow: [M] row scales
- alphaCol: [N] column scales
- D: [M, N], column-major FP32 output (leading dim ldc)

Epilogue math:  D[m, n] = float(acc_i32[m, n]) * alphaRow[m] * alphaCol[n]
*/
#if defined(CUSTOM_KERNEL_HAVE_CUTLASS)
// Forward declaration -- defined in ggml-cuda's common.cuh (transitively included
// by amir_v4.cu and int8_helpers.cu inside the CUTLASS guard).
struct ggml_cuda_pool;

bool matmul_w8a8_cutlass_cuda(
    const int8_t* A,                  // [M, K_gemm], row-major
    const int8_t* B,                  // [N_gemm, K_gemm], row-major physical memory
    const float* alphaRow,            // [M]
    const float* alphaCol,            // [N_gemm]
    float* D,                         // [M, N_gemm], column-major physical memory
    int M,
    int N_gemm,
    int K_gemm,
    int ldc,
    ggml_cuda_pool & pool,            // ggml CUDA pool for the GEMM workspace
    cudaStream_t stream
);

// amir_v5 — CUTLASS 3.x version of the same fused INT8 GEMM + per-row × per-col
// dequant epilogue, using the collective::CollectiveBuilder API so we can target
// Sm100 (Blackwell). Same Arguments interface as the 2.x wrapper above.
bool matmul_w8a8_cutlass3x_cuda(
    const int8_t* A,
    const int8_t* B,
    const float* alphaRow,
    const float* alphaCol,
    float* D,
    int M,
    int N_gemm,
    int K_gemm,
    int ldc,
    ggml_cuda_pool & pool,
    cudaStream_t stream
);
#endif