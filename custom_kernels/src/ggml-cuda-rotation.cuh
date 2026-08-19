#pragma once

/*
 * Custom INT8 CUDA helpers for ggml experiments.
 *
 * This header should contain declarations only.
 * Function bodies should stay in ggml-cuda-int8.cu.
 */

#include "ggml.h"

#include <cuda_runtime.h>
#include <stdint.h>

void apply_hadamard_rotation(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int K,
    cudaStream_t stream
);

// void dequantize_q8_0_to_fp32_cuda(
//     const void * input_q8_0,
//     float * output_fp32,
//     int rows,
//     int cols,
//     cudaStream_t stream
// );

void fusion_dequantize_q8_and_hadamard_cuda(
    const void * input_q8_0,
    float * output,
    int rows,
    int K,
    cudaStream_t stream
);

void fusion_hadamard_quantize_fp32_to_int8_cuda(
    const float * input,
    int8_t * output_i8,
    float * output_scales,
    int rows,
    int K,
    cudaStream_t stream
);