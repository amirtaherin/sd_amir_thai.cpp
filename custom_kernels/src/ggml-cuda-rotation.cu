#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "ggml-cuda.h"
#include "ggml.h"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include "ggml-cuda-rotation.cuh"

#ifndef QK8_0
#define QK8_0 32
#endif

// If ggml_half is uint16_t, we reinterpret it as CUDA half.
struct block_q8_0_simple {
    uint16_t d;
    int8_t qs[QK8_0];
};

__device__ __forceinline__ float q8_0_scale_to_float(uint16_t h) {
    return __half2float(*reinterpret_cast<const __half*>(&h));
}

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                             \
    do {                                                             \
        cudaError_t status = call;                                   \
        if (status != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error: %s:%d: %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(status)); \
            return false;                                            \
        }                                                            \
    } while (0)
#endif

#ifndef CUTLASS_CHECK
#define CUTLASS_CHECK(status)                                      \
    do {                                                           \
        cutlass::Status s = status;                                \
        if (s != cutlass::Status::kSuccess) {                      \
            fprintf(stderr, "CUTLASS error: %s:%d: status = %d\n", \
                    __FILE__, __LINE__, static_cast<int>(s));      \
            return false;                                          \
        }                                                          \
    } while (0)
#endif

// ============================================================
// Hadamard
// ============================================================

constexpr int HADAMARD_BLOCK_SIZE = 256;
constexpr int HADAMARD_THREADS    = 32;
constexpr int VALUES_PER_THREAD   = 8;
constexpr int FUSED_QUANT_THREADS = 256;
constexpr int FUSED_QUANT_WARPS   = FUSED_QUANT_THREADS / 32;

template <typename T>
struct alignas(16) Vec8 {
    T x[8];
};

__device__ __forceinline__ void hadamard_8(float x[8]) {
#pragma unroll
    for (int stride = 1; stride < 8; stride <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int lo  = j & (stride - 1);
            int idx = (j - lo) * 2 + lo;

            float a = x[idx];
            float b = x[idx + stride];

            x[idx]          = a + b;
            x[idx + stride] = a - b;
        }
    }
}

/*
In this function, one warp processes one 256-element block.
// lane 0  -> [0..7]
// lane 1  -> [8..15]
// ...
// lane 31 -> [248..255]
*/
__device__ __forceinline__ void hadamard_256(
    const float* __restrict__ input,
    float x[VALUES_PER_THREAD],
    int lane) {
    Vec8<float> input_vec = reinterpret_cast<const Vec8<float>*>(input)[lane];

#pragma unroll
    for (int i = 0; i < VALUES_PER_THREAD; ++i) {
        x[i] = input_vec.x[i];
    }
    hadamard_8(x);  // H8 inside each lane.

    // H32 across warp lanes.
#pragma unroll
    for (int stride = 1; stride < 32; stride <<= 1) {
        float sign = (lane & stride) ? -1.0f : 1.0f;

#pragma unroll
        for (int i = 0; i < VALUES_PER_THREAD; ++i) {
            float other = __shfl_xor_sync(0xffffffff, x[i], stride);
            x[i]        = sign * x[i] + other;
        }
    }

    // Normalize H256.
    constexpr float scale = 1.0f / 16.0f;

#pragma unroll
    for (int i = 0; i < VALUES_PER_THREAD; ++i) {
        x[i] *= scale;
    }
}

/*
This kernel applies the Hadamard rotation.
*/
__global__ void hadamard_rotation_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int K) {
    int row      = blockIdx.x;
    int block_id = blockIdx.y;
    int lane     = threadIdx.x;

    if (row >= rows) {
        return;
    }

    int base_k = block_id * HADAMARD_BLOCK_SIZE;

    int64_t offset = static_cast<int64_t>(row) * K + base_k;

    float x[VALUES_PER_THREAD];

    hadamard_256(input + offset, x, lane);

    Vec8<float> output_vec;

#pragma unroll
    for (int i = 0; i < VALUES_PER_THREAD; ++i) {
        output_vec.x[i] = x[i];
    }

    reinterpret_cast<Vec8<float>*>(output + offset)[lane] = output_vec;
}

void apply_hadamard_rotation(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int K,
    cudaStream_t stream) {
    if (K % HADAMARD_BLOCK_SIZE != 0) {
        fprintf(stderr,
                "apply_hadamard_rotation: "
                "K must be divisible by %d\n",
                HADAMARD_BLOCK_SIZE);
        return;
    }

    dim3 grid(rows, K / HADAMARD_BLOCK_SIZE);
    dim3 block(HADAMARD_THREADS);

    hadamard_rotation_kernel<<<grid, block, 0, stream>>>(
        input, output, rows, K);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in apply_hadamard_rotation: %s\n", cudaGetErrorString(err));
    }
}

__device__ __forceinline__ float warp_reduce_max_float(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    }
    return v;
}

/*
Function 1: FP32 -> Hadamard -> row max -> scale

This function compute output scales (after Hadamard rotation) for each row.
*/
__global__ void hadamard_compute_row_scale_kernel(
    const float* __restrict__ input,
    float* __restrict__ output_scales,
    int rows,
    int K) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) {
        return;
    }

    int lane    = tid & 31;
    int warp_id = tid >> 5;

    __shared__ float warp_max[FUSED_QUANT_WARPS];

    int num_hadamard_blocks = K / HADAMARD_BLOCK_SIZE;

    float local_max = 0.0f;

    // Each warp handles one or more H256 tiles.
    for (int block_id = warp_id; block_id < num_hadamard_blocks; block_id += FUSED_QUANT_WARPS) {
        int base_k = block_id * HADAMARD_BLOCK_SIZE;

        const float* tile_input = input + static_cast<int64_t>(row) * K + base_k;

        float x[VALUES_PER_THREAD];

        hadamard_256(tile_input, x, lane);

#pragma unroll
        for (int i = 0; i < VALUES_PER_THREAD; ++i) {
            local_max = fmaxf(local_max, fabsf(x[i]));
        }
    }

    // Reduce max inside each warp.
    local_max = warp_reduce_max_float(local_max);
    if (lane == 0) {
        warp_max[warp_id] = local_max;
    }

    __syncthreads();

    // Warp 0 reduces maxima from all warps.
    if (warp_id == 0) {
        float max_value = lane < FUSED_QUANT_WARPS ? warp_max[lane] : 0.0f;

        max_value = warp_reduce_max_float(max_value);

        if (lane == 0) {
            output_scales[row] = max_value > 0.0f ? max_value / 127.0f : 1.0f;
        }
    }
}


/*
Function 2: FP32 -> Hadamard -> INT8
This function actually: rotate -> quantize -> store to output_i8.
*/
__global__ void hadamard_quantize_int8_kernel(
    const float* __restrict__ input,
    int8_t* __restrict__ output_i8,
    const float* __restrict__ input_scales,
    int rows,
    int K) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) {
        return;
    }

    int lane    = tid & 31;
    int warp_id = tid >> 5;

    int num_hadamard_blocks = K / HADAMARD_BLOCK_SIZE;

    float row_scale = input_scales[row];

    for (int block_id = warp_id; block_id < num_hadamard_blocks; block_id += FUSED_QUANT_WARPS) {
        int base_k = block_id * HADAMARD_BLOCK_SIZE;

        int64_t row_offset = static_cast<int64_t>(row) * K;

        const float* tile_input = input + row_offset + base_k;

        int8_t* tile_output = output_i8 + row_offset + base_k;

        float x[VALUES_PER_THREAD];

        hadamard_256(tile_input, x, lane);

        int output_offset = lane * VALUES_PER_THREAD;

#pragma unroll
        for (int i = 0; i < VALUES_PER_THREAD; ++i) {
            float q = nearbyintf(x[i] / row_scale);

            q = fminf(fmaxf(q, -128.0f), 127.0f);

            tile_output[output_offset + i] = static_cast<int8_t>(q);
        }
    }
}


/*
Wrapper function that applies Hadamard rotation and quantization to INT8.
- Pass 1: hadamard_compute_row_scale_kernel
- Pass 2: hadamard_quantize_int8_kernel
*/
void fusion_hadamard_quantize_fp32_to_int8_cuda(
    const float* input,
    int8_t* output_i8,
    float* output_scales,
    int rows,
    int K,
    cudaStream_t stream) {
    if (K % HADAMARD_BLOCK_SIZE != 0) {
        fprintf(
            stderr, "fusion_hadamard_quantize_fp32_to_int8_cuda: "
            "K must be divisible by %d\n", HADAMARD_BLOCK_SIZE);
        return;
    }

    // --------------------------------------------------------
    // Pass 1: Hadamard + find row-wise quantization scale
    // --------------------------------------------------------

    hadamard_compute_row_scale_kernel<<<rows, FUSED_QUANT_THREADS, 0, stream>>>(
                input, output_scales, rows, K);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(
            stderr, "CUDA error in hadamard_compute_row_scale_kernel: %s\n",
            cudaGetErrorString(err));
        return;
    }

    // --------------------------------------------------------
    // Pass 2: Hadamard + quantize directly to INT8
    // --------------------------------------------------------

    hadamard_quantize_int8_kernel<<<rows, FUSED_QUANT_THREADS, 0, stream>>>(
        input, output_i8, output_scales, rows, K);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,
            "CUDA error in hadamard_quantize_int8_kernel: %s\n",
            cudaGetErrorString(err));
    }
}


// // ============================================================
// // Q8_0 -> FP32
// // ============================================================
// __global__ void dequantize_q8_0_to_fp32_kernel(
//     const block_q8_0_simple* __restrict__ input_q8_0,
//     float* __restrict__ output_fp32,
//     int rows,
//     int cols) {
//     int row = blockIdx.x;
//     int tid = threadIdx.x;

//     if (row >= rows)
//         return;

//     // Q8_0 requires cols to be multiple of QK8_0.
//     int blocks_per_row = cols / QK8_0;

//     const block_q8_0_simple* row_blocks = input_q8_0 + row * blocks_per_row;

//     float* row_output = output_fp32 + row * cols;

//     // ------------------------------------------------------------
//     // Dequantize Q8_0 blocks
//     //
//     // real_value = block_scale * q_value
//     // ------------------------------------------------------------
//     for (int b = tid; b < blocks_per_row; b += blockDim.x) {
//         const block_q8_0_simple& blk = row_blocks[b];

//         float d = q8_0_scale_to_float(blk.d);

//         int base_col = b * QK8_0;

// #pragma unroll
//         for (int i = 0; i < QK8_0; ++i) {
//             row_output[base_col + i] =
//                 d * static_cast<float>(blk.qs[i]);
//         }
//     }
// }

// void dequantize_q8_0_to_fp32_cuda(
//     const void* input_q8_0,
//     float* output_fp32,
//     int rows,
//     int cols,
//     cudaStream_t stream) {
//     if (cols % QK8_0 != 0) {
//         fprintf(stderr,
//                 "dequantize_q8_0_to_fp32_cuda: cols must be multiple of QK8_0\n");
//         return;
//     }

//     const int threads = 256;
//     const int blocks  = rows;

//     dequantize_q8_0_to_fp32_kernel<<<blocks, threads, 0, stream>>>(
//         reinterpret_cast<const block_q8_0_simple*>(input_q8_0),
//         output_fp32,
//         rows,
//         cols);

//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         fprintf(stderr, "CUDA error in dequantize_q8_0_to_fp32_cuda: %s\n", cudaGetErrorString(err));
//     }
// }


// ============================================================
// Fused Q8_0 dequantization + Hadamard rotation
// ============================================================

__global__ void dequantize_q8_0_hadamard_kernel(
    const block_q8_0_simple* __restrict__ input_q8_0,
    float* __restrict__ output,
    int rows,
    int K) {
    int row      = blockIdx.x;
    int block_id = blockIdx.y;
    int lane     = threadIdx.x;

    if (row >= rows) {
        return;
    }

    constexpr float scale = 1.0f / 16.0f;

    // Each Hadamard block contains 256 values.
    // Each Q8_0 block contains 32 values.
    // Therefore one Hadamard block contains 8 Q8_0 blocks.
    constexpr int Q8_BLOCKS_PER_HADAMARD =
        HADAMARD_BLOCK_SIZE / QK8_0;

    const int q8_blocks_per_row = K / QK8_0;

    // First Q8_0 block belonging to this H256 tile.
    const block_q8_0_simple* tile_blocks =
        input_q8_0 + static_cast<int64_t>(row) * q8_blocks_per_row + block_id * Q8_BLOCKS_PER_HADAMARD;

    // ------------------------------------------------------------
    // Step 1: Q8_0 -> FP32 directly into registers
    //
    // Each Q8_0 block contains 32 values.
    // Four warp lanes consume one Q8_0 block:
    //
    // lane 0 -> q[0..7]
    // lane 1 -> q[8..15]
    // lane 2 -> q[16..23]
    // lane 3 -> q[24..31]
    //
    // lane 4 -> next Q8_0 block, etc.
    // ------------------------------------------------------------

    int q8_block_id = lane >> 2;
    int q8_offset   = (lane & 3) * VALUES_PER_THREAD;

    const block_q8_0_simple& q8_block =
        tile_blocks[q8_block_id];

    float d = q8_0_scale_to_float(q8_block.d);

    float x[VALUES_PER_THREAD];

#pragma unroll
    for (int i = 0; i < VALUES_PER_THREAD; ++i) {
        x[i] =
            d * static_cast<float>(
                    q8_block.qs[q8_offset + i]);
    }

    // ------------------------------------------------------------
    // Step 2: local H8
    // ------------------------------------------------------------

    hadamard_8(x);

    // ------------------------------------------------------------
    // Step 3: warp-level H32
    //
    // H256 = H32 x H8
    // ------------------------------------------------------------

#pragma unroll
    for (int stride = 1; stride < 32; stride <<= 1) {
        float sign = (lane & stride) ? -1.0f : 1.0f;

#pragma unroll
        for (int i = 0; i < VALUES_PER_THREAD; ++i) {
            float other =
                __shfl_xor_sync(
                    0xffffffff,
                    x[i],
                    stride);

            x[i] = sign * x[i] + other;
        }
    }

    // ------------------------------------------------------------
    // Step 4: normalize and write rotated FP32
    // ------------------------------------------------------------

    int base_k = block_id * HADAMARD_BLOCK_SIZE;

    int64_t output_offset =
        static_cast<int64_t>(row) * K + base_k + lane * VALUES_PER_THREAD;

#pragma unroll
    for (int i = 0; i < VALUES_PER_THREAD; ++i) {
        output[output_offset + i] =
            x[i] * scale;
    }
}

void fusion_dequantize_q8_and_hadamard_cuda(
    const void* input_q8_0,
    float* output,
    int rows,
    int K,
    cudaStream_t stream) {
    if (K % HADAMARD_BLOCK_SIZE != 0) {
        fprintf(
            stderr,
            "fusion_dequantize_q8_and_hadamard_cuda: "
            "K must be divisible by %d\n",
            HADAMARD_BLOCK_SIZE);
        return;
    }

    if (K % QK8_0 != 0) {
        fprintf(
            stderr,
            "fusion_dequantize_q8_and_hadamard_cuda: "
            "K must be divisible by QK8_0\n");
        return;
    }

    dim3 grid(
        rows,
        K / HADAMARD_BLOCK_SIZE);

    dim3 block(HADAMARD_THREADS);

    dequantize_q8_0_hadamard_kernel<<<
        grid,
        block,
        0,
        stream>>>(
        reinterpret_cast<const block_q8_0_simple*>(
            input_q8_0),
        output,
        rows,
        K);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in fusion_dequantize_q8_and_hadamard_cuda: %s\n",
                cudaGetErrorString(err));
    }
}
