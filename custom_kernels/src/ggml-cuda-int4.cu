#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "ggml-quants.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdlib>
#include <cstring>
#include <math.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/numeric_types.h"

#include "ggml-cuda-int4.cuh"


#define QK4_0 32

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                \
  do {                                                                  \
    cudaError_t status = call;                                          \
    if (status != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error: %s:%d: %s\n",                        \
              __FILE__, __LINE__, cudaGetErrorString(status));          \
      return false;                                                     \
    }                                                                   \
  } while (0)
#endif

#ifndef CUTLASS_CHECK
#define CUTLASS_CHECK(status)                                           \
  do {                                                                  \
    cutlass::Status s = status;                                         \
    if (s != cutlass::Status::kSuccess) {                               \
      fprintf(stderr, "CUTLASS error: %s:%d: status = %d\n",            \
              __FILE__, __LINE__, static_cast<int>(s));                 \
      return false;                                                     \
    }                                                                   \
  } while (0)
#endif


static __device__ __forceinline__ uint8_t pack_s4_pair(int lo, int hi) {
    // signed int4 range: [-8, 7]
    lo = max(-8, min(7, lo));
    hi = max(-8, min(7, hi));

    uint8_t ulo = static_cast<uint8_t>(lo) & 0x0f;
    uint8_t uhi = static_cast<uint8_t>(hi) & 0x0f;

    return ulo | (uhi << 4);
}

static __device__ __forceinline__ int unpack_ggml_q4_0_value(uint8_t byte, bool high) {
    int q = high ? ((byte >> 4) & 0x0f) : (byte & 0x0f);

    // GGML Q4_0 convention:
    // real = d * (q - 8)
    return q - 8;
}


static __device__ __forceinline__ float ggml_fp16_to_fp32_kernel(ggml_fp16_t x) {
    return __half2float(*reinterpret_cast<const __half *>(&x));
}


template<int BLOCK_SIZE>
__global__ void convert_q4_0_to_int4_row_wise_kernel(
    const char * __restrict__ src_q4,
    uint8_t * __restrict__ dst_i4_packed,
    float * __restrict__ row_scales,
    int rows,
    int K
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    constexpr int qk = QK4_0;
    const int blocks_per_row = K / qk;

    const block_q4_0 * blocks = reinterpret_cast<const block_q4_0 *>(src_q4);

    const block_q4_0 * row_blocks = blocks + row * blocks_per_row;

    __shared__ float smem[BLOCK_SIZE];

    float local_max = 0.0f;

    // First pass: find max abs value in reconstructed FP32 row.
    for (int k = threadIdx.x; k < K; k += BLOCK_SIZE) {
        const int block_id = k / qk;
        const int within   = k % qk;

        const block_q4_0 * blk = &row_blocks[block_id];

        // GGML Q4_0 stores 32 values in 16 bytes.
        // First 16 values are low nibbles.
        // Next 16 values are high nibbles.
        const int byte_id = within % 16;
        const bool high   = within >= 16;

        uint8_t byte = static_cast<uint8_t>(blk->qs[byte_id]);
        int q_signed = unpack_ggml_q4_0_value(byte, high);

        float d = ggml_fp16_to_fp32_kernel(blk->d);
        float x = d * static_cast<float>(q_signed);

        local_max = fmaxf(local_max, fabsf(x));
    }

    smem[threadIdx.x] = local_max;
    __syncthreads();

    // Block reduction for max.
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    float max_abs = smem[0];

    // Signed int4 has positive max 7.
    float scale = max_abs > 0.0f ? max_abs / 7.0f : 1.0f;

    if (threadIdx.x == 0) {
        row_scales[row] = scale;
    }

    __syncthreads();

    // Second pass: pack two signed INT4 values into one byte.
    const int packed_K = K / 2;

    for (int p = threadIdx.x; p < packed_K; p += BLOCK_SIZE) {
        int k0 = 2 * p;
        int k1 = k0 + 1;

        float x0;
        float x1;

        {
            const int block_id = k0 / qk;
            const int within   = k0 % qk;

            const block_q4_0 * blk = &row_blocks[block_id];

            const int byte_id = within % 16;
            const bool high   = within >= 16;

            uint8_t byte = static_cast<uint8_t>(blk->qs[byte_id]);
            int q_signed = unpack_ggml_q4_0_value(byte, high);

            float d = ggml_fp16_to_fp32_kernel(blk->d);
            x0 = d * static_cast<float>(q_signed);
        }

        {
            const int block_id = k1 / qk;
            const int within   = k1 % qk;

            const block_q4_0 * blk = &row_blocks[block_id];

            const int byte_id = within % 16;
            const bool high   = within >= 16;

            uint8_t byte = static_cast<uint8_t>(blk->qs[byte_id]);
            int q_signed = unpack_ggml_q4_0_value(byte, high);

            float d = ggml_fp16_to_fp32_kernel(blk->d);
            x1 = d * static_cast<float>(q_signed);
        }

        int q0 = __float2int_rn(x0 / scale);
        int q1 = __float2int_rn(x1 / scale);

        q0 = max(-8, min(7, q0));
        q1 = max(-8, min(7, q1));

        dst_i4_packed[row * packed_K + p] = pack_s4_pair(q0, q1);
    }
}


void convert_q4_0_to_int4_row_wise_cuda(
    const char * src_q4,
    uint8_t * dst_i4_packed,
    float * row_scales,
    int rows,
    int K,
    cudaStream_t stream
) {
    GGML_ASSERT(K % QK4_0 == 0);
    GGML_ASSERT(K % 2 == 0);

    constexpr int BLOCK_SIZE = 256;

    convert_q4_0_to_int4_row_wise_kernel<BLOCK_SIZE>
        <<<rows, BLOCK_SIZE, 0, stream>>>(
            src_q4,
            dst_i4_packed,
            row_scales,
            rows,
            K
        );
}

template<int BLOCK_SIZE>
__global__ void quantize_fp32_to_int4_row_wise_kernel(
    const float * __restrict__ src,
    uint8_t * __restrict__ dst_i4_packed,
    float * __restrict__ row_scales,
    int rows,
    int K
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    __shared__ float smem[BLOCK_SIZE];

    const float * src_row = src + row * K;

    float local_max = 0.0f;

    for (int k = threadIdx.x; k < K; k += BLOCK_SIZE) {
        local_max = fmaxf(local_max, fabsf(src_row[k]));
    }

    smem[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    float max_abs = smem[0];
    float scale = max_abs > 0.0f ? max_abs / 7.0f : 1.0f;

    if (threadIdx.x == 0) {
        row_scales[row] = scale;
    }

    __syncthreads();

    const int packed_K = K / 2;

    for (int p = threadIdx.x; p < packed_K; p += BLOCK_SIZE) {
        int k0 = 2 * p;
        int k1 = k0 + 1;

        float x0 = src_row[k0];
        float x1 = src_row[k1];

        int q0 = __float2int_rn(x0 / scale);
        int q1 = __float2int_rn(x1 / scale);

        q0 = max(-8, min(7, q0));
        q1 = max(-8, min(7, q1));

        dst_i4_packed[row * packed_K + p] = pack_s4_pair(q0, q1);
    }
}

void quantize_fp32_to_int4_row_wise_cuda(
    const float * src,
    uint8_t * dst_i4_packed,
    float * row_scales,
    int rows,
    int K,
    cudaStream_t stream
) {
    GGML_ASSERT(K % 2 == 0);

    constexpr int BLOCK_SIZE = 256;

    quantize_fp32_to_int4_row_wise_kernel<BLOCK_SIZE>
        <<<rows, BLOCK_SIZE, 0, stream>>>(
            src,
            dst_i4_packed,
            row_scales,
            rows,
            K
        );
}

bool int4_matmul_cutlass_cuda(
    const uint8_t * A_i4_packed,
    const uint8_t * B_i4_packed,
    int32_t * C_i32,
    int M,
    int N,
    int K,
    int ldc,
    cudaStream_t stream
) {
    using ElementA = cutlass::int4b_t;
    using ElementB = cutlass::int4b_t;
    using ElementC = int32_t;
    using ElementAccumulator = int32_t;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::ColumnMajor;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementA,
        LayoutA,
        ElementB,
        LayoutB,
        ElementC,
        LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,

        // Good starting point for INT4 on Ampere.
        cutlass::gemm::GemmShape<128, 128, 128>,
        cutlass::gemm::GemmShape<64, 64, 128>,
        cutlass::gemm::GemmShape<8, 8, 32>,

        cutlass::epilogue::thread::LinearCombination<
            ElementC,
            1,
            ElementAccumulator,
            ElementAccumulator>,

        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3
    >;

    Gemm gemm_op;

    typename Gemm::Arguments args(
        {M, N, K},
        {
            reinterpret_cast<const ElementA *>(A_i4_packed),
            K
        },
        {
            reinterpret_cast<const ElementB *>(B_i4_packed),
            K
        },
        {
            C_i32,
            ldc
        },
        {
            C_i32,
            ldc
        },
        {
            1,
            0
        }
    );

    cutlass::Status status = gemm_op.can_implement(args);

    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS INT4 GEMM cannot implement this problem. Status = %d\n",
                static_cast<int>(status));
        return false;
    }

    status = gemm_op.initialize(args, nullptr, stream);

    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS INT4 GEMM initialize failed. Status = %d\n",
                static_cast<int>(status));
        return false;
    }

    status = gemm_op(stream);

    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS INT4 GEMM launch failed. Status = %d\n",
                static_cast<int>(status));
        return false;
    }

    return true;
}


static __device__ __forceinline__ int q4_0_get_signed_value(uint8_t byte, int within_block) {
    /*
        GGML Q4_0 layout for one block of 32 values:

            qs[0] low  -> element 0
            qs[1] low  -> element 1
            ...
            qs[15] low -> element 15

            qs[0] high  -> element 16
            qs[1] high  -> element 17
            ...
            qs[15] high -> element 31

        stored unsigned q in [0, 15]
        signed value = q - 8
    */

    int q_u4;

    if (within_block < 16) {
        q_u4 = byte & 0x0f;
    } else {
        q_u4 = (byte >> 4) & 0x0f;
    }

    return q_u4 - 8;
}


template<int BLOCK_SIZE>
__global__ void convert_q4_0_to_int8_row_wise_kernel(
    const char * __restrict__ src_q4,
    int8_t * __restrict__ dst_i8,
    float * __restrict__ row_scales,
    int rows,
    int K
) {
    const int row = blockIdx.x;

    if (row >= rows) {
        return;
    }

    constexpr int qk = QK4_0;

    const int blocks_per_row = K / qk;

    const block_q4_0 * blocks =
        reinterpret_cast<const block_q4_0 *>(src_q4);

    const block_q4_0 * row_blocks =
        blocks + row * blocks_per_row;

    __shared__ float smem[BLOCK_SIZE];

    float local_max = 0.0f;

    // ------------------------------------------------------------
    // Pass 1: reconstruct Q4_0 values and find row max abs
    // ------------------------------------------------------------
    for (int k = threadIdx.x; k < K; k += BLOCK_SIZE) {
        const int block_id = k / qk;
        const int within   = k % qk;

        const block_q4_0 * blk = &row_blocks[block_id];

        const int byte_id = within % 16;
        const uint8_t byte = static_cast<uint8_t>(blk->qs[byte_id]);

        const int q_s4 = q4_0_get_signed_value(byte, within);

        const float d = ggml_fp16_to_fp32_kernel(blk->d);
        const float x = d * static_cast<float>(q_s4);

        local_max = fmaxf(local_max, fabsf(x));
    }

    smem[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const float max_abs = smem[0];

    // Symmetric INT8: [-127, 127]
    const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;

    if (threadIdx.x == 0) {
        row_scales[row] = scale;
    }

    __syncthreads();

    // ------------------------------------------------------------
    // Pass 2: reconstruct Q4_0 and requantize to INT8
    // ------------------------------------------------------------
    for (int k = threadIdx.x; k < K; k += BLOCK_SIZE) {
        const int block_id = k / qk;
        const int within   = k % qk;

        const block_q4_0 * blk = &row_blocks[block_id];

        const int byte_id = within % 16;
        const uint8_t byte = static_cast<uint8_t>(blk->qs[byte_id]);

        const int q_s4 = q4_0_get_signed_value(byte, within);

        const float d = ggml_fp16_to_fp32_kernel(blk->d);
        const float x = d * static_cast<float>(q_s4);

        int q_i8 = __float2int_rn(x / scale);
        q_i8 = max(-127, min(127, q_i8));

        dst_i8[row * K + k] = static_cast<int8_t>(q_i8);
    }
}

void convert_q4_0_to_int8_row_wise_cuda(
    const char * src_q4,
    int8_t * dst_i8,
    float * row_scales,
    int rows,
    int K,
    cudaStream_t stream
) {
    GGML_ASSERT(src_q4 != nullptr);
    GGML_ASSERT(dst_i8 != nullptr);
    GGML_ASSERT(row_scales != nullptr);
    GGML_ASSERT(K % QK4_0 == 0);

    constexpr int BLOCK_SIZE = 256;

    convert_q4_0_to_int8_row_wise_kernel<BLOCK_SIZE>
        <<<rows, BLOCK_SIZE, 0, stream>>>(
            src_q4,
            dst_i8,
            row_scales,
            rows,
            K
        );
}


static __device__ __forceinline__ int spin_sign(int k, int seed) {
    uint32_t x = static_cast<uint32_t>(k) ^ static_cast<uint32_t>(seed);

    x ^= x >> 17;
    x *= 0xed5ad4bbU;
    x ^= x >> 11;
    x *= 0xac4c1b51U;
    x ^= x >> 15;
    x *= 0x31848babU;
    x ^= x >> 14;

    return (x & 1U) ? 1 : -1;
}


template<int BLOCK_SIZE>
__global__ void dequantize_q4_0_to_f32_kernel(
    const char * __restrict__ src_q4,
    float * __restrict__ dst_f32,
    int rows,
    int K
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    constexpr int qk = QK4_0;
    const int blocks_per_row = K / qk;

    const block_q4_0 * blocks =
        reinterpret_cast<const block_q4_0 *>(src_q4);

    const block_q4_0 * row_blocks =
        blocks + row * blocks_per_row;

    for (int k = threadIdx.x; k < K; k += BLOCK_SIZE) {
        const int block_id = k / qk;
        const int within   = k % qk;
        const int byte_id  = within % 16;

        const block_q4_0 * blk = &row_blocks[block_id];

        uint8_t byte = static_cast<uint8_t>(blk->qs[byte_id]);
        int q_s4 = q4_0_get_signed_value(byte, within);

        float d = ggml_fp16_to_fp32_kernel(blk->d);

        dst_f32[row * K + k] = d * static_cast<float>(q_s4);
    }
}

void dequantize_q4_0_to_f32_cuda(
    const char * src_q4,
    float * dst_f32,
    int rows,
    int K,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;

    dequantize_q4_0_to_f32_kernel<BLOCK_SIZE>
        <<<rows, BLOCK_SIZE, 0, stream>>>(
            src_q4,
            dst_f32,
            rows,
            K
        );
}


template<int BLOCK_SIZE>
__global__ void quantize_f32_to_int4_row_wise_kernel(
    const float * __restrict__ src,
    uint8_t * __restrict__ dst_i4,
    float * __restrict__ row_scales,
    int rows,
    int K
) {
    const int row = blockIdx.x;

    if (row >= rows) {
        return;
    }

    const float * src_row = src + row * K;

    __shared__ float smem[BLOCK_SIZE];

    float local_max = 0.0f;

    for (int k = threadIdx.x; k < K; k += BLOCK_SIZE) {
        local_max = fmaxf(local_max, fabsf(src_row[k]));
    }

    smem[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] =
                fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const float max_abs = smem[0];

    // signed int4 positive max = 7
    const float scale = max_abs > 0.0f ? max_abs / 7.0f : 1.0f;

    if (threadIdx.x == 0) {
        row_scales[row] = scale;
    }

    __syncthreads();

    const int packed_K = K / 2;

    for (int p = threadIdx.x; p < packed_K; p += BLOCK_SIZE) {
        int k0 = 2 * p;
        int k1 = k0 + 1;

        int q0 = __float2int_rn(src_row[k0] / scale);
        int q1 = __float2int_rn(src_row[k1] / scale);

        q0 = max(-8, min(7, q0));
        q1 = max(-8, min(7, q1));

        dst_i4[row * packed_K + p] = pack_s4_pair(q0, q1);
    }
}

void quantize_f32_to_int4_row_wise_cuda(
    const float * src,
    uint8_t * dst_i4,
    float * row_scales,
    int rows,
    int K,
    cudaStream_t stream
) {
    GGML_ASSERT(K % 2 == 0);

    constexpr int BLOCK_SIZE = 256;

    quantize_f32_to_int4_row_wise_kernel<BLOCK_SIZE>
        <<<rows, BLOCK_SIZE, 0, stream>>>(
            src,
            dst_i4,
            row_scales,
            rows,
            K
        );
}

template<int BLOCK_SIZE, int BLOCK_H>
__global__ void block_fwht_rotate_rows_inplace_kernel(
    float * __restrict__ x,
    int rows,
    int K
) {
    const int row = blockIdx.x;
    const int block_h_id = blockIdx.y;

    if (row >= rows) {
        return;
    }

    const int base_k = block_h_id * BLOCK_H;

    if (base_k + BLOCK_H > K) {
        return;
    }

    __shared__ float smem[BLOCK_H];

    float * x_row = x + row * K;

    // ------------------------------------------------------------
    // Load one K-block
    // ------------------------------------------------------------
    for (int i = threadIdx.x; i < BLOCK_H; i += BLOCK_SIZE) {
        const int k = base_k + i;
        smem[i] = x_row[k];
    }

    __syncthreads();

    // ------------------------------------------------------------
    // In-place FWHT inside shared memory
    // ------------------------------------------------------------
    for (int len = 1; len < BLOCK_H; len <<= 1) {
        for (int i = threadIdx.x; i < BLOCK_H; i += BLOCK_SIZE) {
            const int j = i ^ len;

            if ((i & len) == 0) {
                float a = smem[i];
                float b = smem[j];

                smem[i] = a + b;
                smem[j] = a - b;
            }
        }

        __syncthreads();
    }

    const float norm = rsqrtf(static_cast<float>(BLOCK_H));

    // ------------------------------------------------------------
    // Normalize + store back
    // ------------------------------------------------------------
    for (int i = threadIdx.x; i < BLOCK_H; i += BLOCK_SIZE) {
        const int k = base_k + i;
        x_row[k] = smem[i] * norm;
    }
}

bool block_fwht_rotate_rows_inplace_cuda(
    float * x,
    int rows,
    int K,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int BLOCK_H    = 256;

    if (K % BLOCK_H != 0) {
        return false;
    }

    dim3 grid(rows, K / BLOCK_H);
    dim3 block(BLOCK_SIZE);

    block_fwht_rotate_rows_inplace_kernel<BLOCK_SIZE, BLOCK_H>
        <<<grid, block, 0, stream>>>(
            x,
            rows,
            K
        );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in block_fwht_rotate_rows_inplace_cuda: %s\n",
                cudaGetErrorString(err));
        return false;
    }

    return true;
}


template<int BLOCK_SIZE, int BLOCK_H>
__global__ void block_fwht_rotate_rows_kernel(
    const float * __restrict__ x_in,
    float * __restrict__ x_out,
    int rows,
    int K
) {
    const int row = blockIdx.x;
    const int block_h_id = blockIdx.y;

    if (row >= rows) {
        return;
    }

    const int base_k = block_h_id * BLOCK_H;

    if (base_k + BLOCK_H > K) {
        return;
    }

    __shared__ float smem[BLOCK_H];

    const float * x_in_row = x_in + row * K;
    float * x_out_row = x_out + row * K;

    // ------------------------------------------------------------
    // Load one K-block from input
    // ------------------------------------------------------------
    for (int i = threadIdx.x; i < BLOCK_H; i += BLOCK_SIZE) {
        const int k = base_k + i;
        smem[i] = x_in_row[k];
    }

    __syncthreads();

    // ------------------------------------------------------------
    // In-place FWHT inside shared memory
    // ------------------------------------------------------------
    for (int len = 1; len < BLOCK_H; len <<= 1) {
        for (int i = threadIdx.x; i < BLOCK_H; i += BLOCK_SIZE) {
            const int j = i ^ len;

            if ((i & len) == 0) {
                float a = smem[i];
                float b = smem[j];

                smem[i] = a + b;
                smem[j] = a - b;
            }
        }

        __syncthreads();
    }

    const float norm = rsqrtf(static_cast<float>(BLOCK_H));

    // ------------------------------------------------------------
    // Normalize + store to output
    // ------------------------------------------------------------
    for (int i = threadIdx.x; i < BLOCK_H; i += BLOCK_SIZE) {
        const int k = base_k + i;
        x_out_row[k] = smem[i] * norm;
    }
}

bool block_fwht_rotate_rows_cuda(
    const float * x_in,
    float * x_out,
    int rows,
    int K,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int BLOCK_H    = 256;

    if (x_in == x_out) {
        fprintf(stderr, "block_fwht_rotate_rows_cuda: x_in and x_out must be different pointers\n");
        return false;
    }

    if (K % BLOCK_H != 0) {
        return false;
    }

    dim3 grid(rows, K / BLOCK_H);
    dim3 block(BLOCK_SIZE);

    block_fwht_rotate_rows_kernel<BLOCK_SIZE, BLOCK_H>
        <<<grid, block, 0, stream>>>(
            x_in,
            x_out,
            rows,
            K
        );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in block_fwht_rotate_rows_out_cuda: %s\n",
                cudaGetErrorString(err));
        return false;
    }

    return true;
}


// =============================================================================
/*
Below is code to compute the incoherence score of a tensor on CUDA. Basically, we have two version.

Version 1: compute_incoherence_kernel and compute_incoherence_score_cuda. 
- This version is simple and straightforward.
- But, it is slow. 

Version 2: compute_incoherence_score_cuda_fast
- This version is more complex, but it is faster. It uses a two-pass approach to compute the incoherence score.
*/
// ==============================================================================

float get_quantization_incoherent_threshold() {
    // Default incoherent  is 10.0f
    const char * env = std::getenv("QUANTIZATION_INCOHERENT_THRESHOLD");
    if (env != nullptr) {
        try {
            float threshold = std::stof(env);
            return threshold;
        } catch (const std::exception & e) {    
            fprintf(stderr, "Warning: Invalid value for QUANTIZATION_INCOHERENT_THRESHOLD: %s\n", env);
        }
    } 
    return 10.0f;
}

template<int BLOCK_SIZE>
__global__ void compute_incoherence_kernel(
    const float * __restrict__ x,
    float * __restrict__ score_out,
    int64_t numel
) {
    const int tid = threadIdx.x;

    __shared__ float smem_max_abs[BLOCK_SIZE];
    __shared__ float smem_sum_sq[BLOCK_SIZE];

    float local_max_abs = 0.0f;
    float local_sum_sq  = 0.0f;

    // One CUDA block scans the whole src1.
    for (int64_t i = tid; i < numel; i += BLOCK_SIZE) {
        float v = x[i];
        float av = fabsf(v);

        local_max_abs = fmaxf(local_max_abs, av);
        local_sum_sq += v * v;
    }

    smem_max_abs[tid] = local_max_abs;
    smem_sum_sq[tid]  = local_sum_sq;

    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem_max_abs[tid] = fmaxf(smem_max_abs[tid], smem_max_abs[tid + s]);
            smem_sum_sq[tid] += smem_sum_sq[tid + s];
        }

        __syncthreads();
    }

    if (tid == 0) {
        float max_abs = smem_max_abs[0];
        float rms = sqrtf(smem_sum_sq[0] / (float) numel);
        float eps = 1.0e-12f;

        score_out[0] = max_abs / fmaxf(rms, eps);
    }
}


void compute_incoherence_score_cuda(
    const float * src1_ddf_i,
    float * score_device,
    int64_t N,
    int64_t K,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;

    const int64_t numel = N * K;

    compute_incoherence_kernel<BLOCK_SIZE>
        <<<1, BLOCK_SIZE, 0, stream>>>(
            src1_ddf_i,
            score_device,
            numel
        );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in compute_incoherence_score_cuda: %s\n",
                cudaGetErrorString(err));
        return;
    }
}


template<int BLOCK_SIZE>
__device__ __forceinline__ void block_reduce_max_sum(
    float & local_max_abs,
    float & local_sum_sq,
    float * smem_max_abs,
    float * smem_sum_sq
) {
    const int tid = threadIdx.x;

    smem_max_abs[tid] = local_max_abs;
    smem_sum_sq[tid]  = local_sum_sq;

    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem_max_abs[tid] = fmaxf(
                smem_max_abs[tid],
                smem_max_abs[tid + s]
            );

            smem_sum_sq[tid] += smem_sum_sq[tid + s];
        }

        __syncthreads();
    }

    local_max_abs = smem_max_abs[0];
    local_sum_sq  = smem_sum_sq[0];
}


template<int BLOCK_SIZE>
__global__ void compute_incoherence_partial_kernel(
    const float * __restrict__ x,
    float * __restrict__ partial_max_abs,
    float * __restrict__ partial_sum_sq,
    int64_t numel
) {
    const int tid = threadIdx.x;

    __shared__ float smem_max_abs[BLOCK_SIZE];
    __shared__ float smem_sum_sq[BLOCK_SIZE];

    float local_max_abs = 0.0f;
    float local_sum_sq  = 0.0f;

    const int64_t global_tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride     = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = global_tid; i < numel; i += stride) {
        float v = x[i];
        float av = fabsf(v);

        local_max_abs = fmaxf(local_max_abs, av);
        local_sum_sq += v * v;
    }

    block_reduce_max_sum<BLOCK_SIZE>(
        local_max_abs,
        local_sum_sq,
        smem_max_abs,
        smem_sum_sq
    );

    if (tid == 0) {
        partial_max_abs[blockIdx.x] = local_max_abs;
        partial_sum_sq[blockIdx.x]  = local_sum_sq;
    }
}

template<int BLOCK_SIZE>
__global__ void compute_incoherence_final_kernel(
    const float * __restrict__ partial_max_abs,
    const float * __restrict__ partial_sum_sq,
    float * __restrict__ score_out,
    int num_blocks,
    int64_t numel
) {
    const int tid = threadIdx.x;

    __shared__ float smem_max_abs[BLOCK_SIZE];
    __shared__ float smem_sum_sq[BLOCK_SIZE];

    float local_max_abs = 0.0f;
    float local_sum_sq  = 0.0f;

    for (int i = tid; i < num_blocks; i += BLOCK_SIZE) {
        local_max_abs = fmaxf(local_max_abs, partial_max_abs[i]);
        local_sum_sq += partial_sum_sq[i];
    }

    block_reduce_max_sum<BLOCK_SIZE>(
        local_max_abs,
        local_sum_sq,
        smem_max_abs,
        smem_sum_sq
    );

    if (tid == 0) {
        float max_abs = local_max_abs;
        float rms = sqrtf(local_sum_sq / (float) numel);
        float eps = 1.0e-12f;

        score_out[0] = max_abs / fmaxf(rms, eps);
    }
}

void compute_incoherence_score_cuda_fast(
    const float * src1_ddf_i,
    float * score_device,
    float * partial_max_abs,
    float * partial_sum_sq,
    int64_t N,
    int64_t K,
    int num_blocks,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;

    const int64_t numel = N * K;

    if (numel <= 0) {
        cudaMemsetAsync(score_device, 0, sizeof(float), stream);
        return;
    }

    compute_incoherence_partial_kernel<BLOCK_SIZE>
        <<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            src1_ddf_i,
            partial_max_abs,
            partial_sum_sq,
            numel
        );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in compute_incoherence_partial_kernel: %s\n",
                cudaGetErrorString(err));
        return;
    }

    compute_incoherence_final_kernel<BLOCK_SIZE>
        <<<1, BLOCK_SIZE, 0, stream>>>(
            partial_max_abs,
            partial_sum_sq,
            score_device,
            num_blocks,
            numel
        );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in compute_incoherence_final_kernel: %s\n",
                cudaGetErrorString(err));
        return;
    }
}