// Custom INT8 helper kernels — implementation.
// Compiled into the ggml-cuda library via custom_kernels/CMakeLists.txt.

#include "int8_helpers.cuh"

#include "ggml.h"
#include "ggml-cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_fp16.h>

// Q8_0 block layout constant
#define QK8_0 32

// ----------------------------------------------------------------------------
// Cache shared by amir_v1 / amir_v2 / future cached variants.
// Single definition here; declared extern in the header.
// ----------------------------------------------------------------------------
std::unordered_map<const void *, cached_int8_weight> g_amir_int8_weight_cache;

// ----------------------------------------------------------------------------
// Debug helper
// ----------------------------------------------------------------------------
void print_ggml_tensor_info(const struct ggml_tensor * t, const char * name) {
    if (t == NULL) {
        printf("%s: NULL tensor\n", name);
        return;
    }

    printf("Tensor %s\n", name);
    printf("  type: %s\n", ggml_type_name(t->type));
    printf("  ne: [%lld, %lld, %lld, %lld]\n",
           (long long)t->ne[0], (long long)t->ne[1],
           (long long)t->ne[2], (long long)t->ne[3]);
    printf("  nb: [%zu, %zu, %zu, %zu]\n",
           t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
    printf("\n");
}

// ----------------------------------------------------------------------------
// Row-wise FP32 -> INT8 quantization
// ----------------------------------------------------------------------------
__global__ void quantize_fp32_to_int8_row_wise_kernel(
    const float * input, int8_t * output, float * scales,
    int rows, int cols) {

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    extern __shared__ float sdata[];

    const float * row_input  = input  + row * cols;
    int8_t      * row_output = output + row * cols;

    // Step 1: max absolute value in this row (reduction)
    float local_max = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(row_input[col]));
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }

    float max_abs = sdata[0];
    float scale   = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    if (tid == 0) scales[row] = scale;
    __syncthreads();

    // Step 2: quantize
    for (int col = tid; col < cols; col += blockDim.x) {
        float q = nearbyintf(row_input[col] / scale);
        q = fminf(fmaxf(q, -128.0f), 127.0f);
        row_output[col] = static_cast<int8_t>(q);
    }
}

void quantize_fp32_to_int8_row_wise_cuda(
    const float * input, int8_t * output, float * scales,
    int rows, int cols, cudaStream_t stream) {

    const int threads = 512;
    const int blocks  = rows;
    const size_t shared_mem = threads * sizeof(float);

    quantize_fp32_to_int8_row_wise_kernel<<<blocks, threads, shared_mem, stream>>>(
        input, output, scales, rows, cols);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in quantize_fp32_to_int8_row_wise_cuda: %s\n",
                cudaGetErrorString(err));
    }
}

// ----------------------------------------------------------------------------
// Column-wise FP32 -> INT8 quantization
// ----------------------------------------------------------------------------
__global__ void quantize_fp32_to_int8_col_wise_kernel(
    const float * input, int8_t * output, float * scales,
    int rows, int cols) {

    int col = blockIdx.x;
    int tid = threadIdx.x;
    if (col >= cols) return;

    extern __shared__ float sdata[];

    float local_max = 0.0f;
    for (int row = tid; row < rows; row += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(input[row + col * rows]));
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }

    float max_abs = sdata[0];
    float scale   = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    if (tid == 0) scales[col] = scale;
    __syncthreads();

    for (int row = tid; row < rows; row += blockDim.x) {
        float q = nearbyintf(input[row + col * rows] / scale);
        q = fminf(fmaxf(q, -127.0f), 127.0f);
        output[row + col * rows] = static_cast<int8_t>(q);
    }
}

void quantize_fp32_to_int8_col_wise_cuda(
    const float * input, int8_t * output, float * scales,
    int rows, int cols, cudaStream_t stream) {

    const int threads = 512;
    const int blocks  = cols;
    const size_t shared_mem = threads * sizeof(float);

    quantize_fp32_to_int8_col_wise_kernel<<<blocks, threads, shared_mem, stream>>>(
        input, output, scales, rows, cols);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in quantize_fp32_to_int8_col_wise_cuda: %s\n",
                cudaGetErrorString(err));
    }
}

// ----------------------------------------------------------------------------
// INT32 -> FP32 dequantization (original, uncoalesced)
// ----------------------------------------------------------------------------
__global__ void dequantize_i32_to_f32_kernel(
    const int32_t * input_i32, float * output_f32,
    const float * row_scales, const float * col_scales,
    int rows, int cols, int ldc) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= cols) return;

    int idx = row + col * ldc;
    output_f32[idx] = (float) input_i32[idx] * (row_scales[row] * col_scales[col]);
}

void dequantize_i32_to_f32_cuda(
    const int32_t * input_i32, float * output_f32,
    const float * row_scales, const float * col_scales,
    int rows, int cols, int ldc, cudaStream_t stream) {

    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);

    dequantize_i32_to_f32_kernel<<<grid, block, 0, stream>>>(
        input_i32, output_f32, row_scales, col_scales, rows, cols, ldc);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in dequantize_i32_to_f32_cuda: %s\n",
                cudaGetErrorString(err));
    }
}

// ----------------------------------------------------------------------------
// amir_v2: coalesced INT32 -> FP32 dequantization
//
// The original kernel maps threadIdx.x -> column, so within a warp the access
// stride is ldc (uncoalesced). Here we use a 1D grid-stride loop over the
// logical [rows x cols] region in column-major order: consecutive threads map
// to consecutive (row, then col) elements -> coalesced loads/stores.
// ----------------------------------------------------------------------------
__global__ void dequantize_i32_to_f32_coalesced_kernel(
    const int32_t * __restrict__ input_i32,
    float * __restrict__ output_f32,
    const float * __restrict__ row_scales,
    const float * __restrict__ col_scales,
    int rows, int cols, int ldc) {

    const long total = (long) rows * (long) cols;

    for (long t = (long) blockIdx.x * blockDim.x + threadIdx.x;
         t < total;
         t += (long) gridDim.x * blockDim.x) {

        int row = (int) (t % rows);
        int col = (int) (t / rows);

        long idx = (long) row + (long) col * (long) ldc;

        output_f32[idx] = (float) input_i32[idx] * (row_scales[row] * col_scales[col]);
    }
}

void dequantize_i32_to_f32_coalesced_cuda(
    const int32_t * input_i32, float * output_f32,
    const float * row_scales, const float * col_scales,
    int rows, int cols, int ldc, cudaStream_t stream) {

    const long total = (long) rows * (long) cols;
    const int threads = 256;
    long want_blocks = (total + threads - 1) / threads;
    int blocks = (int) (want_blocks > 65535 ? 65535 : want_blocks);
    if (blocks < 1) blocks = 1;

    dequantize_i32_to_f32_coalesced_kernel<<<blocks, threads, 0, stream>>>(
        input_i32, output_f32, row_scales, col_scales, rows, cols, ldc);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in dequantize_i32_to_f32_coalesced_cuda: %s\n",
                cudaGetErrorString(err));
    }
}

// ----------------------------------------------------------------------------
// Q8_0 -> INT8 + row scales (used to derive INT8 weights from ggml Q8_0 weights)
// ----------------------------------------------------------------------------
struct block_q8_0_simple {
    uint16_t d;
    int8_t   qs[QK8_0];
};

__device__ __forceinline__ float q8_0_scale_to_float(uint16_t h) {
    return __half2float(*reinterpret_cast<const __half *>(&h));
}

__global__ void convert_q8_0_to_int8_row_wise_kernel(
    const block_q8_0_simple * __restrict__ input_q8_0,
    int8_t * __restrict__ output_i8,
    float  * __restrict__ output_scales,
    int rows, int cols) {

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    extern __shared__ float sdata[];

    int blocks_per_row = cols / QK8_0;
    const block_q8_0_simple * row_blocks = input_q8_0 + row * blocks_per_row;
    int8_t * row_output = output_i8 + row * cols;

    // Step 1: max abs real value in this row (real = block_scale * q)
    float local_max = 0.0f;
    for (int b = tid; b < blocks_per_row; b += blockDim.x) {
        const block_q8_0_simple & blk = row_blocks[b];
        float d = q8_0_scale_to_float(blk.d);
        #pragma unroll
        for (int i = 0; i < QK8_0; ++i) {
            local_max = fmaxf(local_max, fabsf(d * (float) blk.qs[i]));
        }
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }

    float max_abs   = sdata[0];
    float row_scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    if (tid == 0) output_scales[row] = row_scale;
    __syncthreads();

    // Step 2: re-quantize each element with the new row_scale
    for (int b = tid; b < blocks_per_row; b += blockDim.x) {
        const block_q8_0_simple & blk = row_blocks[b];
        float d = q8_0_scale_to_float(blk.d);
        int base_col = b * QK8_0;
        #pragma unroll
        for (int i = 0; i < QK8_0; ++i) {
            float real_value = d * (float) blk.qs[i];
            float q = nearbyintf(real_value / row_scale);
            q = fminf(fmaxf(q, -128.0f), 127.0f);
            row_output[base_col + i] = static_cast<int8_t>(q);
        }
    }
}

void convert_q8_0_to_int8_row_wise_cuda(
    const void * input_q8_0, int8_t * output_i8, float * output_scales,
    int rows, int cols, cudaStream_t stream) {

    if (cols % QK8_0 != 0) {
        fprintf(stderr,
                "convert_q8_0_to_int8_row_wise_cuda: cols must be multiple of QK8_0\n");
        return;
    }

    const int threads = 256;
    const int blocks  = rows;
    const size_t shared_mem = threads * sizeof(float);

    convert_q8_0_to_int8_row_wise_kernel<<<blocks, threads, shared_mem, stream>>>(
        reinterpret_cast<const block_q8_0_simple *>(input_q8_0),
        output_i8, output_scales, rows, cols);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in convert_q8_0_to_int8_row_wise_cuda: %s\n",
                cudaGetErrorString(err));
    }
}


#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"

/*
    Fused INT8 GEMM + row/column scaling + FP32 output.

    Input:
        A        : int8  [M, K_gemm], row-major
        B        : int8  [N_gemm, K_gemm], row-major physical memory
        alphaRow : float [M]
        alphaCol : float [N_gemm]

    Output:
        D        : float [M, N_gemm], column-major physical memory

    Logical math:
        D = A @ B^T

        A      = [M, K_gemm]
        B      = [N_gemm, K_gemm]
        B^T    = [K_gemm, N_gemm]
        D      = [M, N_gemm]

    Epilogue:
        D[m, n] = float(acc_i32[m, n]) * alphaRow[m] * alphaCol[n]

    Physical output layout:
        D[m, n] is stored at:

            D[m + n * ldc]

    This matches your working INT8 -> INT32 CUTLASS path and your old
    dequantization kernel.
*/
template <typename TileShape, typename WarpShape, int kStages>
bool matmul_w8a8_cutlass_f32_ptr(
    const int8_t* A,        // [M, K_gemm], row-major
    const int8_t* B,        // [N_gemm, K_gemm], row-major physical memory
    const float* alphaRow,  // [M]
    const float* alphaCol,  // [N_gemm]
    float* D,               // [M, N_gemm], column-major physical memory
    int32_t M,
    int32_t N_gemm,
    int32_t K_gemm,
    int32_t ldc,
    cudaStream_t stream
) {
    if (!A || !B || !alphaRow || !alphaCol || !D) {
        fprintf(stderr, "matmul_w8a8_cutlass_f32_ptr: null pointer input\n");
        return false;
    }

    if (M <= 0 || N_gemm <= 0 || K_gemm <= 0) {
        fprintf(stderr, "matmul_w8a8_cutlass_f32_ptr: invalid shape\n");
        return false;
    }

    if (ldc < M) {
        fprintf(stderr,
                "matmul_w8a8_cutlass_f32_ptr: invalid ldc. ldc=%d, M=%d\n",
                ldc, M);
        return false;
    }

    if (K_gemm % 32 != 0) {
        fprintf(stderr,
                "matmul_w8a8_cutlass_f32_ptr: K_gemm must be multiple of 32. K_gemm=%d\n",
                K_gemm);
        return false;
    }

    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementScale = float;
    using ElementC = float;
    using ElementOutput = float;
    using ElementAccumulator = int32_t;
    using ElementCompute = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // 16 int8
    constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;  // 16 int8

    // Safe scalar FP32 output store
    constexpr int AlignmentC = 1;

    constexpr int EVTEpilogueStages = 1;

    using namespace cute;

    using OutputTileThreadMap =
        cutlass::epilogue::threadblock::OutputTileThreadLayout<
            TileShape,
            WarpShape,
            ElementC,
            AlignmentC,
            EVTEpilogueStages>;

    using Accum =
        cutlass::epilogue::threadblock::VisitorAccFetch;

    /*
        alphaRow[m]

        This broadcasts one scale per output row.
    */
    using RowScaleBroadcast =
        cutlass::epilogue::threadblock::VisitorColBroadcast<
            OutputTileThreadMap,
            ElementScale,
            cute::Stride<_1, _0, int32_t>>;

    /*
        alphaCol[n]

        This broadcasts one scale per output column.
    */
    using ColScaleBroadcast =
        cutlass::epilogue::threadblock::VisitorRowBroadcast<
            OutputTileThreadMap,
            ElementScale,
            cute::Stride<_0, _1, int32_t>>;

    /*
        First multiply:

            acc * alphaRow[m]
    */
    using ComputeRowScale =
        cutlass::epilogue::threadblock::VisitorCompute<
            cutlass::multiplies,
            ElementCompute,
            ElementCompute,
            cutlass::FloatRoundStyle::round_to_nearest>;

    using EVTRowScale =
        cutlass::epilogue::threadblock::Sm80EVT<
            ComputeRowScale,
            Accum,
            RowScaleBroadcast>;

    /*
        Second multiply:

            (acc * alphaRow[m]) * alphaCol[n]
    */
    using ComputeColScale =
        cutlass::epilogue::threadblock::VisitorCompute<
            cutlass::multiplies,
            ElementCompute,
            ElementCompute,
            cutlass::FloatRoundStyle::round_to_nearest>;

    using EVTRowColScale =
        cutlass::epilogue::threadblock::Sm80EVT<
            ComputeColScale,
            EVTRowScale,
            ColScaleBroadcast>;

    /*
        Store FP32 output in column-major physical layout:

            D[m, n] -> D[m + n * ldc]

        Therefore stride is:

            stride_m     = 1
            stride_n     = ldc
            stride_batch = ldc * N_gemm
    */
    using StoreD =
        cutlass::epilogue::threadblock::VisitorAuxStore<
            OutputTileThreadMap,
            ElementOutput,
            cutlass::FloatRoundStyle::round_to_nearest,
            cute::Stride<_1, int64_t, int64_t>>;

    using EVTD =
        cutlass::epilogue::threadblock::Sm80EVT<
            StoreD,
            EVTRowColScale>;

    using Kernel =
        typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
            ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignmentA,
            ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignmentB,
            ElementC, LayoutC, AlignmentC,
            ElementAccumulator,
            ElementCompute,
            cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm80,
            TileShape,
            WarpShape,
            cutlass::gemm::GemmShape<16, 8, 32>,
            EVTD,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
            kStages,
            cutlass::arch::OpMultiplyAddSaturate,
            EVTEpilogueStages
        >::GemmKernel;

    using DeviceGemm =
        cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

    typename EVTD::Arguments callback_args{
        {
            {
                {},
                {alphaRow, ElementScale(0), {_1{}, _0{}, int32_t(M)}},
                {}
            },
            {alphaCol, ElementScale(0), {_0{}, _1{}, int32_t(N_gemm)}},
            {}
        },
        {
            D,
            {_1{}, int64_t{ldc}, int64_t{ldc} * int64_t{N_gemm}}
        }
    };

    typename DeviceGemm::Arguments arguments(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N_gemm, K_gemm},
        1,
        callback_args,

        A,
        B,
        nullptr,
        nullptr,

        int64_t(M) * int64_t(K_gemm),
        int64_t(N_gemm) * int64_t(K_gemm),
        0,
        0,

        int64_t(K_gemm),  // lda: A row-major [M, K_gemm]
        int64_t(K_gemm),  // ldb: B viewed as column-major [K_gemm, N_gemm]
        0,
        0
    );

    DeviceGemm gemm_op;

    size_t workspace_size = DeviceGemm::get_workspace_size(arguments);

    void* workspace = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    }

    cutlass::Status status;

    status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        if (workspace) cudaFree(workspace);
        CUTLASS_CHECK(status);
    }

    status = gemm_op.initialize(arguments, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        if (workspace) cudaFree(workspace);
        CUTLASS_CHECK(status);
    }

    status = gemm_op(stream);

    /*
        Debugging helper.

        You can remove this after the kernel is stable.
        This helps catch CUTLASS errors immediately instead of later in NORM.
    */
#if 0
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUTLASS launch error: %s\n", cudaGetErrorString(err));
        if (workspace) cudaFree(workspace);
        return false;
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUTLASS runtime error: %s\n", cudaGetErrorString(err));
        if (workspace) cudaFree(workspace);
        return false;
    }
#endif

    if (workspace) {
        cudaFree(workspace);
    }

    CUTLASS_CHECK(status);

    return true;
}


bool matmul_w8a8_cutlass_cuda(
    const int8_t* A,
    const int8_t* B,
    const float* alphaRow,
    const float* alphaCol,
    float* D,
    int M,
    int N_gemm,
    int K_gemm,
    int32_t ldc,
    cudaStream_t stream
) {
    using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    constexpr int kStages = 3;

    return matmul_w8a8_cutlass_f32_ptr<TileShape, WarpShape, kStages>(
        A,
        B,
        alphaRow,
        alphaCol,
        D,
        M,
        N_gemm,
        K_gemm,
        ldc,
        stream
    );
}