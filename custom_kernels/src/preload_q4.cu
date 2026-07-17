// =====================================================================================
// amir_q4_v1 preload — pre-convert every Q4_0 diffusion weight at model load.
//
// Called from src/stable-diffusion.cpp after the diffusion model has been
// loaded into the CUDA backend, gated by CUSTOM_Q4_KERNEL_VERSION >= 2.
//
// For each Q4_0 weight tensor we produce and cache TWO derivatives:
//
//   1. Rotated INT4 (for the SpinQuant path):
//      Q4_0 -> FP32 -> in-place block-FWHT rotation -> INT4 packed + row scales
//
//   2. Plain INT8 (for the INT8 fallback path):
//      Q4_0 -> INT8 + row scales
//
// Both derivatives are needed because the runtime dispatch inside
// custom_ggml_q4_kernel_amir_v1 chooses one or the other per matmul call
// depending on activation incoherence. Caching both means the per-call cost
// is exactly the same as the equivalent Q8 amir_v3 -- no weight-side work.
//
// Memory cost per weight tensor (in addition to the original Q4_0 storage):
//   INT4 rotated : (M * K) / 2 bytes   ~= 0.5 * original
//   Row scales   : M * 4 bytes         (negligible)
//   INT8         : M * K bytes         ~= original size * 2 (Q4_0 is 0.5 B/elem)
//   Row scales   : M * 4 bytes         (negligible)
// Total: roughly 2.5x the raw Q4_0 weight footprint. On Thor with 128 GB
// unified memory and a ~19 GB Q4_0 diffusion model this is well within budget.
// =====================================================================================

#include "common.cuh"
#include "int8_helpers.cuh"   // cached_int8_weight (unused here) + shared symbols
#include "ggml-cuda-int4.cuh" // Q4 helpers + cached_q4_weight + g_amir_q4_weight_cache
#include "custom_kernels.h"

#include "ggml.h"

#include <cstdio>

namespace custom_kernels {

int preload_q4_0_weights(const std::map<std::string, struct ggml_tensor *> & tensors) {
    int converted   = 0;
    int already_in  = 0;
    int skipped     = 0;

    // Default stream. Sync at the end so subsequent matmuls see populated cache.
    cudaStream_t stream = (cudaStream_t) 0;

    for (auto & kv : tensors) {
        const struct ggml_tensor * t = kv.second;
        if (t == nullptr || t->data == nullptr) {
            continue;
        }
        if (t->type != GGML_TYPE_Q4_0) {
            skipped++;
            continue;
        }

        // Skip if already cached (e.g., called twice, or lazy fill happened first).
        if (g_amir_q4_weight_cache.find((const void *) t->data) !=
            g_amir_q4_weight_cache.end()) {
            already_in++;
            continue;
        }

        const int64_t M = t->ne[1];  // rows (output features)
        const int64_t K = t->ne[0];  // cols (input  features)

        // The SpinQuant block size is 256; if K isn't divisible by it, the
        // rotated INT4 derivative isn't usable. We still cache the plain INT8
        // derivative -- the runtime dispatch will always route to the INT8
        // path for this tensor.
        const bool can_rotate = (K % 256 == 0);

        cached_q4_weight cw;
        cw.M = (int32_t) M;
        cw.K = (int32_t) K;

        // --------------------------------------------------------------
        // 1. Rotated INT4 derivative (skipped if K not divisible by 256)
        // --------------------------------------------------------------
        if (can_rotate) {
            // Scratch FP32 buffer for dequant + rotation.
            float * d_fp32 = nullptr;
            CUDA_CHECK(cudaMalloc((void **) &d_fp32, sizeof(float) * M * K));

            // Q4_0 -> FP32
            dequantize_q4_0_to_f32_cuda(
                (const char *) t->data, d_fp32, (int) M, (int) K, stream);

            // In-place block FWHT rotation.
            bool ok = block_fwht_rotate_rows_inplace_cuda(
                d_fp32, (int) M, (int) K, stream);
            GGML_ASSERT(ok);

            // Rotated FP32 -> packed signed INT4 + row scales.
            CUDA_CHECK(cudaMalloc((void **) &cw.d_i4_rot,
                                  sizeof(uint8_t) * M * (K / 2)));
            CUDA_CHECK(cudaMalloc((void **) &cw.d_scales_rot,
                                  sizeof(float) * M));
            quantize_f32_to_int4_row_wise_cuda(
                d_fp32, cw.d_i4_rot, cw.d_scales_rot,
                (int) M, (int) K, stream);

            // Free the scratch buffer only after the stream has consumed it.
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaFree(d_fp32));
        }

        // --------------------------------------------------------------
        // 2. Plain INT8 derivative (always cached — used by the INT8
        //    fallback path when incoherence score > threshold)
        // --------------------------------------------------------------
        CUDA_CHECK(cudaMalloc((void **) &cw.d_i8,
                              sizeof(int8_t) * M * K));
        CUDA_CHECK(cudaMalloc((void **) &cw.d_scales_i8,
                              sizeof(float) * M));
        convert_q4_0_to_int8_row_wise_cuda(
            (const char *) t->data, cw.d_i8, cw.d_scales_i8,
            (int) M, (int) K, stream);

        g_amir_q4_weight_cache.emplace((const void *) t->data, cw);
        converted++;
    }

    // Ensure all conversions are visible to subsequent kernels on any stream.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    fprintf(stderr,
            "[custom_kernels] amir_q4_v1 preload: converted=%d, already_cached=%d, "
            "skipped_non_q4=%d, total_seen=%zu\n",
            converted, already_in, skipped, tensors.size());

    return converted;
}

}  // namespace custom_kernels
