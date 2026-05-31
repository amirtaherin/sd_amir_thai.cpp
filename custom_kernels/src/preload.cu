// =====================================================================================
// amir_v3 (exp018) -- load-time INT8 weight conversion.
//
// Walks the diffusion model's parameter tensors after they have been loaded
// into the CUDA backend and pre-converts every Q8_0 tensor to INT8 + per-row
// scales, populating the cache used by amir_v2's matmul dispatch.
//
// This removes the lazy "cache fills on step 1" spike that amir_v2 exhibits
// (4.67 s vs steady 2.98 s/step in exp017). All conversions are synchronized
// before this function returns, so subsequent matmuls find a warm cache.
//
// Runtime matmul behaviour is otherwise identical to amir_v2 -- amir_v3 is
// purely a "fill earlier" change.
// =====================================================================================

#include "common.cuh"
#include "int8_helpers.cuh"
#include "custom_kernels.h"

#include "ggml.h"

#include <cstdio>

namespace custom_kernels {

int preload_q8_0_weights(const std::map<std::string, struct ggml_tensor *> & tensors) {

    int converted   = 0;
    int already_in  = 0;
    int skipped     = 0;

    // Use the default stream: strong synchronization with anything launched
    // later on other streams. We also explicitly synchronize at the end before
    // returning, so subsequent matmuls reliably see populated cache entries.
    cudaStream_t stream = (cudaStream_t) 0;

    for (auto & kv : tensors) {
        const struct ggml_tensor * t = kv.second;
        if (t == nullptr || t->data == nullptr) {
            continue;
        }
        if (t->type != GGML_TYPE_Q8_0) {
            skipped++;
            continue;
        }

        // Skip if amir_v2's lazy path already populated this entry.
        if (g_amir_int8_weight_cache.find((const void *) t->data) !=
            g_amir_int8_weight_cache.end()) {
            already_in++;
            continue;
        }

        const int64_t M = t->ne[1];  // rows (output features)
        const int64_t K = t->ne[0];  // cols (input  features)

        cached_int8_weight cw;
        CUDA_CHECK(cudaMalloc((void **) &cw.d_i8,     sizeof(int8_t) * M * K));
        CUDA_CHECK(cudaMalloc((void **) &cw.d_scales, sizeof(float)  * M));

        convert_q8_0_to_int8_row_wise_cuda(
            t->data, cw.d_i8, cw.d_scales, (int) M, (int) K, stream);

        g_amir_int8_weight_cache.emplace((const void *) t->data, cw);
        converted++;
    }

    // Make sure all conversions are visible to subsequent kernels on any stream.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    fprintf(stderr,
            "[custom_kernels] amir_v3 preload: converted=%d, already_cached=%d, skipped_non_q8=%d, "
            "total_seen=%zu\n",
            converted, already_in, skipped, tensors.size());

    return converted;
}

}  // namespace custom_kernels
