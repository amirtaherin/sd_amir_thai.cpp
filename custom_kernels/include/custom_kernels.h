#pragma once

// Public API of the custom_kernels library, for consumers outside ggml-cuda
// (e.g., src/stable-diffusion.cpp). Internal helpers stay in src/.

#include <map>
#include <string>

// Forward declaration -- we don't pull in ggml.h here.
struct ggml_tensor;

namespace custom_kernels {

// amir_v3 (CUSTOM_KERNEL_VERSION >= 4): pre-convert every Q8_0 tensor in
// `tensors` to INT8 + per-row scales and populate the cache used by amir_v2.
// Safe to call once after the diffusion model is loaded into the CUDA backend,
// before sampling starts -- eliminates the lazy step-1 cache-fill spike.
//
// Tensors with type != GGML_TYPE_Q8_0 or with null data are skipped.
// Returns the number of tensors converted. Synchronizes the CUDA stream
// before returning so callers can immediately enqueue matmuls.
int preload_q8_0_weights(const std::map<std::string, struct ggml_tensor *> & tensors);

}  // namespace custom_kernels
