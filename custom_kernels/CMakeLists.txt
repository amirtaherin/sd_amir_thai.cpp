# custom_kernels — out-of-tree INT8 matmul variants for ggml-cuda.
#
# We inject our .cu files into the ggml-cuda library compilation instead of
# putting them inside the ggml/ submodule. This lets us version multiple
# kernel variants here (thai_vu / amir_v1 / amir_v2 / ...), each as its own
# committed file, without touching leejet/ggml.
#
# The ggml-cuda.cu dispatch hook is a small patch (ggml-dispatch.patch)
# applied by apply_patch.sh after `git submodule update`.

# CUDA backend required.
if(NOT GGML_CUDA AND NOT SD_CUDA)
    return()
endif()
if(NOT TARGET ggml-cuda)
    message(STATUS "custom_kernels: ggml-cuda target not found, skipping.")
    return()
endif()

# ----------------------------------------------------------------------------
# Which kernel variant to compile in.
#   1 = thai_vu  (per-call Q8_0 -> INT8 conversion; exp015)
#   2 = amir_v1  (cached weight conversion;           exp016)
#   3 = amir_v2  (cached + coalesced dequant;         exp017)  [default]
# ----------------------------------------------------------------------------
set(CUSTOM_KERNEL_VERSION "3" CACHE STRING
    "Custom INT8 kernel variant for ggml-cuda Q8_0 matmuls: 1=thai_vu, 2=amir_v1, 3=amir_v2")
set_property(CACHE CUSTOM_KERNEL_VERSION PROPERTY STRINGS "1" "2" "3")

# ----------------------------------------------------------------------------
# Sources — always compile all variants so swapping is just a recompile of the
# dispatch hook, not of every variant. Unused variants are dead code in the binary.
# ----------------------------------------------------------------------------
set(CUSTOM_KERNEL_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/src/int8_helpers.cu
    ${CMAKE_CURRENT_LIST_DIR}/src/thai_vu.cu
    ${CMAKE_CURRENT_LIST_DIR}/src/amir_v1.cu
    ${CMAKE_CURRENT_LIST_DIR}/src/amir_v2.cu
)

# Inject into the ggml-cuda library so our files compile with the same flags,
# include paths, and link dependencies (cudart, cublas, ...) as the rest of
# ggml-cuda. The ggml-cuda dispatch hook (added by ggml-dispatch.patch) then
# calls into the variant selected by CUSTOM_KERNEL_VERSION.
target_sources(ggml-cuda PRIVATE ${CUSTOM_KERNEL_SOURCES})

# Expose ggml-cuda's own source directory and our header directory on its
# include path, so our .cu files can `#include "common.cuh"`, `#include
# "int8_helpers.cuh"`, etc.
target_include_directories(ggml-cuda PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../ggml/src/ggml-cuda
    ${CMAKE_CURRENT_LIST_DIR}/src
)

# Tell the dispatch hook (in the patched ggml-cuda.cu) which variant is active.
target_compile_definitions(ggml-cuda PRIVATE CUSTOM_KERNEL_VERSION=${CUSTOM_KERNEL_VERSION})

message(STATUS "custom_kernels: variant=${CUSTOM_KERNEL_VERSION} "
               "(1=thai_vu, 2=amir_v1, 3=amir_v2)")
