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
#   1 = thai_vu  (per-call Q8_0 -> INT8 conversion;            exp015)
#   2 = amir_v1  (cached weight conversion;                    exp016)
#   3 = amir_v2  (cached + coalesced dequant;                  exp017)
#   4 = amir_v3  (amir_v2 + load-time INT8 weight conversion;  exp018)  [default]
#   5 = amir_v4  (CUTLASS 2.x EVT fused INT8 GEMM + dequant epilogue, Sm90;
#                 implementation by Thai Vu — requires CUTLASS_DIR)
#   6 = amir_v5  (CUTLASS 3.x rewrite of amir_v4 targeting Sm100 / Blackwell
#                 collective-builder API — requires CUTLASS_DIR)
# ----------------------------------------------------------------------------
set(CUSTOM_KERNEL_VERSION "4" CACHE STRING
    "Custom INT8 kernel variant: 1=thai_vu, 2=amir_v1, 3=amir_v2, 4=amir_v3, 5=amir_v4, 6=amir_v5")
set_property(CACHE CUSTOM_KERNEL_VERSION PROPERTY STRINGS "1" "2" "3" "4" "5" "6")

# ----------------------------------------------------------------------------
# Sources — always compile all variants so swapping is just a recompile of the
# dispatch hook, not of every variant. Unused variants are dead code in the binary.
# preload.cu is the amir_v3-specific entry point called from stable-diffusion.cpp.
# amir_v4.cu holds Thai Vu's CUTLASS fused-epilogue implementation; its CUTLASS
# call is guarded by CUSTOM_KERNEL_HAVE_CUTLASS (set below when CUTLASS_DIR is
# provided). amir_v4_skeleton.cu (the original template) is intentionally NOT
# in this list — kept as documentation only.
# ----------------------------------------------------------------------------
set(CUSTOM_KERNEL_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/src/int8_helpers.cu
    ${CMAKE_CURRENT_LIST_DIR}/src/thai_vu.cu
    ${CMAKE_CURRENT_LIST_DIR}/src/amir_v1.cu
    ${CMAKE_CURRENT_LIST_DIR}/src/amir_v2.cu
    ${CMAKE_CURRENT_LIST_DIR}/src/amir_v4.cu
    ${CMAKE_CURRENT_LIST_DIR}/src/amir_v5.cu
    ${CMAKE_CURRENT_LIST_DIR}/src/preload.cu
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
    ${CMAKE_CURRENT_LIST_DIR}/include
)

# Tell the dispatch hook (in the patched ggml-cuda.cu) which variant is active.
target_compile_definitions(ggml-cuda PRIVATE CUSTOM_KERNEL_VERSION=${CUSTOM_KERNEL_VERSION})

# ----------------------------------------------------------------------------
# Q4 kernel family (thai_vu_q4, Q4_0 weights + FP32 activations).
#
#   0 = disabled (default) — Q4_0 weights fall through to stock mul_mat_q.
#   1 = thai_vu_q4 (SpinQuant-style FWHT rotation + CUTLASS INT4 GEMM, with
#       runtime fallback to INT8 GEMM when the activation incoherence score
#       exceeds a threshold — see src/int4_note.md). Requires CUTLASS_DIR.
#   2 = amir_q4_v1 (thai_vu_q4 with weight caching: rotated INT4 and plain INT8
#       derivatives of every Q4_0 weight are precomputed at model load and
#       reused for every matmul. Q4 analog of amir_v3 for Q8). Requires CUTLASS_DIR.
#
# This knob is independent of CUSTOM_KERNEL_VERSION: v=6 (amir_v5) still
# handles Q8_0 weights, Q4 sets which kernel handles Q4_0 weights.
#
# Compile with, e.g.:
#     cmake .. -DSD_CUDA=ON \
#              -DCUSTOM_KERNEL_VERSION=6 \
#              -DCUSTOM_Q4_KERNEL_VERSION=1 \
#              -DCUTLASS_DIR=/path/to/cutlass \
#              -DCMAKE_CUDA_ARCHITECTURES=110a-real     # Blackwell TMA (see amir_v5)
# ----------------------------------------------------------------------------
set(CUSTOM_Q4_KERNEL_VERSION "0" CACHE STRING
    "Q4_0 custom kernel: 0=disabled, 1=thai_vu_q4, 2=amir_q4_v1 (thai_vu_q4 + weight caching)")
set_property(CACHE CUSTOM_Q4_KERNEL_VERSION PROPERTY STRINGS "0" "1" "2")

if (CUSTOM_Q4_KERNEL_VERSION GREATER_EQUAL 1)
    set(_CUSTOM_Q4_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/ggml-cuda-int4.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/int4_library.cu
    )
    if (CUSTOM_Q4_KERNEL_VERSION GREATER_EQUAL 2)
        # amir_q4_v1: cached-weight matmul entry + load-time preload.
        list(APPEND _CUSTOM_Q4_SOURCES
            ${CMAKE_CURRENT_LIST_DIR}/src/amir_q4_v1.cu
            ${CMAKE_CURRENT_LIST_DIR}/src/preload_q4.cu
        )
    endif()
    list(APPEND CUSTOM_KERNEL_SOURCES ${_CUSTOM_Q4_SOURCES})
    target_sources(ggml-cuda PRIVATE ${_CUSTOM_Q4_SOURCES})
    target_compile_definitions(ggml-cuda PRIVATE
        CUSTOM_Q4_KERNEL_VERSION=${CUSTOM_Q4_KERNEL_VERSION})
endif()

# ----------------------------------------------------------------------------
# CUTLASS: needed by amir_v4 (CUSTOM_KERNEL_VERSION=5), amir_v5 (v=6), and
# thai_vu_q4 (CUSTOM_Q4_KERNEL_VERSION=1). Point at it with
#     -DCUTLASS_DIR=/path/to/cutlass-checkout
# (clone from https://github.com/NVIDIA/cutlass). When CUTLASS_DIR is provided
# we add its include dir to the ggml-cuda target and define
# CUSTOM_KERNEL_HAVE_CUTLASS, which gates all CUTLASS code paths.
# Q8 variants 1..4 and Q4 variant 0 build fine without CUTLASS.
# ----------------------------------------------------------------------------
set(CUTLASS_DIR "" CACHE PATH
    "Path to a CUTLASS checkout (https://github.com/NVIDIA/cutlass). Required for CUSTOM_KERNEL_VERSION>=5 or CUSTOM_Q4_KERNEL_VERSION>=1.")

if (CUTLASS_DIR AND EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h")
    target_include_directories(ggml-cuda PRIVATE
        ${CUTLASS_DIR}/include
        ${CUTLASS_DIR}/tools/util/include   # for cutlass::util::* helpers if used
    )
    target_compile_definitions(ggml-cuda PRIVATE CUSTOM_KERNEL_HAVE_CUTLASS=1)
    message(STATUS "custom_kernels: CUTLASS found at ${CUTLASS_DIR}")
elseif (CUSTOM_KERNEL_VERSION GREATER_EQUAL 5 OR CUSTOM_Q4_KERNEL_VERSION GREATER_EQUAL 1)
    message(FATAL_ERROR
        "custom_kernels: CUSTOM_KERNEL_VERSION=${CUSTOM_KERNEL_VERSION} and/or "
        "CUSTOM_Q4_KERNEL_VERSION=${CUSTOM_Q4_KERNEL_VERSION} requires CUTLASS.\n"
        "  Clone CUTLASS:  git clone --depth 1 https://github.com/NVIDIA/cutlass /path/to/cutlass\n"
        "  Reconfigure:    cmake .. -DSD_CUDA=ON -DCUTLASS_DIR=/path/to/cutlass ...\n"
        "  See custom_kernels/src/amir_v4_implementation.md for the full build flow.")
endif()

# Propagate to the stable-diffusion library too -- amir_v3 calls
# custom_kernels::preload_q8_0_weights from src/stable-diffusion.cpp, gated by
# the same macro. Skipped if the stable-diffusion target hasn't been declared
# yet (e.g., when this directory is consumed standalone).
if (TARGET stable-diffusion)
    target_compile_definitions(stable-diffusion PRIVATE CUSTOM_KERNEL_VERSION=${CUSTOM_KERNEL_VERSION})
    if (CUSTOM_Q4_KERNEL_VERSION GREATER_EQUAL 1)
        target_compile_definitions(stable-diffusion PRIVATE
            CUSTOM_Q4_KERNEL_VERSION=${CUSTOM_Q4_KERNEL_VERSION})
    endif()
    target_include_directories(stable-diffusion PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)
endif()

message(STATUS "custom_kernels: Q8 variant=${CUSTOM_KERNEL_VERSION} "
               "(1=thai_vu, 2=amir_v1, 3=amir_v2, 4=amir_v3, 5=amir_v4[CUTLASS 2.x Sm90], 6=amir_v5[CUTLASS 3.x Sm100])")
message(STATUS "custom_kernels: Q4 variant=${CUSTOM_Q4_KERNEL_VERSION} "
               "(0=disabled, 1=thai_vu_q4[SpinQuant+INT4], 2=amir_q4_v1[thai_vu_q4+cache])")
