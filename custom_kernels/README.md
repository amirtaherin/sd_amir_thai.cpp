# `custom_kernels/` — Out-of-Tree INT8 Matmul Variants

This directory holds custom CUDA kernel variants for `ggml-cuda` that replace
the stock **MMQ** path for **Q8_0 weights × F32 activations** with a
quantize → INT8 cuBLAS GEMM (`i8·i8 → s32`) → dequantize pipeline.

The point of putting them here, **outside the `ggml/` submodule**, is so that
we can version and commit them in `sd_amir_thai.cpp` without forking
`leejet/ggml`. The only thing that lives inside the ggml submodule is a tiny
dispatch hook (~65-line patch in `ggml-dispatch.patch`), applied to a clean
`a8db410` checkout by `apply_patch.sh`.

## Layout

```
custom_kernels/
├── CMakeLists.txt            injects sources into the ggml-cuda target
├── apply_patch.sh            applies the dispatch hook to ggml-cuda.cu (idempotent)
├── ggml-dispatch.patch       the dispatch hook (forward decls + #if-selected branch)
├── README.md                 (this file)
└── src/
    ├── int8_helpers.{cuh,cu} shared helpers: quant / dequant / Q8_0->INT8 / cache map
    ├── thai_vu.cu            v1 of the variants — per-call weight conversion (exp015)
    ├── amir_v1.cu            v2 of the variants — cached weight conversion   (exp016)
    └── amir_v2.cu            v3 of the variants — cached + coalesced dequant (exp017)
```

## Variants (current)

| Macro value | Name | Behaviour | Experiment |
|------------:|------|-----------|------------|
| `1` | `thai_vu` | Re-converts Q8_0 → INT8 weights on every matmul call. | `exp015` |
| `2` | `amir_v1` | Caches the Q8_0 → INT8 weight conversion per weight tensor. | `exp016` |
| `3` | `amir_v2` | `amir_v1` + memory-coalesced INT32 → FP32 dequant. **Beats cuBLAS.** | `exp017` |

Non-Q8_0 quantized weights (e.g. the Q4_K Mistral text encoder) fall back to
stock `ggml_cuda_mul_mat_q`, so a full sd-cli run works without `--clip-on-cpu`.

## Build

After `git submodule update --init ggml` (in the parent repo):

```bash
# One-time: apply the dispatch hook to the ggml submodule
./custom_kernels/apply_patch.sh

# Configure with CUDA + your chosen variant (default = 3 = amir_v2)
mkdir -p build && cd build
cmake .. -DSD_CUDA=ON -DCUSTOM_KERNEL_VERSION=3
cmake --build . --config Release -j
```

To switch variants, reconfigure with a different `-DCUSTOM_KERNEL_VERSION=…`
and rebuild — no source edits required.

## How it's wired into the ggml-cuda library

`custom_kernels/CMakeLists.txt` does, after `add_subdirectory(ggml)` brings
in the `ggml-cuda` target:

1. `target_sources(ggml-cuda PRIVATE …/src/*.cu)` — every `.cu` here compiles
   into the ggml-cuda library, inheriting its compile flags and link
   dependencies (cudart, cublas).
2. `target_include_directories(ggml-cuda PRIVATE …/ggml/src/ggml-cuda …/src)` —
   our files can `#include "common.cuh"` etc. just like the in-tree files.
3. `target_compile_definitions(ggml-cuda PRIVATE CUSTOM_KERNEL_VERSION=…)` —
   the dispatch hook in `ggml-cuda.cu` (added by the patch) selects the active
   variant via `#if CUSTOM_KERNEL_VERSION == …`.

## Adding a new variant (e.g. `amir_v3`)

1. Add `src/amir_v3.cu` defining `void custom_kernels_mmq_amir_v3(...)` with
   the same signature as the others (see `amir_v2.cu`).
2. Add it to `CUSTOM_KERNEL_SOURCES` in `CMakeLists.txt`.
3. Extend `ggml-dispatch.patch`: add a forward declaration of
   `custom_kernels_mmq_amir_v3` and a new `#elif CUSTOM_KERNEL_VERSION == 4`
   branch in the dispatch hook. Bump the `set_property(CACHE … STRINGS …)`
   list in `CMakeLists.txt`.
4. Rebuild with `-DCUSTOM_KERNEL_VERSION=4`.

## Why not just commit the kernels into a ggml fork?

That's also a fine option — see `notes/custom_kernel.md` for the discussion.
We picked this layout because it (a) avoids creating and maintaining a fork
of `leejet/ggml`, (b) keeps all variants side-by-side as committable files in
this repo, (c) makes the integration with stock ggml a single small patch we
can re-apply when upstream moves.
