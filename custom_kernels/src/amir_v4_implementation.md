# amir_v4 — CUTLASS Fused-Epilogue Implementation

**Author / implementation:** Thai Vu.
**Design / planning doc:** `cutlass_improvement.md` (same directory).
**Project context:** `../../../../progress.md` and `../../../../notes/custom_kernel.md`.

This note documents the actual implementation that landed in `amir_v4.cu` +
the CUTLASS section of `int8_helpers.cu` + the declaration in
`int8_helpers.cuh`. It is the companion to `cutlass_improvement.md`, which
described the *plan*; this file describes *what was built* and how to use it.

> **Naming note.** Thai Vu's email shipped this code as `amir_v3.cu`
> (defining `custom_kernels_mmq_amir_v3`) because no `amir_v3.cu` file
> existed in the tree — our "amir_v3" is a preload-only variant of amir_v2's
> matmul with no new matmul file. On integration the file was renamed to
> `amir_v4.cu` and the entry point to `custom_kernels_mmq_amir_v4` to match
> the variant scheme described in `progress.md` and the slide decks. The
> original skeleton sketched in `amir_v4_skeleton.cu` is preserved next to
> this file as a reference.

---

## What changed vs amir_v3

`amir_v4` keeps amir_v2 / amir_v3's surrounding work — cached Q8_0 → INT8
weight conversion (`g_amir_int8_weight_cache`, optionally pre-warmed by
amir_v3's `preload.cu`), per-call FP32 → INT8 activation quantization. The
only change is **Step 3**: instead of running
`cublasGemmEx(s8·s8 → s32) + dequantize_i32_to_f32_coalesced_cuda`, amir_v4
calls one CUTLASS kernel that does both at once and writes FP32 directly.

```
amir_v3 (current):                          amir_v4 (this implementation):
                                            
   cached INT8 weight  ┐                       cached INT8 weight  ┐
   activation INT8     ┘── cublasGemmEx ──┐    activation INT8     ┘── CUTLASS
                                          │       row_scale[M]    ────────┐
                                          │       col_scale[N]    ────────┤
                                          ▼                              │
                              INT32 buffer (M×N×4 bytes)                  │ (epilogue
                                          │                               │  in-kernel)
                                          ▼                               ▼
                          dequantize_i32_to_f32_coalesced ────────────► FP32 D
```

The eliminated parts: the standalone `dequantize_i32_to_f32_coalesced_kernel`
(~7 % of GPU time in exp018) and the M×N INT32 intermediate buffer write +
read pair.

---

## CUTLASS layout

The GEMM is a CUTLASS 2.x **Epilogue Visitor Tree (EVT)** built from these
visitors:

```
accumulator (int32, in registers)
       │
       ▼
  VisitorAccFetch                                ← convert acc to fp32 compute type
       │
       ├── VisitorColBroadcast(alphaRow[M])      ← row_scale, broadcast along N
       ▼
  VisitorCompute<multiplies>                     ← acc * alphaRow[m]
       │
       ├── VisitorRowBroadcast(alphaCol[N])      ← col_scale, broadcast along M
       ▼
  VisitorCompute<multiplies>                     ← (acc * alphaRow[m]) * alphaCol[n]
       │
       ▼
  VisitorAuxStore   stride=<1, ldc, ldc·N>       ← write fp32 in column-major
```

> Note on the `VisitorColBroadcast` / `VisitorRowBroadcast` naming: the
> "Col" / "Row" refers to the broadcast *direction*, not the indexed axis.
> `VisitorColBroadcast` with stride `<_1, _0, …>` indexes by row and
> broadcasts across columns → exactly what `alphaRow[m]` needs.

Implements:
```
D[m, n] = float(acc[m, n]) * alphaRow[m] * alphaCol[n]
```
— identical math to amir_v2/v3's GEMM + coalesced dequant pair.

## CUTLASS template parameters

Selected by `matmul_w8a8_cutlass_cuda` (the non-template entry point) and
forwarded to `matmul_w8a8_cutlass_f32_ptr<TileShape, WarpShape, kStages>`:

| Setting          | Value                                  | Notes |
|------------------|----------------------------------------|-------|
| Arch tag         | `cutlass::arch::Sm80` (Ampere)         | Forward-compatible on sm_110 (Blackwell). Functional but not tile-tuned for Blackwell — performance sweep recommended after correctness lands. |
| Op class         | `OpClassTensorOp`                      | IMMA tensor cores. |
| TileShape        | `<128, 128, 64>`                       | Standard sm_80 INT8 tile. |
| WarpShape        | `<64, 64, 64>`                         |   |
| InstructionShape | `<16, 8, 32>`                          | `mma.sync.m16n8k32.s8.s8` — INT8 IMMA. |
| Stages           | 3                                      | Pipelining depth. |
| AlignmentA       | 16 (= 128 / sizeof_bits<int8_t>)       | Optimal for INT8 LDG. |
| AlignmentB       | 16                                     |   |
| AlignmentC       | 1 (scalar fp32 store)                  | Safe; could be widened once correctness is solid. |
| Math op          | `OpMultiplyAddSaturate`                | Saturating INT32 accumulate. |
| Threadblock swizzle | `GemmIdentityThreadblockSwizzle<>` | Default. |

## Data layouts and the B-matrix trick

- `A` = cached INT8 weights, `[M, K]` row-major (`lda = K`). M = `row_diff`.
- `B` = activations quantized by `quantize_fp32_to_int8_row_wise_cuda(...)`
  into shape `[N, K]` row-major. The CUTLASS GEMM is declared with
  `LayoutB = ColumnMajor`. This is correct: a row-major `[N, K]` matrix is
  bit-identical to a column-major `[K, N]` matrix, and the math wants
  `B^T = [K, N]`. CUTLASS reads the column-major `[K, N]` view → effectively
  uses `B^T` for free.
- `D` = output, `[M, N]` column-major with leading dim `ldc`. Stride passed
  to the `VisitorAuxStore` is `<1, ldc, ldc·N>`. This matches what
  ggml expects for the dispatch hook's output buffer.

## Input validation

`matmul_w8a8_cutlass_f32_ptr` (and through it `matmul_w8a8_cutlass_cuda`)
checks before launching:

- All four pointers (`A`, `B`, `alphaRow`, `alphaCol`, `D`) non-null.
- M, N_gemm, K_gemm > 0.
- `ldc >= M`.
- `K_gemm % 32 == 0` (required by the s8 IMMA shape). For Q8_0 weights the
  block size is 32, so K is always a multiple of 32 — this check passes by
  construction for our use.

`can_implement` and `initialize` are checked for CUTLASS-internal feasibility.
Failures go through `CUTLASS_CHECK` (typically abort-on-failure).

A `#if 0`-gated post-launch sync exists in the file for early debugging; flip
to `#if 1` if you want fast feedback when iterating.

---

## How it's wired into the build

1. `amir_v4.cu` defines `custom_kernels_mmq_amir_v4(ggml_backend_cuda_context &, ...)`
   with the same signature as the other variants. Its body is guarded by
   `#if defined(CUSTOM_KERNEL_HAVE_CUTLASS)` — without CUTLASS, the function
   prints an error and aborts at runtime (so dispatch errors are loud, not
   silent).
2. `custom_kernels_mmq_amir_v4` is forward-declared and called from the
   `CUSTOM_KERNEL_VERSION == 5` branch of the dispatch hook in the patched
   `ggml/src/ggml-cuda/ggml-cuda.cu` (see `ggml-dispatch.patch`).
3. The CUTLASS code in `int8_helpers.cu` (the EVT GEMM and its
   non-template wrapper) is wrapped in the same
   `#if defined(CUSTOM_KERNEL_HAVE_CUTLASS)` guard, as is the declaration in
   `int8_helpers.cuh`. Variants 1..4 build without CUTLASS as before.
4. `custom_kernels/CMakeLists.txt` exposes `CUTLASS_DIR` as a cache path
   variable. If set to a valid CUTLASS checkout it adds
   `${CUTLASS_DIR}/include` (and `${CUTLASS_DIR}/tools/util/include`) to
   `ggml-cuda`'s include path and defines `CUSTOM_KERNEL_HAVE_CUTLASS=1`. If
   `CUSTOM_KERNEL_VERSION >= 5` and CUTLASS_DIR is missing, CMake errors out
   with a helpful message.

## Build steps

```bash
# 1) Clone CUTLASS (once):
git clone --depth 1 https://github.com/NVIDIA/cutlass.git \
    ~/Desktop/Diffusion/cutlass        # any path you like

# 2) Configure and build the submodule with amir_v4 active:
cd experiments/custom_kernels/build_thor
cmake .. -DSD_CUDA=ON \
         -DCUSTOM_KERNEL_VERSION=5 \
         -DCUTLASS_DIR=$HOME/Desktop/Diffusion/cutlass
cmake --build . -j

# 3) Quick smoke (Q8_0 / 256×256 / 2 steps):
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 ./bin/sd-cli \
    --diffusion-model ../../../models/flux2-dev-Q8_0.gguf \
    --vae ../../../models/flux2_ae.safetensors \
    --llm ../../../models/Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf \
    -p "a cat" --cfg-scale 1.0 --sampling-method euler --diffusion-fa \
    --seed 42 -H 256 -W 256 --steps 2 -o /tmp/v4.png -v
```

Email-style alternative (works too — just pass the include flag directly):

```bash
cmake .. -DSD_CUDA=ON -DCUSTOM_KERNEL_VERSION=5 \
         -DCUTLASS_DIR=$HOME/Desktop/Diffusion/cutlass \
         -DCMAKE_CUDA_FLAGS="-I$HOME/Desktop/Diffusion/cutlass/include"
```

(`CUTLASS_DIR` is the canonical knob; the `-I` is redundant once `CUTLASS_DIR`
is set, but doesn't hurt.)

To switch back to amir_v3 (no CUTLASS needed): reconfigure with
`-DCUSTOM_KERNEL_VERSION=4` (and `CUTLASS_DIR` can stay set or be empty).

---

## Things worth flagging back to Thai Vu (review notes)

These are small things to consider after the kernel is correct end-to-end:

1. **`Sm80` arch tag on Blackwell sm_110.** Functional, but Blackwell has
   wider preferred tile shapes for INT8. After correctness is locked, a
   tile-shape sweep (and maybe `cutlass::arch::Sm90` / `Sm100`) will likely
   close any remaining gap to cuBLAS HGEMM throughput.
2. **Workspace allocator.** Each call does `cudaMalloc` / `cudaFree` for the
   GEMM workspace. With 640 calls per 4-step generation that's a noticeable
   sync overhead. Allocating from `ctx.pool(id)` via `ggml_cuda_pool_alloc<char>`
   (like `dst_i32` in `amir_v2.cu`) would amortize this away.
3. **`CUTLASS_CHECK` is abort-on-failure.** Fine for development; if we ever
   want graceful fallback to amir_v3, the function returns `bool` already —
   the dispatch in `amir_v4.cu` could call into amir_v3's pipeline on
   failure. Future polish, not required.
4. **`#if 0` post-launch sync block** — useful for triage. Leave as-is.

---

## Expected performance (estimates from `cutlass_improvement.md`)

vs amir_v3 (exp018):

| Metric                        | amir_v3 today | amir_v4 estimated |
|-------------------------------|--------------:|------------------:|
| Sampling (4 steps, nsys)      |        10.23 s|             ~9.0 s|
| Sampling (bare, no nsys)      |         8.66 s|             ~7.5 s|
| s/step (steady)               |          2.53 |              ~2.20|
| Module-total EDP (4 steps)    |    5658 J·s   |     ~3500–4200 J·s|

Targets to verify once the build runs end-to-end:
- `exp020` — nsys profile (same 4-step / 512×512 / seed 42 / "a cat" run as
  exp017/018) to capture the per-kernel breakdown without the standalone
  dequant row.
- `exp021` — tegrastats power / energy / EDP run (matching the exp019
  methodology).

The slide tables (`slides/optimization_for_q8_v2.tex`) get a new column;
`notes/custom_kernel.md` gets the new comparison row.
