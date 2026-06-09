# amir_v4 — CUTLASS Epilogue Fusion (Original Design Note)

> **Status.** This doc captures the *design* of amir_v4 that motivated the
> work. The actual implementation by Thai Vu is documented in
> **`amir_v4_implementation.md`** (same directory), which describes the EVT
> epilogue tree, the CUTLASS template parameters chosen, the build wiring
> (CUTLASS_DIR), and review notes. Read `amir_v4_implementation.md` first if
> you want to *use* amir_v4; read this doc if you want the *why*.

This doc lives next to `amir_v4.cu` and was originally written as the
technical brief for the colleague implementing the CUTLASS variant. It
explains the bottleneck we're attacking, why stock cuBLAS can't fix it,
what CUTLASS gives us, the exact math the epilogue needs to apply, and the
expected performance impact.

Cross-reference: the broader project context is in
`progress.md` (repo root) and `notes/custom_kernel.md`.

---

## The problem in one sentence

After amir_v3, the per-matmul work for Q8_0 is:

```
   weights (cached INT8)  ┐
                          ├──→ cublasGemmEx (i8·i8 → s32)  ──→ INT32 buffer (M×N×4 bytes)
   activations (INT8)    ─┘                                         │
                                                                    ▼
                                                       dequantize_i32_to_f32_coalesced
                                                                    │  multiplies by
                                                                    │  row_scale[i]·col_scale[j]
                                                                    ▼
                                                              FP32 output
```

The **GEMM and the dequant are two separate kernel launches**, with an **INT32
intermediate buffer** between them. That buffer:

- Is materialized to global memory by the GEMM (M×N int32 values).
- Is read back from global memory by the dequant kernel.
- Is then written out as M×N FP32 values.

For typical Flux DiT shapes (M = 1024 tokens, N = 3072–12288 features) this is
**12–50 MB of extra memory traffic per matmul**, ×640 matmuls per 4-step
generation → several GB of avoidable traffic per generation.

The dequant kernel itself is still ~7 % of GPU time (0.93 s in exp018) — small
in absolute terms but the largest remaining custom-kernel cost.

## What we want

```
   weights (cached INT8)  ┐
                          ├──→ ONE kernel: GEMM + dequant FUSED  ──→ FP32 output
   activations (INT8)    ─┘                                            (no INT32 buffer)
   row_scales[M]         ─┤
   col_scales[N]         ─┘
```

A **single GEMM kernel** that does INT8 × INT8 multiply-accumulate into INT32
registers inside the SM, then in the kernel's **epilogue stage** — before
writing to global memory — multiplies each output element by
`row_scale[i] · col_scale[j]` and writes **FP32 directly**. No INT32 buffer
ever touches DRAM.

## Why we can't get this from stock cuBLAS / cuBLASLt

Three relevant constraints on CUDA 13.2:

1. **`cublasGemmEx` with `CUBLAS_COMPUTE_32I`** (the API we use today) forces
   the output type to be INT32 for INT8 inputs. Scalar `alpha`/`beta` only.
2. **`cublasLtMatmul`** is more flexible: it supports INT8 → FP32 output paths
   and exposes `A_SCALE_POINTER`, `B_SCALE_POINTER`, `D_SCALE_POINTER`
   attributes that can apply *scale vectors* in the epilogue.
3. *However*, on CUDA 13.x those **per-row / per-col FP32 vector scales for
   INT8 outputs are only wired up for FP8 / block-scaled formats** (the MXFP8
   / e4m3 / e5m2 family). For straight INT8 GEMM with FP32 output, only
   scalar α is exposed.

So stock cuBLAS(Lt) gives us either INT32 output (then we dequant separately —
current state) or FP8 with vector scales (different data type, not what we
want). The fix is to drop down to CUTLASS and write the epilogue ourselves.

## What CUTLASS is

[CUTLASS](https://github.com/NVIDIA/cutlass) is NVIDIA's open-source C++
template library for high-performance GEMM. The same algorithms cuBLAS uses
internally are exposed as CUTLASS templates — instantiate with custom data
types, custom tile shapes, custom epilogues. It compiles into normal CUDA
kernels.

A CUTLASS GEMM has three stages:

1. **Mainloop:** the actual tensor-core matmul. Per thread block, it loads
   tiles of A and B from global → shared memory, runs IMMA/HMMA instructions
   across `k`, and accumulates into per-thread INT32 register tiles.
2. **Epilogue:** runs **inside the same kernel** after the mainloop finishes
   a tile. It takes the INT32 accumulator (already in registers), applies
   whatever transformation the user wires up, and writes to the output
   tensor `D`.
3. **Threadblock swizzle / scheduler:** chooses which tiles go to which SMs.

The mainloop is fixed by the data types (s8 × s8 → s32 IMMA). **The epilogue is
where we customize.**

## The custom epilogue

Math we need in the epilogue:

```
D[i, j] = float(acc[i, j]) * row_scale[i] * col_scale[j]
```

`acc[i,j]` is the INT32 accumulator from the mainloop. `row_scale` is a
length-M vector (per output row); `col_scale` is a length-N vector (per
output column). Their outer product is the dequant factor matrix.

CUTLASS ships epilogues that are close:

- **`LinearCombination`** — the default. `D = α · acc + β · C`. Scalar α only
  — exactly the cuBLASLt limitation.
- **`LinearCombinationPlusBroadcast`** / `…GenericPlusBroadcast` — supports an
  additive bias broadcast vector along M or N. Close but additive, not
  multiplicative.
- **`EpilogueWithBroadcast` family (3.x)** — generalized; you can declare
  per-row and per-col scale vectors and write the arithmetic in terms of
  them.

For our case ("per-row × per-col scale multiply with FP32 output"), the
cleanest implementation paths:

- **CUTLASS 3.x with `cute` / `epilogue::collective`** — define a collective
  epilogue that, for each tile, broadcasts the M-vector `row_scale` (along N)
  and the N-vector `col_scale` (along M), multiplies the FP32-converted
  accumulator by both, and stores. This is the modern path and what NVIDIA
  recommends for sm_90+ (Hopper / Blackwell).
- **CUTLASS 2.x with custom functor** — write a custom `ThreadEpilogueOp` (a
  function object that runs per-thread on the accumulator fragment), or a
  "GemmWithBroadcast" device kernel. More boilerplate but well-trodden; see
  `cutlass/examples/35_*` family.

## Skeleton (CUTLASS 2.x flavor, for orientation)

Not real code — the moving parts to expect:

```cpp
using ElementA       = int8_t;                            // weights (INT8)
using ElementB       = int8_t;                            // activations (INT8)
using ElementAcc     = int32_t;                           // accumulator
using ElementD       = float;                             // output (FP32)
using ElementCompute = float;                             // epilogue arithmetic

using ArchTag        = cutlass::arch::Sm90 /* or sm_100/110 */;
using OpClass        = cutlass::arch::OpClassTensorOp;    // IMMA tensor cores
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;   // tune
using WarpShape        = cutlass::gemm::GemmShape< 64,  64, 64>;   // tune
using InstructionShape = cutlass::gemm::GemmShape< 16,   8, 32>;   // s8 IMMA shape

// Custom epilogue: D = (float(acc) * row_scale[i]) * col_scale[j]
struct RowColScaleEpilogue {
    /* per-thread functor: given INT32 accumulator fragment +
       row_scale tile + col_scale tile, returns float fragment */
};

using Gemm = cutlass::gemm::device::GemmUniversal<
    ElementA, cutlass::layout::RowMajor,
    ElementB, cutlass::layout::ColumnMajor,
    ElementD, cutlass::layout::RowMajor,
    ElementAcc,
    OpClass, ArchTag,
    ThreadblockShape, WarpShape, InstructionShape,
    RowColScaleEpilogue,
    /* swizzle, stages, alignment ... */
>;

// At call site (replaces the cublasGemmEx + dequantize_i32_to_f32 pair):
Gemm gemm_op;
typename Gemm::Arguments args{
    {M, N, K},
    /* tensor A */ {A_int8_ptr, K},
    /* tensor B */ {B_int8_ptr, K},
    /* tensor D */ {D_fp32_ptr, ldc},
    /* row_scale vector (length M) */ row_scales_ptr,
    /* col_scale vector (length N) */ col_scales_ptr,
};
gemm_op(args, workspace, stream);
```

The hard parts are (a) picking tile shapes for Blackwell sm_110, (b) wiring
the two scale vectors through CUTLASS's epilogue argument plumbing, and (c)
validating numerics.

## How this drops into our code

In `custom_kernels/src/`:

- `amir_v4.cu` (this directory) — the entry point with the same signature as
  amir_v2/amir_v3. Currently a SKELETON: it falls back to amir_v2's pipeline
  (cuBLAS INT8 GEMM + coalesced dequant). The TODO inside
  `#if defined(AMIR_V4_USE_CUTLASS)` is where the CUTLASS kernel goes.
- The Q8_0→INT8 weight cache, the amir_v3 preload at model load, and the
  per-call activation quantize all already exist and are shared — **don't
  re-implement them**. amir_v4 only replaces the GEMM-plus-dequant pair.
- `CUSTOM_KERNEL_VERSION=5` already dispatches to `custom_kernels_mmq_amir_v4`
  (in `ggml-dispatch.patch`). Selecting v=5 today runs the fallback (≡ v=4
  amir_v3 behaviour, since v4 inherits v3's preload via the
  `CUSTOM_KERNEL_VERSION >= 4` guard in `src/stable-diffusion.cpp`).

### Build integration TODO

When CUTLASS is added:

1. Vendor CUTLASS (preferred — pin a known-good commit). Either
   `git submodule add` under `thirdparty/cutlass`, or system install.
2. In `custom_kernels/CMakeLists.txt`, when `CUSTOM_KERNEL_VERSION >= 5`:
   - `target_include_directories(ggml-cuda PRIVATE <cutlass>/include
     <cutlass>/tools/util/include)`
   - `target_compile_definitions(ggml-cuda PRIVATE AMIR_V4_USE_CUTLASS)`
3. Verify the file compiles. CUTLASS template instantiations are heavy —
   expect a slow `amir_v4.cu` compile (30 s–2 min).

## Expected performance impact

Conservative estimates, all relative to amir_v3 (exp018):

**Direct savings:**
- The standalone dequant kernel goes away: **−7 % of GPU time = −0.93 s** off
  sampling.
- The INT32 intermediate buffer goes away: avoid M×N × 4 B written-then-read.
  For Flux DiT shapes, this is **a few MB to ~50 MB per matmul**, ×640
  matmuls per 4-step gen. Some overlaps with compute; on Thor's unified
  memory the savings typically compound by another 1–3 % beyond the kernel
  time saved.

**Estimated sampling time (4 steps, 512 × 512):**

| Metric              | amir_v3 today | amir_v4 estimated |
|---------------------|--------------:|------------------:|
| Sampling (nsys)     |     10.23 s   |      ~9.0  s      |
| Sampling (bare run) |      8.66 s   |      ~7.5  s      |
| s/step (steady)     |      2.53     |      ~2.20        |

That's another **−12 % to −15 %** on top of amir_v3, and **~36 % faster than
FORCE_CUBLAS** (vs amir_v3's 27 %).

**Energy & EDP (extrapolating from exp019's amir_v3 numbers):**
- Less GPU time at similar power → fewer joules on every rail.
- Less memory traffic → lower MSS rail power too.
- Module-total EDP estimated to drop from **5658 J·s → ~3500–4200 J·s** —
  another 25–40 % EDP reduction on top of amir_v3's already-large win.

These estimates assume the CUTLASS kernel hits cuBLAS-class throughput for
the GEMM mainloop. If the colleague's tile/scheduler choice isn't optimal
for sm_110, that target shifts.

## What can go wrong (and how to catch it early)

1. **Sm_110 CUTLASS support.** CUTLASS upstream usually tracks new
   architectures within a few months of silicon, but tile-shape autotuning
   lags. The first working version may be 1.5–2× slower than cuBLAS until
   tile shapes are tuned. **Mitigation:** start with the closest sm_90 /
   sm_100 preset; profile; sweep tile shapes after functional correctness
   is established.
2. **Numerical accuracy.** The math is identical to amir_v3's GEMM+dequant
   pair (same INT32 accumulator, same FP32 scaling). Output should be
   bit-exact to amir_v3 modulo nondeterminism in atomic accumulation.
   **Mitigation:** write a small standalone test that compares amir_v3 and
   amir_v4 outputs for a few representative weight tensors before
   integrating end-to-end.
3. **Epilogue plumbing.** Wiring two scale vectors (per-M and per-N) through
   CUTLASS's epilogue argument path is where bugs hide. The
   `examples/35_gemm_softmax`, `examples/45_dual_gemm`, and `examples/52_*`
   directories in CUTLASS have nearby patterns. **Mitigation:** get a
   "RowColScale" epilogue working on a small problem (M=128, N=128, K=128)
   before tackling Flux dimensions.
4. **Workspace allocation.** CUTLASS GEMMs sometimes need a scratch
   workspace. Use `ggml_cuda_pool_alloc<char>` and pass it via the
   `Arguments` struct, like we do for `dst_i32` today.

## Recommended starting points (for the colleague)

- Clone CUTLASS, build the examples on Thor first to verify the toolchain.
- Read these CUTLASS examples for relevant epilogue patterns:
  - `examples/35_gemm_softmax` — fused-epilogue with downstream op
  - `examples/45_dual_gemm` — multiple outputs with shared epilogue
  - `examples/52_hopper_gather_scatter_fusion` — modern (3.x) epilogue style
- Use our `amir_v2.cu` / `amir_v4.cu` as references for the math, GEMM
  arguments (M, N, K, lda, ldb, ldc, transposes), and the two scale vectors
  (`cw.d_scales` from the weight cache, `src1_scales` from the activation
  quant).
- The cache map (`g_amir_int8_weight_cache`) and the activation quantize
  call are shared infrastructure — do not replace them, just swap out the
  GEMM-plus-dequant pair.
- Validate the matmul output against amir_v3 (bit-identity) on a few weight
  tensors before doing full sd-cli runs.
- Once it works end-to-end: run the standard 4-step Q8_0 / 512×512 / seed 42
  nsys profile (this becomes `exp020`) and the tegrastats run (`exp021`).
  The slide tables get a new column.
