# 1. INT4 Kernel

The main entry point is located in `int4_library.cu`.
The function `custom_ggml_q4_kernel_spin` implements a custom `Q4_0` matrix multiplication.


The kernel first computes an incoherence score for `src1`.

**1. Path 1: Q8 compute**: If the incoherence score of `src1` is larger than the  threshold, the kernel falls back to:
   `custom_ggml_q4_weight_q8_compute_kernel`

**2. Path 2: incompatible SpinQuant size** If `K` is not compatible with the SpinQuant rotation size (K % 256 != 0), the kernel also falls back to: `custom_ggml_q4_weight_q8_compute_kernel`

**3. INT4 + rotation path**: uses the INT4 + rotation path.


---
# 2. Mathematical of Rotation
The rotation is applied on the right side of both matrices:

```text
src0_rot = src0 @ R
src1_rot = src1 @ R
```

Since `R` is orthogonal:

```text
R @ R^T = I
```

the matrix product is preserved:

```text
src0_rot @ src1_rot^T
= (src0 @ R) @ (src1 @ R)^T
= src0 @ R @ R^T @ src1^T
= src0 @ src1^T
```

Therefore, in exact arithmetic, the rotation does not change the result.