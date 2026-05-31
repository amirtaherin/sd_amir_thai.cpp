#!/usr/bin/env bash
# Apply the custom_kernels dispatch hook into the ggml submodule.
#
# Run this once after `git submodule update --init ggml`. Idempotent: detects
# if the patch is already applied and is a no-op in that case.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GGML_DIR="$ROOT_DIR/ggml"
PATCH_FILE="$ROOT_DIR/custom_kernels/ggml-dispatch.patch"

if [[ ! -d "$GGML_DIR/.git" && ! -f "$GGML_DIR/.git" ]]; then
    echo "ERROR: ggml submodule not initialized at $GGML_DIR" >&2
    echo "  Run: git submodule update --init ggml" >&2
    exit 1
fi

if [[ ! -f "$PATCH_FILE" ]]; then
    echo "ERROR: patch file missing: $PATCH_FILE" >&2
    exit 1
fi

# Already applied? (heuristic: look for our forward declaration)
if grep -q "custom_kernels_mmq_amir_v2" "$GGML_DIR/src/ggml-cuda/ggml-cuda.cu" 2>/dev/null; then
    echo "custom_kernels: dispatch patch already applied to ggml-cuda.cu (no-op)."
    exit 0
fi

cd "$GGML_DIR"
if git apply --check "$PATCH_FILE" 2>/dev/null; then
    git apply "$PATCH_FILE"
    echo "custom_kernels: applied $PATCH_FILE to ggml/src/ggml-cuda/ggml-cuda.cu"
else
    echo "ERROR: cannot cleanly apply $PATCH_FILE." >&2
    echo "  ggml may have moved away from the pinned commit. Check .gitmodules / submodule SHA." >&2
    exit 1
fi
