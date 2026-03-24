"""
Patch transformers/models/qwen3/modeling_qwen3.py — Qwen3RotaryEmbedding.forward.

Bug: `inv_freq_expanded.float() @ position_ids_expanded.float()` is a batched
matrix multiply with k=1 (inner dimension = 1 — a degenerate outer-product).
On Blackwell (GB10) + CUDA 13.0 + torch 2.10.0 the cuBLAS path for batched
Sgemm with k=1 is broken: cublasSgemm is called with an invalid leading
dimension → CUBLAS_STATUS_INVALID_VALUE.

  inv_freq_expanded  : [bs, head_dim//2, 1]
  position_ids_expanded: [bs, 1, seq_len]
  result of @        : [bs, head_dim//2, seq_len]

The matmul [bs, N, 1] @ [bs, 1, M] = [bs, N, M] is mathematically identical to
a broadcast element-wise multiply:
  [bs, N, 1] * [bs, 1, M] → [bs, N, M]  (NumPy/PyTorch broadcasting rules)

Fix: replace `@` (cuBLAS batched Sgemm) with `*` (broadcast multiply).
Zero functional change — same values, same dtype, avoids cuBLAS entirely.

Affects: transformers 4.51.x – 4.57.x (line 327 in 4.57.6).
Guard: if the pattern is missing the script exits 0 and prints a note, so a
future upstream fix won't silently produce a double-patch or a build failure.
"""

import sys

TRANSFORMERS_QWEN3 = (
    "/usr/local/lib/python3.12/dist-packages"
    "/transformers/models/qwen3/modeling_qwen3.py"
)

# The broken line — batched Sgemm with k=1 crashes on Blackwell + CUDA 13.0.
OLD = (
    "freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)"
)

# Broadcast multiply: [bs, N, 1] * [bs, 1, M] = [bs, N, M] — no cuBLAS call.
NEW = (
    "freqs = (inv_freq_expanded.float() * position_ids_expanded.float()).transpose(1, 2)"
)

txt = open(TRANSFORMERS_QWEN3).read()

if OLD not in txt:
    print(
        f"Pattern not found in {TRANSFORMERS_QWEN3} — transformers may have already "
        "fixed this upstream, or the installed version differs.  Skipping patch.",
        file=sys.stderr,
    )
    sys.exit(0)

open(TRANSFORMERS_QWEN3, "w").write(txt.replace(OLD, NEW, 1))
print(
    f"Patched {TRANSFORMERS_QWEN3}:\n"
    "  replaced @ (cuBLAS batched Sgemm, k=1) with * (broadcast multiply)\n"
    "  to prevent CUBLAS_STATUS_INVALID_VALUE on Blackwell (GB10) + CUDA 13.0 + torch 2.10.0."
)
