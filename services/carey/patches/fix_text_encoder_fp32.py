"""
Patch acestep/core/generation/handler/init_service_loader.py
— keep the Qwen3 text encoder on CPU (not moved to CUDA).

Bug: on Blackwell (GB10) + CUDA 13.0 + torch 2.10.0, BOTH bfloat16 AND
float32 cuBLAS GEMM calls fail with CUBLAS_STATUS_INVALID_VALUE inside the
Qwen3 attention stack (q_proj, k_proj, v_proj, o_proj, and MLP gates):

  bfloat16: cublasGemmEx CUDA_R_16BF → CUBLAS_STATUS_INVALID_VALUE
  float32:  cublasSgemm              → CUBLAS_STATUS_INVALID_VALUE

The root cause appears to be a cuBLAS compatibility issue specific to this
architecture/driver/torch combination for all GEMM variants.

Fix: leave the text encoder on CPU (AutoModel.from_pretrained defaults to CPU).
A companion patch (fix_conditioning_embed_cpu.py) makes the inference call move
input IDs to CPU and the output back to CUDA transparently.

The DiT model, VAE, and all other components are unaffected — they remain on
CUDA in bfloat16.

Memory: Qwen3-Embedding-0.6B in float32 on CPU ≈ 2.4 GB system RAM.
On the DGX Spark (128 GB unified memory) this is completely fine.

Guard: exits 0 with a note if the pattern is absent.
"""

import sys

PATH = "/app/acestep/core/generation/handler/init_service_loader.py"

# Original lines — move text encoder to CUDA and cast to model dtype.
OLD = (
    "            self.text_encoder = self.text_encoder.to(device).to(self.dtype)\n"
    "        else:\n"
    "            self.text_encoder = self.text_encoder.to(\"cpu\").to(self.dtype)"
)

# Leave text encoder on CPU; companion patch handles CPU↔CUDA in inference.
NEW = (
    "            pass  # text encoder stays on CPU: cuBLAS GEMM (fp32+bf16) broken on Blackwell+CUDA13\n"
    "        else:\n"
    "            pass  # text encoder already on CPU by default"
)

txt = open(PATH).read()

if OLD not in txt:
    print(
        f"Pattern not found in {PATH} — init_service_loader may have changed "
        "or this fix was already applied.  Skipping patch.",
        file=sys.stderr,
    )
    sys.exit(0)

open(PATH, "w").write(txt.replace(OLD, NEW, 1))
print(
    f"Patched {PATH}:\n"
    "  text encoder left on CPU — cuBLAS GEMM (both fp32 and bf16) broken\n"
    "  on Blackwell (GB10) + CUDA 13.0 + torch 2.10.0.\n"
    "  Companion patch fix_conditioning_embed_cpu.py handles CPU↔CUDA I/O."
)
