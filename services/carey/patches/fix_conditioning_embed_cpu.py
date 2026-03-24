"""
Patch acestep/core/generation/handler/conditioning_embed.py
— route text encoder calls through CPU.

Companion to fix_text_encoder_fp32.py which leaves the Qwen3 text encoder on
CPU to avoid cuBLAS GEMM failures on Blackwell (GB10) + CUDA 13.0 + torch 2.10.

Patches two methods in ConditioningEmbedMixin:

  infer_text_embeddings:
    input_ids → .cpu() → text_encoder → .to(device, dtype)

  infer_lyric_embeddings:
    lyric_token_ids → .cpu() → embed_tokens → .to(device, dtype)

Both move inputs to CPU before the call, then move the float32 output back to
the main CUDA device in self.dtype (bfloat16), so the rest of the pipeline
(DiT conditioning) receives tensors in the expected device and dtype.

Performance: Qwen3-Embedding-0.6B on CPU typically finishes in < 1 s on the
DGX Spark's ARM cores with 128 GB unified memory. This is called once per
generation request, so it does not meaningfully affect throughput.

Guard: exits 0 with a note if either pattern is absent.
"""

import sys

PATH = "/app/acestep/core/generation/handler/conditioning_embed.py"

# ── Patch 1: infer_text_embeddings ──────────────────────────────────────────
OLD1 = (
    "    def infer_text_embeddings(self, text_token_idss):\n"
    "        \"\"\"Infer text-token embeddings via text encoder.\"\"\"\n"
    "        with torch.inference_mode():\n"
    "            return self.text_encoder(input_ids=text_token_idss, lyric_attention_mask=None).last_hidden_state"
)

NEW1 = (
    "    def infer_text_embeddings(self, text_token_idss):\n"
    "        \"\"\"Infer text-token embeddings via text encoder.\"\"\"\n"
    "        with torch.inference_mode():\n"
    "            # text_encoder is on CPU (cuBLAS GEMM broken on Blackwell+CUDA13)\n"
    "            out = self.text_encoder(input_ids=text_token_idss.cpu(), lyric_attention_mask=None).last_hidden_state\n"
    "            return out.to(device=self.device, dtype=self.dtype)"
)

# ── Patch 2: infer_lyric_embeddings ─────────────────────────────────────────
OLD2 = (
    "    def infer_lyric_embeddings(self, lyric_token_ids):\n"
    "        \"\"\"Infer lyric-token embeddings via text encoder embedding table.\"\"\"\n"
    "        with torch.inference_mode():\n"
    "            return self.text_encoder.embed_tokens(lyric_token_ids)"
)

NEW2 = (
    "    def infer_lyric_embeddings(self, lyric_token_ids):\n"
    "        \"\"\"Infer lyric-token embeddings via text encoder embedding table.\"\"\"\n"
    "        with torch.inference_mode():\n"
    "            # text_encoder is on CPU (cuBLAS GEMM broken on Blackwell+CUDA13)\n"
    "            return self.text_encoder.embed_tokens(lyric_token_ids.cpu()).to(device=self.device, dtype=self.dtype)"
)

txt = open(PATH).read()
patched = txt
missing = []

if OLD1 not in patched:
    missing.append("infer_text_embeddings")
else:
    patched = patched.replace(OLD1, NEW1, 1)

if OLD2 not in patched:
    missing.append("infer_lyric_embeddings")
else:
    patched = patched.replace(OLD2, NEW2, 1)

if missing:
    print(
        f"Pattern(s) not found in {PATH}: {missing}\n"
        "The file may have changed or these patches were already applied.  "
        "Skipping affected patches.",
        file=sys.stderr,
    )

if patched == txt:
    sys.exit(0)

open(PATH, "w").write(patched)
applied = [n for n, o in [("infer_text_embeddings", OLD1), ("infer_lyric_embeddings", OLD2)] if o not in txt]
print(
    f"Patched {PATH}:\n"
    "  infer_text_embeddings  — input_ids → CPU, output → CUDA self.dtype\n"
    "  infer_lyric_embeddings — lyric_ids → CPU, output → CUDA self.dtype"
)
