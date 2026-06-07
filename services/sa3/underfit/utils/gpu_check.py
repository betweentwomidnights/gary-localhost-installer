"""Detect GPU capability quirks and pre-warn the user instead of letting torch
spew autotune warnings mid-run.

Currently catches: pre-Ampere GPUs (compute capability < 8.0) hitting
flex_attention's Triton kernel, which requires SM80+. The kernel falls back
to a slow eager path and emits W0522-style logs that look terrifying but
are harmless. We replace that wall with a single readable warning."""
from __future__ import annotations


def _silence_compile_noise() -> None:
    """Silence torch's autotune / dynamo / inductor log channels.

    These emit W/E-level messages when torch.compile falls back to eager —
    a graceful degradation, NOT a real failure. The fallback runs the eager
    path and training completes normally. The messages look terrifying
    (multi-screen stack traces) but they're noise.

    We silence three layers:
      1. Standard Python loggers under torch._dynamo / torch._inductor
      2. torch._logging.set_logs() — torch's own glog-style logger (the
         source of the `W0522` / `E0522` prefixed lines)
      3. Best-effort suppression for child loggers (triton_heuristics,
         select_algorithm, convert_frame) that may have explicit levels.

    Some output still leaks through — the MLIR / LLVM crash banners on
    flex_attention compile failure come from C++ stderr writes, not Python
    logging. Those would require dup2() of fd 2, which is too invasive."""
    import logging
    for mod in (
        "torch",
        "torch._dynamo",
        "torch._inductor",
        "torch._inductor.select_algorithm",
        "torch._inductor.runtime",
        "torch._inductor.runtime.triton_heuristics",
        "torch._dynamo.convert_frame",
    ):
        logging.getLogger(mod).setLevel(logging.CRITICAL)
    try:
        import torch._logging
        torch._logging.set_logs(
            dynamo=logging.CRITICAL,
            inductor=logging.CRITICAL,
        )
    except Exception:
        pass


def check_attention_backends() -> dict:
    """Probe which attention implementations are importable and print a
    one-line summary. Helps clear up the misleading 'flash_attn not installed'
    warnings that SA3 prints by saying explicitly which path *will* run.

    SA3's per-call routing (transformer.py):
      1. flex_attention  — when a block_mask / score_mod is passed (inpainting
                            and varied seq lengths trigger this).
      2. flash_attn varlen — packed-batch, only if flash_attn AND bert_padding
                              utilities both import.
      3. flash_attn      — plain, when neither of the above applies.
      4. SDPA            — torch.nn.functional fallback (always available).
    """
    paths = {"flash_attn": False, "flash_attn_varlen": False,
             "flex_attention": False, "sdpa": True}
    try:
        import flash_attn  # noqa: F401
        paths["flash_attn"] = True
        try:
            from flash_attn.bert_padding import unpad_input  # noqa: F401
            paths["flash_attn_varlen"] = True
        except ImportError:
            pass
    except ImportError:
        pass
    try:
        from torch.nn.attention.flex_attention import flex_attention  # noqa: F401
        paths["flex_attention"] = True
    except ImportError:
        pass

    if paths["flash_attn_varlen"]:
        primary = "Flash Attention (varlen)"
    elif paths["flash_attn"]:
        primary = "Flash Attention"
    elif paths["flex_attention"]:
        primary = "FlexAttention (PyTorch compiled)"
    else:
        primary = "SDPA (PyTorch eager)"
    available = [k for k, v in paths.items() if v]
    print(f"Attention: primary path = {primary}", flush=True)
    print(f"  available: {', '.join(available)}", flush=True)
    if not paths["flash_attn"]:
        print(
            "  flash-attn not installed (optional; biggest speedup on Ampere+ with batch>1).",
            flush=True,
        )
    return paths


def check_attention_compute_capability() -> bool | None:
    """Return True if the GPU supports flex_attention's compiled path
    (compute capability >= 8.0), False if older, None if no GPU.

    Always silences torch.compile/autotune log noise (the W0522/E0522 walls).
    Prints a friendly heads-up if the GPU is pre-Ampere — those users will
    hit the eager fallback path more often than modern-GPU users would.
    """
    _silence_compile_noise()

    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None

    major, minor = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    compatible = major >= 8

    if compatible:
        print(f"GPU: {name} (sm{major}{minor}, compute capability {major}.{minor})", flush=True)
        return True

    print(
        f"⚠️  Older GPU detected: {name} (sm{major}{minor}). "
        f"flex_attention / flash_attention may not work; expect slower runs than L4 / A100 / H100.",
        flush=True,
    )
    return False
