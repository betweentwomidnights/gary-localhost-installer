"""ACE-Step package."""

from .rocm_compat import install_windows_rocm_distributed_fallback


if install_windows_rocm_distributed_fallback():
    print(
        "[ACE-Step] Installed single-process torch.distributed compatibility "
        "for Windows ROCm",
        flush=True,
    )
