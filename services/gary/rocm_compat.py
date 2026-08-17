"""Compatibility helpers for incomplete Windows ROCm PyTorch builds."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any


_COMPAT_MARKER = "_gary_windows_rocm_single_process_compat"


def install_windows_rocm_distributed_fallback(
    torch_module: Any = None,
    *,
    os_name: str | None = None,
) -> bool:
    """Supply the distributed surface imported by Flashy in single-process mode."""
    if torch_module is None:
        import torch as torch_module

    current_os = os.name if os_name is None else os_name
    torch_version = getattr(torch_module, "version", None)
    if current_os != "nt" or not getattr(torch_version, "hip", None):
        return False

    distributed = getattr(torch_module, "distributed", None)
    if distributed is None or getattr(distributed, _COMPAT_MARKER, False):
        return False

    if hasattr(distributed, "ReduceOp") and hasattr(distributed, "all_reduce"):
        return False

    if not hasattr(distributed, "ReduceOp"):
        distributed.ReduceOp = SimpleNamespace(
            SUM="sum",
            AVG="avg",
            AVERAGE="avg",
            PRODUCT="product",
            MIN="min",
            MAX="max",
            BAND="band",
            BOR="bor",
            BXOR="bxor",
            PREMUL_SUM="premul_sum",
        )
    if not hasattr(distributed, "is_initialized"):
        distributed.is_initialized = lambda: False
    if not hasattr(distributed, "get_world_size"):
        distributed.get_world_size = lambda *args, **kwargs: 1

    def single_process_all_reduce(tensor: Any, *args: Any, **kwargs: Any) -> Any:
        if distributed.is_initialized() and distributed.get_world_size() > 1:
            raise RuntimeError(
                "The Windows ROCm compatibility path only supports one process."
            )
        return tensor

    distributed.all_reduce = single_process_all_reduce
    setattr(distributed, _COMPAT_MARKER, True)
    return True
