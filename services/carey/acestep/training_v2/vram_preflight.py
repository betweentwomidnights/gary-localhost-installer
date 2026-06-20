"""Conservative CUDA memory budgeting before ACE-Step training begins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


_MIB = 1024 ** 2


@dataclass(frozen=True)
class VramPreflightResult:
    total_mb: float
    free_mb: float
    allocated_mb: float
    reserved_mb: float
    trainable_parameters: int
    gradient_mb: float
    optimizer_state_mb: float
    optimizer_workspace_mb: float
    activation_mb: float
    system_safety_mb: float
    required_headroom_mb: float
    margin_mb: float
    safe: bool


def estimate_vram_preflight(
    *,
    total_mb: float,
    free_mb: float,
    allocated_mb: float,
    reserved_mb: float,
    trainable_parameters: int,
    trainable_bytes: int,
    max_duration: float,
    batch_size: int,
    gradient_checkpointing: bool,
) -> VramPreflightResult:
    """Estimate lazy allocations and activation headroom not present at setup.

    AdamW creates its moment tensors lazily on the first optimizer step. The
    estimate assumes two fp32 moments per trainable value, plus gradients and
    one parameter-sized optimizer workspace. Activation reserve scales from a
    conservative checkpointed 240-second, batch-one baseline.
    """
    gradient_mb = trainable_bytes / _MIB
    optimizer_state_mb = trainable_parameters * 8 / _MIB
    optimizer_workspace_mb = trainable_bytes / _MIB

    duration_scale = max(60.0, float(max_duration)) / 240.0
    activation_mb = 900.0 * duration_scale * max(1, int(batch_size))
    if not gradient_checkpointing:
        activation_mb *= 1.75

    # Leave room for WDDM/the display driver and non-PyTorch CUDA allocations.
    system_safety_mb = max(768.0, total_mb * 0.08)
    required_headroom_mb = (
        gradient_mb
        + optimizer_state_mb
        + optimizer_workspace_mb
        + activation_mb
        + system_safety_mb
    )
    margin_mb = free_mb - required_headroom_mb
    return VramPreflightResult(
        total_mb=total_mb,
        free_mb=free_mb,
        allocated_mb=allocated_mb,
        reserved_mb=reserved_mb,
        trainable_parameters=trainable_parameters,
        gradient_mb=gradient_mb,
        optimizer_state_mb=optimizer_state_mb,
        optimizer_workspace_mb=optimizer_workspace_mb,
        activation_mb=activation_mb,
        system_safety_mb=system_safety_mb,
        required_headroom_mb=required_headroom_mb,
        margin_mb=margin_mb,
        safe=margin_mb >= 0.0,
    )


def capture_cuda_vram_preflight(
    model: nn.Module,
    config: Any,
) -> VramPreflightResult | None:
    """Capture real post-setup CUDA state and evaluate remaining headroom."""
    device = torch.device(config.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None

    index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.synchronize(index)
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    trainable = [param for param in model.parameters() if param.requires_grad]
    trainable_parameters = sum(param.numel() for param in trainable)
    trainable_bytes = sum(param.numel() * param.element_size() for param in trainable)

    return estimate_vram_preflight(
        total_mb=total_bytes / _MIB,
        free_mb=free_bytes / _MIB,
        allocated_mb=torch.cuda.memory_allocated(index) / _MIB,
        reserved_mb=torch.cuda.memory_reserved(index) / _MIB,
        trainable_parameters=trainable_parameters,
        trainable_bytes=trainable_bytes,
        max_duration=getattr(config, "max_duration", 240.0),
        batch_size=getattr(config, "batch_size", 1),
        gradient_checkpointing=getattr(config, "gradient_checkpointing", True),
    )


def vram_preflight_summary(result: VramPreflightResult) -> str:
    """One-line summary suitable for logs and training updates."""
    verdict = "SAFE" if result.safe else "BLOCKED"
    return (
        f"[VRAM preflight] {verdict}: {result.free_mb:.0f} MiB free after setup; "
        f"{result.required_headroom_mb:.0f} MiB estimated first-step reserve; "
        f"margin {result.margin_mb:+.0f} MiB"
    )


def vram_preflight_detail(result: VramPreflightResult) -> str:
    return (
        f"gradients {result.gradient_mb:.0f} MiB, "
        f"AdamW states {result.optimizer_state_mb:.0f} MiB, "
        f"optimizer workspace {result.optimizer_workspace_mb:.0f} MiB, "
        f"activations {result.activation_mb:.0f} MiB, "
        f"WDDM safety {result.system_safety_mb:.0f} MiB"
    )
