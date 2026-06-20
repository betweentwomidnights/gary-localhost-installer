"""Loss weighting helpers for ACE-Step flow-matching training.

Min-SNR-gamma follows Hang et al., "Efficient Diffusion Training via
Min-SNR Weighting Strategy" (ICCV 2023).  ACE-Step's interpolation is
``x_t = t * noise + (1 - t) * data``, so its signal-to-noise ratio is
``((1 - t) / t) ** 2``.
"""

from __future__ import annotations

import torch


def flow_min_snr_weights(timesteps: torch.Tensor, gamma: float = 5.0) -> torch.Tensor:
    """Return per-sample Min-SNR weights for ACE-Step flow timesteps."""
    if gamma <= 0:
        raise ValueError("Min-SNR gamma must be greater than zero")

    t = timesteps.float().clamp(min=1e-4, max=1.0 - 1e-4)
    snr = (((1.0 - t) / t) ** 2).clamp(max=1e6)
    gamma_tensor = torch.full_like(snr, float(gamma))
    return torch.minimum(snr, gamma_tensor) / snr.clamp(min=1e-6)
