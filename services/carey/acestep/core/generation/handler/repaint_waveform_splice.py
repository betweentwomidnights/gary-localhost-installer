"""Post-decode waveform splicing for repaint tasks.

After VAE decode, non-repaint regions carry reconstruction error.  This module
replaces those regions with the original user waveform and applies a crossfade
at boundaries so the splice is inaudible.
"""

from typing import List, Optional

import torch


def _build_waveform_crossfade_mask(
    total_samples: int,
    start_sample: int,
    end_sample: int,
    crossfade_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a float mask that is 1.0 inside the repaint region with linear ramps.

    Returns:
        1-D float tensor of length *total_samples*.
    """
    mask = torch.zeros(total_samples, device=device)
    start_sample = max(0, start_sample)
    end_sample = min(total_samples, end_sample)
    if start_sample >= end_sample:
        return mask

    # Flat 1.0 interior
    mask[start_sample:end_sample] = 1.0

    # Fade-in ramp at the left boundary
    if crossfade_samples > 0 and start_sample > 0:
        fade_len = min(crossfade_samples, start_sample, end_sample - start_sample)
        if fade_len > 0:
            ramp = torch.linspace(0.0, 1.0, fade_len + 1, device=device)[1:]
            mask[start_sample:start_sample + fade_len] = ramp

    # Fade-out ramp at the right boundary
    if crossfade_samples > 0 and end_sample < total_samples:
        fade_len = min(crossfade_samples, total_samples - end_sample, end_sample - start_sample)
        if fade_len > 0:
            ramp = torch.linspace(1.0, 0.0, fade_len + 1, device=device)[1:]
            mask[end_sample - fade_len:end_sample] = ramp

    return mask


def apply_repaint_waveform_splice(
    pred_wavs: torch.Tensor,
    src_wavs: torch.Tensor,
    repainting_starts: Optional[List[float]],
    repainting_ends: Optional[List[float]],
    sample_rate: int,
    crossfade_duration: float = 0.0,
) -> torch.Tensor:
    """Replace non-repaint regions of decoded audio with the original waveform.

    Args:
        pred_wavs: Decoded prediction tensor ``[B, C, T]``.
        src_wavs: Original source waveform tensor, same shape as *pred_wavs* or
            ``[B, T]`` (will be unsqueezed).
        repainting_starts: Per-item start times in seconds (``None`` = no splice).
        repainting_ends: Per-item end times in seconds.
        sample_rate: Audio sample rate.
        crossfade_duration: Crossfade length in seconds at splice boundaries.

    Returns:
        Spliced waveform tensor with the same shape as *pred_wavs*.
    """
    if repainting_starts is None or repainting_ends is None:
        return pred_wavs

    if src_wavs.dim() == 2:
        src_wavs = src_wavs.unsqueeze(1)

    # Ensure same device
    src_wavs = src_wavs.to(pred_wavs.device)

    batch_size = pred_wavs.shape[0]
    total_samples = pred_wavs.shape[-1]
    crossfade_samples = int(crossfade_duration * sample_rate)

    result = pred_wavs.clone()
    for i in range(batch_size):
        start = repainting_starts[i] if i < len(repainting_starts) else None
        end = repainting_ends[i] if i < len(repainting_ends) else None
        if start is None or end is None:
            continue
        if end <= start:
            continue

        start_sample = int(start * sample_rate)
        end_sample = min(int(end * sample_rate), total_samples)

        # mask: 1.0 = use predicted (repaint region), 0.0 = use source
        mask = _build_waveform_crossfade_mask(
            total_samples, start_sample, end_sample, crossfade_samples, pred_wavs.device
        )

        # Ensure src_wavs covers the needed length
        src_len = src_wavs.shape[-1]
        if src_len < total_samples:
            pad = torch.zeros(
                src_wavs.shape[0], src_wavs.shape[1], total_samples - src_len,
                device=src_wavs.device, dtype=src_wavs.dtype,
            )
            src_padded = torch.cat([src_wavs, pad], dim=-1)
        else:
            src_padded = src_wavs[..., :total_samples]

        # Blend: repaint region keeps prediction, outside keeps source
        result[i] = mask * pred_wavs[i] + (1.0 - mask) * src_padded[i]

    return result
