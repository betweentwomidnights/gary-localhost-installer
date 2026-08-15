"""Audio file I/O for the MelodyFlow localhost service."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import soundfile as sf
import torch
import torchaudio


PathLike = Union[str, Path]


def load_audio(path: PathLike, target_sr: int, device: str) -> torch.Tensor:
    samples, sample_rate = sf.read(
        str(path), dtype="float32", always_2d=True
    )
    waveform = torch.from_numpy(samples.T.copy())

    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)

    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2, :]

    return waveform.unsqueeze(0).to(device)


def save_audio(path: PathLike, waveform: torch.Tensor, sample_rate: int) -> None:
    rendered = waveform.detach().to(torch.float32).cpu()
    if rendered.dim() == 1:
        rendered = rendered.unsqueeze(0)
    if rendered.dim() != 2:
        raise ValueError(f"Expected [channels, samples] audio, got {rendered.shape}")

    sf.write(
        str(path),
        rendered.clamp(-1.0, 1.0).transpose(0, 1).numpy(),
        sample_rate,
        format="WAV",
        subtype="PCM_16",
    )
