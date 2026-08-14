from __future__ import annotations

from os import PathLike
from typing import Union

import torch
import torchaudio


AudioPath = Union[str, PathLike[str]]


def load_audio_file(audio_path: AudioPath) -> tuple[torch.Tensor, int]:
    """Decode audio without requiring TorchCodec for common file formats."""
    try:
        import soundfile as sf

        samples, sample_rate = sf.read(
            audio_path,
            dtype="float32",
            always_2d=True,
        )
        return torch.from_numpy(samples.T.copy()), int(sample_rate)
    except Exception as soundfile_error:
        try:
            return torchaudio.load(str(audio_path))
        except Exception as torchaudio_error:
            raise RuntimeError(
                f"Could not decode {audio_path} with soundfile ({soundfile_error}) "
                f"or torchaudio ({torchaudio_error})"
            ) from torchaudio_error
