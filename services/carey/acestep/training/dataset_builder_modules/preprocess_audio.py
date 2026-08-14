import torch
import torchaudio


def _load_audio_without_torchcodec(audio_path: str):
    """Decode common training formats without TorchCodec when possible."""
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
            return torchaudio.load(audio_path)
        except Exception as torchaudio_error:
            raise RuntimeError(
                f"Could not decode {audio_path} with soundfile ({soundfile_error}) "
                f"or torchaudio ({torchaudio_error})"
            ) from torchaudio_error


def load_audio_stereo(audio_path: str, target_sample_rate: int, max_duration: float):
    """Load audio, resample, convert to stereo, and truncate."""
    audio, sr = _load_audio_without_torchcodec(audio_path)

    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)

    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2, :]

    max_samples = int(max_duration * target_sample_rate)
    if audio.shape[1] > max_samples:
        audio = audio[:, :max_samples]

    return audio, sr
