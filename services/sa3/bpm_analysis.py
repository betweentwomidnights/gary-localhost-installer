"""Lightweight tempo estimation for training sidecar preparation.

This is intentionally conservative: it provides a useful BPM candidate without
trying to solve meter, half-time, or double-time perfectly. Human correction in
the sidecar UI remains the gold standard.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BpmCandidate:
    bpm: float
    score: float
    family: tuple[float, ...]


@dataclass(frozen=True)
class BpmEstimate:
    bpm: float
    confidence: float
    candidates: tuple[BpmCandidate, ...]


@dataclass(frozen=True)
class BpmDecision:
    bpm: int | None
    source: str
    lm_bpm: float | None = None
    local_bpm: float | None = None
    filename_bpm: int | None = None


def estimate_bpm(
    audio_path: Path,
    *,
    target_sr: int = 11025,
    frame_size: int = 1024,
    hop_size: int = 512,
    min_bpm: float = 60.0,
    max_bpm: float = 220.0,
    top_k: int = 8,
) -> BpmEstimate | None:
    """Estimate BPM via RMS-onset autocorrelation."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import find_peaks, resample_poly

    audio, sample_rate = sf.read(str(audio_path), always_2d=True)
    if audio.size == 0:
        return None

    mono = audio.mean(axis=1).astype("float32")
    if sample_rate != target_sr:
        mono = resample_poly(mono, target_sr, sample_rate).astype("float32")
        sample_rate = target_sr

    frame_count = 1 + (len(mono) - frame_size) // hop_size
    if frame_count <= 2:
        return None

    frames = np.lib.stride_tricks.as_strided(
        mono,
        shape=(frame_count, frame_size),
        strides=(mono.strides[0] * hop_size, mono.strides[0]),
    )
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    onset = np.maximum(0, np.diff(rms, prepend=rms[0]))
    onset -= onset.mean()
    if not np.any(onset):
        return None

    autocorr = np.correlate(onset, onset, mode="full")[len(onset) - 1 :]
    frames_per_second = sample_rate / hop_size
    min_lag = max(1, int(round(frames_per_second * 60 / max_bpm)))
    max_lag = min(
        len(autocorr) - 1,
        int(round(frames_per_second * 60 / min_bpm)),
    )
    if max_lag <= min_lag:
        return None

    window = autocorr[min_lag : max_lag + 1]
    peaks, _ = find_peaks(window, distance=max(1, min_lag // 2))
    candidates: list[BpmCandidate] = []
    for peak in peaks:
        lag = int(peak + min_lag)
        bpm = 60 * frames_per_second / lag
        score = float(window[peak])
        candidates.append(
            BpmCandidate(
                bpm=_normalize_training_bpm(bpm),
                score=score,
                family=_tempo_family(bpm),
            )
        )
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    candidates = [candidate for candidate in candidates if candidate.score > 0]
    if not candidates:
        return None

    top = candidates[0]
    runner_up = abs(candidates[1].score) if len(candidates) > 1 else 1e-9
    confidence = top.score / max(runner_up, 1e-9)
    return BpmEstimate(
        bpm=top.bpm,
        confidence=float(confidence),
        candidates=tuple(candidates[:top_k]),
    )


def choose_bpm(
    *,
    filename_bpm: int | None = None,
    lm_bpm: Any = None,
    local_estimate: BpmEstimate | None = None,
    disagreement_threshold: float = 5.0,
    minimum_local_confidence: float = 1.2,
) -> BpmDecision:
    """Choose the BPM to write to a sidecar.

    Precedence:
    - Filename BPM is ground truth when present.
    - If local and LM tempos agree, keep the LM value.
    - If they disagree enough, use the local estimate.
    """
    parsed_lm = _parse_bpm_float(lm_bpm)
    if filename_bpm is not None:
        return BpmDecision(
            bpm=filename_bpm,
            source="filename",
            lm_bpm=parsed_lm,
            local_bpm=local_estimate.bpm if local_estimate else None,
            filename_bpm=filename_bpm,
        )

    if local_estimate is None:
        return BpmDecision(
            bpm=_round_bpm(parsed_lm),
            source="lm" if parsed_lm is not None else "missing",
            lm_bpm=parsed_lm,
        )

    local_bpm = local_estimate.bpm
    if parsed_lm is None:
        return BpmDecision(
            bpm=_round_bpm(local_bpm),
            source="local",
            local_bpm=local_bpm,
        )

    if local_estimate.confidence < minimum_local_confidence:
        return BpmDecision(
            bpm=_round_bpm(parsed_lm),
            source="lm_local_low_confidence",
            lm_bpm=parsed_lm,
            local_bpm=local_bpm,
        )

    if _same_tempo_family(parsed_lm, local_bpm, tolerance=disagreement_threshold):
        return BpmDecision(
            bpm=_round_bpm(parsed_lm),
            source="lm_agrees_with_local",
            lm_bpm=parsed_lm,
            local_bpm=local_bpm,
        )

    return BpmDecision(
        bpm=_round_bpm(local_bpm),
        source="local_overrode_lm",
        lm_bpm=parsed_lm,
        local_bpm=local_bpm,
    )


def _normalize_training_bpm(bpm: float) -> float:
    while bpm > 220:
        bpm /= 2
    return float(bpm)


def _tempo_family(bpm: float) -> tuple[float, ...]:
    values = {round(float(bpm), 3)}
    current = float(bpm)
    while current / 2 >= 40:
        current /= 2
        values.add(round(current, 3))
    current = float(bpm)
    while current * 2 <= 240:
        current *= 2
        values.add(round(current, 3))
    return tuple(sorted(values))


def _same_tempo_family(a: float, b: float, *, tolerance: float) -> bool:
    return any(abs(a - candidate) <= tolerance for candidate in _tempo_family(b))


def _parse_bpm_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        bpm = float(value)
    except (TypeError, ValueError):
        return None
    return bpm if 1 <= bpm <= 400 else None


def _round_bpm(value: float | None) -> int | None:
    if value is None:
        return None
    bpm = int(round(value))
    return bpm if 1 <= bpm <= 400 else None
