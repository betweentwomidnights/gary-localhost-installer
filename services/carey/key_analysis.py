"""Lightweight musical key estimation for training sidecar preparation.

This intentionally stays dependency-light by using the scipy/soundfile stack
Carey already ships. It should be treated as a suggestion, not a musicological
truth machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
NOTE_ALIASES = {
    "DB": "C#",
    "EB": "D#",
    "GB": "F#",
    "AB": "G#",
    "BB": "A#",
}

# Krumhansl-Kessler profiles in C major / C minor orientation.
MAJOR_PROFILE = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
MINOR_PROFILE = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)


@dataclass(frozen=True)
class KeyCandidate:
    keyscale: str
    score: float


@dataclass(frozen=True)
class KeyEstimate:
    keyscale: str
    confidence: float
    candidates: tuple[KeyCandidate, ...]


@dataclass(frozen=True)
class KeyDecision:
    keyscale: str
    source: str
    lm_keyscale: str = ""
    local_keyscale: str = ""
    confidence: float = 0.0


def estimate_key(
    audio_path: Path,
    *,
    target_sr: int = 11025,
    frame_size: int = 8192,
    hop_size: int = 2048,
    min_frequency: float = 55.0,
    max_frequency: float = 2500.0,
    top_db: float = 55.0,
    top_k: int = 8,
) -> KeyEstimate | None:
    """Estimate key using a chroma vector and key-profile correlation."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly, stft

    audio, sample_rate = sf.read(str(audio_path), always_2d=True)
    if audio.size == 0:
        return None
    mono = audio.mean(axis=1).astype("float32")
    if sample_rate != target_sr:
        mono = resample_poly(mono, target_sr, sample_rate).astype("float32")
        sample_rate = target_sr

    _, _, spectrum = stft(
        mono,
        fs=sample_rate,
        nperseg=frame_size,
        noverlap=frame_size - hop_size,
        window="hann",
        boundary=None,
        padded=False,
    )
    magnitudes = np.abs(spectrum)
    if magnitudes.size == 0:
        return None

    threshold = magnitudes.max(axis=0, keepdims=True) * (10 ** (-top_db / 20))
    magnitudes = np.where(magnitudes >= threshold, magnitudes, 0.0)
    freqs = np.fft.rfftfreq(frame_size, d=1 / sample_rate)
    chroma = np.zeros(12, dtype=float)
    mask = (freqs >= min_frequency) & (freqs <= max_frequency)
    for freq, row in zip(freqs[mask], magnitudes[mask]):
        if freq <= 0:
            continue
        midi = int(round(69 + 12 * math.log2(float(freq) / 440.0)))
        chroma[midi % 12] += float(row.sum()) / math.sqrt(float(freq))

    total = chroma.sum()
    if total <= 0:
        return None
    chroma = chroma / total
    candidates = _score_key_profiles(chroma)
    if not candidates:
        return None
    top = candidates[0]
    runner_up = candidates[1].score if len(candidates) > 1 else 0.0
    return KeyEstimate(
        keyscale=top.keyscale,
        confidence=float(top.score - runner_up),
        candidates=tuple(candidates[:top_k]),
    )


def choose_key(
    *,
    lm_keyscale: Any = None,
    local_estimate: KeyEstimate | None = None,
    minimum_local_confidence: float = 0.15,
) -> KeyDecision:
    parsed_lm = normalize_keyscale(lm_keyscale)
    if local_estimate is None:
        return KeyDecision(
            keyscale=parsed_lm,
            source="lm" if parsed_lm else "missing",
            lm_keyscale=parsed_lm,
        )

    local = local_estimate.keyscale
    if parsed_lm and normalize_keyscale(parsed_lm) == normalize_keyscale(local):
        return KeyDecision(
            keyscale=parsed_lm,
            source="lm_agrees_with_local",
            lm_keyscale=parsed_lm,
            local_keyscale=local,
            confidence=local_estimate.confidence,
        )
    if local_estimate.confidence >= minimum_local_confidence:
        return KeyDecision(
            keyscale=local,
            source="local_overrode_lm" if parsed_lm else "local",
            lm_keyscale=parsed_lm,
            local_keyscale=local,
            confidence=local_estimate.confidence,
        )
    return KeyDecision(
        keyscale=parsed_lm,
        source="lm_local_low_confidence" if parsed_lm else "missing",
        lm_keyscale=parsed_lm,
        local_keyscale=local,
        confidence=local_estimate.confidence,
    )


def normalize_keyscale(value: Any) -> str:
    text = " ".join(str(value or "").replace("_", " ").split()).strip()
    if not text or text.upper() == "N/A":
        return ""
    parts = text.replace("-", " ").split()
    if len(parts) < 2:
        return ""
    note = parts[0].upper().replace("♭", "B").replace("♯", "#")
    note = NOTE_ALIASES.get(note, note)
    if note not in NOTE_NAMES:
        return ""
    mode = parts[1].lower()
    if mode in {"maj", "major"}:
        mode = "major"
    elif mode in {"min", "minor"}:
        mode = "minor"
    else:
        return ""
    return f"{note} {mode}"


def _score_key_profiles(chroma: Any) -> list[KeyCandidate]:
    import numpy as np

    chroma_n = _normalize_vector(chroma)
    major = np.asarray(MAJOR_PROFILE, dtype=float)
    minor = np.asarray(MINOR_PROFILE, dtype=float)
    results: list[KeyCandidate] = []
    for tonic, note in enumerate(NOTE_NAMES):
        for mode, profile in (("major", major), ("minor", minor)):
            shifted = _normalize_vector(np.roll(profile, tonic))
            score = float(np.dot(chroma_n, shifted))
            results.append(KeyCandidate(f"{note} {mode}", score))
    results.sort(key=lambda candidate: candidate.score, reverse=True)
    return results


def _normalize_vector(value: Any) -> Any:
    import numpy as np

    vector = np.asarray(value, dtype=float)
    vector = vector - vector.mean()
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector
