"""Dataset and sidecar helpers for Gary's ACE-Step trainer.

The JSON shape matches the dataset format consumed by Carey's existing
two-pass preprocessor. Canonical ``.txt`` sidecars keep ``lyrics:`` last so
multi-line lyrics cannot consume later metadata fields.
"""

from __future__ import annotations

import json
import re
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}
_TRUE_VALUES = {"1", "true", "yes", "on"}
_SCALAR_KEYS = {
    "caption",
    "genre",
    "bpm",
    "bpm_source",
    "lm_bpm",
    "local_bpm",
    "filename_bpm",
    "key",
    "keyscale",
    "key_source",
    "lm_keyscale",
    "local_keyscale",
    "signature",
    "timesignature",
    "time_signature",
    "language",
    "is_instrumental",
    "custom_tag",
    "prompt_override",
}


def discover_audio_files(dataset_dir: Path) -> list[Path]:
    root = dataset_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not files:
        supported = ", ".join(sorted(AUDIO_EXTENSIONS))
        raise FileNotFoundError(f"No audio files found in {root}. Supported: {supported}")
    return files


def parse_key_value_sidecar(path: Path) -> dict[str, str]:
    """Parse a Side-Step-style sidecar with multiline values."""
    if not path.is_file():
        return {}
    content = path.read_text(encoding="utf-8-sig")
    if not content.strip():
        return {}

    result: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    def commit() -> None:
        if current_key is not None:
            result[current_key] = "\n".join(current_lines).strip()

    for line in content.splitlines():
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_ ]*):\s*(.*)$", line)
        candidate = match.group(1).strip().lower().replace(" ", "_") if match else ""
        if match and (candidate in _SCALAR_KEYS or candidate == "lyrics"):
            commit()
            current_key = candidate
            current_lines = [match.group(2)] if match.group(2) else []
        elif current_key is not None:
            current_lines.append(line.rstrip())
    commit()
    return result


def load_sidecar_metadata(audio_path: Path) -> dict[str, str]:
    """Load canonical or split caption/lyrics sidecars for one audio file."""
    canonical = audio_path.with_suffix(".txt")
    if canonical.is_file():
        parsed = parse_key_value_sidecar(canonical)
        if parsed:
            return parsed

    stem = audio_path.with_suffix("")
    caption_path = Path(f"{stem}.caption.txt")
    lyrics_path = Path(f"{stem}.lyrics.txt")
    caption = _read_optional_text(caption_path)
    lyrics = _read_optional_text(lyrics_path)
    if caption or lyrics:
        return {"caption": caption, "lyrics": lyrics}

    if canonical.is_file():
        legacy_lyrics = _read_optional_text(canonical)
        if legacy_lyrics:
            return {"lyrics": legacy_lyrics}
    return {}


def write_canonical_sidecar(
    path: Path,
    *,
    caption: str = "",
    genre: str = "",
    lyrics: str = "",
    bpm: Any = None,
    bpm_source: str = "",
    lm_bpm: Any = None,
    local_bpm: Any = None,
    filename_bpm: Any = None,
    keyscale: str = "",
    key_source: str = "",
    lm_keyscale: str = "",
    local_keyscale: str = "",
    timesignature: str = "",
    language: str = "",
    is_instrumental: bool = False,
    custom_tag: str = "",
) -> None:
    """Write metadata with ``lyrics:`` as the final field."""
    lines: list[str] = []
    values = (
        ("caption", caption),
        ("genre", genre),
        ("bpm", bpm),
        ("bpm_source", bpm_source),
        ("lm_bpm", lm_bpm),
        ("local_bpm", local_bpm),
        ("filename_bpm", filename_bpm),
        ("keyscale", keyscale),
        ("key_source", key_source),
        ("lm_keyscale", lm_keyscale),
        ("local_keyscale", local_keyscale),
        ("timesignature", timesignature),
        ("language", language),
    )
    for key, value in values:
        scalar = _clean_scalar(value)
        if scalar:
            lines.append(f"{key}: {scalar}")
    lines.append(f"is_instrumental: {'true' if is_instrumental else 'false'}")
    tag = _clean_scalar(custom_tag)
    if tag:
        lines.append(f"custom_tag: {tag}")

    lyric_text = str(lyrics or "").strip()
    if not lyric_text and is_instrumental:
        lyric_text = "[Instrumental]"
    lyric_lines = lyric_text.splitlines() or [""]
    lines.append(f"lyrics: {lyric_lines[0]}")
    lines.extend(lyric_lines[1:])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dataset_json(
    dataset_dir: Path,
    output_path: Path,
    *,
    name: str,
    trigger: str = "",
    tag_position: str = "prepend",
    genre_ratio: int = 0,
    instrumental_default: bool = False,
) -> dict[str, Any]:
    if tag_position not in {"prepend", "append", "replace"}:
        raise ValueError("tag_position must be prepend, append, or replace")
    if not 0 <= genre_ratio <= 100:
        raise ValueError("genre_ratio must be between 0 and 100")

    samples: list[dict[str, Any]] = []
    metadata_count = 0
    for audio_path in discover_audio_files(dataset_dir):
        meta = load_sidecar_metadata(audio_path)
        if meta:
            metadata_count += 1

        caption = meta.get("caption", "").strip()
        if not caption:
            caption = audio_path.stem.replace("_", " ").replace("-", " ")

        lyrics = meta.get("lyrics", "").strip()
        instrumental = _parse_bool(
            meta.get("is_instrumental"),
            default=instrumental_default or not lyrics or "[instrumental]" in lyrics.lower(),
        )
        if not lyrics and instrumental:
            lyrics = "[Instrumental]"

        sample_tag = trigger.strip() or meta.get("custom_tag", "").strip()
        samples.append(
            {
                "audio_path": str(audio_path.resolve()),
                "filename": audio_path.name,
                "caption": caption,
                "genre": meta.get("genre", "").strip(),
                "lyrics": lyrics,
                "bpm": _parse_bpm(meta.get("bpm")),
                "keyscale": (meta.get("keyscale") or meta.get("key") or "").strip(),
                "timesignature": (
                    meta.get("timesignature")
                    or meta.get("time_signature")
                    or meta.get("signature")
                    or ""
                ).strip(),
                "duration": audio_duration_seconds(audio_path),
                "language": meta.get("language", "unknown").strip() or "unknown",
                "is_instrumental": instrumental,
                "custom_tag": sample_tag,
                "prompt_override": _prompt_override(meta.get("prompt_override")),
            }
        )

    payload = {
        "metadata": {
            "name": name,
            "custom_tag": trigger.strip(),
            "tag_position": tag_position,
            "genre_ratio": genre_ratio,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "num_samples": len(samples),
            "all_instrumental": all(sample["is_instrumental"] for sample in samples),
        },
        "samples": samples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "path": str(output_path),
        "samples": len(samples),
        "with_metadata": metadata_count,
        "payload": payload,
    }


def audio_duration_seconds(path: Path) -> float:
    try:
        import soundfile as sf

        return round(float(sf.info(str(path)).duration), 3)
    except Exception:
        pass

    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wav:
                rate = wav.getframerate()
                return round(wav.getnframes() / rate, 3) if rate else 0.0
        except (OSError, wave.Error):
            pass
    return 0.0


def _read_optional_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig").strip()
    except (OSError, UnicodeError):
        return ""


def _clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    return value.strip().lower() in _TRUE_VALUES


def _parse_bpm(value: str | None) -> int | None:
    if not value:
        return None
    try:
        bpm = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return bpm if 1 <= bpm <= 400 else None


def _prompt_override(value: str | None) -> str | None:
    normalized = (value or "").strip().lower()
    return normalized if normalized in {"caption", "genre"} else None
