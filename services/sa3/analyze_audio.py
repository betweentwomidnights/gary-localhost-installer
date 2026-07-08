#!/usr/bin/env python3
"""Suggest BPM and musical key for a single audio file (SA3 sidecar helper).

Runs on CPU using the vendored lightweight estimators. scipy is installed on
demand the first time this is used so existing SA3 environments don't need a full
rebuild (it is also declared in requirements.txt for fresh builds). The result is
a single JSON object printed to stdout; all diagnostics go to stderr so stdout
stays clean for the caller.

Output on success:
    {"ok": true, "bpm": 145, "keyscale": "C minor",
     "bpm_source": "local", "key_source": "local",
     "bpm_confidence": 1.83, "key_confidence": 0.21,
     "suggestion": "145 bpm, C minor"}

BPM and key are emitted in the bare format gary4juce appends at inference
("145 bpm", "C minor"), so a filled sidecar and a generated prompt agree.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCIPY_REQUIREMENT = "scipy>=1.14"


def ensure_scipy() -> None:
    """Install scipy into this env on first use; point to 'rebuild env' if it fails."""
    if importlib.util.find_spec("scipy") is not None:
        return
    print(
        f"[environment-setup] Installing analysis dependency: {SCIPY_REQUIREMENT}",
        file=sys.stderr,
        flush=True,
    )
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", SCIPY_REQUIREMENT],
            stdout=sys.stderr,  # keep stdout reserved for the JSON result
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Could not install {SCIPY_REQUIREMENT} automatically ({exc}). "
            "Use 'rebuild env' for SA3 and try again."
        ) from exc
    importlib.invalidate_caches()
    if importlib.util.find_spec("scipy") is None:
        raise RuntimeError(
            f"{SCIPY_REQUIREMENT} is still unavailable after automatic install. "
            "Use 'rebuild env' for SA3 and try again."
        )


def build_suggestion(bpm: int | None, keyscale: str) -> str:
    """Join bpm/key into the bare, comma-separated tail gary4juce uses."""
    parts = []
    if bpm is not None:
        parts.append(f"{bpm} bpm")
    if keyscale:
        parts.append(keyscale)
    return ", ".join(parts)


def analyze(audio_path: Path) -> dict:
    from bpm_analysis import choose_bpm, estimate_bpm
    from key_analysis import choose_key, estimate_key

    bpm_estimate = estimate_bpm(audio_path)
    key_estimate = estimate_key(audio_path)

    # No LM captioning in SA3, so the local estimate is the only source. Drop the
    # key confidence gate: this is an explicit "suggest, I'll verify" action, so a
    # best guess the user can correct beats silently returning nothing.
    bpm_decision = choose_bpm(local_estimate=bpm_estimate)
    key_decision = choose_key(local_estimate=key_estimate, minimum_local_confidence=0.0)

    return {
        "ok": True,
        "bpm": bpm_decision.bpm,
        "keyscale": key_decision.keyscale,
        "bpm_source": bpm_decision.source,
        "key_source": key_decision.source,
        "bpm_confidence": round(bpm_estimate.confidence, 4) if bpm_estimate else None,
        "key_confidence": round(key_estimate.confidence, 4) if key_estimate else None,
        "suggestion": build_suggestion(bpm_decision.bpm, key_decision.keyscale),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(description="Suggest BPM/key for an SA3 sidecar.")
    parser.add_argument("audio_path", help="Path to the audio file to analyze")
    args = parser.parse_args()

    audio_path = Path(args.audio_path)
    if not audio_path.is_file():
        print(json.dumps({"ok": False, "error": f"file not found: {audio_path}"}))
        return 1

    try:
        ensure_scipy()
        result = analyze(audio_path)
    except Exception as exc:  # surface a clean JSON error rather than a traceback
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
