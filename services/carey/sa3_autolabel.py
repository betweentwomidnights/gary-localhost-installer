#!/usr/bin/env python3
"""Auto-label an SA3 dataset folder from ACE-Step's understand_music.

Runs in the Carey env (needs ace-step, the caption LM, and scipy). Reuses the exact
captioning machinery from ``train_lora_job.py`` -- the temporary LM-only ACE server,
the per-track analysis request, and the bpm/key reconciliation -- but writes plain
SA3 ``.txt`` sidecars using only the genre plus the local helpers' bpm/key. The
caption, lyrics, language, and everything else understand_music returns are dropped.

Progress and cancellation use the same status.json / cancel-file protocol as the
training job, so the Tauri layer polls and cancels it the same way. Existing sidecars
are overwritten.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SERVICE_DIR = Path(__file__).resolve().parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from train_lora_job import (  # noqa: E402 - path is set up above
    Cancelled,
    check_cancel,
    decide_sidecar_bpm,
    decide_sidecar_key,
    ensure_carey_model_loaded,
    ensure_carey_stopped,
    request_music_analysis,
    start_caption_server,
    stop_caption_server,
    update_status,
    wait_for_carey,
)

# Mirror SA3's dataset audio discovery (is_sa3_dataset_audio_file in the Rust side).
AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a")


def discover_sa3_audio(dataset_dir: Path) -> list[Path]:
    """Recurse the dataset, sorted by relative path lowercase to match the modal."""
    files = [
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    files.sort(key=lambda path: path.relative_to(dataset_dir).as_posix().lower())
    return files


def format_sidecar(style: str, genre: str, bpm: int | None, keyscale: str) -> str:
    """Assemble an SA3 sidecar in barebones or official-SA3 style."""
    genre = (genre or "").strip()
    keyscale = (keyscale or "").strip()
    bpm_text = f"{int(bpm)} bpm" if bpm else ""

    if style == "labeled":
        parts = ["TrackType: Music", "VocalType: Instrumental"]
        if genre:
            parts.append(f"Genre: {genre}")
        if bpm:
            parts.append(f"BPM: {int(bpm)}")
        if keyscale:
            parts.append(f"Key: {keyscale}")
        return ", ".join(parts)

    # barebones: "genre, 145 bpm, C minor"
    return ", ".join(part for part in (genre, bpm_text, keyscale) if part)


def build_reuse_args(cli: argparse.Namespace) -> SimpleNamespace:
    """A namespace compatible with the reused train_lora_job caption helpers.

    Fields read directly (not via getattr) must all be present: the status/cancel
    paths, the carey endpoint, the caption LM, and the DiT model key.
    """
    return SimpleNamespace(
        job_id=cli.job_id,
        name=cli.name or "sa3-autolabel",
        run_dir=Path(cli.run_dir),
        log_path=Path(cli.log_path),
        cancel_path=Path(cli.cancel_path),
        status_path=Path(cli.status_path),
        current_job_path=Path(cli.current_job_path),
        dataset_dir=Path(cli.dataset_dir),
        carey_url=cli.carey_url,
        inference_carey_url="http://127.0.0.1:8003",
        caption_lm_model=cli.caption_lm_model,
        model=cli.model,
        caption_timeout=cli.caption_timeout,
        caption_startup_timeout=cli.caption_startup_timeout,
        model_load_timeout=cli.model_load_timeout,
        carey_stop_timeout=cli.carey_stop_timeout,
        caption_window_seconds=cli.caption_window_seconds,
        caption_fallback_window_seconds=120.0,
        analysis_duration=cli.analysis_duration,
        overwrite_captions=True,
        # bpm/key reconciliation: understand_music supplies the LM value, the local
        # helpers sanity-check and override it. These mirror the training defaults.
        bpm_analysis=True,
        key_analysis=True,
        bpm_disagreement_threshold=5.0,
        bpm_min_confidence=1.2,
        key_min_confidence=0.15,
        instrumental=True,
        trigger="",
    )


def parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-label an SA3 dataset folder.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--style", choices=("bare", "labeled"), default="bare")
    parser.add_argument("--caption-lm-model", default="acestep-5Hz-lm-1.7B")
    parser.add_argument("--model", default="base", help="ACE DiT config key for analysis")
    parser.add_argument("--carey-url", default="http://127.0.0.1:8013")
    parser.add_argument("--job-id", default="sa3-autolabel")
    parser.add_argument("--name", default="sa3-autolabel")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--status-path", required=True)
    parser.add_argument("--cancel-path", required=True)
    parser.add_argument("--current-job-path", required=True)
    parser.add_argument("--caption-timeout", type=float, default=900.0)
    parser.add_argument("--caption-startup-timeout", type=float, default=900.0)
    parser.add_argument("--model-load-timeout", type=float, default=900.0)
    parser.add_argument("--carey-stop-timeout", type=float, default=180.0)
    parser.add_argument("--caption-window-seconds", type=float, default=0.0)
    parser.add_argument("--analysis-duration", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    cli = parse_cli(argv)
    args = build_reuse_args(cli)
    dataset_dir = Path(cli.dataset_dir)

    files = discover_sa3_audio(dataset_dir)
    total = len(files)
    update_status(
        args,
        status="running",
        phase="starting",
        message="Preparing auto-label",
        total=total,
        done=0,
        currentPath="",
        style=cli.style,
    )
    if not files:
        update_status(args, status="completed", phase="done", message="No audio files found", total=0, done=0)
        return 0

    import httpx

    server = None
    try:
        ensure_carey_stopped(args)
        server = start_caption_server(args)
        with httpx.Client(timeout=httpx.Timeout(args.caption_timeout)) as client:
            wait_for_carey(args, client, server)
            update_status(args, phase="loading-model", message="Loading caption models")
            ensure_carey_model_loaded(args, client)

            done = 0
            for audio_path in files:
                check_cancel(args)
                update_status(
                    args,
                    status="running",
                    phase="analyzing",
                    message=f"Analyzing {audio_path.name}",
                    currentPath=str(audio_path),
                    done=done,
                    total=total,
                )
                result: dict[str, Any] = request_music_analysis(
                    args,
                    client,
                    audio_path,
                    caption_window_seconds=args.caption_window_seconds,
                )
                genre = str(result.get("genre") or result.get("genres") or "")
                bpm_decision = decide_sidecar_bpm(args, audio_path, result)
                key_decision = decide_sidecar_key(args, audio_path, result)
                text = format_sidecar(cli.style, genre, bpm_decision.bpm, key_decision.keyscale)
                if text.strip():
                    audio_path.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
                done += 1
                update_status(args, done=done, currentPath="")
                print(
                    f"[autolabel] {done}/{total} {audio_path.name} -> {text or '(empty)'} "
                    f"[genre={genre or 'n/a'} bpm={bpm_decision.bpm or 'n/a'}({bpm_decision.source}) "
                    f"key={key_decision.keyscale or 'n/a'}({key_decision.source})]",
                    flush=True,
                )
    except Cancelled:
        update_status(args, status="cancelled", phase="cancelled", message="Auto-label cancelled")
        print("[autolabel] cancelled", flush=True)
        return 1
    except Exception as exc:  # surface a clean status; the log has the traceback
        update_status(args, status="failed", phase="error", message=str(exc), error=str(exc))
        print(f"[autolabel] error: {exc}", flush=True)
        return 1
    finally:
        if server is not None:
            stop_caption_server(args, server)

    update_status(
        args,
        status="completed",
        phase="done",
        message=f"Auto-labeled {total} track{'' if total == 1 else 's'}",
        done=total,
        currentPath="",
    )
    print(f"[autolabel] done: {total} track(s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
