#!/usr/bin/env python3
"""Smoke-test SA3 loop generation through the localhost API."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path


def request_json(method: str, url: str, payload: dict | None = None, timeout: int = 30) -> dict:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = {"error": text}
        parsed["http_status"] = exc.code
        return parsed


def wav_info(path: Path) -> dict:
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        return {
            "channels": channels,
            "frames": frames,
            "sample_rate": sample_rate,
            "duration": frames / float(sample_rate),
        }


def run_once(args: argparse.Namespace, run_index: int) -> dict:
    payload = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "bpm": args.bpm,
        "bars": args.bars,
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "sampler_type": args.sampler,
        "seed": args.seed,
    }
    print(f"\n[sa3-smoke] run={run_index} POST /generate/loop")
    accepted = request_json("POST", f"{args.base_url}/generate/loop", payload, timeout=30)
    if not accepted.get("success"):
        raise RuntimeError(f"generate/loop failed to start: {json.dumps(accepted, indent=2)}")

    session_id = accepted["session_id"]
    expected_duration = accepted["loop_duration"]
    print(
        "[sa3-smoke] session={session} seed={seed} bars={bars} bpm={bpm} "
        "loop={loop:.3f}s gen={gen:.3f}s".format(
            session=session_id,
            seed=accepted["seed"],
            bars=accepted["bars"],
            bpm=accepted["bpm"],
            loop=accepted["loop_duration"],
            gen=accepted["gen_duration"],
        )
    )

    started = time.perf_counter()
    last_line = ""
    while True:
        status = request_json("GET", f"{args.base_url}/poll_status/{session_id}", timeout=30)
        elapsed = time.perf_counter() - started
        line = (
            f"[sa3-smoke] t={elapsed:7.1f}s status={status.get('status')} "
            f"progress={status.get('progress')} step={status.get('step', '-')}"
        )
        if line != last_line:
            print(line, flush=True)
            last_line = line

        if status.get("status") == "completed":
            audio_data = status.get("audio_data") or ""
            if not audio_data:
                raise RuntimeError("completed response did not include audio_data")
            audio_bytes = base64.b64decode(audio_data)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.output_dir / (
                f"sa3_smoke_4bar_120bpm_run{run_index}_{int(time.time())}.wav"
            )
            out_path.write_bytes(audio_bytes)
            info = wav_info(out_path)
            meta = status.get("meta", {})
            measured = info["duration"]
            drift = measured - expected_duration
            print(f"[sa3-smoke] wrote={out_path}")
            print(
                "[sa3-smoke] measured={measured:.6f}s expected={expected:.6f}s "
                "drift={drift:+.6f}s sr={sr} channels={channels} bytes={bytes_len}".format(
                    measured=measured,
                    expected=expected_duration,
                    drift=drift,
                    sr=info["sample_rate"],
                    channels=info["channels"],
                    bytes_len=len(audio_bytes),
                )
            )
            print(
                "[sa3-smoke] elapsed={elapsed:.3f}s api_generation_seconds={api}".format(
                    elapsed=elapsed,
                    api=meta.get("generation_seconds"),
                )
            )
            return {
                "path": str(out_path),
                "elapsed": elapsed,
                "api_generation_seconds": meta.get("generation_seconds"),
                "duration": measured,
                "expected_duration": expected_duration,
                "sample_rate": info["sample_rate"],
                "channels": info["channels"],
                "seed": meta.get("seed", accepted["seed"]),
            }

        if status.get("status") == "failed" or status.get("success") is False:
            raise RuntimeError(f"generation failed: {json.dumps(status, indent=2)}")

        if elapsed > args.timeout:
            raise TimeoutError(f"timed out after {args.timeout}s waiting for {session_id}")

        time.sleep(args.poll_interval)


def default_output_dir() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "Gary4JUCE-ROCm" / "sa3" / "outputs"
    return Path.cwd() / "smoke-tests" / "outputs"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a 4-bar 120 BPM SA3 loop through localhost.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8006")
    parser.add_argument("--prompt", default="tight electronic drum and bass loop, four bars, 120 bpm, clean mix")
    parser.add_argument("--negative-prompt", default="low quality, distorted, vocals")
    parser.add_argument("--bpm", type=float, default=120.0)
    parser.add_argument("--bars", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--sampler", default="pingpong")
    parser.add_argument("--seed", type=int, default=5070)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    args = parser.parse_args()

    health = request_json("GET", f"{args.base_url}/health", timeout=10)
    print(f"[sa3-smoke] health={json.dumps(health, sort_keys=True)}")
    if health.get("status") != "healthy":
        raise RuntimeError(f"SA3 service is not healthy at {args.base_url}")

    results = [run_once(args, index + 1) for index in range(args.repeat)]
    print("\n[sa3-smoke] summary")
    for index, result in enumerate(results, 1):
        print(
            "  run={index} elapsed={elapsed:.3f}s api={api}s duration={duration:.6f}s file={path}".format(
                index=index,
                elapsed=result["elapsed"],
                api=result["api_generation_seconds"],
                duration=result["duration"],
                path=result["path"],
            )
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"[sa3-smoke] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
