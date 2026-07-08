#!/usr/bin/env python3
"""Build an SA3 prompt dice pool from a LoRA training caption folder."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


# bpm and key are re-added by gary4juce from its own dropdowns before a request is
# submitted, so they must not leak into the dice pool (that would double them at
# inference). The plugin writes them bare ("145 bpm", "C minor"), while the official
# SA3 / Underfit guides suggest labeled forms ("BPM: 145", "Key: C minor"); we strip
# either style. The match is anchored to the tail and peeled repeatedly, so bpm and
# key are removed in any order.
_NOTE = r"[A-G][#b♯♭]?"
_MODE = r"(?:maj(?:or)?|min(?:or)?)"
_TRAILING_TAG = re.compile(
    r"[,;]?\s*(?:"
    r"bpm\s*[:=]?\s*\d+(?:\.\d+)?"  # BPM: 145 / bpm 145
    r"|\d+(?:\.\d+)?\s*bpm"  # 145 bpm
    rf"|(?:key|scale)\s*[:=]\s*{_NOTE}\s+{_MODE}"  # Key: C minor / Scale: F# maj
    rf"|(?<![A-Za-z]){_NOTE}\s+(?:major|minor)"  # bare C minor / F# major
    r")\s*$",
    re.IGNORECASE,
)


def prompt_from_caption(text: str) -> str:
    """Strip trailing bpm/key tags the host re-adds from its own controls.

    Peels any trailing bpm/key token (labeled or bare) until the tail is stable, so
    a caption ending in e.g. "..., 145 bpm, C minor" loses both regardless of order.
    """
    prompt = text.strip()
    while True:
        stripped = _TRAILING_TAG.sub("", prompt).strip(" ,;\t\r\n")
        if stripped == prompt:
            return prompt
        prompt = stripped


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="LoRA registry name")
    parser.add_argument("--captions-dir", required=True, help="Folder with SA3 training .txt sidecars")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts"),
        help="Prompt JSON output directory",
    )
    parser.add_argument("--bucket", default="instrumental", help="Prompt dice bucket")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing prompt JSON")
    args = parser.parse_args()

    if not os.path.isdir(args.captions_dir):
        sys.exit(f"captions-dir not found: {args.captions_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.name}.json")
    if os.path.exists(out_path) and not args.force:
        sys.exit(f"{out_path} exists; refusing to clobber curated prompts (use --force)")

    seen = set()
    prompts: list[str] = []
    captions_dir = Path(args.captions_dir)
    txts = sorted(
        captions_dir.rglob("*.txt"),
        key=lambda path: path.relative_to(captions_dir).as_posix().lower(),
    )
    for filename in txts:
        with filename.open(encoding="utf-8-sig", errors="replace") as handle:
            prompt = prompt_from_caption(handle.read())
        if not prompt:
            continue
        key = prompt.lower()
        if key in seen:
            continue
        seen.add(key)
        prompts.append(prompt)

    payload = {
        "version": 1,
        "source": {
            "lora": args.name,
            "captions_dir": args.captions_dir,
            "files": len(txts),
            "unique_prompts": len(prompts),
        },
        "dice": {args.bucket: prompts},
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(
        f"wrote {out_path}: {len(prompts)} unique prompts from {len(txts)} captions "
        f"-> dice.{args.bucket}"
    )
    if prompts:
        suffix = " ..." if len(prompts) > 12 else ""
        print("  " + " | ".join(prompts[:12]) + suffix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
