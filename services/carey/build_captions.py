#!/usr/bin/env python3
"""Build captions.json for the /captions endpoint.

Reads sidecar .txt files from LoRA training dirs and example JSONs from
ACE-Step, extracting only the caption text. Output is a single JSON file
keyed by pool name.

Usage:
    python build_captions.py \
        --lora koan:/home/kev/saos_training \
        --default /home/kev/ace/ACE-Step-1.5/examples/text2music \
        -o captions.json
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path


def parse_sidecar(path: Path) -> tuple[str | None, str | None]:
    """Extract (caption, genre) from a sidecar .txt file."""
    caption, genre = None, None
    for line in path.read_text().splitlines():
        if line.startswith("caption:") and caption is None:
            caption = line.split("caption:", 1)[1].strip() or None
        elif line.startswith("genre:") and genre is None:
            genre = line.split("genre:", 1)[1].strip() or None
        if caption and genre:
            break
    return caption, genre


def parse_example_json(path: Path) -> str | None:
    try:
        data = json.loads(path.read_text())
        return (data.get("caption") or "").strip() or None
    except (json.JSONDecodeError, OSError):
        return None


def genre_variants(genre: str) -> list[str]:
    """Expand a comma-separated genre line into the full line plus 2- and
    3-token subsets drawn from that same line. Subsets only mix tokens from
    one line at a time so unrelated styles never cross-pollinate."""
    tokens = [t.strip() for t in genre.split(",") if t.strip()]
    if not tokens:
        return []
    variants = [", ".join(tokens)]
    for size in (2, 3):
        if len(tokens) >= size:
            for combo in combinations(tokens, size):
                variants.append(", ".join(combo))
    return variants


def collect_sidecars(directory: Path) -> list[str]:
    """Return deduped pool of full captions + genre-tag strings.

    `caption:` lines feed the dice button as long prose prompts. `genre:`
    lines are split into tokens and expanded into 2- and 3-token subsets
    (plus the full line) so users get quick stylistic re-rolls that match
    how the LoRA was trained.
    """
    seen: set[str] = set()
    pool: list[str] = []
    for f in sorted(directory.glob("*.txt")):
        if f.suffix != ".txt" or ".v4bak" in f.name or f.name.endswith(".v4bak"):
            continue
        caption, genre = parse_sidecar(f)
        entries = []
        if caption:
            entries.append(caption)
        if genre:
            entries.extend(genre_variants(genre))
        for entry in entries:
            if entry not in seen:
                seen.add(entry)
                pool.append(entry)
    return pool


def collect_examples(directory: Path) -> list[str]:
    captions = []
    for f in sorted(directory.glob("*.json")):
        cap = parse_example_json(f)
        if cap:
            captions.append(cap)
    return captions


def main():
    parser = argparse.ArgumentParser(description="Build captions.json")
    parser.add_argument(
        "--lora", action="append", default=[],
        help="name:path pair, e.g. koan:/home/kev/saos_training",
    )
    parser.add_argument(
        "--default",
        help="Path to ACE-Step examples/text2music directory",
    )
    parser.add_argument("-o", "--output", default="captions.json")
    args = parser.parse_args()

    pools: dict[str, list[str]] = {}

    for entry in args.lora:
        if ":" not in entry:
            print(f"Error: --lora must be name:path, got '{entry}'", file=sys.stderr)
            sys.exit(1)
        name, path_str = entry.split(":", 1)
        path = Path(path_str)
        if not path.is_dir():
            print(f"Error: {path} is not a directory", file=sys.stderr)
            sys.exit(1)
        captions = collect_sidecars(path)
        print(f"  {name}: {len(captions)} captions from {path}")
        pools[name] = captions

    if args.default:
        path = Path(args.default)
        if not path.is_dir():
            print(f"Error: {path} is not a directory", file=sys.stderr)
            sys.exit(1)
        captions = collect_examples(path)
        print(f"  default: {len(captions)} captions from {path}")
        pools["default"] = captions

    out = Path(args.output)
    out.write_text(json.dumps(pools, indent=2, ensure_ascii=False))
    print(f"Wrote {out} ({sum(len(v) for v in pools.values())} total captions)")


if __name__ == "__main__":
    main()
