"""Compare native metadata suggestions with the pre-migration Python helper.

Run with an existing SA3 Python environment (numpy/scipy/soundfile). No models or
GPU are used. WAV files and sidecars are never changed; results go to --output.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "sa3"))
from analyze_audio import analyze


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted((path for path in args.dataset.rglob("*") if path.is_file() and path.suffix.lower() == ".wav"), key=lambda path: str(path).casefold())
    if not files:
        parser.error("dataset contains no WAV files")
    if args.output.exists():
        parser.error("output must be a new result file")
    rows = []
    failed = False
    for path in files:
        before = digest(path)
        reference = analyze(path)
        result = subprocess.run([str(args.tool.resolve()), "--in", str(path.resolve())], capture_output=True, text=True, encoding="utf-8", check=True)
        native = json.loads(result.stdout)
        matches = all(native[field] == reference[field] for field in ["bpm", "keyscale", "suggestion", "bpm_source", "key_source"])
        for field in ["bpm_confidence", "key_confidence"]:
            a, b = native[field], reference[field]
            matches &= (a == b) if a is None or b is None else math.isclose(a, b, rel_tol=1e-4, abs_tol=0.0002)
        unchanged = before == digest(path)
        failed |= not (matches and unchanged)
        rows.append({"file":str(path),"sourceSha256":before,"unchanged":unchanged,"matches":matches,"reference":reference,"native":native})
        print(f"{len(rows)}/{len(files)}: {path.name}: {'PASS' if matches and unchanged else 'FAIL'}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"{sum(row['matches'] and row['unchanged'] for row in rows)}/{len(rows)} matched with unchanged source hashes")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
