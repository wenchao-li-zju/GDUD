#!/usr/bin/env python3
"""Merge mono splits into family-level joint multilingual splits."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


TOKEN_ID_RE = re.compile(r"^[0-9]+$")


def count_sentences(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as in_file:
        for line in in_file:
            if line.startswith("# sent_id"):
                count += 1
    return count


def count_tokens(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as in_file:
        for line in in_file:
            if not line.strip() or line.startswith("#"):
                continue
            columns = line.rstrip("\n").split("\t")
            if len(columns) == 10 and TOKEN_ID_RE.match(columns[0]):
                count += 1
    return count


def append_files(inputs: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as out_file:
        for input_path in inputs:
            text = input_path.read_text(encoding="utf-8").rstrip()
            if text:
                out_file.write(text)
                out_file.write("\n\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-language mono splits into family-level joint splits."
    )
    parser.add_argument(
        "--mono-dir",
        default="experiments_0616/splits/mono",
        type=Path,
        help="Directory containing mono splits organized as family/language/foldN.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments_0616/splits/joint",
        type=Path,
        help="Directory where joint split files will be written.",
    )
    parser.add_argument(
        "--manifest",
        default="experiments_0616/splits/joint_manifest.csv",
        type=Path,
        help="CSV file summarizing joint split counts.",
    )
    parser.add_argument("--folds", default=5, type=int, help="Number of folds.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    families = sorted(path for path in args.mono_dir.iterdir() if path.is_dir())

    if not families:
        raise SystemExit(f"No family directories found under {args.mono_dir}")

    rows: list[dict[str, str | int]] = []

    for family_dir in families:
        family = family_dir.name
        languages = sorted(path.name for path in family_dir.iterdir() if path.is_dir())
        if not languages:
            continue

        for fold_index in range(args.folds):
            fold_name = f"fold{fold_index}"
            train_inputs = [
                family_dir / language / fold_name / "train.conllu" for language in languages
            ]
            dev_inputs = [
                family_dir / language / fold_name / "dev.conllu" for language in languages
            ]

            missing = [path for path in train_inputs + dev_inputs if not path.exists()]
            if missing:
                missing_text = "\n".join(str(path) for path in missing)
                raise SystemExit(f"Missing split files:\n{missing_text}")

            joint_dir = args.output_dir / family / fold_name
            train_output = joint_dir / "train.conllu"
            dev_output = joint_dir / "dev.conllu"

            append_files(train_inputs, train_output)
            append_files(dev_inputs, dev_output)

            rows.append(
                {
                    "family": family,
                    "fold": fold_index,
                    "languages": " ".join(languages),
                    "train_sentences": count_sentences(train_output),
                    "dev_sentences": count_sentences(dev_output),
                    "train_tokens": count_tokens(train_output),
                    "dev_tokens": count_tokens(dev_output),
                }
            )

        print(f"{family}: {len(languages)} languages, {args.folds} joint folds")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote joint splits to {args.output_dir}")
    print(f"Wrote manifest to {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
