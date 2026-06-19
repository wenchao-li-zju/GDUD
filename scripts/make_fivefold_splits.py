#!/usr/bin/env python3
"""Create fixed 5-fold train/dev/test splits for GDUD CoNLL-U files."""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path


TOKEN_ID_RE = re.compile(r"^[0-9]+$")


def read_sentences(path: Path) -> list[str]:
    """Read a CoNLL-U file as sentence blocks, preserving block text."""
    text = path.read_text(encoding="utf-8")
    blocks = []
    current = []

    for line in text.splitlines(keepends=True):
        if line.strip():
            current.append(line)
            continue

        if current:
            blocks.append("".join(current).rstrip("\n") + "\n\n")
            current = []

    if current:
        blocks.append("".join(current).rstrip("\n") + "\n\n")

    return blocks


def count_tokens(sentence: str) -> int:
    tokens = 0
    for line in sentence.splitlines():
        if not line or line.startswith("#"):
            continue
        columns = line.split("\t")
        if len(columns) == 10 and TOKEN_ID_RE.match(columns[0]):
            tokens += 1
    return tokens


def write_sentences(path: Path, sentences: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(sentences), encoding="utf-8")


def make_folds(sentences: list[str], k: int, seed: int) -> list[list[str]]:
    shuffled = sentences[:]
    random.Random(seed).shuffle(shuffled)
    return [shuffled[i::k] for i in range(k)]


def split_language(
    src: Path,
    input_dir: Path,
    output_dir: Path,
    k: int,
    seed: int,
) -> list[dict[str, str | int]]:
    family = src.parent.name
    language = src.stem.removesuffix("_core")
    sentences = read_sentences(src)

    if len(sentences) < k:
        raise ValueError(f"{src} has only {len(sentences)} sentences; cannot create {k} folds")

    folds = make_folds(sentences, k, seed)
    rows: list[dict[str, str | int]] = []

    for fold_index in range(k):
        test = folds[fold_index]
        train_all = [
            sentence
            for index, fold in enumerate(folds)
            if index != fold_index
            for sentence in fold
        ]

        random.Random(f"{seed}:{family}:{language}:fold{fold_index}:dev").shuffle(train_all)
        dev_size = max(1, len(train_all) // 10)
        dev = train_all[:dev_size]
        train = train_all[dev_size:]

        split_dir = output_dir / family / language / f"fold{fold_index}"
        write_sentences(split_dir / "train.conllu", train)
        write_sentences(split_dir / "dev.conllu", dev)
        write_sentences(split_dir / "test.conllu", test)

        rows.append(
            {
                "family": family,
                "language": language,
                "source_file": str(src.relative_to(input_dir)),
                "fold": fold_index,
                "train_sentences": len(train),
                "dev_sentences": len(dev),
                "test_sentences": len(test),
                "train_tokens": sum(count_tokens(sentence) for sentence in train),
                "dev_tokens": sum(count_tokens(sentence) for sentence in dev),
                "test_tokens": sum(count_tokens(sentence) for sentence in test),
            }
        )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create fixed 5-fold train/dev/test splits from core CoNLL-U files."
    )
    parser.add_argument(
        "--input-dir",
        default="experiments_0616/data_core",
        type=Path,
        help="Directory containing family subdirectories with *_core.conllu files.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments_0616/splits/mono",
        type=Path,
        help="Directory where split files will be written.",
    )
    parser.add_argument(
        "--manifest",
        default="experiments_0616/splits/mono_manifest.csv",
        type=Path,
        help="CSV file summarizing split sentence/token counts.",
    )
    parser.add_argument("--folds", default=5, type=int, help="Number of folds.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = sorted(args.input_dir.rglob("*_core.conllu"))

    if not files:
        raise SystemExit(f"No *_core.conllu files found under {args.input_dir}")

    rows: list[dict[str, str | int]] = []
    for src in files:
        language_rows = split_language(src, args.input_dir, args.output_dir, args.folds, args.seed)
        rows.extend(language_rows)

        first_row = language_rows[0]
        print(
            f"{first_row['family']}/{first_row['language']}: "
            f"{sum(row['test_sentences'] for row in language_rows)} sentences, "
            f"{args.folds} folds"
        )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote splits for {len(files)} languages to {args.output_dir}")
    print(f"Wrote manifest to {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
