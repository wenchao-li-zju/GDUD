#!/usr/bin/env python3
"""Create core-relation CoNLL-U files by removing DEPREL subtypes.

Example:
    python scripts/make_core_deprels.py \
        --input-dir treebanks_amended_0608 \
        --output-dir treebanks_core_0608
"""

from __future__ import annotations

import argparse
from pathlib import Path


def core_relation(label: str) -> str:
    """Return the UD core relation by removing any subtype after ':'."""
    if not label or label == "_":
        return label
    return label.split(":", 1)[0]


def core_deps(deps: str) -> str:
    """Remove relation subtypes in the DEPS column while preserving heads."""
    if not deps or deps == "_":
        return deps

    items = []
    for item in deps.split("|"):
        parts = item.split(":", 1)
        if len(parts) == 1:
            items.append(item)
        else:
            head, rel = parts
            items.append(f"{head}:{core_relation(rel)}")
    return "|".join(items)


def convert_file(src: Path, dst: Path, convert_enhanced: bool) -> tuple[int, int]:
    """Convert one CoNLL-U file and return changed DEPREL/DEPS counts."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    changed_deprel = 0
    changed_deps = 0

    with src.open(encoding="utf-8") as in_file, dst.open("w", encoding="utf-8") as out_file:
        for line in in_file:
            if not line.strip() or line.startswith("#"):
                out_file.write(line)
                continue

            columns = line.rstrip("\n").split("\t")
            if len(columns) != 10:
                out_file.write(line)
                continue

            new_deprel = core_relation(columns[7])
            if new_deprel != columns[7]:
                changed_deprel += 1
                columns[7] = new_deprel

            if convert_enhanced:
                new_deps = core_deps(columns[8])
                if new_deps != columns[8]:
                    changed_deps += 1
                    columns[8] = new_deps

            out_file.write("\t".join(columns) + "\n")

    return changed_deprel, changed_deps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-convert CoNLL-U DEPREL subtypes to UD core relations."
    )
    parser.add_argument(
        "--input-dir",
        default="treebanks_amended_0608",
        type=Path,
        help="Directory containing amended .conllu files.",
    )
    parser.add_argument(
        "--output-dir",
        default="treebanks_core_0608",
        type=Path,
        help="Directory where converted .conllu files will be written.",
    )
    parser.add_argument(
        "--convert-enhanced",
        action="store_true",
        help="Also remove subtypes in the DEPS column.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = sorted(args.input_dir.rglob("*.conllu"))

    if not files:
        raise SystemExit(f"No .conllu files found under {args.input_dir}")

    total_deprel = 0
    total_deps = 0

    for src in files:
        rel_path = src.relative_to(args.input_dir)
        dst = args.output_dir / rel_path
        changed_deprel, changed_deps = convert_file(src, dst, args.convert_enhanced)
        total_deprel += changed_deprel
        total_deps += changed_deps
        print(
            f"{rel_path}: DEPREL {changed_deprel}"
            + (f", DEPS {changed_deps}" if args.convert_enhanced else "")
        )

    print(f"\nWrote {len(files)} files to {args.output_dir}")
    print(f"Total DEPREL changes: {total_deprel}")
    if args.convert_enhanced:
        print(f"Total DEPS changes: {total_deps}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
