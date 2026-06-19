#!/usr/bin/env python3
"""Create a paper-style A/B result table with mean +/- SD over 5 folds."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


LANGUAGE_ORDER = {
    "Mongolic": ["Daur", "Kalmyk", "Khalkha", "Ordos", "Tu"],
    "Tungusic": [
        "Even",
        "Evenki",
        "Manchu",
        "Nanai",
        "Negidal",
        "Oroch",
        "Udihe",
        "Ulch",
    ],
}

FOLDS = {"0", "1", "2", "3", "4"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize experiment A/B LAS, UAS, and UPOS as mean +/- SD over folds."
    )
    parser.add_argument(
        "--mono-parse-file",
        default=Path("experiments_0616/results/mono_by_fold.csv"),
        type=Path,
    )
    parser.add_argument(
        "--joint-parse-file",
        default=Path("experiments_0616/results/joint_by_fold.csv"),
        type=Path,
    )
    parser.add_argument(
        "--mono-upos-file",
        default=Path("experiments_0616/results/upos_mono_by_fold.csv"),
        type=Path,
    )
    parser.add_argument(
        "--joint-upos-file",
        default=Path("experiments_0616/results/upos_joint_by_fold.csv"),
        type=Path,
    )
    parser.add_argument(
        "--csv-output",
        default=Path("experiments_0616/results/ab_mean_sd_table.csv"),
        type=Path,
    )
    parser.add_argument(
        "--markdown-output",
        default=Path("experiments_0616/results/ab_mean_sd_table.md"),
        type=Path,
    )
    parser.add_argument(
        "--bold-best",
        action="store_true",
        help="Bold the better mono/joint score for each metric in the Markdown table.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing result file: {path}")
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def index_by_language(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    duplicates: list[tuple[str, str, str]] = []

    for row in rows:
        key = (row["family"], row["language"], row["fold"])
        if key in seen:
            duplicates.append(key)
        seen.add(key)
        grouped[(row["family"], row["language"])].append(row)

    if duplicates:
        formatted = ", ".join(f"{family}/{language}/fold{fold}" for family, language, fold in duplicates)
        raise SystemExit(f"Duplicate language-fold rows found: {formatted}")

    for (family, language), items in grouped.items():
        folds = {row["fold"] for row in items}
        if folds != FOLDS:
            missing = ", ".join(sorted(FOLDS - folds))
            extra = ", ".join(sorted(folds - FOLDS))
            detail = []
            if missing:
                detail.append(f"missing folds: {missing}")
            if extra:
                detail.append(f"extra folds: {extra}")
            raise SystemExit(f"{family}/{language} does not have exactly folds 0-4 ({'; '.join(detail)})")

    return grouped


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def format_mean_sd(values: list[float]) -> str:
    mean, sd = mean_sd(values)
    return f"{mean:.2f}±{sd:.2f}"


def mean_value(values: list[float]) -> float:
    return statistics.mean(values)


def values_for(grouped: dict[tuple[str, str], list[dict[str, str]]], family: str, language: str, metric: str) -> list[float]:
    rows = grouped[(family, language)]
    return [float(row[metric]) for row in sorted(rows, key=lambda row: int(row["fold"]))]


def token_total(grouped: dict[tuple[str, str], list[dict[str, str]]], family: str, language: str) -> int:
    rows = grouped[(family, language)]
    return sum(int(row["tokens"]) for row in rows)


def maybe_bold(value: str, should_bold: bool) -> str:
    return f"**{value}**" if should_bold else value


def build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    mono_parse = index_by_language(read_rows(args.mono_parse_file))
    joint_parse = index_by_language(read_rows(args.joint_parse_file))
    mono_upos = index_by_language(read_rows(args.mono_upos_file))
    joint_upos = index_by_language(read_rows(args.joint_upos_file))

    expected = {(family, language) for family, languages in LANGUAGE_ORDER.items() for language in languages}
    actual_sets = [set(index) for index in [mono_parse, joint_parse, mono_upos, joint_upos]]
    for actual in actual_sets:
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise SystemExit(f"Unexpected language set. Missing={missing}; extra={extra}")

    output_rows = []
    for family, languages in LANGUAGE_ORDER.items():
        for language in languages:
            mono_las = values_for(mono_parse, family, language, "las")
            joint_las = values_for(joint_parse, family, language, "las")
            mono_uas = values_for(mono_parse, family, language, "uas")
            joint_uas = values_for(joint_parse, family, language, "uas")
            mono_pos = values_for(mono_upos, family, language, "upos")
            joint_pos = values_for(joint_upos, family, language, "upos")

            row = {
                "Family": family,
                "Language": language,
                "Tokens": str(token_total(mono_parse, family, language)),
                "Mono LAS": format_mean_sd(mono_las),
                "Joint LAS": format_mean_sd(joint_las),
                "Mono UAS": format_mean_sd(mono_uas),
                "Joint UAS": format_mean_sd(joint_uas),
                "Mono UPOS": format_mean_sd(mono_pos),
                "Joint UPOS": format_mean_sd(joint_pos),
                "_mono_las_mean": f"{mean_value(mono_las):.10f}",
                "_joint_las_mean": f"{mean_value(joint_las):.10f}",
                "_mono_uas_mean": f"{mean_value(mono_uas):.10f}",
                "_joint_uas_mean": f"{mean_value(joint_uas):.10f}",
                "_mono_upos_mean": f"{mean_value(mono_pos):.10f}",
                "_joint_upos_mean": f"{mean_value(joint_pos):.10f}",
            }
            output_rows.append(row)

    return output_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "Family",
        "Language",
        "Tokens",
        "Mono LAS",
        "Joint LAS",
        "Mono UAS",
        "Joint UAS",
        "Mono UPOS",
        "Joint UPOS",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_markdown(path: Path, rows: list[dict[str, str]], bold_best: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Family",
        "Language",
        "Tokens",
        "Mono LAS",
        "Joint LAS",
        "Mono UAS",
        "Joint UAS",
        "Mono UPOS",
        "Joint UPOS",
    ]
    lines = [
        "# Experiment A/B Results",
        "",
        "Dependency parsing and POS tagging results (mean ± SD over 5 folds).",
        "",
        "|" + "|".join(headers) + "|",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]

    previous_family = None
    for row in rows:
        display = dict(row)
        if bold_best:
            for metric in ["las", "uas", "upos"]:
                mono_key = f"Mono {metric.upper()}"
                joint_key = f"Joint {metric.upper()}"
                mono_mean = float(row[f"_mono_{metric}_mean"])
                joint_mean = float(row[f"_joint_{metric}_mean"])
                display[mono_key] = maybe_bold(row[mono_key], mono_mean >= joint_mean)
                display[joint_key] = maybe_bold(row[joint_key], joint_mean > mono_mean)

        family = display["Family"] if display["Family"] != previous_family else ""
        previous_family = display["Family"]
        values = [
            family,
            display["Language"],
            display["Tokens"],
            display["Mono LAS"],
            display["Joint LAS"],
            display["Mono UAS"],
            display["Joint UAS"],
            display["Mono UPOS"],
            display["Joint UPOS"],
        ]
        lines.append("|" + "|".join(values) + "|")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = build_rows(args)
    write_csv(args.csv_output, rows)
    write_markdown(args.markdown_output, rows, args.bold_best)
    print(f"Wrote CSV table to {args.csv_output}")
    print(f"Wrote Markdown table to {args.markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
