#!/usr/bin/env python3
"""Summarize mono, joint, and transfer GDUD parsing results.

Outputs:
  - summary_mono_joint.csv: language-level mean/std and joint-minus-mono
  - mono_joint_by_fold.csv: fold-level aligned mono/joint comparison
  - final_summary.csv: language-level mono/joint/transfer summary
  - transfer_comparison_by_fold.csv: fold-level transfer-vs-mono comparison
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ("uas", "las")
NA = "NA"


def read_latest_rows(path: Path, expected_experiment: str | None = None) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing input file: {path}")

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"No rows found in {path}")

    latest: dict[tuple[str, str, str, int], dict[str, str]] = {}
    for row in rows:
        experiment = row["experiment"]
        if expected_experiment and experiment != expected_experiment:
            continue
        key = (experiment, row["family"], row["language"], int(row["fold"]))
        latest[key] = row

    return list(latest.values())


def mean(values: list[float]) -> float:
    return statistics.mean(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def format_float(value: float) -> str:
    return f"{value:.2f}"


def group_by_language(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["language"])].append(row)
    return grouped


def group_transfer_rows(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["source"], row["family"], row["language"])].append(row)
    return grouped


def build_fold_rows(
    mono_rows: list[dict[str, str]],
    joint_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    mono_by_key = {
        (row["family"], row["language"], int(row["fold"])): row for row in mono_rows
    }
    joint_by_key = {
        (row["family"], row["language"], int(row["fold"])): row for row in joint_rows
    }

    missing_joint = sorted(set(mono_by_key) - set(joint_by_key))
    missing_mono = sorted(set(joint_by_key) - set(mono_by_key))
    if missing_joint or missing_mono:
        if missing_joint:
            print("Missing joint rows for:")
            for key in missing_joint:
                print(f"  {key}")
        if missing_mono:
            print("Missing mono rows for:")
            for key in missing_mono:
                print(f"  {key}")
        raise SystemExit("Mono/joint result keys do not align.")

    fold_rows = []
    for family, language, fold in sorted(mono_by_key):
        mono = mono_by_key[(family, language, fold)]
        joint = joint_by_key[(family, language, fold)]
        row = {
            "family": family,
            "language": language,
            "fold": fold,
            "tokens": mono["tokens"],
            "mono_uas": mono["uas"],
            "mono_las": mono["las"],
            "joint_uas": joint["uas"],
            "joint_las": joint["las"],
            "delta_uas": format_float(float(joint["uas"]) - float(mono["uas"])),
            "delta_las": format_float(float(joint["las"]) - float(mono["las"])),
            "mono_prediction_file": mono["prediction_file"],
            "joint_prediction_file": joint["prediction_file"],
        }
        fold_rows.append(row)
    return fold_rows


def build_summary_rows(
    mono_rows: list[dict[str, str]],
    joint_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    mono_grouped = group_by_language(mono_rows)
    joint_grouped = group_by_language(joint_rows)

    missing_joint = sorted(set(mono_grouped) - set(joint_grouped))
    missing_mono = sorted(set(joint_grouped) - set(mono_grouped))
    if missing_joint or missing_mono:
        if missing_joint:
            print("Missing joint language summaries for:")
            for key in missing_joint:
                print(f"  {key}")
        if missing_mono:
            print("Missing mono language summaries for:")
            for key in missing_mono:
                print(f"  {key}")
        raise SystemExit("Mono/joint language sets do not align.")

    summary_rows = []
    for family, language in sorted(mono_grouped):
        mono_language_rows = sorted(mono_grouped[(family, language)], key=lambda row: int(row["fold"]))
        joint_language_rows = sorted(joint_grouped[(family, language)], key=lambda row: int(row["fold"]))

        mono_folds = [int(row["fold"]) for row in mono_language_rows]
        joint_folds = [int(row["fold"]) for row in joint_language_rows]
        if mono_folds != joint_folds:
            raise SystemExit(
                f"Fold mismatch for {family}/{language}: mono={mono_folds}, joint={joint_folds}"
            )

        row: dict[str, str | int] = {
            "family": family,
            "language": language,
            "folds": len(mono_folds),
            "tokens": sum(int(row["tokens"]) for row in mono_language_rows),
        }

        for metric in METRICS:
            mono_values = [float(result[metric]) for result in mono_language_rows]
            joint_values = [float(result[metric]) for result in joint_language_rows]
            deltas = [joint - mono for mono, joint in zip(mono_values, joint_values)]

            row[f"mono_{metric}_mean"] = format_float(mean(mono_values))
            row[f"mono_{metric}_std"] = format_float(sample_std(mono_values))
            row[f"joint_{metric}_mean"] = format_float(mean(joint_values))
            row[f"joint_{metric}_std"] = format_float(sample_std(joint_values))
            row[f"delta_{metric}_mean"] = format_float(mean(deltas))
            row[f"delta_{metric}_std"] = format_float(sample_std(deltas))
            row[f"joint_better_{metric}_folds"] = sum(delta > 0 for delta in deltas)

        summary_rows.append({key: str(value) for key, value in row.items()})

    return summary_rows


def build_transfer_fold_rows(
    mono_rows: list[dict[str, str]],
    transfer_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    mono_by_key = {
        (row["family"], row["language"], int(row["fold"])): row for row in mono_rows
    }

    fold_rows = []
    for transfer in sorted(
        transfer_rows,
        key=lambda row: (row["source"], row["family"], row["language"], int(row["fold"])),
    ):
        key = (transfer["family"], transfer["language"], int(transfer["fold"]))
        if key not in mono_by_key:
            raise SystemExit(
                "Missing mono row for transfer target "
                f"{transfer['source']} -> {transfer['family']}/{transfer['language']} "
                f"fold{transfer['fold']}"
            )

        mono = mono_by_key[key]
        row = {
            "source": transfer["source"],
            "family": transfer["family"],
            "language": transfer["language"],
            "fold": transfer["fold"],
            "tokens": transfer["tokens"],
            "mono_uas": mono["uas"],
            "mono_las": mono["las"],
            "transfer_uas": transfer["uas"],
            "transfer_las": transfer["las"],
            "delta_uas": format_float(float(transfer["uas"]) - float(mono["uas"])),
            "delta_las": format_float(float(transfer["las"]) - float(mono["las"])),
            "mono_prediction_file": mono["prediction_file"],
            "transfer_prediction_file": transfer["prediction_file"],
        }
        fold_rows.append(row)

    return fold_rows


def transfer_summary_lookup(
    transfer_rows: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    grouped = group_transfer_rows(transfer_rows)
    lookup: dict[tuple[str, str], dict[str, str]] = {}

    for (source, family, language), rows in grouped.items():
        rows = sorted(rows, key=lambda row: int(row["fold"]))
        summary: dict[str, str] = {
            "transfer_source": source,
            "transfer_folds": str(len(rows)),
        }

        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            summary[f"transfer_{metric}_mean"] = format_float(mean(values))
            summary[f"transfer_{metric}_std"] = format_float(sample_std(values))

        lookup[(family, language)] = summary

    return lookup


def build_final_summary_rows(
    mono_rows: list[dict[str, str]],
    joint_rows: list[dict[str, str]],
    transfer_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    rows = build_summary_rows(mono_rows, joint_rows)
    transfer_lookup = transfer_summary_lookup(transfer_rows)
    transfer_fold_rows = build_transfer_fold_rows(mono_rows, transfer_rows)
    transfer_deltas: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"uas": [], "las": []}
    )

    for row in transfer_fold_rows:
        key = (row["family"], row["language"])
        transfer_deltas[key]["uas"].append(float(row["delta_uas"]))
        transfer_deltas[key]["las"].append(float(row["delta_las"]))

    final_rows = []
    for row in rows:
        key = (row["family"], row["language"])
        transfer = transfer_lookup.get(key)

        final_row = dict(row)
        if transfer:
            final_row.update(transfer)
            for metric in METRICS:
                deltas = transfer_deltas[key][metric]
                final_row[f"transfer_delta_{metric}_mean"] = format_float(mean(deltas))
                final_row[f"transfer_delta_{metric}_std"] = format_float(sample_std(deltas))
                final_row[f"transfer_better_{metric}_folds"] = str(sum(delta > 0 for delta in deltas))
        else:
            final_row.update(
                {
                    "transfer_source": NA,
                    "transfer_folds": "0",
                    "transfer_uas_mean": NA,
                    "transfer_uas_std": NA,
                    "transfer_delta_uas_mean": NA,
                    "transfer_delta_uas_std": NA,
                    "transfer_better_uas_folds": NA,
                    "transfer_las_mean": NA,
                    "transfer_las_std": NA,
                    "transfer_delta_las_mean": NA,
                    "transfer_delta_las_std": NA,
                    "transfer_better_las_folds": NA,
                }
            )

        final_rows.append(final_row)

    return final_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise SystemExit(f"No rows to write for {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize mono, joint, and transfer results.")
    parser.add_argument(
        "--mono-file",
        default=Path("experiments_0616/results/mono_by_fold_dedup.csv"),
        type=Path,
    )
    parser.add_argument(
        "--joint-file",
        default=Path("experiments_0616/results/joint_by_fold.csv"),
        type=Path,
    )
    parser.add_argument(
        "--transfer-file",
        default=Path("experiments_0616/results/transfer_by_fold.csv"),
        type=Path,
    )
    parser.add_argument(
        "--summary-file",
        default=Path("experiments_0616/results/summary_mono_joint.csv"),
        type=Path,
    )
    parser.add_argument(
        "--fold-file",
        default=Path("experiments_0616/results/mono_joint_by_fold.csv"),
        type=Path,
    )
    parser.add_argument(
        "--final-summary-file",
        default=Path("experiments_0616/results/final_summary.csv"),
        type=Path,
    )
    parser.add_argument(
        "--transfer-fold-file",
        default=Path("experiments_0616/results/transfer_comparison_by_fold.csv"),
        type=Path,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mono_rows = read_latest_rows(args.mono_file, expected_experiment="mono")
    joint_rows = read_latest_rows(args.joint_file, expected_experiment="joint")
    transfer_rows = read_latest_rows(args.transfer_file, expected_experiment="transfer")

    fold_rows = build_fold_rows(mono_rows, joint_rows)
    summary_rows = build_summary_rows(mono_rows, joint_rows)
    transfer_fold_rows = build_transfer_fold_rows(mono_rows, transfer_rows)
    final_summary_rows = build_final_summary_rows(mono_rows, joint_rows, transfer_rows)

    write_csv(args.fold_file, fold_rows)
    write_csv(args.summary_file, summary_rows)
    write_csv(args.transfer_fold_file, transfer_fold_rows)
    write_csv(args.final_summary_file, final_summary_rows)

    print(f"Read {len(mono_rows)} mono rows from {args.mono_file}")
    print(f"Read {len(joint_rows)} joint rows from {args.joint_file}")
    print(f"Read {len(transfer_rows)} transfer rows from {args.transfer_file}")
    print(f"Wrote fold-level comparison to {args.fold_file}")
    print(f"Wrote language summary to {args.summary_file}")
    print(f"Wrote transfer fold-level comparison to {args.transfer_fold_file}")
    print(f"Wrote final summary to {args.final_summary_file}")
    print()
    print("Language summary:")
    for row in summary_rows:
        print(
            f"{row['family']}/{row['language']}: "
            f"Mono LAS {row['mono_las_mean']} +/- {row['mono_las_std']}, "
            f"Joint LAS {row['joint_las_mean']} +/- {row['joint_las_std']}, "
            f"Delta {row['delta_las_mean']} "
            f"({row['joint_better_las_folds']}/{row['folds']} folds)"
        )

    print()
    print("Transfer summary:")
    for row in final_summary_rows:
        if row["transfer_source"] == NA:
            continue
        print(
            f"{row['transfer_source']} -> {row['family']}/{row['language']}: "
            f"Transfer LAS {row['transfer_las_mean']} +/- {row['transfer_las_std']}, "
            f"Delta vs Mono {row['transfer_delta_las_mean']} "
            f"({row['transfer_better_las_folds']}/{row['transfer_folds']} folds)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
