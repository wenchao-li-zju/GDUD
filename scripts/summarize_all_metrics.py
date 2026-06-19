#!/usr/bin/env python3
"""Summarize UPOS, UAS, and LAS results for GDUD experiments A/B/C.

The output names are intentionally explicit:
  - all_metrics_language_summary.csv: one row per language
  - all_metrics_by_fold.csv: one row per language/fold/setting
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


FAMILY_ORDER = {
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
EXPERIMENTS = ("mono", "joint", "transfer")
METRICS = ("upos", "uas", "las")
NA = "NA"


def read_latest_rows(path: Path, expected_experiment: str) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing input file: {path}")

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"No rows found in {path}")

    latest: dict[tuple[str, str, str, str, int], dict[str, str]] = {}
    for row in rows:
        if row["experiment"] != expected_experiment:
            continue
        key = (
            row["experiment"],
            row.get("source", ""),
            row["family"],
            row["language"],
            int(row["fold"]),
        )
        latest[key] = row

    return list(latest.values())


def read_results(args: argparse.Namespace) -> dict[str, dict[str, list[dict[str, str]]]]:
    return {
        "mono": {
            "upos": read_latest_rows(args.upos_mono_file, "mono"),
            "parse": read_latest_rows(args.parse_mono_file, "mono"),
        },
        "joint": {
            "upos": read_latest_rows(args.upos_joint_file, "joint"),
            "parse": read_latest_rows(args.parse_joint_file, "joint"),
        },
        "transfer": {
            "upos": read_latest_rows(args.upos_transfer_file, "transfer"),
            "parse": read_latest_rows(args.parse_transfer_file, "transfer"),
        },
    }


def mean(values: list[float]) -> float:
    return statistics.mean(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def fmt(value: float) -> str:
    return f"{value:.2f}"


def language_sort_key(row: dict[str, str]) -> tuple[int, int]:
    family = row["family"]
    language = row["language"]
    families = list(FAMILY_ORDER)
    family_index = families.index(family) if family in FAMILY_ORDER else len(families)
    languages = FAMILY_ORDER.get(family, [])
    language_index = languages.index(language) if language in languages else len(languages)
    return family_index, language_index


def result_key(row: dict[str, str]) -> tuple[str, str, int]:
    return (row["family"], row["language"], int(row["fold"]))


def transfer_key(row: dict[str, str]) -> tuple[str, str, str, int]:
    return (row["source"], row["family"], row["language"], int(row["fold"]))


def index_by_language_fold(rows: list[dict[str, str]]) -> dict[tuple[str, str, int], dict[str, str]]:
    return {result_key(row): row for row in rows}


def index_transfer_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, int], dict[str, str]]:
    return {transfer_key(row): row for row in rows}


def check_alignment(results: dict[str, dict[str, list[dict[str, str]]]]) -> None:
    for experiment in ("mono", "joint"):
        upos_keys = set(index_by_language_fold(results[experiment]["upos"]))
        parse_keys = set(index_by_language_fold(results[experiment]["parse"]))
        if upos_keys != parse_keys:
            missing_parse = sorted(upos_keys - parse_keys)
            missing_upos = sorted(parse_keys - upos_keys)
            if missing_parse:
                print(f"{experiment}: parse rows missing for:")
                for key in missing_parse:
                    print(f"  {key}")
            if missing_upos:
                print(f"{experiment}: upos rows missing for:")
                for key in missing_upos:
                    print(f"  {key}")
            raise SystemExit(f"{experiment} UPOS and parse keys do not align.")

    upos_transfer_keys = set(index_transfer_rows(results["transfer"]["upos"]))
    parse_transfer_keys = set(index_transfer_rows(results["transfer"]["parse"]))
    if upos_transfer_keys != parse_transfer_keys:
        missing_parse = sorted(upos_transfer_keys - parse_transfer_keys)
        missing_upos = sorted(parse_transfer_keys - upos_transfer_keys)
        if missing_parse:
            print("transfer: parse rows missing for:")
            for key in missing_parse:
                print(f"  {key}")
        if missing_upos:
            print("transfer: upos rows missing for:")
            for key in missing_upos:
                print(f"  {key}")
        raise SystemExit("transfer UPOS and parse keys do not align.")


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
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


def combine_fold_rows(
    results: dict[str, dict[str, list[dict[str, str]]]]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for experiment in ("mono", "joint"):
        upos_by_key = index_by_language_fold(results[experiment]["upos"])
        parse_by_key = index_by_language_fold(results[experiment]["parse"])
        for key in sorted(upos_by_key):
            upos = upos_by_key[key]
            parse = parse_by_key[key]
            rows.append(
                {
                    "experiment": experiment,
                    "source": upos["source"],
                    "family": upos["family"],
                    "language": upos["language"],
                    "fold": upos["fold"],
                    "tokens": upos["tokens"],
                    "upos": upos["upos"],
                    "uas": parse["uas"],
                    "las": parse["las"],
                    "upos_prediction_file": upos["prediction_file"],
                    "parse_prediction_file": parse["prediction_file"],
                }
            )

    upos_transfer = index_transfer_rows(results["transfer"]["upos"])
    parse_transfer = index_transfer_rows(results["transfer"]["parse"])
    for key in sorted(upos_transfer):
        upos = upos_transfer[key]
        parse = parse_transfer[key]
        rows.append(
            {
                "experiment": "transfer",
                "source": upos["source"],
                "family": upos["family"],
                "language": upos["language"],
                "fold": upos["fold"],
                "tokens": upos["tokens"],
                "upos": upos["upos"],
                "uas": parse["uas"],
                "las": parse["las"],
                "upos_prediction_file": upos["prediction_file"],
                "parse_prediction_file": parse["prediction_file"],
            }
        )

    return rows


def summarize_experiment(
    experiment: str,
    upos_rows: list[dict[str, str]],
    parse_rows: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    summary: dict[tuple[str, str], dict[str, str]] = {}

    if experiment == "transfer":
        upos_grouped = group_transfer_rows(upos_rows)
        parse_grouped = group_transfer_rows(parse_rows)

        for source, family, language in sorted(upos_grouped):
            upos_group = sorted(
                upos_grouped[(source, family, language)], key=lambda row: int(row["fold"])
            )
            parse_group = sorted(
                parse_grouped[(source, family, language)], key=lambda row: int(row["fold"])
            )
            summary[(family, language)] = summarize_row_groups(
                upos_group, parse_group, source=source
            )
        return summary

    upos_grouped = group_rows(upos_rows)
    parse_grouped = group_rows(parse_rows)
    for family, language in sorted(upos_grouped):
        upos_group = sorted(upos_grouped[(family, language)], key=lambda row: int(row["fold"]))
        parse_group = sorted(parse_grouped[(family, language)], key=lambda row: int(row["fold"]))
        source = upos_group[0]["source"]
        summary[(family, language)] = summarize_row_groups(upos_group, parse_group, source=source)

    return summary


def summarize_row_groups(
    upos_group: list[dict[str, str]],
    parse_group: list[dict[str, str]],
    source: str,
) -> dict[str, str]:
    if [int(row["fold"]) for row in upos_group] != [int(row["fold"]) for row in parse_group]:
        raise SystemExit("UPOS and parse fold lists do not align within a language group.")

    upos_values = [float(row["upos"]) for row in upos_group]
    uas_values = [float(row["uas"]) for row in parse_group]
    las_values = [float(row["las"]) for row in parse_group]

    return {
        "source": source,
        "folds": str(len(upos_group)),
        "tokens": str(sum(int(row["tokens"]) for row in upos_group)),
        "upos_mean": fmt(mean(upos_values)),
        "upos_std": fmt(sample_std(upos_values)),
        "uas_mean": fmt(mean(uas_values)),
        "uas_std": fmt(sample_std(uas_values)),
        "las_mean": fmt(mean(las_values)),
        "las_std": fmt(sample_std(las_values)),
    }


def build_language_summary(
    results: dict[str, dict[str, list[dict[str, str]]]]
) -> list[dict[str, str]]:
    summaries = {
        experiment: summarize_experiment(
            experiment,
            results[experiment]["upos"],
            results[experiment]["parse"],
        )
        for experiment in EXPERIMENTS
    }

    language_keys = sorted(
        set(summaries["mono"]) | set(summaries["joint"]) | set(summaries["transfer"]),
        key=lambda key: language_sort_key({"family": key[0], "language": key[1]}),
    )

    rows = []
    for family, language in language_keys:
        row = {
            "family": family,
            "language": language,
            "folds": summaries["mono"].get((family, language), {}).get("folds", NA),
            "tokens": summaries["mono"].get((family, language), {}).get("tokens", NA),
        }

        for experiment in EXPERIMENTS:
            summary = summaries[experiment].get((family, language))
            prefix = experiment
            if not summary:
                row[f"{prefix}_source"] = NA
                row[f"{prefix}_folds"] = "0"
                for metric in METRICS:
                    row[f"{prefix}_{metric}_mean"] = NA
                    row[f"{prefix}_{metric}_std"] = NA
                continue

            row[f"{prefix}_source"] = summary["source"]
            row[f"{prefix}_folds"] = summary["folds"]
            for metric in METRICS:
                row[f"{prefix}_{metric}_mean"] = summary[f"{metric}_mean"]
                row[f"{prefix}_{metric}_std"] = summary[f"{metric}_std"]

        add_delta_columns(row)
        rows.append(row)

    return rows


def add_delta_columns(row: dict[str, str]) -> None:
    for metric in METRICS:
        mono = row.get(f"mono_{metric}_mean", NA)
        joint = row.get(f"joint_{metric}_mean", NA)
        transfer = row.get(f"transfer_{metric}_mean", NA)

        row[f"joint_minus_mono_{metric}"] = (
            fmt(float(joint) - float(mono)) if mono != NA and joint != NA else NA
        )
        row[f"transfer_minus_mono_{metric}"] = (
            fmt(float(transfer) - float(mono)) if mono != NA and transfer != NA else NA
        )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise SystemExit(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize all GDUD experiment metrics.")
    parser.add_argument(
        "--upos-mono-file",
        default=Path("experiments_0616/results/upos_mono_by_fold.csv"),
        type=Path,
    )
    parser.add_argument(
        "--upos-joint-file",
        default=Path("experiments_0616/results/upos_joint_by_fold.csv"),
        type=Path,
    )
    parser.add_argument(
        "--upos-transfer-file",
        default=Path("experiments_0616/results/upos_transfer_by_fold.csv"),
        type=Path,
    )
    parser.add_argument(
        "--parse-mono-file",
        default=Path("experiments_0616/results/mono_by_fold_dedup.csv"),
        type=Path,
    )
    parser.add_argument(
        "--parse-joint-file",
        default=Path("experiments_0616/results/joint_by_fold.csv"),
        type=Path,
    )
    parser.add_argument(
        "--parse-transfer-file",
        default=Path("experiments_0616/results/transfer_by_fold.csv"),
        type=Path,
    )
    parser.add_argument(
        "--language-summary-file",
        default=Path("experiments_0616/results/all_metrics_language_summary.csv"),
        type=Path,
    )
    parser.add_argument(
        "--fold-summary-file",
        default=Path("experiments_0616/results/all_metrics_by_fold.csv"),
        type=Path,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = read_results(args)
    check_alignment(results)

    fold_rows = combine_fold_rows(results)
    language_rows = build_language_summary(results)

    write_csv(args.fold_summary_file, fold_rows)
    write_csv(args.language_summary_file, language_rows)

    print(f"Wrote fold-level all-metrics table to {args.fold_summary_file}")
    print(f"Wrote language-level all-metrics summary to {args.language_summary_file}")
    print()
    print("Input rows:")
    for experiment in EXPERIMENTS:
        print(
            f"  {experiment}: "
            f"UPOS={len(results[experiment]['upos'])}, "
            f"UAS/LAS={len(results[experiment]['parse'])}"
        )

    print()
    print("Language summary:")
    for row in language_rows:
        transfer_text = ""
        if row["transfer_source"] != NA:
            transfer_text = (
                f", Transfer UPOS {row['transfer_upos_mean']} +/- {row['transfer_upos_std']}, "
                f"Transfer LAS {row['transfer_las_mean']} +/- {row['transfer_las_std']}"
            )
        print(
            f"{row['family']}/{row['language']}: "
            f"Mono UPOS {row['mono_upos_mean']} +/- {row['mono_upos_std']}, "
            f"Joint UPOS {row['joint_upos_mean']} +/- {row['joint_upos_std']}, "
            f"Mono LAS {row['mono_las_mean']} +/- {row['mono_las_std']}, "
            f"Joint LAS {row['joint_las_mean']} +/- {row['joint_las_std']}"
            f"{transfer_text}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
