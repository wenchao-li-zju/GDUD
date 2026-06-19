#!/usr/bin/env python3
"""Run full-source transfer: train on all Nanai, test full target treebanks.

This script is separate from the 5-fold transfer scripts.  It uses the full
Nanai core treebank as both train and dev because Stanza requires a dev file,
while the target languages remain completely unseen during training.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


SOURCE = "Nanai"
FAMILY = "Tungusic"
TARGETS = ["Ulch", "Negidal", "Oroch"]


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def token_rows(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        columns = line.split("\t")
        if len(columns) == 10 and columns[0].isdigit():
            rows.append(columns)
    return rows


def score_parse(gold_path: Path, pred_path: Path) -> dict[str, float | int]:
    gold = token_rows(gold_path)
    pred = token_rows(pred_path)
    if len(gold) != len(pred):
        raise ValueError(f"Gold/pred token mismatch: {gold_path}={len(gold)}, {pred_path}={len(pred)}")
    if not gold:
        raise ValueError(f"No regular CoNLL-U token rows found in {gold_path}")

    uas = sum(1 for gold_row, pred_row in zip(gold, pred) if gold_row[6] == pred_row[6])
    las = sum(
        1
        for gold_row, pred_row in zip(gold, pred)
        if gold_row[6] == pred_row[6] and gold_row[7] == pred_row[7]
    )
    return {"tokens": len(gold), "uas": uas / len(gold) * 100, "las": las / len(gold) * 100}


def score_upos(gold_path: Path, pred_path: Path) -> dict[str, float | int]:
    gold = token_rows(gold_path)
    pred = token_rows(pred_path)
    if len(gold) != len(pred):
        raise ValueError(f"Gold/pred token mismatch: {gold_path}={len(gold)}, {pred_path}={len(pred)}")
    if not gold:
        raise ValueError(f"No regular CoNLL-U token rows found in {gold_path}")

    upos = sum(1 for gold_row, pred_row in zip(gold, pred) if gold_row[3] == pred_row[3])
    return {"tokens": len(gold), "upos": upos / len(gold) * 100}


def run_command(command: list[str], env: dict[str, str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, env=env, check=True)


def make_env(cache_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    hf_cache = cache_dir / "huggingface"
    torch_cache = cache_dir / "torch"
    hf_cache.mkdir(parents=True, exist_ok=True)
    torch_cache.mkdir(parents=True, exist_ok=True)
    env["HF_HOME"] = str(hf_cache)
    env["TRANSFORMERS_CACHE"] = str(hf_cache)
    env["TORCH_HOME"] = str(torch_cache)
    return env


def device_args(device: str) -> list[str]:
    return ["--cpu"] if device == "cpu" else ["--cuda"]


def parser_dims(args: argparse.Namespace) -> list[str]:
    if not args.small_model:
        return []
    return [
        "--word_emb_dim",
        "50",
        "--char_emb_dim",
        "50",
        "--char_hidden_dim",
        "100",
        "--hidden_dim",
        "100",
        "--deep_biaff_hidden_dim",
        "100",
        "--deep_biaff_output_dim",
        "50",
    ]


def tagger_dims(args: argparse.Namespace) -> list[str]:
    if not args.small_model:
        return []
    return [
        "--word_emb_dim",
        "50",
        "--char_emb_dim",
        "50",
        "--char_hidden_dim",
        "100",
        "--hidden_dim",
        "100",
        "--deep_biaff_hidden_dim",
        "100",
        "--composite_deep_biaff_hidden_dim",
        "100",
    ]


def append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def latest_rows(path: Path, key_fields: list[str]) -> dict[tuple[str, ...], dict[str, str]]:
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    latest = {}
    for row in rows:
        latest[tuple(row[field] for field in key_fields)] = row
    return latest


def write_combined(parse_file: Path, upos_file: Path, combined_file: Path) -> None:
    parse_rows = latest_rows(parse_file, ["source", "language"])
    upos_rows = latest_rows(upos_file, ["source", "language"])
    keys = sorted(set(parse_rows) | set(upos_rows))
    if not keys:
        return

    rows = []
    for key in keys:
        parse = parse_rows.get(key)
        upos = upos_rows.get(key)
        base = parse or upos
        assert base is not None
        rows.append(
            {
                "experiment": "full_source_transfer",
                "family": base["family"],
                "source": base["source"],
                "language": base["language"],
                "tokens": base["tokens"],
                "upos": upos["upos"] if upos else "NA",
                "uas": parse["uas"] if parse else "NA",
                "las": parse["las"] if parse else "NA",
                "upos_prediction_file": upos["prediction_file"] if upos else "NA",
                "parse_prediction_file": parse["prediction_file"] if parse else "NA",
            }
        )

    combined_file.parent.mkdir(parents=True, exist_ok=True)
    with combined_file.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_parser(args: argparse.Namespace, source_file: Path, env: dict[str, str]) -> Path:
    model_dir = args.root / "models" / "full_source_transfer" / SOURCE
    pred_dir = args.root / "predictions" / "full_source_transfer" / SOURCE / "dev"
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / "parser.pt"

    if args.overwrite or not model_file.exists():
        command = [
            sys.executable,
            "-m",
            "stanza.models.parser",
            "--mode",
            "train",
            "--train_file",
            str(source_file),
            "--eval_file",
            str(source_file),
            "--output_file",
            str(pred_dir / "dev_pred.conllu"),
            "--lang",
            SOURCE.lower(),
            "--shorthand",
            "gdud_nanai_full_transfer",
            "--save_dir",
            str(model_dir),
            "--save_name",
            "parser.pt",
            "--seed",
            str(args.seed),
            "--batch_size",
            str(args.batch_size),
            *parser_dims(args),
            *device_args(args.device),
            "--no_pretrain",
            "--no_second_optim",
            "--no_checkpoint",
            "--max_steps",
            str(args.max_steps),
            "--eval_interval",
            str(args.eval_interval),
            "--log_step",
            str(args.log_step),
        ]
        run_command(command, env)
    else:
        print(f"Parser model exists, skipping training: {model_file}", flush=True)

    return model_file


def train_tagger(args: argparse.Namespace, source_file: Path, env: dict[str, str]) -> Path:
    model_dir = args.root / "models_upos" / "full_source_transfer" / SOURCE
    pred_dir = args.root / "predictions_upos" / "full_source_transfer" / SOURCE / "dev"
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / "tagger.pt"

    if args.overwrite or not model_file.exists():
        command = [
            sys.executable,
            "-m",
            "stanza.models.tagger",
            "--mode",
            "train",
            "--train_file",
            str(source_file),
            "--eval_file",
            str(source_file),
            "--output_file",
            str(pred_dir / "dev_pred.conllu"),
            "--lang",
            SOURCE.lower(),
            "--shorthand",
            "gdud_nanai_full_transfer_upos",
            "--save_dir",
            str(model_dir),
            "--save_name",
            "tagger.pt",
            "--seed",
            str(args.seed),
            "--batch_size",
            str(args.batch_size),
            *tagger_dims(args),
            *device_args(args.device),
            "--no_pretrain",
            "--no_second_optim",
            "--max_steps",
            str(args.max_steps),
            "--eval_interval",
            str(args.eval_interval),
            "--fix_eval_interval",
            "--log_step",
            str(args.log_step),
        ]
        run_command(command, env)
    else:
        print(f"UPOS tagger model exists, skipping training: {model_file}", flush=True)

    return model_file


def predict_parse(
    args: argparse.Namespace,
    model_file: Path,
    target: str,
    target_file: Path,
    env: dict[str, str],
) -> None:
    pred_dir = args.root / "predictions" / "full_source_transfer" / f"{SOURCE}_to_{target}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_file = pred_dir / "test_pred.conllu"
    model_dir = model_file.parent

    command = [
        sys.executable,
        "-m",
        "stanza.models.parser",
        "--mode",
        "predict",
        "--eval_file",
        str(target_file),
        "--output_file",
        str(pred_file),
        "--lang",
        SOURCE.lower(),
        "--shorthand",
        "gdud_nanai_full_transfer",
        "--save_dir",
        str(model_dir),
        "--save_name",
        "parser.pt",
        "--seed",
        str(args.seed),
        "--batch_size",
        str(args.batch_size),
        *parser_dims(args),
        *device_args(args.device),
        "--no_pretrain",
    ]
    run_command(command, env)
    scores = score_parse(target_file, pred_file)
    row = {
        "experiment": "full_source_transfer",
        "family": FAMILY,
        "source": SOURCE,
        "language": target,
        "train_file": str(args.source_file),
        "dev_file": str(args.source_file),
        "test_file": str(target_file),
        "model_file": str(model_file),
        "prediction_file": str(pred_file),
        "tokens": str(scores["tokens"]),
        "uas": f"{scores['uas']:.2f}",
        "las": f"{scores['las']:.2f}",
    }
    append_row(args.parse_results_file, row)
    print(f"RESULT parse full-source {SOURCE} -> {target}: UAS={row['uas']} LAS={row['las']}")


def predict_upos(
    args: argparse.Namespace,
    model_file: Path,
    target: str,
    target_file: Path,
    env: dict[str, str],
) -> None:
    pred_dir = args.root / "predictions_upos" / "full_source_transfer" / f"{SOURCE}_to_{target}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_file = pred_dir / "test_pred.conllu"
    model_dir = model_file.parent

    command = [
        sys.executable,
        "-m",
        "stanza.models.tagger",
        "--mode",
        "predict",
        "--eval_file",
        str(target_file),
        "--output_file",
        str(pred_file),
        "--lang",
        SOURCE.lower(),
        "--shorthand",
        "gdud_nanai_full_transfer_upos",
        "--save_dir",
        str(model_dir),
        "--save_name",
        "tagger.pt",
        "--seed",
        str(args.seed),
        "--batch_size",
        str(args.batch_size),
        *tagger_dims(args),
        *device_args(args.device),
        "--no_pretrain",
    ]
    run_command(command, env)
    scores = score_upos(target_file, pred_file)
    row = {
        "experiment": "full_source_transfer",
        "family": FAMILY,
        "source": SOURCE,
        "language": target,
        "train_file": str(args.source_file),
        "dev_file": str(args.source_file),
        "test_file": str(target_file),
        "model_file": str(model_file),
        "prediction_file": str(pred_file),
        "tokens": str(scores["tokens"]),
        "upos": f"{scores['upos']:.2f}",
    }
    append_row(args.upos_results_file, row)
    print(f"RESULT UPOS full-source {SOURCE} -> {target}: UPOS={row['upos']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full-source transfer: Nanai full core treebank -> full target treebanks."
    )
    parser.add_argument("--root", default=Path("experiments_0616"), type=Path)
    parser.add_argument(
        "--source-file",
        default=Path("experiments_0616/data_core/Tungusic/Nanai_core.conllu"),
        type=Path,
    )
    parser.add_argument(
        "--targets",
        default=",".join(TARGETS),
        help="Comma-separated targets. Default: Ulch,Negidal,Oroch.",
    )
    parser.add_argument(
        "--tasks",
        default="parse,upos",
        help="Comma-separated tasks: parse,upos.",
    )
    parser.add_argument("--max-steps", default=3000, type=int)
    parser.add_argument("--eval-interval", default=100, type=int)
    parser.add_argument("--log-step", default=20, type=int)
    parser.add_argument("--batch-size", default=1000, type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--small-model", action="store_true")
    parser.add_argument(
        "--parse-results-file",
        default=Path("experiments_0616/results/full_source_transfer_parse.csv"),
        type=Path,
    )
    parser.add_argument(
        "--upos-results-file",
        default=Path("experiments_0616/results/full_source_transfer_upos.csv"),
        type=Path,
    )
    parser.add_argument(
        "--combined-results-file",
        default=Path("experiments_0616/results/full_source_transfer_all_metrics.csv"),
        type=Path,
    )
    parser.add_argument(
        "--cache-dir",
        default=Path("experiments_0616/cache"),
        type=Path,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = parse_csv_list(args.targets)
    tasks = set(parse_csv_list(args.tasks))
    unknown_targets = sorted(set(targets) - set(TARGETS))
    unknown_tasks = sorted(tasks - {"parse", "upos"})
    if unknown_targets:
        raise SystemExit(f"Unknown target(s): {', '.join(unknown_targets)}")
    if unknown_tasks:
        raise SystemExit(f"Unknown task(s): {', '.join(unknown_tasks)}")
    if not args.source_file.exists():
        raise SystemExit(f"Missing source file: {args.source_file}")

    target_files = {
        target: args.root / "data_core" / FAMILY / f"{target}_core.conllu" for target in targets
    }
    for target, target_file in target_files.items():
        if not target_file.exists():
            raise SystemExit(f"Missing target file for {target}: {target_file}")

    env = make_env(args.cache_dir)

    print(f"Full-source transfer source: {args.source_file}")
    print(f"Targets: {', '.join(targets)}")
    print(f"Tasks: {', '.join(sorted(tasks))}")
    print("Note: source file is used as both train and dev; target files are never used for training.")

    parser_model = train_parser(args, args.source_file, env) if "parse" in tasks else None
    tagger_model = train_tagger(args, args.source_file, env) if "upos" in tasks else None

    for target in targets:
        target_file = target_files[target]
        if parser_model is not None:
            predict_parse(args, parser_model, target, target_file, env)
        if tagger_model is not None:
            predict_upos(args, tagger_model, target, target_file, env)

    write_combined(args.parse_results_file, args.upos_results_file, args.combined_results_file)
    if args.combined_results_file.exists():
        print(f"Wrote combined full-source transfer table to {args.combined_results_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
