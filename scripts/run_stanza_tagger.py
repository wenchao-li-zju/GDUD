#!/usr/bin/env python3
"""Run one Stanza POS tagger train+predict experiment and score UPOS."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


def token_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        columns = line.split("\t")
        if len(columns) == 10 and columns[0].isdigit():
            rows.append(columns)
    return rows


def score_upos(gold_path: Path, pred_path: Path) -> dict[str, float | int]:
    gold = token_rows(gold_path)
    pred = token_rows(pred_path)

    if len(gold) != len(pred):
        raise ValueError(
            f"Gold/pred token count mismatch: {gold_path} has {len(gold)}, "
            f"{pred_path} has {len(pred)}"
        )

    tokens = len(gold)
    if tokens == 0:
        raise ValueError(f"No regular CoNLL-U token rows found in {gold_path}")

    upos = sum(1 for gold_row, pred_row in zip(gold, pred) if gold_row[3] == pred_row[3])
    return {
        "tokens": tokens,
        "upos": upos / tokens * 100,
    }


def run_command(args: list[str], env: dict[str, str]) -> None:
    print("\n$ " + " ".join(args), flush=True)
    subprocess.run(args, env=env, check=True)


def build_common_tagger_args(args: argparse.Namespace) -> list[str]:
    return [
        "--lang",
        args.lang,
        "--shorthand",
        args.shorthand,
        "--save_dir",
        str(args.model_dir),
        "--save_name",
        args.save_name,
        "--seed",
        str(args.seed),
        "--batch_size",
        str(args.batch_size),
        "--word_emb_dim",
        str(args.word_emb_dim),
        "--char_emb_dim",
        str(args.char_emb_dim),
        "--char_hidden_dim",
        str(args.char_hidden_dim),
        "--hidden_dim",
        str(args.hidden_dim),
        "--deep_biaff_hidden_dim",
        str(args.deep_biaff_hidden_dim),
        "--composite_deep_biaff_hidden_dim",
        str(args.composite_deep_biaff_hidden_dim),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Stanza POS tagger train+predict experiment."
    )
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--dev-file", required=True, type=Path)
    parser.add_argument("--test-file", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--pred-dir", required=True, type=Path)
    parser.add_argument("--results-file", required=True, type=Path)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--source", default="")
    parser.add_argument("--lang", default="xx")
    parser.add_argument("--shorthand", required=True)
    parser.add_argument("--save-name", default="tagger.pt")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max-steps", default=50000, type=int)
    parser.add_argument("--eval-interval", default=100, type=int)
    parser.add_argument("--log-step", default=20, type=int)
    parser.add_argument("--batch-size", default=1000, type=int)
    parser.add_argument("--word-emb-dim", default=75, type=int)
    parser.add_argument("--char-emb-dim", default=100, type=int)
    parser.add_argument("--char-hidden-dim", default=400, type=int)
    parser.add_argument("--hidden-dim", default=400, type=int)
    parser.add_argument("--deep-biaff-hidden-dim", default=400, type=int)
    parser.add_argument("--composite-deep-biaff-hidden-dim", default=400, type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--cache-dir",
        default=Path("experiments_0616/cache"),
        type=Path,
        help="Project-local cache directory for HF/torch caches.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain even if the model file already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    for path in (args.train_file, args.dev_file, args.test_file):
        if not path.exists():
            raise SystemExit(f"Missing required file: {path}")

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.pred_dir.mkdir(parents=True, exist_ok=True)
    args.results_file.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    hf_cache = args.cache_dir / "huggingface"
    torch_cache = args.cache_dir / "torch"
    hf_cache.mkdir(parents=True, exist_ok=True)
    torch_cache.mkdir(parents=True, exist_ok=True)
    env["HF_HOME"] = str(hf_cache)
    env["TRANSFORMERS_CACHE"] = str(hf_cache)
    env["TORCH_HOME"] = str(torch_cache)

    model_path = args.model_dir / args.save_name
    dev_pred = args.pred_dir / "dev_pred.conllu"
    test_pred = args.pred_dir / "test_pred.conllu"

    common = build_common_tagger_args(args)
    device_args = ["--cpu"] if args.device == "cpu" else ["--cuda"]

    if args.overwrite or not model_path.exists():
        train_cmd = [
            sys.executable,
            "-m",
            "stanza.models.tagger",
            "--mode",
            "train",
            "--train_file",
            str(args.train_file),
            "--eval_file",
            str(args.dev_file),
            "--output_file",
            str(dev_pred),
            *common,
            *device_args,
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
        run_command(train_cmd, env)
    else:
        print(f"Model exists, skipping training: {model_path}", flush=True)

    predict_cmd = [
        sys.executable,
        "-m",
        "stanza.models.tagger",
        "--mode",
        "predict",
        "--eval_file",
        str(args.test_file),
        "--output_file",
        str(test_pred),
        *common,
        *device_args,
        "--no_pretrain",
    ]
    run_command(predict_cmd, env)

    scores = score_upos(args.test_file, test_pred)
    row = {
        "experiment": args.experiment,
        "family": args.family,
        "language": args.language,
        "fold": args.fold,
        "source": args.source,
        "train_file": str(args.train_file),
        "dev_file": str(args.dev_file),
        "test_file": str(args.test_file),
        "model_file": str(model_path),
        "prediction_file": str(test_pred),
        "tokens": scores["tokens"],
        "upos": f"{scores['upos']:.2f}",
    }

    exists = args.results_file.exists()
    with args.results_file.open("a", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)

    print(
        "\nRESULT "
        f"{args.experiment} {args.family}/{args.language} fold{args.fold}: "
        f"UPOS={row['upos']} tokens={row['tokens']}"
    )
    print(f"Appended results to {args.results_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
