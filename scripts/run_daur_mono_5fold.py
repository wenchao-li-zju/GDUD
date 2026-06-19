#!/usr/bin/env python3
"""Run the Daur monolingual 5-fold Stanza dependency parser experiment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Daur monolingual 5-fold experiment.")
    parser.add_argument("--max-steps", default=50000, type=int)
    parser.add_argument("--eval-interval", default=100, type=int)
    parser.add_argument("--log-step", default=20, type=int)
    parser.add_argument("--batch-size", default=1000, type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--folds", default="0,1,2,3,4", help="Comma-separated fold ids.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--small-model",
        action="store_true",
        help="Use the smaller dimensions from the smoke test for quick runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    folds = [int(fold.strip()) for fold in args.folds.split(",") if fold.strip()]
    root = Path("experiments_0616")

    dims = []
    if args.small_model:
        dims = [
            "--word-emb-dim",
            "50",
            "--char-emb-dim",
            "50",
            "--char-hidden-dim",
            "100",
            "--hidden-dim",
            "100",
            "--deep-biaff-hidden-dim",
            "100",
            "--deep-biaff-output-dim",
            "50",
        ]

    for fold in folds:
        split_dir = root / "splits" / "mono" / "Mongolic" / "Daur" / f"fold{fold}"
        cmd = [
            sys.executable,
            "scripts/run_stanza_depparse.py",
            "--experiment",
            "mono",
            "--family",
            "Mongolic",
            "--language",
            "Daur",
            "--fold",
            str(fold),
            "--source",
            "Daur",
            "--train-file",
            str(split_dir / "train.conllu"),
            "--dev-file",
            str(split_dir / "dev.conllu"),
            "--test-file",
            str(split_dir / "test.conllu"),
            "--model-dir",
            str(root / "models" / "mono" / "Mongolic" / "Daur" / f"fold{fold}"),
            "--pred-dir",
            str(root / "predictions" / "mono" / "Mongolic" / "Daur" / f"fold{fold}"),
            "--results-file",
            str(root / "results" / "mono_by_fold.csv"),
            "--lang",
            "daur",
            "--shorthand",
            "gdud_daur",
            "--max-steps",
            str(args.max_steps),
            "--eval-interval",
            str(args.eval_interval),
            "--log-step",
            str(args.log_step),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            *dims,
        ]
        if args.overwrite:
            cmd.append("--overwrite")

        print(f"\n=== Daur mono fold{fold} ===", flush=True)
        subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
