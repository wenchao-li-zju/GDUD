#!/usr/bin/env python3
"""Run Tungusic transfer experiments: Nanai -> Ulch/Negidal/Oroch."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SOURCE_LANGUAGE = "Nanai"
FAMILY = "Tungusic"
TARGET_LANGUAGES = ["Ulch", "Negidal", "Oroch"]


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Nanai -> Ulch/Negidal/Oroch transfer experiments."
    )
    parser.add_argument("--max-steps", default=50000, type=int)
    parser.add_argument("--eval-interval", default=100, type=int)
    parser.add_argument("--log-step", default=20, type=int)
    parser.add_argument("--batch-size", default=1000, type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--folds", default="0,1,2,3,4", help="Comma-separated fold ids.")
    parser.add_argument(
        "--targets",
        default=",".join(TARGET_LANGUAGES),
        help="Comma-separated targets. Default: Ulch,Negidal,Oroch.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--small-model",
        action="store_true",
        help="Use the smaller dimensions from the smoke test for quicker runs.",
    )
    parser.add_argument(
        "--results-file",
        default=Path("experiments_0616/results/transfer_by_fold.csv"),
        type=Path,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    folds = [int(fold.strip()) for fold in args.folds.split(",") if fold.strip()]
    targets = parse_csv_list(args.targets)

    unknown_targets = sorted(set(targets) - set(TARGET_LANGUAGES))
    if unknown_targets:
        raise SystemExit(
            "Unknown transfer target(s): "
            + ", ".join(unknown_targets)
            + f". Allowed targets: {', '.join(TARGET_LANGUAGES)}"
        )

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

    jobs = []
    for fold in folds:
        for target in targets:
            jobs.append((fold, target))

    print(f"Selected {len(jobs)} transfer prediction job(s).", flush=True)
    print(f"Training jobs implied: {len(folds)} Nanai source model(s).", flush=True)

    overwritten_models = set()

    for index, (fold, target) in enumerate(jobs, start=1):
        source_split = root / "splits" / "mono" / FAMILY / SOURCE_LANGUAGE / f"fold{fold}"
        target_split = root / "splits" / "mono" / FAMILY / target / f"fold{fold}"
        model_dir = root / "models" / "transfer" / f"{SOURCE_LANGUAGE}_to_targets" / f"fold{fold}"
        pred_dir = root / "predictions" / "transfer" / f"{SOURCE_LANGUAGE}_to_{target}" / f"fold{fold}"
        shorthand = f"gdud_{SOURCE_LANGUAGE.lower()}_transfer"

        cmd = [
            sys.executable,
            "scripts/run_stanza_depparse.py",
            "--experiment",
            "transfer",
            "--family",
            FAMILY,
            "--language",
            target,
            "--fold",
            str(fold),
            "--source",
            SOURCE_LANGUAGE,
            "--train-file",
            str(source_split / "train.conllu"),
            "--dev-file",
            str(source_split / "dev.conllu"),
            "--test-file",
            str(target_split / "test.conllu"),
            "--model-dir",
            str(model_dir),
            "--pred-dir",
            str(pred_dir),
            "--results-file",
            str(args.results_file),
            "--lang",
            SOURCE_LANGUAGE.lower(),
            "--shorthand",
            shorthand,
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

        model_key = fold
        if args.overwrite and model_key not in overwritten_models:
            cmd.append("--overwrite")

        print(
            f"\n=== [{index}/{len(jobs)}] {SOURCE_LANGUAGE} transfer fold{fold} -> {target} ===",
            flush=True,
        )
        subprocess.run(cmd, check=True)
        overwritten_models.add(model_key)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
