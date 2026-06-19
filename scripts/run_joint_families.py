#!/usr/bin/env python3
"""Run joint multilingual Stanza depparse experiments by language family.

For each family/fold, this trains one joint model on the merged family
train/dev split, then predicts each language's corresponding mono test split.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


LANGUAGES = {
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


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def selected_families(args: argparse.Namespace) -> list[str]:
    if not args.families:
        return list(LANGUAGES)

    requested = parse_csv_list(args.families)
    unknown = sorted(set(requested) - set(LANGUAGES))
    if unknown:
        raise SystemExit(f"Unknown family/families: {', '.join(unknown)}")
    return requested


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run joint multilingual 5-fold experiments for Mongolic/Tungusic."
    )
    parser.add_argument("--max-steps", default=50000, type=int)
    parser.add_argument("--eval-interval", default=100, type=int)
    parser.add_argument("--log-step", default=20, type=int)
    parser.add_argument("--batch-size", default=1000, type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--folds", default="0,1,2,3,4", help="Comma-separated fold ids.")
    parser.add_argument(
        "--families",
        default="",
        help="Optional comma-separated subset: Mongolic,Tungusic.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--small-model",
        action="store_true",
        help="Use the smaller dimensions from the smoke test for quicker runs.",
    )
    parser.add_argument(
        "--results-file",
        default=Path("experiments_0616/results/joint_by_fold.csv"),
        type=Path,
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

    jobs = []
    for family in selected_families(args):
        for fold in folds:
            for language in LANGUAGES[family]:
                jobs.append((family, fold, language))

    print(f"Selected {len(jobs)} joint prediction job(s).", flush=True)
    print(
        "Training jobs implied: "
        f"{len(selected_families(args)) * len(folds)} family-fold model(s).",
        flush=True,
    )

    overwritten_models = set()

    for index, (family, fold, language) in enumerate(jobs, start=1):
        joint_split = root / "splits" / "joint" / family / f"fold{fold}"
        mono_split = root / "splits" / "mono" / family / language / f"fold{fold}"
        model_dir = root / "models" / "joint" / family / f"fold{fold}"
        pred_dir = root / "predictions" / "joint" / family / language / f"fold{fold}"
        shorthand = f"gdud_{family.lower()}_joint"

        cmd = [
            sys.executable,
            "scripts/run_stanza_depparse.py",
            "--experiment",
            "joint",
            "--family",
            family,
            "--language",
            language,
            "--fold",
            str(fold),
            "--source",
            f"{family}_joint",
            "--train-file",
            str(joint_split / "train.conllu"),
            "--dev-file",
            str(joint_split / "dev.conllu"),
            "--test-file",
            str(mono_split / "test.conllu"),
            "--model-dir",
            str(model_dir),
            "--pred-dir",
            str(pred_dir),
            "--results-file",
            str(args.results_file),
            "--lang",
            family.lower(),
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
        model_key = (family, fold)
        if args.overwrite and model_key not in overwritten_models:
            cmd.append("--overwrite")

        print(
            f"\n=== [{index}/{len(jobs)}] {family} joint fold{fold} -> {language} ===",
            flush=True,
        )
        subprocess.run(cmd, check=True)
        overwritten_models.add(model_key)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
