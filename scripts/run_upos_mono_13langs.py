#!/usr/bin/env python3
"""Run monolingual 5-fold Stanza UPOS tagger experiments for GDUD."""

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


def selected_languages(args: argparse.Namespace) -> list[tuple[str, str]]:
    requested = set(parse_csv_list(args.languages)) if args.languages else None
    allowed_families = set(parse_csv_list(args.families)) if args.families else None
    selected = []

    for family, languages in LANGUAGES.items():
        if allowed_families and family not in allowed_families:
            continue
        for language in languages:
            if requested and language not in requested:
                continue
            selected.append((family, language))

    if requested:
        known = {language for languages in LANGUAGES.values() for language in languages}
        unknown = sorted(requested - known)
        if unknown:
            raise SystemExit(f"Unknown language(s): {', '.join(unknown)}")

    if allowed_families:
        unknown = sorted(allowed_families - set(LANGUAGES))
        if unknown:
            raise SystemExit(f"Unknown family/families: {', '.join(unknown)}")

    if not selected:
        raise SystemExit("No languages selected.")

    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run UPOS monolingual 5-fold experiments for the 13-language setting."
    )
    parser.add_argument("--max-steps", default=50000, type=int)
    parser.add_argument("--eval-interval", default=100, type=int)
    parser.add_argument("--log-step", default=20, type=int)
    parser.add_argument("--batch-size", default=1000, type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--folds", default="0,1,2,3,4", help="Comma-separated fold ids.")
    parser.add_argument("--families", default="", help="Optional subset: Mongolic,Tungusic.")
    parser.add_argument("--languages", default="", help="Optional subset, e.g. Daur,Ulch.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--small-model",
        action="store_true",
        help="Use small dimensions for quicker runs.",
    )
    parser.add_argument(
        "--results-file",
        default=Path("experiments_0616/results/upos_mono_by_fold.csv"),
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
            "--composite-deep-biaff-hidden-dim",
            "100",
        ]

    jobs = []
    for family, language in selected_languages(args):
        for fold in folds:
            jobs.append((family, language, fold))

    print(f"Selected {len(jobs)} UPOS mono job(s).", flush=True)
    for index, (family, language, fold) in enumerate(jobs, start=1):
        split_dir = root / "splits" / "mono" / family / language / f"fold{fold}"
        model_dir = root / "models_upos" / "mono" / family / language / f"fold{fold}"
        pred_dir = root / "predictions_upos" / "mono" / family / language / f"fold{fold}"
        shorthand = f"gdud_{language.lower()}_upos"

        cmd = [
            sys.executable,
            "scripts/run_stanza_tagger.py",
            "--experiment",
            "mono",
            "--family",
            family,
            "--language",
            language,
            "--fold",
            str(fold),
            "--source",
            language,
            "--train-file",
            str(split_dir / "train.conllu"),
            "--dev-file",
            str(split_dir / "dev.conllu"),
            "--test-file",
            str(split_dir / "test.conllu"),
            "--model-dir",
            str(model_dir),
            "--pred-dir",
            str(pred_dir),
            "--results-file",
            str(args.results_file),
            "--lang",
            language.lower(),
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
        if args.overwrite:
            cmd.append("--overwrite")

        print(f"\n=== [{index}/{len(jobs)}] UPOS mono {family}/{language} fold{fold} ===", flush=True)
        subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
