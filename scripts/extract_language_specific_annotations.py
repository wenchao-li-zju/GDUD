#!/usr/bin/env python3
"""Extract language-specific FEATS and DEPRELs from validation reports."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ValidationSource:
    family: str
    language: str
    path: Path


VALIDATION_SOURCES = [
    ValidationSource("Mongolic", "Daur", Path("Original/Mongolic/Daur/Daur-UD_validation")),
    ValidationSource("Mongolic", "Kalmyk", Path("Original/Mongolic/Kalmyk/Kalmyk-UD_validation")),
    ValidationSource(
        "Mongolic",
        "Khalkha",
        Path("Original/Mongolic/Mongolian (Khalkha)/Mongolian (Khalkha)_UD_validation"),
    ),
    ValidationSource(
        "Mongolic",
        "Ordos",
        Path(
            "Original/Mongolic/Peripheral Mongolian (Ordos)/"
            "Peripheral Mongolian (Ordos)_UD_validation"
        ),
    ),
    ValidationSource("Mongolic", "Tu", Path("Original/Mongolic/Tu/Tu-UD_validation")),
    ValidationSource("Tungusic", "Even", Path("Original/Tungusic/Even/Even-UD_validation")),
    ValidationSource("Tungusic", "Evenki", Path("Original/Tungusic/Evenki/Evenki-UD_validation")),
    ValidationSource("Tungusic", "Manchu", Path("Original/Tungusic/Manchu/Manchu-UD_validation")),
    ValidationSource("Tungusic", "Nanai", Path("Original/Tungusic/Nanai/Nanai-UD_validation")),
    ValidationSource("Tungusic", "Negidal", Path("Original/Tungusic/Negidal /Negidal_UD_validation")),
    ValidationSource("Tungusic", "Oroch", Path("Original/Tungusic/Oroch/Oroch-UD_validation")),
    ValidationSource("Tungusic", "Udihe", Path("Original/Tungusic/Udihe/Udihe-UD_validation")),
    ValidationSource("Tungusic", "Ulch", Path("Original/Tungusic/Ulch/Ulch-UD_validation")),
]


def extract_appendices(path: Path) -> tuple[list[str], list[str]]:
    if not path.exists():
        raise SystemExit(f"Missing validation file: {path}")

    features: list[str] = []
    relations: list[str] = []
    section: str | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Appendix A:") or line.startswith(
            "Complete language-specific feature list"
        ):
            section = "features"
            continue
        if line.startswith("Appendix B:") or line.startswith(
            "Complete language-specific dependency list"
        ):
            section = "relations"
            continue
        if section == "features" and line.lower().startswith("dependency relation"):
            section = None
            continue
        if section and line.startswith("  "):
            item = line.strip()
            if item and item.lower() != "none":
                if section == "features":
                    features.append(item)
                else:
                    relations.append(item)

    return sorted(set(features)), sorted(set(relations))


def join_items(items: list[str]) -> str:
    return ", ".join(items) if items else "None"


def build_markdown(rows: list[dict[str, str | list[str]]]) -> str:
    lines = [
        "# Language-Specific Features and Dependency Relations",
        "",
        "Source: `Original/` validation reports, Appendix A and Appendix B.",
        "",
        "Bonan and Dongxiang are excluded because they are not included in the current 13-language experiment.",
        "",
        "For FEATS coverage, Appendix A raw labels are first normalized to UD-style Feature=Value bundles, and a token is counted as UD-core only if none of these normalized bundles is matched by the token's normalized FEATS.",
        "",
        "## Appendix A: Language-Specific Feature=Value Pairs",
        "",
        "| Family | Language | Feature=Value pairs | Source validation file |",
        "|---|---|---|---|",
    ]

    for row in rows:
        lines.append(
            f"| {row['family']} | {row['language']} | "
            f"{join_items(row['features'])} | `{row['source']}` |"
        )

    lines.extend(
        [
            "",
            "## Appendix B: Language-Specific Dependency Relations",
            "",
            "| Family | Language | Dependency relations | Source validation file |",
            "|---|---|---|---|",
        ]
    )

    for row in rows:
        lines.append(
            f"| {row['family']} | {row['language']} | "
            f"{join_items(row['relations'])} | `{row['source']}` |"
        )

    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Appendix A/B annotations from Original validation reports."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("language_specific_features_and_dependency_relations.md"),
        help="Markdown file to write.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, str | list[str]]] = []
    for source in VALIDATION_SOURCES:
        features, relations = extract_appendices(source.path)
        rows.append(
            {
                "family": source.family,
                "language": source.language,
                "features": features,
                "relations": relations,
                "source": str(source.path),
            }
        )

    args.output.write_text(build_markdown(rows), encoding="utf-8")

    print(f"Wrote {args.output}")
    for row in rows:
        print(
            f"{row['family']}/{row['language']}: "
            f"{len(row['features'])} feature values, {len(row['relations'])} relations"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
