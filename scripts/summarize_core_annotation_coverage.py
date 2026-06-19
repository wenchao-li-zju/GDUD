#!/usr/bin/env python3
"""Compute per-language UD core DEPREL and FEATS coverage.

DEPREL coverage is the percentage of regular tokens whose DEPREL is one of the
37 UD core dependency relations. FEATS coverage is the percentage of regular
tokens for which every morphological feature-value pair is outside the
language-specific non-core list.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


UD_CORE_DEPRELS = {
    "acl",
    "advcl",
    "advmod",
    "amod",
    "appos",
    "aux",
    "case",
    "cc",
    "ccomp",
    "clf",
    "compound",
    "conj",
    "cop",
    "csubj",
    "dep",
    "det",
    "discourse",
    "dislocated",
    "expl",
    "fixed",
    "flat",
    "goeswith",
    "iobj",
    "list",
    "mark",
    "nmod",
    "nsubj",
    "nummod",
    "obj",
    "obl",
    "orphan",
    "parataxis",
    "punct",
    "reparandum",
    "root",
    "vocative",
    "xcomp",
}

TOKEN_ID_RE = re.compile(r"^[0-9]+$")
FEATURE_PAIR_RE = re.compile(r"(?<![^\s,|])([^\s,|`]+=[^\s,|`]*)")


CASE_CODE_MAP = {
    "NOM": "Nom",
    "ACC": "Acc",
    "GEN": "Gen",
    "DAT": "Dat",
    "ABL": "Abl",
    "INS": "Ins",
    "COM": "Com",
    "LOC": "Loc",
    "DIR": "Dir",
    "ALL": "All",
    "LAT": "Lat",
    "TERM": "Ter",
    "TER": "Ter",
}

NUMBER_CODE_MAP = {
    "SG": "Sing",
    "SING": "Sing",
    "PL": "Plur",
    "PLUR": "Plur",
}

PERSON_NUMBER_RE = re.compile(r"^([123])(?:H)?(SG|PL)(?:\.([A-Z]+))?$")
PSOR_RE = re.compile(r"^([123])POSS$")


@dataclass(frozen=True)
class NormalizedPattern:
    raw: str
    feats: frozenset[str]
    method: str

    def label(self) -> str:
        return f"{self.raw}=>{'+'.join(sorted(self.feats))}"


@dataclass
class CoverageStats:
    family: str
    language: str
    file: str
    sentences: int = 0
    tokens: int = 0
    core_deprel_tokens: int = 0
    core_feats_tokens: int = 0
    non_core_deprels: set[str] = field(default_factory=set)
    non_core_feats_seen: set[str] = field(default_factory=set)
    non_core_raw_seen: set[str] = field(default_factory=set)
    non_core_patterns_seen: set[str] = field(default_factory=set)
    non_core_feat_inventory_size: int = 0
    normalized_pattern_inventory_size: int = 0
    unmapped_raw_feat_values: set[str] = field(default_factory=set)
    normalized_patterns_in_data: set[str] = field(default_factory=set)
    normalized_patterns_not_in_data: set[str] = field(default_factory=set)
    non_core_feat_tokens: int = 0
    malformed_feat_tokens: int = 0


def pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "NA"
    return f"{numerator / denominator * 100:.2f}"


def ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "NA"
    return f"{numerator / denominator:.2f}"


def expand_feature_pair(pair: str) -> set[str]:
    """Return exact and comma-expanded Feature=Value pairs."""
    if "=" not in pair:
        return set()

    feature, values = pair.split("=", 1)
    expanded = {pair}
    for value in values.split(","):
        value = value.strip()
        if value:
            expanded.add(f"{feature}={value}")
    return expanded


def join_feats(feats: set[str] | frozenset[str]) -> str:
    return "+".join(sorted(feats))


def canonical_pair(feature: str, value: str) -> str:
    """Return a lightly normalized UD-style Feature=Value pair."""
    feature = feature.strip()
    value = value.strip()

    feature_aliases = {
        "Reflexive": "Reflex",
        "Interrog": "PronType",
        "Neg": "Polarity",
        "Infinitive": "VerbForm",
        "Distal": "Deixis",
        "Incl": "Clusivity",
        "Modality": "Mood",
        "Habitual": "Aspect",
    }
    feature = feature_aliases.get(feature, feature)

    value_aliases = {
        ("Number", "SG"): "Sing",
        ("Number", "PL"): "Plur",
        ("Number[psor]", "SG"): "Sing",
        ("Number[psor]", "PL"): "Plur",
        ("Definite", "YES"): "Def",
        ("Definite", "Indef"): "Ind",
        ("Reflex", "YES"): "Yes",
        ("PronType", "Yes"): "Int",
        ("Polarity", "Yes"): "Neg",
        ("Clusivity", "Incl"): "In",
        ("Clusivity", "Yes"): "In",
        ("Clusivity", "Excl"): "Ex",
        ("Mood", "Nec"): "Nec",
        ("Aspect", "Yes"): "Hab",
        ("Deixis", "Yes"): "Dist",
        ("VerbForm", ""): "Inf",
    }
    value = value_aliases.get((feature, value), value)
    return f"{feature}={value}"


def normalize_data_feature_pair(pair: str) -> set[str]:
    """Normalize a Feature=Value pair found in data_core."""
    if "=" not in pair:
        return set()
    feature, value = pair.split("=", 1)
    return {canonical_pair(feature, value)}


def normalize_raw_label(label: str) -> list[frozenset[str]]:
    """Map one Appendix A raw label to UD-style feature bundles.

    Each returned bundle is matched as a subset of the normalized token FEATS.
    Ambiguous raw labels are intentionally left unmapped instead of guessed.
    """
    label = label.strip()
    if not label or label == "None":
        return []

    if "=" in label:
        feature, value = label.split("=", 1)
        feature = feature.strip()
        value = value.strip()
        if value:
            return normalize_nonempty_feature_value(feature, value)
        return normalize_empty_value_label(feature)

    return normalize_empty_value_label(label)


def normalize_nonempty_feature_value(feature: str, value: str) -> list[frozenset[str]]:
    direct_maps = {
        ("Definite", "YES"): {"Definite=Def"},
        ("Focus", "FOC"): {"Focus=Yes"},
        ("Habitual", "Yes"): {"Aspect=Hab"},
        ("Honorific", "HON"): {"Polite=Form"},
        ("Interrog", "Yes"): {"PronType=Int"},
        ("Modality", "Nec"): {"Mood=Nec"},
        ("Neg", "Yes"): {"Polarity=Neg"},
        ("Reflexive", "YES"): {"Reflex=Yes"},
        ("Reflexive", "Yes"): {"Reflex=Yes"},
        ("Infinitive", ""): {"VerbForm=Inf"},
        ("Loc", ""): {"Case=Loc"},
        ("Distal", "Yes"): {"Deixis=Dist"},
        ("FEATS", "Instr.Ref"): {"Case=Ins", "Reflex=Yes"},
    }
    if (feature, value) in direct_maps:
        return [frozenset(direct_maps[(feature, value)])]

    if feature == "Number[psor]":
        return [frozenset({canonical_pair(feature, value)})]

    if feature == "Incl":
        return [frozenset({canonical_pair(feature, value)})]

    return [frozenset({canonical_pair(feature, value)})]


def normalize_empty_value_label(label: str) -> list[frozenset[str]]:
    upper_label = label.upper()

    if upper_label in CASE_CODE_MAP:
        return [frozenset({f"Case={CASE_CODE_MAP[upper_label]}"})]

    if upper_label in NUMBER_CODE_MAP:
        return [frozenset({f"Number={NUMBER_CODE_MAP[upper_label]}"})]

    match = PERSON_NUMBER_RE.match(label)
    if match:
        person, number, case_code = match.groups()
        feats = {f"Person={person}", f"Number={NUMBER_CODE_MAP[number]}"}
        if case_code in CASE_CODE_MAP:
            feats.add(f"Case={CASE_CODE_MAP[case_code]}")
        return [frozenset(feats)]

    match = PSOR_RE.match(label)
    if match:
        return [frozenset({f"Person[psor]={match.group(1)}", "Poss=Yes"})]

    direct_maps = {
        "2PL.POSS": {"Person[psor]=2", "Number[psor]=Plur", "Poss=Yes"},
        "ADV": {"VerbForm=Conv"},
        "CVB": {"VerbForm=Conv"},
        "CVBsim": {"VerbForm=Conv", "ConvType=Sim"},
        "D.DIST": {"Deixis=Dist"},
        "D.PROX": {"Deixis=Prox"},
        "DIM": {"Degree=Dim"},
        "DPST": {"Tense=Past"},
        "Emph": {"Emph=Yes"},
        "EX.NEG": {"Polarity=Neg"},
        "FOC": {"Focus=Yes"},
        "FUT.P": {"Tense=Fut", "VerbForm=Part"},
        "HAB.P": {"Aspect=Hab", "VerbForm=Part"},
        "IMP": {"Mood=Imp"},
        "IMP.HON": {"Mood=Imp", "Polite=Form"},
        "IMP.INT": {"Mood=Imp", "Mood=Int"},
        "NEG": {"Polarity=Neg"},
        "NEG-DIR": {"Polarity=Neg", "Voice=Dir"},
        "ORD": {"NumType=Ord"},
        "PASS": {"Voice=Pass"},
        "PL-ACC": {"Number=Plur", "Case=Acc"},
        "PRF": {"Aspect=Perf"},
        "PRF.P": {"Aspect=Perf", "VerbForm=Part"},
        "PROG": {"Aspect=Prog"},
        "PROG-FUT.P": {"Aspect=Prog", "Tense=Fut", "VerbForm=Part"},
        "Q": {"PartType=Int"},
        "RHET": {"PartType=Int"},
        "TAG": {"PartType=Int"},
        "VOL": {"Mood=Des"},
        "¬DIR": {"Voice=Inv"},
    }
    if label in direct_maps:
        return [frozenset(direct_maps[label])]

    return []


def extract_feature_items(cell: str) -> list[str]:
    if cell == "None":
        return []
    return [item.strip() for item in cell.split(",") if item.strip()]


def load_non_core_feat_patterns(
    path: Path,
) -> tuple[dict[str, list[NormalizedPattern]], dict[str, set[str]]]:
    """Load and normalize Appendix A non-core FEATS from markdown/plain text."""
    if not path.exists():
        raise SystemExit(f"Missing non-core FEATS file: {path}")

    text = path.read_text(encoding="utf-8")
    raw_by_language: dict[str, set[str]] = {}
    in_appendix_a = False

    for line in text.splitlines():
        if line.startswith("## Appendix A:"):
            in_appendix_a = True
            continue
        if line.startswith("## Appendix B:"):
            in_appendix_a = False
            continue
        if not in_appendix_a:
            continue

        stripped = line.strip()
        if not stripped.startswith("|") or stripped.startswith("|---"):
            continue

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 3 or cells[0] == "Family" or cells[1] == "Language":
            continue

        language = cells[1]
        feature_cell = cells[2]
        raw_by_language.setdefault(language, set()).update(extract_feature_items(feature_cell))

    patterns_by_language: dict[str, list[NormalizedPattern]] = {}
    unmapped_by_language: dict[str, set[str]] = {}
    for language, raw_items in raw_by_language.items():
        patterns: list[NormalizedPattern] = []
        unmapped: set[str] = set()
        seen_keys: set[tuple[str, frozenset[str]]] = set()
        for raw in sorted(raw_items):
            bundles = normalize_raw_label(raw)
            if not bundles:
                unmapped.add(raw)
                continue
            method = "exact" if all("=" in raw and raw in bundle for bundle in bundles) else "normalized"
            for bundle in bundles:
                key = (raw, bundle)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                patterns.append(NormalizedPattern(raw=raw, feats=bundle, method=method))
        patterns_by_language[language] = patterns
        unmapped_by_language[language] = unmapped

    return patterns_by_language, unmapped_by_language


def iter_feats(feats: str) -> tuple[set[str], bool]:
    """Return exact and atomic feature-value pairs, plus malformed flag."""
    if feats == "_":
        return set(), False

    pairs: set[str] = set()
    malformed = False
    for item in feats.split("|"):
        if "=" not in item:
            malformed = True
            continue
        for expanded in expand_feature_pair(item):
            pairs.update(normalize_data_feature_pair(expanded))
    return pairs, malformed


def summarize_file(
    path: Path,
    data_core_dir: Path,
    patterns_by_language: dict[str, list[NormalizedPattern]],
    unmapped_by_language: dict[str, set[str]],
) -> CoverageStats:
    family = path.parent.name
    language = path.stem.removesuffix("_core")
    stats = CoverageStats(
        family=family,
        language=language,
        file=str(path.relative_to(data_core_dir)),
    )
    patterns = patterns_by_language.get(
        language,
        patterns_by_language.get("__GLOBAL__", []),
    )
    unmapped_raw = unmapped_by_language.get(
        language,
        unmapped_by_language.get("__GLOBAL__", set()),
    )
    patterns_in_data: set[str] = set()

    with path.open(encoding="utf-8") as in_file:
        for line in in_file:
            line = line.rstrip("\n")
            if line.startswith("# sent_id"):
                stats.sentences += 1
                continue
            if not line or line.startswith("#"):
                continue

            columns = line.split("\t")
            if len(columns) != 10 or not TOKEN_ID_RE.match(columns[0]):
                continue

            stats.tokens += 1

            deprel = columns[7]
            if deprel in UD_CORE_DEPRELS:
                stats.core_deprel_tokens += 1
            else:
                stats.non_core_deprels.add(deprel)

            feats, malformed = iter_feats(columns[5])
            seen_patterns = [pattern for pattern in patterns if pattern.feats <= feats]
            if malformed:
                stats.malformed_feat_tokens += 1
            if seen_patterns:
                stats.non_core_feat_tokens += 1
                for pattern in seen_patterns:
                    patterns_in_data.add(pattern.label())
                    stats.non_core_raw_seen.add(pattern.raw)
                    stats.non_core_patterns_seen.add(pattern.label())
                    stats.non_core_feats_seen.update(pattern.feats)

            if not malformed and not seen_patterns:
                stats.core_feats_tokens += 1

    stats.non_core_feat_inventory_size = len({pattern.raw for pattern in patterns}) + len(
        unmapped_raw
    )
    stats.normalized_pattern_inventory_size = len(patterns)
    stats.unmapped_raw_feat_values = unmapped_raw
    all_pattern_labels = {pattern.label() for pattern in patterns}
    stats.normalized_patterns_in_data = patterns_in_data
    stats.normalized_patterns_not_in_data = all_pattern_labels - patterns_in_data
    return stats


def stats_to_row(stats: CoverageStats) -> dict[str, str | int]:
    return {
        "family": stats.family,
        "language": stats.language,
        "file": stats.file,
        "sentences": stats.sentences,
        "tokens": stats.tokens,
        "avg_sentence_length": ratio(stats.tokens, stats.sentences),
        "ud_core_deprel_tokens": stats.core_deprel_tokens,
        "ud_core_deprel_pct": pct(stats.core_deprel_tokens, stats.tokens),
        "ud_core_feats_tokens": stats.core_feats_tokens,
        "ud_core_feats_pct": pct(stats.core_feats_tokens, stats.tokens),
        "non_core_deprel_types": " ".join(sorted(stats.non_core_deprels)) or "NA",
        "raw_non_core_feat_inventory_size": stats.non_core_feat_inventory_size,
        "normalized_non_core_pattern_count": stats.normalized_pattern_inventory_size,
        "unmapped_raw_non_core_feat_values": " ".join(sorted(stats.unmapped_raw_feat_values))
        or "NA",
        "normalized_non_core_patterns_in_data": " ".join(
            sorted(stats.normalized_patterns_in_data)
        )
        or "NA",
        "normalized_non_core_patterns_not_in_data": " ".join(
            sorted(stats.normalized_patterns_not_in_data)
        )
        or "NA",
        "non_core_feat_tokens": stats.non_core_feat_tokens,
        "non_core_raw_feat_values_seen": " ".join(sorted(stats.non_core_raw_seen)) or "NA",
        "non_core_normalized_feats_seen": " ".join(sorted(stats.non_core_feats_seen)) or "NA",
        "malformed_feat_tokens": stats.malformed_feat_tokens,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize UD core DEPREL and FEATS coverage for *_core.conllu files."
    )
    parser.add_argument(
        "--data-core-dir",
        type=Path,
        default=Path("experiments_0616/data_core"),
        help="Directory containing family subdirectories with *_core.conllu files.",
    )
    parser.add_argument(
        "--non-core-feats-file",
        type=Path,
        default=Path("language_specific_features_and_dependency_relations.md"),
        help="Markdown/plain-text list of non-core Feature=Value entries.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments_0616/results/core_annotation_coverage_by_language.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--mapping-output",
        type=Path,
        default=Path("experiments_0616/results/non_core_feats_normalization_by_language.csv"),
        help="Output CSV path for raw-to-normalized non-core FEATS mapping audit.",
    )
    return parser.parse_args()


def mapping_rows(
    patterns_by_language: dict[str, list[NormalizedPattern]],
    unmapped_by_language: dict[str, set[str]],
    rows: list[dict[str, str | int]],
) -> list[dict[str, str]]:
    in_data_by_language: dict[str, set[str]] = {}
    for row in rows:
        labels = str(row["normalized_non_core_patterns_in_data"])
        in_data_by_language[str(row["language"])] = set() if labels == "NA" else set(labels.split())

    audit_rows: list[dict[str, str]] = []
    for language in sorted(set(patterns_by_language) | set(unmapped_by_language)):
        in_data = in_data_by_language.get(language, set())
        for pattern in sorted(patterns_by_language.get(language, []), key=lambda item: item.label()):
            label = pattern.label()
            audit_rows.append(
                {
                    "language": language,
                    "raw_feature_value": pattern.raw,
                    "normalized_ud_bundle": join_feats(pattern.feats),
                    "normalization_method": pattern.method,
                    "matched_in_data_core": "yes" if label in in_data else "no",
                }
            )
        for raw in sorted(unmapped_by_language.get(language, set())):
            audit_rows.append(
                {
                    "language": language,
                    "raw_feature_value": raw,
                    "normalized_ud_bundle": "UNMAPPED",
                    "normalization_method": "unmapped",
                    "matched_in_data_core": "no",
                }
            )

    return audit_rows


def write_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    files = sorted(args.data_core_dir.rglob("*_core.conllu"))
    if not files:
        raise SystemExit(f"No *_core.conllu files found under {args.data_core_dir}")

    patterns_by_language, unmapped_by_language = load_non_core_feat_patterns(
        args.non_core_feats_file
    )
    total_non_core_feats = sum(
        len({pattern.raw for pattern in patterns}) + len(unmapped_by_language.get(language, set()))
        for language, patterns in patterns_by_language.items()
    )
    total_patterns = sum(len(patterns) for patterns in patterns_by_language.values())
    total_unmapped = sum(len(values) for values in unmapped_by_language.values())
    if total_non_core_feats == 0:
        print(
            f"Warning: no Feature=Value entries found in {args.non_core_feats_file}; "
            "UD core FEATS coverage will be 100% unless malformed FEATS are present.",
            file=sys.stderr,
        )

    rows = [
        stats_to_row(
            summarize_file(path, args.data_core_dir, patterns_by_language, unmapped_by_language)
        )
        for path in files
    ]

    write_csv(args.output, rows)
    audit_rows = mapping_rows(patterns_by_language, unmapped_by_language, rows)
    write_csv(args.mapping_output, audit_rows)

    print(
        f"Loaded {total_non_core_feats} raw non-core FEATS values "
        f"from {args.non_core_feats_file}"
    )
    print(f"Normalized to {total_patterns} FEATS patterns; {total_unmapped} raw labels unmapped")
    print(f"Wrote {len(rows)} language rows to {args.output}")
    print(f"Wrote {len(audit_rows)} mapping rows to {args.mapping_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
