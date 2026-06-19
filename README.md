Companion repository for the paper"GDUD: A Reproducible Pipeline for
Building Universal Dependencies Treebanks from Grammatical Descriptions,
with 13 Low-Resource Tungusic and Mongolic Languages."

GDUD builds Universal Dependencies (UD) treebanks for low-resource and
endangered languages from example sentences in published grammars, rather
than from running text. This repository covers thirteen Tungusic and
Mongolic languages.

Languages
Mongolic:Daur, Kalmyk, Mongolian (Khalkha), Tu, Peripheral Mongolian (Ordos)
Tungusic:Even, Evenki, Manchu, Nanai, Negidal, Ulch, Oroch, Udihe

Repository contents
- `sample_treebank/` — sample sentences (a few per language) in CoNLL-U,
  illustrating the annotation scheme.
- `scripts/` — conversion, validation, and parsing/evaluation scripts.
- `splits_public/` — fold manifests (sentence/token counts per fold) for the
  parsing experiments.
- `results/` — per-fold parsing and tagging scores.

Data availability
This repository provides sample sentences only, together with the
annotation materials and all experiment code, splits manifests, and results
needed to follow the evaluation procedure.

The complete thirteen treebanks are being prepared for release through
the [Universal Dependencies](https://universaldependencies.org/) repository
on a per-language basis, in collaboration with the UD team, targeting UD
release 2.19 (November 2026). During the review period, the full treebanks
are available from the authors on request.

The train/dev/test partitions used in the experiments are derived from the
full treebanks; `splits_public/` contains only the corresponding manifests
(counts), not the partitioned text.

License
- Data (sample treebanks): CC BY 4.0
- Code (scripts): MIT

Individual grammatical sources retain their original licenses; full
per-language source citations are listed in the paper's appendices.

Citation
If you use these materials, please cite the paper (full reference to be added
upon publication) and the original grammatical sources for each language
listed in the paper's appendices.
