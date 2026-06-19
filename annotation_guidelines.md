GDUD Annotation Guidelines

Shared annotation schema for building Universal Dependencies (UD) treebanks
from glossed grammatical examples, applied to thirteen Tungusic and Mongolic
languages. These guidelines correspond to Section 3.2 of the accompanying
paper and document the decisions that keep annotation consistent across the
two families while preserving language-specific distinctions.

Languages.Mongolic: Daur, Kalmyk, Khalkha, Tu, Ordos. Tungusic: Even,
Evenki, Manchu, Nanai, Negidal, Ulch, Oroch, Udihe.

---

1. General principles

1. Examples are re-glossed following the Leipzig Glossing Rules; token
   boundaries and morpheme order follow the source analysis.
2. Each token receives UPOS, morphological features (FEATS), and a dependency
   relation (DEPREL) under one shared schema.
3. Where a construction maps cleanly onto a core UD label, the core label is
   used. Language-specific phenomena are kept as carefully delimited
   extensions (subtype relations, or non-universal features).
4. A morphological distinction is annotated only where it is explicit in the
   source gloss/description; features are evaluated at the value level.
5. The same phenomenon is annotated the same way across all thirteen
   languages, so that parallel constructions receive parallel analyses.

---

2. Feature and relation naming conventions

Universal features and relations follow UD v2. Naming is harmonized across
treebanks wherever the underlying category is the same.

2.1 Voice (universal feature; values restricted to the UD inventory)

| Category | Value |
|---|---|
| Causative | `Voice=Cau` |
| Reciprocal | `Voice=Rcp` |
| Passive | `Voice=Pass` |
| Middle | `Voice=Mid` |
| Active | `Voice=Act` |

Reflexive is not encoded as a Voice value. It is annotated with the
universal feature `Reflex=Yes`.

2.2 Derivation and other non-universal features

Valency-changing or stem-deriving morphology that falls outside the UD Voice
inventory is recorded with the language-specific feature `Deriv=...`
(e.g. `Deriv=Inch` inchoative, `Deriv=Mult` multiplicative, `Deriv=Res`
resultative, `Deriv=Detr` detransitive, `Deriv=Impers` impersonal). These
remain morphological features on a single verbal head; they do not introduce
extra syntactic nodes.

Aspect uses the UD inventory (`Aspect=Imp`, `Perf`, `Hab`, `Iter`, `Prog`,
…); spelling is unified across treebanks (e.g. imperfective is always `Imp`,
iterative always `Iter`).

2.3 Preserving genuinely distinct categories

Where a surface-similar label encodes a genuinely different category, the
distinction is kept rather than collapsed. For example, Evenki `Distal=Yes`
marks a remote-past tense (glossed PST.DIST, layered on `Tense=Past`) and
is deliberately kept separate from the spatial `Deixis` feature used in other
languages.

2.4 Dependency subtypes

Language-specific subtypes (e.g. `obl:tmod`, `acl:relcl`, `obl:agent`,
`nmod:poss`, `det:poss`, `flat:name`, `ccomp:quote`) are retained in the
released treebanks. The complete per-language inventories of features and
relations are in `feature_inventories/`.

---

3. Construction-specific decisions

3.1 Clause chaining and converbs

Clause chains consist of several non-finite predicates (converbs,
participles, auxiliary stacking) organized around finite or quasi-finite
anchors, often without overt subordinators.

- Chains are segmented into successive event blocks rather than forced
  into a single head-final hierarchy.
- The core predicate of the first event block is the `root`.
- Non-finite predicates within a block attach to their block head as `advcl`;
  when they share a subject and form a tight sequence, as `conj`.
- Each subsequent event block links to the first at the same level via `conj`.
- The root may be a converb when the first block is headed by one (a
  simultaneity or anteriority converb may carry the discourse-central event).

Auxiliaries that host tense/mood/interrogativity attach to the lexical verb
as `aux` (the lexical verb stays the clausal root); their inflectional
information is recorded as features.

3.2 Nominalization and clausal embedding

A nominalized or participial predicate that retains verbal argument
structure is treated as the head of a clause, not as a plain nominal head.

- Reported content / propositional subject → `csubj` (e.g. Manchu
  participle + nominalizer `-re-nge`).
- Complement clause → `ccomp` (e.g. Mongolic case-inflected verbal noun
  `-x-iig` under a verb of cognition).
- Relative clause → `acl:relcl` (verbal noun modifying a head noun).

A case suffix on a verbal noun is recorded as a feature (e.g. `Case=Acc`) on
the predicate, not as evidence for a nominal-headed phrase. Purely
referential, non-clausal nominalizations are the only cases treated as
nominal heads.

3.3 Copular and zero-copula constructions

Only a morphologically realized copula is a verbal head.

- (a) Overt copula as root, nominal predicate as its subject (e.g. Manchu
  `oho`).
- (b) Zero-copula nominal predication: the predicative noun is the `root`; no
  implicit copula is posited.
- (c) Adjectival predication with no copular morphology: the adjective is the
  `root`; a standard of comparison (ablative) attaches as `obl`.

3.4 Possessor agreement

- Dependent-marked possession (Manchu): the possessor carries genitive
  morphology and attaches as `nmod:poss`; the head noun is not person-indexed.
- Possessor with explicit person/number (Mongolic): possessive features are
  recorded on the possessor (e.g. `Poss=Yes|Person[psor]=1|Number[psor]=Plur`),
  attached as `det:poss` or `nmod:poss` as appropriate.

3.5 Voice and derivational morphology

Both families build voice/valency inside the verb by stacking derivational
suffixes. Such forms are annotated as a single verbal head; the valency
change is recorded in FEATS (see §2.1–2.2), and the licensed argument attaches
to that same head with an ordinary core relation.

- Causative: `Voice=Cau`; added causer = subject, demoted subject =
  `obl:agent` (instrumental).
- Reciprocal: `Voice=Rcp`; the clause stays mono-clausal.
- Reflexive / inchoative / and other stem derivations: `Reflex=Yes`,
  `Deriv=...`.

Stacked derivations on one stem are kept as one token; the morphology does
not support segmenting them into separate syntactic nodes.

3.6 Discourse particles and clitics

Particles and clitics that mark clause type, information structure, or
speaker stance sit outside argument structure.

- They are attached to the verbal head with `discourse`, and the core
  relations of the clause are left untouched.
- Clause-final and tag particles (including stacked ones) are each segmented
  and attached separately as `discourse`.
- A focus/additive clitic is segmented as its own token and attached as
  `discourse`, even when it cliticizes low (to a noun or pronoun); the host
  keeps its own relation to the predicate.

---

4. Resources in this repository

- `sample_treebank/` — sample sentences per language (CoNLL-U).
- `feature_inventories/` — complete FEATS values and DEPREL lists per language
  (universal vs. language-specific; core vs. subtype).
- `scripts/`, `splits_public/`, `results/` — experiment code, fold manifests,
  and per-fold scores.

For full source citations per language, see the appendices of the paper.
