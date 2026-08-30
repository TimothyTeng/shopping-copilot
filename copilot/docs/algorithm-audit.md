# Algorithm audit & experiment record

*2026-08-30. Consolidated reference for the algorithms in the agent, the
alternatives considered, and the experiments run this session. Point-in-time
measurements are quoted with the suite they came from; re-measure after code
around them changes (ablation results go stale).*

The agent is a pure-stdlib conversational retriever over a frozen 50k-product
catalog: no LLM, no embeddings, no network on the scored path. One turn flows
through eight algorithmic stages.

---

## 1. Algorithm inventory

Each stage: the algorithm in place, its known failure mode, and the alternative
algorithms that would produce a *functional* difference (not a tuning delta).

### 1.1 Text normalization — `src/normalize.py`
- **Used:** regex lowercasing + non-alphanumeric collapse; exact-token matching;
  a hand-built ~300-word `DIALOGUE_STOP` list. Query and catalog must normalize
  identically or phrase grounding silently fails.
- **Failure mode:** no typo tolerance — every layer keys on exact tokens.
  *(Addressed this session — see §2.1.)*
- **Alternatives:** stemming/lemmatization (Porter/Snowball); fuzzy edit-distance
  matching (symspell / BK-tree — the direct typo fix); phonetic keys (Double
  Metaphone) for brand misspellings.

### 1.2 Constraint extraction — `src/extract.py`
- **Used:** Layer A regex templates (latency shortcut) + Layer C "salience" —
  contiguous runs of IDF-informative, catalog-grounded, unseen tokens, with
  gap-bridging. Structure-blind by design, so paraphrase survives.
- **Failure mode:** interior glue tokens become required AND-terms
  (`"plus design"` forces `plus`); greedy leftmost segmentation.
- **Alternatives:** POS / noun-phrase chunking (reintroduces the sentence-
  structure dependency Layer C avoids); CRF/BIO sequence labeler (learned, needs
  data); named-attribute gazetteers (the reserved "Layer B cue families" — the
  cheapest predictable win).

### 1.3 Category resolution — `src/category.py`, `src/category_clf.py`
- **Used:** three tiers — (1) exact substring bucket-name match (conf 1.0, the
  benchmark path); (2) char-3–5-gram linear classifier (SGD log-loss, offline-
  trained, shipped as 0.7 MB pruned postings); (3) BM25 majority-vote fallback.
- **Failure mode:** resolves to sibling buckets on paraphrase (~17/26 stress
  probes). *But this is not the lever it appears to be — see §2.3.*
- **Alternatives:** dense-embedding nearest-centroid; hierarchical soft
  assignment (a distribution over buckets, re-weighted per turn) — the latter is
  the low-risk structural change if category work resumes.

### 1.4 Indexing — `src/index.py`, `src/bm25.py`
- **Used:** unigram inverted index; phrase matching as a *verified bonus* on a
  narrowed candidate set (bigram index rejected at ~180 MB for ~10% selectivity).
  Separate BM25 term-frequency index, built only when a mode needs it.
- **Alternatives:** positional/bigram index (phrase as filter, not bonus); ANN
  vector index (HNSW/FAISS — the only path to non-lexical recall, but LSA vector
  retrieval already rescued 0/200 as a rescorer); BM25+/BM25L/DFR/query-
  likelihood scoring swaps (cheap A/B via the `retrieval` switch).

### 1.5 Ranking / scoring — `src/rank.py`
- **Used:** log-product over constraint slots (conjunctive) with concavity
  `gamma`, phrase/category bonuses, shown-penalty, `_prior` tiebreak; plus
  `bm25`, `rrf`, and `auto` modes. The load-bearing algorithm.
- **Failure mode:** ~18 of ~38 residual errors are exact-score ties. *Partly
  fixable after all — see §2.2.*
- **Alternatives:** learning-to-rank (LambdaMART — overfitting risk given the
  query=document coupling); CombSUM/CombMNZ vs RRF for fusion; probabilistic
  satisfaction model; MMR diversity re-rank (built this session, §2.2).

### 1.6 Constraint state / retraction — `src/state.py`
- **Used:** accumulate-only slots; scoped override; negation-cue retraction;
  token-subset revocation.
- **Alternatives:** POMDP-style belief tracking (decaying per-constraint
  confidence); semantic contradiction detection instead of regex cues.

### 1.7 Question policy — `src/policy.py`
- **Used:** `"other"` weakly dominates (proven from the simulator's filter), so
  "what to ask" is near-constant; "when to speak" is a disclosure gate.
- **Alternatives:** information-gain question selection (measured *worse* vs this
  simulator, but standard for real users); bandit over gate thresholds.

### 1.8 Classifier training — `tools/train_category.py`
- **Used:** SGD log-loss on TF-IDF char-n-grams, top-140-per-class pruning,
  short-query augmentation.
- **Alternatives:** fastText supervised; distilled tiny embedding classifier.

---

## 2. Experiments this session

Method note that recurred: **measure the ceiling or the interval before believing
a direction.** It flipped the expected sign twice and rescued one delta.

### 2.1 Fuzzy token repair — SHIPPED (switch, on for `demo chat`)
`src/fuzzy.py`, `fuzzy_repair` (default off). Absent alpha tokens (`df==0`) map
to the nearest catalog term within a bounded Damerau-Levenshtein distance, via a
trigram candidate index. Damerau, not Levenshtein, because the common typo is an
adjacent transposition (`rainocat`→`raincoat`).

| suite | baseline | fuzzy | note |
|---|---|---|---|
| Official evaluator | 0.9383 | 0.9383 | no-op **by construction** (benchmark tokens are quoted from the target, `df>0`) |
| Perturb L0–L4 ×2 | — | — | +0.0000 in all 8 cells |
| Natural stress (n=26) | 0.4577 | 0.4946 | +0.037; per-probe diff: exactly 1 of 26 moved |
| — `typo` tag | 0.0000 | 0.9600 | MISS → rank 1 |

On for the `demo chat` / bm25 surface (where people type typos); off on the
graded path (a no-op there anyway). `fuzzy_repair_present` extends to
present-but-rare tokens (catches `lightwieght`, `df=1`) at the risk of corrupting
valid rare words — parked, unmeasured.

### 2.2 MMR tie re-rank — SHIPPED (now the graded default)
`src/rank.py` `_mmr_rerank`, `tie_rerank=mmr`. Maximal Marginal Relevance over
the head, similarity = title-token Jaccard, wired into the conjunctive return
only. Structurally cannot demote the single top-scored item.

The intuition was that a diversity objective would help *natural language* (where
hit@10 has headroom). It was the reverse:

| surface | baseline | mmr | note |
|---|---|---|---|
| Benchmark | 0.9383 | **0.9456** | MRR 0.883 → 0.911, hit stays 1.000 |
| — browsing (n=80) | 0.819 | 0.889 | +0.070 MRR — the gain lives here |
| Stress, conjunctive | 0.4946 | 0.4354 | *hurts* — diversity needs trustworthy relevance |
| Chat, bm25 | — | — | **inert** (MMR only touches conjunctive) |

Flat across λ 0.5–0.9 (0.9445–0.9459): a plateau, not a fit. The +0.0074 is
below the ±0.03 noise floor, but that floor is the *unpaired* absolute-score
band; a paired bootstrap (§2.4) puts the delta at **95% CI [+0.0017, +0.0138],
P(delta>0)=0.997 — significant**, so it is now the default. This partly corrects
the "ties are information-theoretically unfixable" claim: unfixable by adding
constraint *information*, but a query-independent diversity prior reorders about
half of them the right way.

### 2.3 Semantic category resolution — REJECTED on ceiling (not built)
Scoped as the fix for the ~17/26 wrong-bucket diagnosis, then rejected before
building the model. Forcing the category to the *truth* (any resolver's ceiling),
on the stress set with fuzzy on:

| forced category | score | Hit@10 | MRR |
|---|---|---|---|
| classifier (current) | 0.4946 | 0.615 | 0.282 |
| full oracle (true key **and** bucket) | 0.4578 | 0.538 | 0.334 |
| true bucket for **ranking only** | 0.5480 | 0.654 | 0.373 |

The full oracle *hurts*. The code couples resolution to extraction via
`suppress = category_key.split()`, so a more accurate key strips more of the
shopper's own category words out of constraint mining — and on free text those
carry coverage, so accuracy cancels itself. Isolating the ranking bucket recovers
+0.053, but capturing it needs a near-perfect resolver **and**
`suppress_category_tokens=False` (added as a switch; benchmark −0.001,
stress-neutral until the resolver is accurate). `category_bonus` swept 0.0–2.0 is
flat (~0.494). Net: the ceiling for *all* category-resolution work is +0.053 on
n=26 — inside the noise band — behind a mechanism fix. A 40–100 MB embedding
model is not justified by its own ceiling. The real free-text bottleneck is
vocabulary (shopper words ≠ product words), which the stress harness's 0.98
"product's own words" oracle already localises.

### 2.4 Bootstrap confidence intervals — BUILT
`tools/bootstrap.py` + `harness ci`. `score_ci` (absolute) and `paired_delta_ci`
(one resample scores both configs, so shared session variance cancels — the
correct test for A-vs-B). Percentile method, seeded, pure stdlib. Validated:
self-vs-self delta is exactly 0; a provably-neutral change reports [0, 0].

```bash
python -m tools.harness ci                          # CI on the current score
python -m tools.harness ci --compare tie_rerank=mmr # is the delta real?
```

Byproduct: the single-config composite CI is **±0.0104**, not ±0.03. The ±0.03
working-rule floor is the *hit-rate* band; with hit saturated (zero variance) the
composite floor is ~3× tighter. Use `harness ci` per-comparison instead of the
rule of thumb — especially for hit-saturated changes, where the paired test is
the right instrument.

---

## 3. Net state

| item | outcome |
|---|---|
| Graded score | **0.9383 → 0.9456** (MMR default; significant at 95%) |
| Fuzzy repair | on for `demo chat`; NL typos 0.00 → 0.96, benchmark provably neutral |
| Semantic category | rejected on ceiling before building |
| Bootstrap CIs | built (`harness ci`) and used to settle MMR |

All switches remain in `src/config.py` with the measurement recorded at each
knob. Nothing on the scored path touches the network; the competition kit stayed
read-only throughout (`git -C techjam-conversational-search status --short`).

## 4. Where the headroom actually is

Retrieval (Hit@10) is finished on the benchmark. The remaining benchmark points
are ordering, and the ties MMR does not reach are genuine exact-score ties. The
open frontier is **free-text vocabulary** — shopper words vs product words — the
one axis the benchmark structurally cannot reward (its simulator quotes the
target). Fuzzy closed the typo slice; the parked attacks (LSA vector retrieval:
0/200; hand synonym map: ~nothing) are recorded as measured negatives. The next
untried lever with a plausible ceiling is weighted term-association / a network-
free semantic layer on the *retrieval* side (not category), gated so it can only
augment a complete Tier-0 result.
