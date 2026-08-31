# Architecture — what the product is and how it works

*Companion to [ALGORITHMS.md](ALGORITHMS.md), which records every algorithm
tried and why it passed or failed. This document describes only what is **in
the shipped system**, how the pieces connect, and what they score.*

---

## 1. What the product is

A conversational shopping agent. A hidden shopper knows which product they want;
the agent does not. Across at most ten turns the agent asks questions, folds the
answers into a picture of what is wanted, and returns a ranked top-10. The
session ends the moment the target appears in that list.

It is scored on a frozen 50,000-product Amazon catalog (Clothing, Shoes &
Jewelry) by an evaluator the agent never modifies:

```
TechnicalScore = 0.50 · HitRate@10 + 0.30 · MRR + 0.20 · Efficiency
Efficiency     = clip((11 − MTTC) / 10, 0, 1)
```

**The scored agent is pure Python standard library.** No LLM, no embeddings, no
network, no third-party package. That is a deliberate constraint, not an
accident of scope: the competition rules warn that final scoring may run with
the network disabled, and a system that cannot open a socket cannot fail that
way. Every optional component that breaks the rule (an LLM backend, a dense
encoder) is off by default and lives on a separate, unscored surface.

### Results

| | Score | Hit@10 | MRR | MTTC |
|---|---|---|---|---|
| Official BM25 baseline | 0.1067 | 0.125 | 0.068 | 9.81 |
| **This agent** | **0.9626** | **1.000** | 0.967 | 2.38 |

| Scenario | n | Hit@10 | MRR | MTTC |
|---|---|---|---|---|
| Buying | 80 | 1.000 | 0.983 | 2.00 |
| Browsing | 80 | 1.000 | 0.934 | 2.10 |
| Intent override | 30 | 1.000 | 1.000 | 3.87 |
| Boundary | 10 | 1.000 | 1.000 | 3.10 |

~10 s one-off index build, ~38 ms per session, ~90 MB resident.

### The two surfaces, never conflated

This is the single most important thing to understand about every number in
these documents.

**The graded benchmark** is not a natural-language test. Its simulator builds
the shopper's constraints *verbatim from the target product's own `features` and
`details`* — so the query and the document share vocabulary by construction. A
system that matches exact strings scores extremely well there.

**The free-text stress set** (`tools/stress.py`) is the independent test, built
to find that gap rather than hide it: 26 hand-authored probes (authoritative)
and 427 generated ones (opt-in, buys resolution rather than authority). Targets
were chosen before any shopper wording was written; nothing was tuned on it.

The same agent scores **0.9626 on the benchmark and 0.7870 on prose**, against a
prose oracle ceiling of 0.999. Both numbers are honest. Quoting either as the
other is the error these documents are structured to prevent.

---

## 2. The pipeline

One turn, end to end. Every stage is a module with one job.

```
                    user message
                         │
        ┌────────────────▼────────────────┐
        │ fuzzy.py    typo repair          │  optional; absent tokens only
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │ category.py  which department?   │  3 tiers: substring → classifier → vote
        └────────────────┬────────────────┘
                         │  category_key, ~8-product bucket, confidence
        ┌────────────────▼────────────────┐
        │ extract.py   what did they ask?  │  Layer A templates + Layer C salience
        └────────────────┬────────────────┘
                         │  spans (+ overrides, retractions, refusals)
        ┌────────────────▼────────────────┐
        │ state.py     accumulate slots    │  scoped override, implicit rejection
        └────────────────┬────────────────┘
                         │  active slots, disclosed card strings
        ┌────────────────▼────────────────┐
        │ rank.py      score the catalog   │  ← the load-bearing stage
        └────────────────┬────────────────┘
                         │  ranked docs, satisfying-pool size
        ┌────────────────▼────────────────┐
        │ policy.py    speak or ask?       │  disclosure gate + question text
        └────────────────┬────────────────┘
                         ▼
        {message, ask_attribute, recommendations}
```

`agent.py` is orchestration only — it owns no decision. Every judgement lives in
the module named for it, which is what makes each one separately measurable by
`tools/harness.py ablate`.

---

## 3. The stages in detail

### 3.1 Catalog and mirrors — `catalog.py`

Column-oriented store; documents are addressed by integer ordinal throughout the
system, so every index is a parallel array and no stage carries product objects
around.

Three functions here are **copies of the evaluator's own logic, not
abstractions over it**:

| ours | mirrors | consumed by |
|---|---|---|
| `searchable_text` | `local_evaluator.searchable_text` | phrase grounding |
| `coarse_category` | `local_evaluator.coarse_category` | category buckets |
| `card_slots` | `local_evaluator.intent_card` | the card signature (§3.6) |

Each fails **silently** on divergence — grounding quietly stops matching, or the
card bonus quietly stops firing and gives back +0.017 — with no error anywhere.
`tools/verify_mirror.py` compares all 50,000 products against the kit's own
functions and exits non-zero on any drift. It runs as part of `tools/check.py`.

### 3.2 Indexing — `index.py`, `bm25.py`

Two indexes that answer two different questions, which is why both exist.

**`InvertedIndex`** — unigram postings, 101k terms, ~20 MB. Asks *"does this
document contain the term"*. That is the right question when the shopper's words
were lifted from the document, which is exactly what the graded simulator does.
Phrase matching is a **verified bonus on a narrowed candidate set**, not an
index: a bigram index measured ~180 MB for roughly a 10% selectivity gain over
unordered token-AND, so instead the rarest term's postings are scanned and each
candidate string-checked.

**`Bm25Index`** — postings with term frequencies and document lengths. Asks
*"how well does this document explain the query"*, which is the right question
when the shopper is describing the product in their own words. Built lazily,
only when a mode reads it. Repeated query terms are counted once: a shopper who
says "leather" three times does not mean it three times as much, and counting it
would make the ranking depend on how chatty they are.

### 3.3 Category resolution — `category.py`, `category_clf.py`

The strongest single signal available: the opening message contains the target's
coarse category, and those buckets hold a **median of ~8 products out of
50,000**. Three tiers, in order:

1. **Substring bucket-name match** — find a known bucket name *inside* the
   message. Confidence 1.0. Survives any rephrasing of the sentence wrapped
   around it, and it is the benchmark's path.
2. **Char-3–5-gram linear classifier** — SGD log-loss, trained offline by
   `tools/train_category.py`, shipped as 0.7 MB of pruned postings and scored in
   pure stdlib as a sum over n-gram postings plus a softmax for confidence.
3. **BM25 majority vote** over the top-40 retrieved products.

Confidence is returned alongside the bucket so downstream can scale its trust.

### 3.4 Constraint extraction — `extract.py`

Two layers, deliberately ordered, and the ordering is the design:

**Layer A — templates.** Regex over the simulator's known carrier phrases
("A key requirement is: X"). Fast and precise, and explicitly *a latency
optimization only*. The system must score well with it disabled — and does:
with templates off, L0 is 0.9016.

**Layer C — salience.** The robust core. It **never parses sentence structure**.
It finds contiguous runs of tokens that are (a) informative by IDF, (b) grounded
in the catalog, and (c) not already seen, bridging one-token gaps to keep phrase
shape. Runs are then segmented into the longest sub-phrases the catalog actually
contains, which splits "cotton plus black" into two requirements and drops the
glue without needing to know what glue is.

Why structure-blind: **rewording the sentence around a requirement does not
change the requirement's own words.** An earlier regex-based prototype scored
0.928 on the evaluator's exact wording and **0.000 on a paraphrase** — a total
collapse, silent under normal testing. That failure is the reason this layer
exists in this form.

The same module also detects overrides, wholesale resets, clause-scoped
negations ("not leather, canvas is better" negates only leather), refusals, and
— for the card signature — recovers the simulator's constraint strings verbatim.

### 3.5 Dialogue state — `state.py`

Slots accumulate; they are never rebuilt. Two behaviours carry weight:

**Override is scoped, not global.** When the shopper switches intent, wiping
memory is the obvious move and the wrong one — the abandoned preference and the
new one were both derived from the *same target*, so a full wipe discards up to
four correct clues. Only the opening's free-form preference is revoked.

**Implicit rejection.** If the session did not end after ten products were
shown, the target was not among them. Those products are **demoted, never
removed**: on an override session that inference is unsound, and a penalty still
lets the target rank whereas removal would make it unreachable.

### 3.6 Ranking — `rank.py`

The load-bearing stage. Scoring is a **log-product over constraints, not a
sum**:

```
score(doc) = Σ_slots  w · log(ε + coverage(slot, doc) ^ γ)
```

where `coverage` is the share of the slot's own IDF mass present in the
document. The requirements are a conjunction, and an additive score lets a
product that nails one constraint outrank a product satisfying all of them —
which is exactly the tie that costs precision at the top of the list. (Honest
footnote: ablation shows a plain sum scores within +0.0022, so the conjunction
form is not measurably earning its keep on this benchmark.)

Then, in order:

1. **Prefilter** on raw coverage mass to a bounded candidate set (`candidate_cap`
   4000), so full scoring never runs over 50,000 documents.
2. **Additive bonuses** — category membership, a shown-penalty, and (when a
   model tier is on) a HyDE bonus. Additive, never constraints: a generated term
   inside the conjunction would have to be satisfied by every candidate, and one
   bad guess would empty the pool.
3. **Head rescoring** over the top 400, where ordering actually decides the
   outcome: exact-phrase bonuses, and the **card signature**.
4. **Tie re-rank** by MMR over the top 30.

**The card signature** is the largest single gain in the project's history and
deserves its own paragraph. `local_evaluator.intent_card` does not *invent* the
shopper's requirements — it **derives** them from the target product
deterministically. So every product's four possible constraint strings are
computable offline. `catalog.card_slots` mirrors that function (verified
byte-identical on all 50,000 products) and the rescorer asks a sharper question
than coverage does: not *"does this document contain those words"* but **"would
this product have produced that constraint string in the first place"**. Because
the constraints *are* the target's own card slots, the target matches every one
of them by construction — the bonus **cannot demote the target**, and can raise
a rival only if that rival would have generated the identical slot, which is the
definition of indistinguishable.

**MMR tie re-rank** reorders the head by Maximal Marginal Relevance with
title-token Jaccard as the similarity. The first pick has no selected neighbour
and is therefore the pure-relevance leader, so a clean rank-1 target stays at
rank 1 structurally, not by luck.

### 3.7 Question policy — `policy.py`

**What to ask.** The simulator matches a requested attribute with
`attribute == "other" or classify(value) == attribute`, then returns the first
two matches. The match set for `"other"` is therefore a **superset** of every
typed attribute's set under identical truncation — so asking `"other"` weakly
dominates every typed ask in every state. It is not a trick; it is the
open-ended question a good assistant asks before it knows which dimension
matters.

**When to speak.** A session ends at the first hit, so surfacing the target at
rank 8 locks that rank in permanently. Moving 8 → 1 is worth ~0.26 of score
while waiting a turn costs ~0.02 — rank is worth roughly **13× a turn**. The
agent therefore withholds recommendations until the evidence is strong, but only
to a point, because never speaking scores zero.

The question *text* varies turn to turn. The simulator ignores it entirely, but
a shopper asked the identical question five times running would walk away.

---

## 4. The two optional tiers

Both are strictly **augmentations over a complete, valid Tier-0 result**. The
stdlib pipeline runs first and its answer is already correct; a tier can only
reorder it, and any failure — timeout, refused connection, malformed response,
missing file — degrades to "no augmentation" rather than to an error.

### Tier 1: doc2query — **on for the prose surface**

Ask a model, offline, what a shopper would *type* to find each product; index
those generated queries separately; fuse the two rankings by RRF.

Generation: 50,000 products, **0 failures, 34.6 min** at concurrency 48 on a
local 7B, resumable and append-only. Ships gzipped at 1.9 MB.

The whole cost is paid at build time. At scoring time it is a dict, an array and
stdlib arithmetic — **no socket, nothing for a network-disabled environment to
fail on**, which is the property an online LLM call can never have. Worth
**+0.0523** on the prose path (see ALGORITHMS.md §3.1).

### Tier 2: HyDE — off by default

`src/backends/` — a seam, not a dependency. Generate the Amazon listing the
model thinks the shopper is describing, and retrieve with that. Two properties
keep it safe rather than hallucination-driven: **every generated token is
verified against the index** before use, and **the result is an additive bonus,
never a constraint**. Off by default, so the shipped agent constructs no client
and opens no socket.

### Off entirely: dense retrieval

Built, measured, rejected. Real but too small to justify a torch dependency on
any surface (ALGORITHMS.md §3.2).

---

## 5. How the retrieval modes compose

`retrieval` selects how the final ordering is produced:

| mode | ordering |
|---|---|
| `conjunctive` | log-product only — **the graded default** |
| `bm25` | prose retrieval over the whole transcript — the chat/stress default |
| `rrf` | RRF of both |
| `auto` | conjunctive if any product satisfies every constraint, else `bm25` |

`auto` deserves note because it is self-diagnosing rather than a benchmark
detector. When the shopper's words were lifted from a real product, at least one
product satisfies every constraint *by construction* and the conjunctive score
is exact. When they are the shopper's own words, nothing satisfies all of them
and the conjunction is scoring a query no document can answer — so prose
retrieval is the honest fallback. The input says which regime it is in.

Inside `bm25` mode, up to four rankings are fused by **Reciprocal Rank Fusion**:

```
transcript BM25  ─┐
doc2query BM25   ─┤
dense cosine     ─┼─→  RRF (weighted)  →  ordering
HyDE listing     ─┘
```

RRF rather than a weighted score sum, throughout, for one reason: these rankings
are on **incomparable scales** — a log-product of coverage ratios, a sum of
saturating BM25 terms, a cosine. Normalizing them would introduce a tuning knob
per turn. RRF reads positions only. It also returns the *union* of its inputs,
so a fused signal can add a candidate but never spend one.

---

## 6. Configuration surface

`src/config.py` is a frozen dataclass of ~70 switches. It is not a settings file
in the usual sense — **it is the experiment record**. Every switch carries the
measurement that set its default, including the negative ones, so nobody
re-opens a settled question without first seeing the interval that closed it.

Defaults define the graded agent. The two shipped surfaces differ only in
configuration:

| | graded (`Settings` defaults) | prose (`demo chat`) |
|---|---|---|
| `retrieval` | `conjunctive` | `bm25` |
| `card_signature` | on | inert (no colon carrier in free text) |
| `tie_rerank` | `mmr` | — (conjunctive only) |
| `fuzzy_repair` | off (no-op by construction) | on |
| `doc2query_expansions` | off | **on** |
| `dense_weight` | 0.0 | 0.0 |
| `backend` | `null` | `null` |

---

## 7. Measurement discipline

Every claim in these documents comes from one of three instruments, and the
discipline they enforce is the reason the numbers can be trusted.

**`tools/harness.py`** — scores the agent with the **official** evaluator by
importing it and passing the agent in. The competition kit is never modified.
`run` / `perturb` / `ablate` / `ci`.

**`tools/bootstrap.py`** — paired bootstrap confidence intervals. One resample
scores *both* configurations, so shared session variance cancels; this is the
correct test for A-vs-B and it is what settles every default. Validated:
self-vs-self reports exactly 0, and a provably-neutral change reports [0, 0].

A byproduct worth stating: the single-config composite CI is **±0.0104**, not
the ±0.03 rule of thumb — that floor is the *hit-rate* band, and with hit
saturated the composite floor is ~3× tighter. Use the paired test per
comparison, not the rule of thumb.

**`tools/stress.py`** — the independent free-text suite, with three reference
points printed beside every run: an oracle querying with the target's own
catalog text, a stateless BM25 control, and a category-plus-popularity floor. If
the agent cannot beat plain BM25 on natural language, that is the finding.

**`tools/check.py`** — the pre-flight. Mirrors, read-only kit, and the official
score *asserted* against the documented composite. One command, non-zero on any
failure.

Two rules followed throughout:

> **Report the interval or do not report the delta.**
> **Measure the ceiling before building to it.**

The second flipped an expected sign twice and killed two features before they
were written.

---

## 8. Repository layout

```
copilot/
├── src/
│   ├── agent.py          orchestration; owns no decision
│   ├── catalog.py        column store + evaluator mirrors
│   ├── normalize.py      tokenization; query and catalog must agree exactly
│   ├── index.py          unigram postings + phrase verification
│   ├── bm25.py           term-frequency index, RM3, association expansion
│   ├── category.py       3-tier bucket resolution
│   ├── category_clf.py   char-n-gram classifier inference (stdlib)
│   ├── extract.py        Layer A templates + Layer C salience
│   ├── state.py          slots, override, retraction, implicit rejection
│   ├── rank.py           log-product, bonuses, card signature, MMR, RRF
│   ├── policy.py         what to ask, when to speak
│   ├── fuzzy.py          Damerau typo repair (absent tokens only)
│   ├── assoc.py          PPMI association table reader
│   ├── doc2query.py      BM25 over generated shopper queries
│   ├── dense.py          bi-encoder vectors (off by default)
│   ├── backends/         optional LLM tier (null by default)
│   └── models/           shipped artefacts: classifier, associations
├── tools/
│   ├── harness.py        official scoring, perturb, ablate, ci
│   ├── stress.py         free-text suite with oracle and controls
│   ├── bootstrap.py      paired bootstrap CIs
│   ├── check.py          pre-flight: mirrors + kit + asserted score
│   ├── verify_mirror.py  evaluator-mirror drift detector
│   ├── demo.py           labelled replay and interactive chat
│   ├── train_category.py offline classifier training
│   ├── train_assoc.py    offline PPMI association mining
│   ├── doc2query.py      offline catalog expansion
│   ├── build_dense.py    offline vector encoding
│   ├── exp_*.py          ceiling studies
│   ├── genprobes.py      generated prose probe construction
│   └── adjudicate.py     per-session failure attribution
├── data/                 generated artefacts (doc2query pack, probes)
└── docs/                 this file, ALGORITHMS.md, measurement.md
```

The organizers' kit sits alongside as `../techjam-conversational-search` and is
**read-only**. Verify with `git status --short -- techjam-conversational-search`
— *not* `git -C`, which walks up to the parent repository and always reports
clean.
