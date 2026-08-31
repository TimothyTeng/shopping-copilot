# Run record — 2026-08-31

Everything below was measured in one sitting on the current tree
(`docs-architecture-and-dense-experiment`, defaults from `src/config.py`:
`retrieval=conjunctive`, `tie_rerank=mmr`, `card_signature=True`,
`fuzzy_repair=False`, `backend="null"`). Linux, CPython 3.12.3, no network.
Nothing was tuned between runs; the only thing that varies is the flag named in
each table.

The two worlds are kept apart, as always: the official evaluator is a benchmark
whose constraints are lifted verbatim from the target's own catalog text, and
`tools/stress.py` is the independent free-text test. A number from one is never
quoted as evidence about the other.

## 1. Official evaluator

```bash
PYTHONIOENCODING=utf-8 python3 -m tools.harness run
```

```
score 0.9626   hit 1.000   mrr 0.967   mttc 2.38   eff 0.863
  boundary         n=10   hit 1.000  mrr 1.000  mttc 3.10
  browsing         n=80   hit 1.000  mrr 0.934  mttc 2.10
  buying           n=80   hit 1.000  mrr 0.983  mttc 2.00
  intent_override  n=30   hit 1.000  mrr 1.000  mttc 3.87
```

200 sessions in 7.2 s (36 ms/session), index built in 8.2 s over 50,000
products / 101,064 terms. Written to `results/official.json`.

This reproduces the documented headline exactly (0.9626, baseline 0.1067). Hit
is saturated at 1.000 on every scenario, so all four scenario columns differ
only in MRR and MTTC. `intent_override`'s 3.87 MTTC remains the worst-looking
and is near its measured floor of 3.60 — the clock does not start until the
override message is delivered at turn 3 or 4 (`local_evaluator.py:234`).

### The two defaults, re-measured at current settings

`harness ci --compare <switch>` re-scores the whole set with one switch flipped
and bootstraps the paired delta over the same 200 sessions:

| switch off | score | Δ vs default | 95% CI | p(Δ>0) |
|---|---|---|---|---|
| `tie_rerank=none` | 0.9523 | −0.0103 | [−0.0167, −0.0048] | 0.000 |
| `card_signature=false` | 0.9456 | −0.0169 | [−0.0256, −0.0093] | 0.000 |

Both still clear zero. `card_signature` reproduces its recorded +0.0170 exactly.
`tie_rerank` does not: it was recorded at **+0.0074** and removing it now costs
**0.0103**, because that figure was measured *before* the card signature landed
and the two interact — the signature creates the tied clusters MMR then orders.
The right reading is that the recorded number is a build-up history, and this
table is what each switch is worth in the tree as it stands.

## 2. Free-text stress test

Four runs: both probe sets × both retrieval modes. Every line is the official
metric, and every run prints its own paired bootstrap (10,000 resamples,
seed 0) against the two independent BM25 controls.

### 2a. `conjunctive` — the graded default, run on prose

| | n=26 hand-authored | n=427 generated |
|---|---|---|
| agent (ours) | **0.3985** hit 0.500 mrr 0.223 mttc 6.92 | **0.4451** hit 0.541 mrr 0.278 mttc 6.43 |
| BM25, turn-matched | 0.4920 hit 0.615 | 0.6409 hit 0.756 |
| BM25, whole transcript | 0.4775 hit 0.538 | 0.6415 hit 0.691 |
| oracle: the product's own words | 0.9827 hit 1.000 | 0.9990 hit 1.000 |
| verbatim overlap with target | 0.68 | 0.54 |

Paired deltas, agent − control:

| | Δ vs whole-transcript BM25 | Δ vs turn-matched BM25 |
|---|---|---|
| n=26 | −0.0790, CI [−0.2104, +0.0539], p(Δ>0) 0.116 — spans zero | −0.0935, CI [−0.2379, +0.0531], p 0.104 — spans zero |
| n=427 | **−0.1964**, CI [−0.2364, −0.1576], p 0.000 — below zero | **−0.1958**, CI [−0.2337, −0.1590], p 0.000 — below zero |

At n=26 the agent's deficit on the conjunctive path is inside the interval; at
n=427 it is not. The generated set does what it was built to do — it converts a
direction into a measurement, and the measurement is that **the conjunctive
ordering loses to plain BM25 on free text by ~0.20, decisively.** That is the
existing conclusion, now with an interval that excludes zero rather than one
that straddles it.

### 2b. `bm25` — the chat/deployment default

| | n=26 | n=427 |
|---|---|---|
| agent (ours) | **0.6339** hit 0.846 mrr 0.305 mttc 5.04 | **0.7347** hit 0.867 mrr 0.529 mttc 3.85 |
| BM25, turn-matched | 0.4920 | 0.6409 |
| BM25, whole transcript | 0.4775 | 0.6415 |

Paired deltas:

| | Δ vs whole-transcript | Δ vs turn-matched |
|---|---|---|
| n=26 | **+0.1564**, CI [+0.0208, +0.2974], p 0.988 — clears zero | +0.1420, CI [+0.0258, +0.2663], p 0.993 — clears zero |
| n=427 | **+0.0933**, CI [+0.0583, +0.1288], p 1.000 — clears zero | +0.0939, CI [+0.0653, +0.1222], p 1.000 — clears zero |

Both probe sets now agree, and both clear zero: **once retrieval matches the
input, the dialogue earns its keep.** The n=427 delta is about 60% of the n=26
point estimate, which is the usual direction — the small set's estimate was
optimistic, and its interval said so.

The whole retrieval-mode decision is visible in one comparison: the same agent,
same probes, differs by +0.24 (n=26) / +0.29 (n=427) between the two modes, in
the opposite direction from the benchmark's 0.9626 vs 0.4029. Nothing dominates;
the default stays per-surface.

### 2c. By tag (agent only; direction, not measurement)

| tag | `bm25`, n=427 | `conjunctive`, n=427 |
|---|---|---|
| multi_attr (48) | 0.8463 | 0.5504 |
| natural (45) | 0.8384 | 0.6079 |
| vague (70) | 0.7364 | 0.4878 |
| colloquial (73) | 0.7219 | 0.3043 |
| brand (57) | 0.7171 | 0.5799 |
| non_catalog (73) | 0.6931 | 0.3280 |
| **negation (61)** | **0.6502** | 0.3758 |

`negation` remains the weakest tag on the prose path, unchanged at 0.6502
against `multi_attr`'s 0.8463 — the two figures the known-weaknesses list quotes
still hold on a fresh run. `colloquial` is the tag that gains most from bm25
(+0.42), which is the vocabulary story in one row: shopper words that never
appear in a slot are still terms an inverted index can score.

### 2d. One flag checked in passing

`fuzzy_repair=True` on the conjunctive n=26 set: **0.3985 → 0.4354**, moving
exactly one probe (`typo_raincoat`, 0.000 → 0.960, rank 1). Interval on the set
still spans zero. Benchmark-neutral by construction, so this is the same result
the switch was recorded with, reproduced at the current defaults.

### Note on figures quoted elsewhere

`CLAUDE.md`'s retrieval-frontier table lists conjunctive-on-stress as 0.3749 and
bm25 as 0.6411; this tree measures 0.3985 and 0.6339. The bm25 figure matches
the 0.6339 recorded for the pre-HyDE prose path, and the conjunctive figure has
drifted with the defaults that landed since the frontier was swept. The prose
numbers in the weaknesses list (0.6502 negation, 0.8463 multi_attr) reproduce
exactly, which confirms they were bm25-mode numbers. Treat this file's table as
the current one and the frontier table as the shape, not the values.

## 3. The saved conversations

```bash
PYTHONIOENCODING=utf-8 python3 -m tools.transcripts --n 50 --out results/transcripts.json
```

```
benchmark  n=50   score 0.9677  hit 1.000  mrr 0.983  mttc 2.36
stress     n=26   score 0.3985  hit 0.500  mrr 0.223  mttc 6.92
```

- `results/transcripts.json` — every turn of 50 benchmark sessions (stratified
  to the public set's scenario mix, seed 0, L0 wording) and all 26 hand-authored
  probes. Each turn carries the shopper's message, the agent's prose, its
  `ask_attribute`, the resolved `category_key`, the active clue slots, and the
  full top-10 with the target flagged.
- `results/transcripts.html` — the same data as a readable session log, with
  `const DATA` embedded. Verified this run: the embedded payload is byte-identical
  to the freshly generated JSON, so the page is current and the dump is
  deterministic under a fixed seed.

The stress half of the dump is on the **conjunctive** path (it takes the config
default), so it is the 0.3985 world, not the 0.6339 one. Read it as the failure
gallery it is.

### What the 50 benchmark sessions look like

- Target rank: **1 in 49 of 50**; the single exception lands at rank 7.
- Turn of the hit: 2 in 36 sessions, 3 in 7, 4 in 6, 1 in one.
- Mix: 20 browsing, 20 buying, 8 intent_override, 2 boundary.
- **52 of 118 turns return an empty list.** That is the holding-back policy made
  visible: rank is worth ~13× a turn, so the agent asks again rather than commit
  a weak ordering. A reader of the log should expect the first turn or two of
  most sessions to show a question and no products.

A representative one (`public_0169`, boundary, target *Amazon Essentials
Women's Pull-On Knit Jegging*): turn 1 resolves the category `women jeans` and
asks an open question, holding the list back; turn 2 the shopper declines to
state a preference and the agent asks again rather than guess; turn 3 the
shopper discloses `cotton` plus the full fabric string, and the target comes
back at rank 1. Three turns, one hit, nothing shown until it was worth showing.

### The 26 probes, one line each

| probe | tags | rank | turn |
|---|---|---|---|
| tee_startrek | control_easy, brand | 1 | 1 |
| sunglasses_fitover | natural | 1 | 2 |
| necklace_butterfly | natural | 1 | 2 |
| vague_clear_backpack | vague | 1 | 3 |
| control_twin_ballet_flat | control_impossible | 2 | 4 |
| slipper_knit_women | natural, multi_attr | 3 | 3 |
| watch_diver | non_catalog, multi_attr | 6 | 3 |
| necklace_cross | natural | 7 | 3 |
| sweater_men_bomber | colloquial, non_catalog | 7 | 3 |
| sweater_women_merino | natural | 7 | 3 |
| sunglasses_gucci | brand | 8 | 3 |
| jogger_champion | brand | 8 | 3 |
| contradiction_leather_canvas | contradiction | 8 | 4 |
| watch_ladies_daydate | natural | miss | — |
| sneaker_gym_mesh | colloquial, brand | miss | — |
| sneaker_denim_slipon | natural | miss | — |
| dress_vintage_floral | non_catalog, multi_attr | miss | — |
| sandal_beach_arch | natural | miss | — |
| sandal_toddler_closed | natural | miss | — |
| loafer_leather | natural, negation | miss | — |
| flat_leather_walking | vague | miss | — |
| slipper_men_scuff | colloquial | miss | — |
| raincoat_anorak | natural | miss | — |
| switch_shoes_to_pants | category_switch | miss | — |
| negation_turtleneck | negation | miss | — |
| typo_raincoat | typo | miss | — |

Thirteen hits, thirteen misses, and the misses are the documented weaknesses in
one column: both `negation` probes, the `category_switch` probe (category is
resolved once and never revised), the `typo` probe (`fuzzy_repair` is off by
default — turning it on fixes precisely this row), and a run of plain `natural`
wordings where the shopper's vocabulary and the product's simply do not meet.
The probes that hit tend to hit on turn 2 or 3 and then stop; the ones that miss
run all ten turns, which is why the stress MTTC is 6.92 against the benchmark's
2.38.

## 4. Reproducing this file

```bash
PYTHONIOENCODING=utf-8 python3 -m tools.harness run
PYTHONIOENCODING=utf-8 python3 -m tools.stress --track natural
PYTHONIOENCODING=utf-8 python3 -m tools.stress --track natural --retrieval bm25
PYTHONIOENCODING=utf-8 python3 -m tools.stress --track natural --probes data/probes_generated.jsonl
PYTHONIOENCODING=utf-8 python3 -m tools.stress --track natural --retrieval bm25 \
    --probes data/probes_generated.jsonl
PYTHONIOENCODING=utf-8 python3 -m tools.transcripts --n 50 --out results/transcripts.json
```

All six are deterministic at fixed seed. `python3 -m tools.check` runs the
read-only-kit check, `verify_mirror`, and the official score together, and is
the thing to run before trusting any of the above after a code change.
