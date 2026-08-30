# TechJam Challenge 4 — Shopping Copilot

Conversational product search over a frozen 50,000-product Amazon catalog. An
agent asks a simulated shopper questions and returns 10 recommendations per
turn; the session ends when the hidden target appears in the list.

## Layout

```
techjam-conversational-search/   the official kit — READ-ONLY, see below
copilot/                         all our work
  src/                           the agent (no deps beyond stdlib)
  tools/                         harness, simulator, demo, experiments
  README.md                      results, ceiling analysis, rejected ideas
```

Run everything from `copilot/`.

## The competition kit is read-only

`techjam-conversational-search/` is a clone of the organizers' repository.
**Never edit anything inside it.** We import its evaluator and read its catalog;
`evaluate()` takes the agent as an argument, so scoring against the real thing
requires no modification.

Verify after any session:

```bash
git -C techjam-conversational-search status --short
```

Only the two downloaded data files (`data/SHA256SUMS`, `data/catalog.jsonl.gz`)
may appear. Anything else is a bug.

## Commands

All from `C:\Users\Timothy\Documents\Techjam\copilot`.

Score against the official evaluator:

```bash
PYTHONIOENCODING=utf-8 python -m tools.harness run
```

Paraphrase robustness, L0 through L4, with templates on and off:

```bash
PYTHONIOENCODING=utf-8 python -m tools.harness perturb
```

Switch-by-switch ablation:

```bash
PYTHONIOENCODING=utf-8 python -m tools.harness ablate
```

Bootstrap a confidence interval on the score, or on the delta to another config
(paired, so a hit-saturated change is judged on its real, tighter interval):

```bash
PYTHONIOENCODING=utf-8 python -m tools.harness ci --compare tie_rerank=mmr
```

Watch a labelled session, hidden target revealed:

```bash
PYTHONIOENCODING=utf-8 python -m tools.demo replay --scenario intent_override --seed 3
```

Talk to it yourself:

```bash
PYTHONIOENCODING=utf-8 python -m tools.demo chat
```

`PYTHONIOENCODING=utf-8` is required — the Windows console default (cp1252)
crashes on product titles containing emoji or typographic dashes.

## Current state

`score 0.9456  hit@10 1.000  MRR 0.911  MTTC 2.38` — baseline 0.1067.
200 sessions in ~11s, no network, no LLM. (Was 0.9383 before the MMR tie
re-rank became the default; the +0.0074 is significant at 95% by paired
bootstrap, CI [+0.0017, +0.0138] — see `harness ci`.)

## How scoring works

```
TechnicalScore = 0.50·HitRate@10 + 0.30·MRR + 0.20·Efficiency
Efficiency     = clip((11 − MTTC) / 10, 0, 1)
```

A miss counts as turn 11. Max 10 turns. `top_k` is always 10.

Two consequences drive most design decisions:

1. **Rank is worth ~13× a turn.** Moving the target 8→1 gains ~0.26; waiting one
   more turn costs ~0.02. So the agent holds back weak lists rather than
   committing to a bad rank, which a session ending at first hit would lock in.
2. **Hit@10 is already 1.000.** Retrieval is finished. Everything remaining is
   ordering, and ~18 of the ~38 residual errors are exact score ties where both
   products satisfy every disclosed constraint — information-theoretically
   unfixable. Realistic ceiling ≈ 0.95.

## Non-obvious evaluator rules

Read these before concluding a transcript is wrong.

**Hits before the override do not count.** `local_evaluator.py:234` sets
`override_applied = scenario != "intent_override"`, and the hit check at line
252 is `if override_applied and target in ranked`. On an `intent_override`
session the target can sit at rank 1 for turns 1–3 and score *nothing*; the
clock starts only when the override message is delivered, at
`rng.choice([3, 4])`. Measured floor for these sessions is MTTC **3.60**, and we
are at 3.87 — so `intent_override` MTTC looks bad but is near its maximum.

Do not "fix" this by emitting as soon as the override lands. Measured: it hits
the 3.60 floor exactly and **scores worse** (0.9383 → 0.9364), because the turn
gained is worth less than the rank given up.

**The simulator never reads `message`.** It branches only on `ask_attribute`.
Prose wording is therefore free to change and cannot affect the score — verify
any wording change is score-neutral, and treat a score move as a bug.

**`ask_attribute="other"` weakly dominates.** The reply filter is
`attribute == "other" or classify_constraint(value) == attribute`, so `"other"`
matches a superset of every typed attribute under identical truncation.

## Working rules for this project

**Measure before believing, and measure after changing.** This codebase has a
documented history of plausible ideas that measured negative: BM25F (−0.04),
vector search (rescued 0/200), specificity weighting (−0.002), evidence-level
alignment (fixed 1, broke 9), oracle question-steering (0.786, worse than not
steering). Four of those failed for the same structural reason — **TF-IDF
assumes the query was authored independently of the document, and here the
constraints are lifted verbatim from the target's own text.**

**The noise floor at n=200 is ±0.03.** Do not report a delta smaller than that
as an improvement without saying it is inside the noise. Per-scenario deltas on
`intent_override` (n=30) and `boundary` (n=10) are noisier still.

**Re-measure old conclusions after the code around them changes.** `segment_spans`
was disabled on a measurement that later fixes invalidated; re-enabling it was
worth +0.20 on L2 at zero L0 cost. Ablation results go stale.

**Every tunable lives in `src/config.py`**, so an experiment is a one-line diff
and the harness can sweep it without touching logic. Rejected ideas stay as
switches with the measurement recorded in the comment — do not delete them.

**Record negatives.** `README.md` has "Measured and rejected" and "Corrections"
sections. A justified negative result is worth as much as a feature, and both
sections are load-bearing for the writeup.

**Nothing on the scored path may require the network.** `submission_rules.md`
warns final scoring may run network-disabled. Any future LLM backend must be a
pure augmentation over a complete, valid Tier-0 result.

## Gotchas

- **Heredocs in the Bash tool collapse `\\` to `\`** on this machine, which
  silently corrupts regex patches. Use the Edit tool for anything containing
  backslashes.
- **`cd` does not persist between Bash calls.** Use an absolute path in the same
  command: `cd /c/Users/Timothy/Documents/Techjam/copilot && ...`
- `catalog.py` mirrors the evaluator's `SEARCH_FIELDS` flattening **exactly**.
  If they diverge, phrase grounding silently fails against the real scorer.
- Both sides must normalize identically (`src/normalize.py`) for the same
  reason.

## The two scores, and never conflating them

The official evaluator is **not a natural-language test**. Its simulator builds
the shopper's constraints verbatim from the target's own `features`/`details`,
so query and document share vocabulary by construction. Quote 0.9383 as a
benchmark result only.

`tools/stress.py` is the independent test, built to find failures rather than
confirm success. Targets were picked before any shopper wording was written, and
nothing was tuned on it.

```bash
PYTHONIOENCODING=utf-8 python -m tools.stress --show
```

Measured, 26 probes, same official metric:

| | score | Hit@10 |
|---|---|---|
| agent (ours) | **0.3749** | 0.462 |
| stateless BM25 over the whole transcript | **0.4775** | 0.538 |
| oracle: query = the product's own words | 0.9827 | 1.000 |

**Plain BM25 beat the agent on natural language**, and the oracle proves the
targets are findable — so the gap is vocabulary, not ambiguity.

## Retrieval mode is a deployment choice, not a tuning problem

`cfg.retrieval` selects the ordering: `conjunctive` (log-product over slots),
`bm25` (over the raw transcript), `rrf` (weighted fusion), `auto`. Every tool
takes `--retrieval`. The measured frontier:

| conjunctive weight | benchmark | natural language |
|---|---|---|
| 1.0 — `conjunctive` | **0.9383** | 0.3749 |
| 0.95 | 0.9341 | 0.3850 |
| 0.85 | 0.9152 | 0.4103 |
| 0.5 | 0.8503 | 0.5909 |
| 0.0 — `bm25` | 0.4029 | **0.6411** |

Nothing dominates, so the default is per surface: **`conjunctive` on the graded
path** (`harness`, `demo replay`), **`bm25` in `demo chat`**. With BM25 the
agent scores 0.6411 against the stateless control's 0.4775 — so the dialogue
does earn its keep once retrieval matches the input (Hit@10 0.462 → 0.846).

Do not switch the graded default without re-running both suites.

**`auto` is present but not recommended** (benchmark 0.6843). It picks
conjunctive when some product satisfies every constraint. Measured, that signal
is too weak: `satisfied > 0` holds on 56% of benchmark turns and 31% of natural
ones. A stronger separator exists — all spans fully grounded is 42% on the
benchmark and 100% on natural language, because natural spans are short enough
to ground whole — but tuning a benchmark-detector on 26 self-authored probes
would be fitting the test, so it was left alone.

## Known weaknesses

- **L2 noisy = 0.649, Hit@10 0.760.** The largest unresolved gap. It is our own
  synthetic stress test, harsher than a plausible organizer paraphrase.
- **Category resolution *looks* like the biggest natural-language failure but
  is NOT the lever — measured.** It fires wrong on ~17 of 26 stress probes
  ("men's dive watch" → a shirts bucket, "wristwatch for my wife" → the wrong
  watches leaf), so the diagnosis blames it. But an oracle-category ceiling
  study (see README "Measured and rejected") shows fixing it does not pay as the
  code is wired:
    * Full oracle (true key + true bucket): **−0.037** on the stress set.
    * True bucket for *ranking only*, classifier key kept for suppression:
      **+0.053** (0.4946 → 0.5480) — the real ceiling.
    * `category_bonus` swept 0.0–2.0: flat (~0.494), so bonus weight is not it.
    * `suppress_category_tokens=False`: benchmark −0.001, stress neutral now.
  The trap is the coupling `suppress = category_key.split()`: a *more accurate*
  key suppresses more of the shopper's own category words from extraction, and
  on free text those words carry coverage, so accuracy cancels itself. Any
  category work must (a) improve the *ranking* bucket and (b) decouple
  suppression (`suppress_category_tokens`, now a switch) — and even the perfect
  resolver ceiling is +0.053 on n=26, inside the noise band. A semantic /
  embedding resolver was scoped and **rejected on this ceiling**, not built. The
  real free-text bottleneck is vocabulary (shopper words ≠ product words), which
  the stress harness's own 0.98 "product's own words" oracle already localises.
- **Category is resolved once and never revised.** `Agent._observe` only
  resolves when `state.category_key is None`, so "actually never mind, I need
  women's sweatpants" keeps searching shoes for the rest of the session.
- **Constraints only accumulate.** There is no retraction, so "leather... actually
  not leather, canvas" ends up requiring both, and the conjunction matches
  nothing.
- **Typo tolerance is now a switch** (`fuzzy_repair`, default off; `src/fuzzy.py`).
  Absent shopper tokens (`df==0`) are mapped to the nearest catalog term within a
  bounded Damerau-Levenshtein distance via a trigram index, before extraction.
  Benchmark-neutral by construction (every benchmark token has `df>0`, so it is a
  no-op): measured 0.9383 → 0.9383 and +0.0000 across perturb L0–L4. On the stress
  set it took the `typo` probe 0.000 → 0.960 (rank 1) and lifted the set
  0.4577 → 0.4946, with exactly one of 26 probes moved. Was: `rainocat`,
  `waterprrof`, `hoood` grounded to nothing because every layer keys on exact
  tokens.
- **Interior glue tokens become required conjunction terms.** `_coverage` needs
  every token of a slot present, so a piece like `"plus design"` forces `plus`
  into the AND. Identified, not yet fixed.
- **Not built yet:** the `backends/` seam (protocol + null impl) and the parked
  LLM / weighted term-association work for real language. (Bootstrap confidence
  intervals are now built — `harness ci`, `tools/bootstrap.py`.)
