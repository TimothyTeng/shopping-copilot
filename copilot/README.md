# Shopping Copilot — conversational retrieval agent

A conversational shopping agent for the TechJam challenge. It has at most ten
turns to work out what a hidden shopper wants and surface the product they
actually bought into a top-10 list, as early and as highly ranked as possible.

**Draft 2.** Working end to end, scored by the official evaluator. Pure Python
standard library — no LLM, no network, no external services.

For a stage-by-stage inventory of the algorithms, the alternatives considered,
and the experiment record behind the switches, see
[docs/algorithm-audit.md](docs/algorithm-audit.md).

## Results

Scored by the **unmodified** official evaluator on the 200-session public set.

| | Score | Hit@10 | MRR | MTTC |
|---|---|---|---|---|
| Official BM25 baseline | 0.1067 | 0.125 | 0.068 | 9.81 |
| **This agent** | **0.9456** | **1.000** | 0.911 | 2.38 |

The score was 0.9383 until the MMR tie re-rank (`tie_rerank=mmr`) became the
default; the +0.0074 is significant at 95% by paired bootstrap, CI
[+0.0017, +0.0138] (`python -m tools.harness ci --compare tie_rerank=mmr`).

Per scenario — every scenario at perfect recall:

| Scenario | n | Hit@10 | MRR | MTTC |
|---|---|---|---|---|
| Buying | 80 | 1.000 | 0.916 | 1.99 |
| Browsing | 80 | 1.000 | 0.889 | 2.14 |
| Intent override | 30 | 1.000 | 0.926 | 3.87 |
| Boundary | 10 | 1.000 | 1.000 | 3.10 |

Runtime: ~10 s one-off index build, ~74 ms per session, ~90 MB resident.

## How close is this to the maximum?

The target is **never missed** and is already **rank 1 in 162 of 200** sessions.
The whole remaining MRR gap is 38 sessions ordered wrongly inside the top 10 —
and most of those are not fixable:

| Of the sessions where the target is not rank 1 | |
|---|---|
| Exact score tie with the product ranked above it | **23** |
| Of those, both products satisfy **every** constraint token | **18** |
| Median size of a tied group | 13 products |

In those 18 cases the two products are genuinely indistinguishable given what the
shopper disclosed — same category, same material, same closure. No retrieval or
scoring method can separate items that satisfy identical constraints, and no
further question helps: the intent card holds exactly **four** constraints, all
disclosed by turn 3, after which the simulator only says *"I don't have an
additional preference."*

Breaking the 38 down: ~18 information-limited and unreachable, ~5 tied but
separable, ~15 with real score gaps. **The achievable ceiling is roughly 0.95**,
so this agent is at about 80% of the reachable headroom. That is the main reason
effort moved to robustness and real-language handling rather than further tuning.

## Robustness

The public evaluator words every customer reply identically, so a local score
cannot tell you how much of it depends on that exact wording. `tools/harness.py
perturb` replays the same session logic through different phrasings to find out.

| Phrasing | Score | Hit@10 |
|---|---|---|
| L0 — the evaluator's exact wording | 0.935 | 1.000 |
| L1 — paraphrased carrier sentences | 0.926 | 1.000 |
| L2 — paraphrase + conversational noise | 0.649 | 0.760 |
| L4 — carriers removed, bare declaratives | 0.930 | 1.000 |

For comparison, an earlier regex-based prototype scored **0.928 on L0 and 0.000
on L1** — a total collapse, silent under normal testing. That result is why
extraction here is grounded in the catalog rather than in sentence structure.

The template layer is a fast path, not the mechanism: with it disabled, L1, L2 and
L4 are unchanged (the regex simply never fires on reworded input) and only L0
drops, to 0.821.

## Setup

Requires Python 3.10+. The agent itself needs **no third-party packages**; only
the optional `tools/exp_vector.py` experiment needs numpy/scipy/scikit-learn.

```bash
python -m tools.harness run
```

Run from this directory. The official kit must sit alongside it as
`../techjam-conversational-search`, with `data/catalog.jsonl` present
(downloaded from the participant-kit release and SHA256-verified).

```bash
python -m tools.harness run       # official score, per-scenario breakdown
python -m tools.harness perturb   # robustness curve across phrasings
python -m tools.harness ablate    # contribution of each design decision
```

## Try it

Watch a real labelled session play out turn by turn, with the hidden target
revealed up front so you can see the ranking close in on it:

```bash
python -m tools.demo replay --scenario intent_override --seed 3
```

That one is worth seeing: the agent holds back on turn 1, puts the target at
rank 1 on turn 2, then the shopper changes their mind on turn 4 and it recovers
to rank 1 again. Other things to try:

```bash
python -m tools.demo replay                      # random session
python -m tools.demo replay --sample public_0007 # a specific one
python -m tools.demo replay --scenario browsing
python -m tools.demo replay --level L1           # customer paraphrases everything
python -m tools.demo chat                        # type your own request
```

`--level L1` is the interesting one: the customer says the same things in
different words, and you can watch extraction still find the requirements.

## How it works

Each turn: interpret the message → update dialogue state → retrieve → score →
decide whether to speak.

- **`catalog.py`** flattens products exactly the way the evaluator does. The
  hidden requirements are drawn from that same rendering, so any divergence
  breaks matching silently.
- **`index.py`** builds unigram postings (101k terms, ~20 MB). A bigram index was
  measured at ~180 MB for roughly a 10% selectivity gain over unordered token-AND
  and was rejected. Exact phrases are verified on a narrowed candidate set instead.
- **`category.py`** resolves the coarse category by finding a known bucket name
  *inside* the message. Those buckets hold a median of ~8 products out of 50,000
  — the strongest single signal — and substring lookup survives rephrasing.
- **`extract.py`** finds requirements without parsing sentence structure: runs of
  informative, catalog-grounded, previously-unseen tokens. Rewording the sentence
  around a requirement does not change the requirement's own words.
- **`state.py`** accumulates constraints, tracks exhausted attributes, and handles
  intent override.
- **`rank.py`** scores IDF-weighted coverage per constraint, combined as a
  log-product on the theory that the requirements are a conjunction. Honest
  footnote: ablation shows a plain sum scores the same (+0.0022), so the
  conjunction form is not measurably earning its keep. Ties are then broken by a
  product-side prior (`tiebreak`), and products already shown are demoted.
- **`policy.py`** decides what to ask, when to commit to a list, and writes the
  question text.

## Design decisions worth knowing

**Asking `other`.** Measured and explained in *Why the agent always asks the
same thing*, below.

**Holding back weak lists.** A session ends at the first hit, so surfacing the
target at rank 8 locks that rank in forever. Moving 8 → 1 is worth ~0.26 of
score; waiting a turn costs ~0.02. The agent therefore withholds recommendations
until the evidence is strong — but only to a point, since never speaking scores
zero.

**Scoped override.** When the shopper switches intent, wiping memory is the
obvious move and the wrong one: the abandoned preference and the new one both
describe the *same* target, so a full wipe discards up to four correct clues.
Only the preference stated in the opening is revoked. (`override_mode` switches
between `scoped`, `erase`, and `keep`.)

## Limitations

- **L2 noise is the weak point** (0.649, Hit@10 0.760). Heavy conversational
  filler fuses separate requirements into one span that matches nothing.
  `segment_spans` splits spans at catalog-grounding seams and recovers ~0.20 of
  it at no cost on L0, so it is now on by default — but a third of L2 sessions
  still fail. This is the largest known weakness and it is unresolved. Note that
  L2 is our own synthetic stress test, harsher than a plausible organizer
  paraphrase (L1, where the agent holds at 0.926).
- **Free-form human language is much weaker than the benchmark suggests.**
  Visible in `demo chat`: the benchmark's requirements are near-verbatim catalog
  text, so lexical matching works extremely well. A phrase a real shopper would
  use — "waterproof for hiking" — appears nowhere verbatim, coverage collapses,
  and unrelated products surface. The 0.93 is an honest benchmark score and an
  overstatement of real-world quality. Note the asymmetry: semantic search is
  measurably useless *here* (see Measured and rejected) yet is exactly what a
  real product would need — the benchmark rewards the opposite of what ships.
- **MRR 0.857 is the main remaining headroom.** Hit rate is saturated at 1.000
  and turn count is near its floor (~1.85 blended), so precision at the top of
  the list is where the remaining points are.
- **No learned ranking prior yet.** Tie-breaking is popularity only. A fitted
  prior is planned, contained so it can only reorder within a rounding band of
  the constraint score.
- **Statistical caution.** At n=200 the noise floor is roughly ±0.03 on hit rate.
  Differences smaller than that are not real; several numbers above are within it.
- **No model tier, now by evidence rather than by omission.** Semantic retrieval
  was built and measured, and it rescued 0 of 200 sessions. An optional LLM
  remains unexplored. The agent is lexical-only, which is also what makes it safe
  if final scoring runs with network access disabled.

## Measured and rejected

Ideas tried, measured, and left out. Recorded because a justified negative is
worth as much as a feature. Every number is from the official evaluator or a
renderer-swapped replay of it; the noise floor at n=200 is roughly ±0.03.

**Vector / semantic search.** Catalog-trained LSA (TF-IDF → 256-dim truncated
SVD). Reproduce with `python -m tools.exp_vector`.

| Given all four constraints | Hit@10 | MRR | Rank-1 |
|---|---|---|---|
| Lexical (current) | 0.930 | 0.847 | 0.800 |
| Vector only | 0.335 | 0.276 | 0.235 |
| Best of both | 0.930 | 0.847 | 0.800 |

Fast and feasible — 51 MB, 3.68 ms brute-force over all 50,000, no ANN index
needed at this scale. But it rescued **0 of 200** sessions, and best-of-both is
identical to lexical alone.

**Semantic / embedding category resolution.** Scoped as the fix for the "~17 of
26 stress probes resolve to the wrong bucket" diagnosis, then **rejected on a
ceiling study before building the model** — the measurement said a better
predictor could not pay as the code is wired. Forcing the category resolution to
the *truth* (the ceiling any resolver could reach), on the stress set with fuzzy
repair on:

| forced category | score | Hit@10 | MRR |
|---|---|---|---|
| classifier (current) | 0.4946 | 0.615 | 0.282 |
| full oracle (true key **and** bucket) | 0.4578 | 0.538 | 0.334 |
| true bucket for **ranking only** | **0.5480** | 0.654 | 0.373 |

The full oracle *hurts*. The value is real but trapped: the code couples
resolution to extraction via `suppress = category_key.split()`, so the more
accurate the key, the more of the shopper's own category words it removes from
constraint mining — and on free text those words carry coverage. Isolating the
ranking bucket from that coupling recovers **+0.053**, but capturing it needs a
near-perfect resolver *and* `suppress_category_tokens=False` (added as a switch;
benchmark cost −0.001, stress-neutral until the resolver is actually accurate).
`category_bonus` swept 0.0–2.0 is flat (~0.494), so bonus weight is not the
lever either. Net: the ceiling for *all* category-resolution work is +0.053 on
n=26 — inside the noise band — behind a mechanism fix, which does not justify a
40–100 MB embedding model. The lesson is the MMR one from the Corrections
below, applied earlier: measure the ceiling before building to it. The genuine
free-text bottleneck is vocabulary, which the 0.98 "product's own words" oracle
in `tools/stress.py` already localises.

**Emitting as soon as the override lands.** On `intent_override` sessions the
evaluator ignores hits until the override has been delivered
(`local_evaluator.py:252`), so the earliest scorable turn is 3 or 4. We convert
only 4 of the 12 sessions whose override lands at turn 3; the gate is still
holding back. Forcing emission from the override turn onward closes that gap
*exactly* — and scores worse:

| | overall | override MRR | override MTTC | hit turns |
|---|---|---|---|---|
| gate as-is | **0.9383** | 0.934 | 3.87 | `{3: 4, 4: 26}` |
| emit once override seen | 0.9364 | 0.874 | **3.60** (the floor) | `{3: 12, 4: 18}` |

The turn gained is worth +0.0008; the rank given up costs −0.0027. This is the
"rank is worth ~13× a turn" trade measured on a live decision rather than
argued from the formula, and it is the clearest single justification for the
disclosure gate.

**Rarity-scored span segmentation** (`segment_mode="rarity"`). Cutting a run into
requirements greedily takes the first grounded window left to right, so a glue
word can capture the head of the next requirement:

```
"rubber sole plus design in usa"
  greedy -> ["rubber sole", "plus design", "in usa"]     <- wrong
  rarity -> ["rubber sole", "design in usa"]             <- right
```

Replaced with a dynamic program choosing the segmentation that explains the most
tokens with the rarest catalog phrases (`len × log(N / (1 + phrase_df))`, minus a
per-piece cost, minus IDF for each token dropped). It **fixes the defect and
still does not pay**:

| `piece_cost` | L0 | L2 | L4 |
|---|---|---|---|
| greedy (default) | 0.9383 | 0.6493 | 0.9300 |
| 0 | 0.9383 | **0.6755** | 0.9124 |
| 3 | 0.9383 | 0.6504 | 0.9153 |
| 4 | 0.9383 | 0.6186 | **0.9309** |

L0 and L1 are **identical at every setting** — clean carriers barely exercise the
segmenter — and `piece_cost` only trades L2 against L4 along a monotonic
frontier, moving ~7 sessions between two synthetic renderers for 70% more
runtime. Kept as a switch, not a default.

Two sub-results worth keeping. Charging for dropped tokens measured *exactly*
zero across `skip_penalty` 0→2: the search never drops anything, because every
token here is grounded as a unigram and covering always beats dropping. And the
naive objective is biased toward splitting, since each extra piece contributes
its own rarity term for the same tokens — which is what `piece_cost` corrects.

**Per-field BM25 / BM25F.** Cost ~0.04 and dropped rank-1 placements from 162 to
~130. Every variant was tried — title-boosted, uniform, body-only, with and
without length normalization, and near-binary. The cause is structural: TF-IDF
relevance assumes the query was authored *independently* of the document, but
here the constraints are lifted from the target's own text, so "mentions it more
often" is noise rather than evidence. A competitor whose copy dwells on polyester
outranks the product that simply is polyester. Same root cause as the vector
result. Kept behind `bm25f_weight` (default 0).

**Constraint-specificity slot weighting.** −0.002. At full coverage every slot
contributes the same regardless of weight, so it cannot break the exact ties that
cause most remaining errors. Kept behind `specificity_weighting` (default off).

**Evidence-level alignment.** Matching each constraint to its best single feature
bullet rather than the flattened document: fixed 1 of 39, broke 9, left 74% still
tied. When both products contain the bullet, per-unit alignment is 1.0 for both.

**Steering which attribute to ask.** See below — capped at +0.010 even with
oracle knowledge.

**Bigram index.** ~180 MB for roughly a 10% selectivity gain over unordered
token-AND.

**Product/attribute co-occurrence graph.** No relational hop the task needs.

**Confidence-gated category bonus** (`gate_category_bonus`). Scaling `category_bonus`
by resolution confidence, so a shaky fallback bucket cannot rank as hard as a
certain one. Benchmark-neutral (substring hits score confidence 1.0), but on the
natural-language stress set it cost −0.010 against the vote resolver alone
(0.4118 → 0.4022): it also weakens the bonus on *correct* low-confidence votes,
demoting their targets (MRR 0.257 → 0.225). Kept as a switch.

**Per-turn category re-resolution** (`reresolve_category`). Category is resolved
once and frozen, so "actually never mind, I need women's sweatpants" keeps
searching shoes. Re-resolving on every turn (switch only to a confident,
different bucket) was meant to catch it. It cost the benchmark a hit
(0.9383 → 0.9347, Hit@10 1.000 → 0.995 — a mid-session answer voted confidently
into a wrong bucket) and did **not** fix the target case: `category_switch`
stayed 0.000, because turns 1–2 deposit *shoe* constraint slots that are never
retracted, so flipping the category still leaves the ranking poisoned. The real
fix is constraint retraction, not re-resolution. Kept as a switch.

**Wholesale reset** (`enable_reset`). A reset cue ("never mind", "scratch that",
"start over") revokes every slot, clears the category, and re-resolves from the
new message — the mechanism the failed per-turn re-resolution was missing, and it
*does* revoke the stale shoe constraints correctly. It measures neutral (0.4118,
no harm) and is left off, because the one probe it targets still scores 0.000:
after the reset, the vote resolver mis-buckets "women's sweatpants" to `tops tees
t shirts` at confidence 0.15 (the target lives in `pants sweatpants`), so the
target is never in the resolved bucket. The blocker is category-resolution
*precision*, not accumulation — reset cannot pay until that is fixed. Raising
`category_min_confidence` to reject the bad vote does not help either: it strips
the category bonus the retraction win leans on, dropping `contradiction` 0.94 →
0.68. Kept as a switch.

**Ensemble category resolver** (`category_resolver="ensemble"`). Unioning the
classifier's bucket with the vote's, to "keep the best of both", collapses
*exactly* back to vote (0.4577 → 0.4191). The classifier's gain comes from
committing to one focused bucket so only its products get the category boost;
adding vote's bucket re-boosts vote's products, which re-dilute that focus and
push the classifier's newly-found targets back out of the top ten. The two
resolvers are right on *different* probes and nothing at inference says which to
trust, so no bucket combination recovers the one-probe MRR dip without also
erasing the recall gain. Kept as a switch.

## Why the agent always asks the same thing

`ask_attribute` is always `"other"`, and that is a measured decision, not laziness.
The simulator matches a request with `attribute == "other" or classify(value) ==
attribute`, so the `"other"` match set is a **superset** of every typed
attribute's under identical truncation — it weakly dominates in every state.

Tested against an oracle that *cheats* by reading the hidden intent card and
always asking for the sharpest available constraint:

| Ask policy | Score | Hit | MRR | MTTC |
|---|---|---|---|---|
| **Always `other`** | **0.9383** | 1.000 | 0.883 | 2.33 |
| Typed, rarest bucket first | 0.2787 | 0.350 | 0.194 | 8.73 |
| Oracle, sharpest first every turn | 0.7856 | 0.850 | 0.728 | 3.88 |
| Oracle on turn 1, then `other` | 0.9417 | 1.000 | 0.903 | 2.46 |

Typed questions are catastrophic — they return only constraints *of that type*,
often none, while `other` returns any two. Even the cheating oracle loses when it
steers every turn. The absolute ceiling for question steering is **+0.010**, and
it requires knowing the answer, so an LLM choosing the attribute cannot pay.

What *does* vary is the wording. The simulator reads only `ask_attribute` and
ignores the prose, so `policy.compose` writes a fresh question each turn from what
actually changed — acknowledging what was just learned and how far the pool has
narrowed. Zero scoring risk, and it stops the agent sounding like a form.

## Corrections

Earlier conclusions in this project that later measurement overturned. Kept
visible because the reasoning that produced them was plausible and wrong.

- **"The template layer is dead weight."** It is worth **+0.109**, the single
  largest contributor. The earlier reading predated later extraction fixes.
- **"The log-product is why precision is high."** No measured effect — a plain
  sum scores the same (+0.0022).
- **"Implicit rejection is provably free value."** It measured *exactly* zero
  because it was dead code: emissions were quarantined past the turn most
  sessions end on. Rewritten as a demotion penalty, it is now live.
- **"The override cue is fully stripped."** It was not. Regex alternation is
  leftmost-first, not longest-first, so `Actually, ignore` matched before
  `ignore my earlier` and left `my earlier preference` in the message. `earlier`
  has df=10, so the scorer treated it as a *highly* informative requirement.
  Rewriting the cue as one span covering the whole preamble is worth **+0.085
  MRR on `intent_override`** and +0.0031 overall. Found by reading a demo
  transcript, not by any metric — the agent was visibly replying "Okay —
  earlier."
- **"Targets are review-biased, so review count will be the strongest prior."**
  Half right. The target does sit at the ~84th percentile of its tied group by
  review count, so popularity is genuinely predictive — but with a median group
  size of 13 that is only ~2nd place, and every tie-break variant tested landed
  within ±0.008 of the others. Re-measured (`price_pop` 0.9383, `popularity`
  0.9357, `price_bayes` 0.9354, `none` 0.9300, `bayes` 0.9266): the current
  `price_pop` is the best of them, so there is no free tie-break MRR — the
  residual ties are genuinely near-unfixable by prior choice.
- **"The category fallback is fine; only the substring path matters."** The
  token-overlap fallback (`overlap / len(key_tokens)`) counts tokens equally, so
  "men's dive watch" resolved to `men hoodies` on the shared token and a wrong
  bucket then poisoned the ranking — it fired on 17/26 stress probes. Replacing
  the fallback with a **retrieve-then-vote** resolver (`category_resolver="vote"`)
  — poll the products the message actually retrieves via BM25 and take their
  majority coarse category, weighted by match mass — is benchmark-neutral (0.9383
  → 0.9383, the substring path still wins there) and worth **+0.037 on natural
  language** (0.3749 → 0.4118, Hit@10 0.462 → 0.500, a recall gain; the
  `colloquial` tag went 0.000 → 0.320). n=26, so read the direction, not the
  third decimal. *Later superseded by the classifier below.*
- **"A char-n-gram category classifier will overfit and lose."** Eight
  hand-picked queries said so (it sent "shoes" to `card id cases` at confidence
  1.0). The *measured* suite said otherwise. An offline-trained linear classifier
  over character 3–5-grams of product titles — augmented with short word-subsets
  so it sees inputs the length of a real query, weights pruned and shipped as a
  0.7 MB file, inference in pure stdlib (`tools/train_category.py`,
  `src/category_clf.py`) — scores **0.4577 on the stress set** (vote was 0.4118),
  the best Hit@10 the agent has posted (0.500 → 0.577, now above the stateless
  BM25 control's 0.538), and stays benchmark-neutral (0.9383). It trades a little
  ranking precision for recall (MRR 0.289 → 0.244, mostly one probe) and char
  n-grams shrug off typos and plurals for free. Now the default; degrades to
  `vote` if the model file is absent. The lesson is the project's own rule: eight
  queries are an intuition, the suite is the measurement.
- **"Constraints only accumulate, and that is fine."** It is not: "actually not
  leather, canvas is better" ended up requiring *both*, and the conjunction
  matched nothing. A negation cue ("not X", "instead of X", "rather than X") now
  revokes the matching slot (`enable_retraction`, default on) — benchmark-neutral
  (the benchmark uses no such cue) and it took the `contradiction` probe from
  0.000 to **0.940, rank 1** (+0.007 on the stress set, 0.4118 → 0.4191). The
  *wholesale* case ("never mind, women's sweatpants") has a matching mechanism
  (`enable_reset`), but see below.
- **"A typo is unfixable without a spell-corrector dependency."** It is fixable
  in pure stdlib. A shopper token absent from the catalog (`df==0`) is mapped to
  the nearest catalog term within a bounded Damerau-Levenshtein distance, via a
  trigram candidate index (`fuzzy_repair`, `src/fuzzy.py`), before extraction or
  category resolution reads it. Because it only touches *absent* tokens, and
  every benchmark token is quoted from the target (`df>0`), it is a no-op on the
  graded path **by construction** — measured 0.9383 → 0.9383, and neutral at
  every perturb level L0–L4 with templates on and off (+0.0000 in all eight
  cells). On the stress set it took the `typo` probe from **0.000 to 0.960,
  rank 1** (`rainocat`→`raincoat`, `waterprrof`→`waterproof`, `hoood`→`hood`) and
  lifted the whole set 0.4577 → **0.4946** — and a per-probe diff confirms
  *exactly one* of 26 probes moved, so there is no collateral. Damerau not
  Levenshtein because the common typo is an adjacent transposition. Default off
  pending a wider read than n=1 on the target tag; `fuzzy_repair_present` extends
  it to present-but-rare tokens (catches `lightwieght`, `df=1`) at the risk of
  corrupting valid rare words, kept as a switch. The lesson repeats: "needs a
  dependency" was an assumption, and a 100-line trigram index measured otherwise.
- **"The exact-score ties are information-theoretically unfixable."** Half-true.
  They are unfixable by adding constraint *information* — both tied products
  satisfy every disclosed constraint — but they are not unfixable. A diversity
  re-rank (Maximal Marginal Relevance, `tie_rerank=mmr`, title-token Jaccard as
  the similarity) reorders a tied cluster to avoid stacking near-duplicate
  popular siblings above the target, and it moves the benchmark **0.9383 →
  0.9456** (MRR 0.883 → 0.911, hit stays 1.000), the first thing in the project
  to lift the benchmark rather than a stress surface. The gain is concentrated
  in `browsing` (MRR 0.819 → 0.889), the exploratory sessions with the weakest
  constraints and so the most ties, and it is flat across λ 0.5–0.9 — a plateau,
  not a fit. It looked like it should stay a switch — the +0.007 is smaller than
  the ±0.03 noise floor — but that floor is the *unpaired* absolute-score band,
  and a paired bootstrap (`harness ci --compare tie_rerank=mmr`, 10k resamples)
  puts the delta at a 95% CI of **[+0.0017, +0.0138]**, P(delta>0)=0.997:
  significant, so it is **now the default**. One caveat remains: it is the
  *reverse* on free text — on the conjunctive path the stress set falls 0.4946 →
  0.4354, because trading relevance for diversity only pays when the near-ties
  are genuine equals, which is the benchmark and not a real shopper's words. It
  is wired into the conjunctive return only, so it is inert on the bm25 chat
  surface. The correction to the earlier claim stands: a query-independent prior
  reached ties that no amount of constraint parsing could — and the lesson twins
  with the noise-floor rule, which is a heuristic for unpaired deltas, not a
  substitute for the paired test when hit is saturated.

## The competition kit is read-only

`../techjam-conversational-search` is never modified. The harness imports the
official `evaluate()` and passes this agent in as an argument, so scores are
authentic without editing a single file of the kit. `starter/agent.py` is only
written at packaging time.
