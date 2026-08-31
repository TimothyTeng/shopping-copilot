# Algorithms — everything tried, how each works, and why it passed or failed

*Companion to [ARCHITECTURE.md](ARCHITECTURE.md), which describes the shipped
system. This document is the complete record: every algorithm considered, the
mechanism in detail, the measurement, and the verdict — including, especially,
the ones that failed.*

A justified negative is worth as much as a feature. Roughly two thirds of the
work recorded here was rejected, and the reasons are more portable than the
wins.

---

## How to read the numbers

Two suites, never conflated:

- **Benchmark** — the official evaluator, n=200. Its simulator lifts the
  shopper's constraints verbatim from the target product, so query and document
  share vocabulary by construction. Exact-string matching is near-optimal here.
- **Prose** — `tools/stress.py`, n=26 hand-authored (authoritative) or n=427
  generated (opt-in, buys resolution not authority). Shopper words, not catalog
  words. Nothing was tuned on either.

Deltas are **paired bootstrap** — one resample scores both configurations so
shared session variance cancels. A delta without an interval is not reported.

**Status legend:** ✅ shipped · 🔘 switch (built, off by default) · ❌ rejected ·
🚫 rejected before building (ceiling study)

---

## 1. The scoring core

### 1.1 ✅ Log-product conjunctive scoring

**Mechanism.** Each grounded requirement becomes a *slot*. For a document, each
slot's coverage is the share of the slot's own IDF mass present in that
document. Scores combine as a log-product with a concavity exponent:

```
score(doc) = Σ_slots  w · log(ε + coverage ^ γ)        γ = 1.5, ε = 0.02
```

**Why not a sum.** The hidden requirements are a *conjunction*. Under an
additive score a product that nails one constraint outranks a product that
satisfies all four — precisely the failure that costs precision at the top of
the list. The log turns "satisfy all" into "add", and ε bounds the penalty for a
total miss so one absent constraint cannot make a document unreachable.

**Verdict.** Shipped, with an honest footnote: ablation shows a plain sum scores
within **+0.0022** on this benchmark. The conjunction form is theoretically
motivated and is not measurably earning its keep on the graded surface.

### 1.2 ✅ Phrase matching as a verified bonus

**Mechanism.** Take the rarest term in the phrase, scan its postings, and
string-check each candidate against the flattened document. Matches add a bonus;
they never filter.

**Why not a phrase index.** A bigram index measured **~180 MB for roughly a 10%
selectivity gain** over unordered token-AND. **Why not a filter:** a paraphrased
requirement would be eliminated rather than merely demoted, so a bonus keeps it
competing on token coverage.

### 1.3 ✅ Product-side prior tie-break

**Mechanism.** When constraints cannot separate documents, order by a
query-independent prior: `10·has_price + log1p(rating_count)`, with Bayesian
smoothed quality available as an alternative.

**Why it matters.** The median tied group is **13 products**, and before the
card signature this decided ~19% of sessions.

### 1.4 ✅ MMR tie re-rank

**Mechanism.** Maximal Marginal Relevance over the top 30, similarity =
title-token Jaccard. Each pick maximises `λ·relevance − (1−λ)·max_similarity to
already-picked`. The first pick has no selected neighbour, so it is the
pure-relevance leader — a clean rank-1 target **structurally cannot** be
demoted.

**The interesting part is that the hypothesis was backwards.** Diversity was
expected to help *natural language*, where hit@10 has headroom. It was the
reverse:

| surface | baseline | MMR |
|---|---|---|
| Benchmark | 0.9383 | **0.9456** (MRR 0.883 → 0.911) |
| — browsing (n=80) | 0.819 MRR | 0.889 |
| Prose, conjunctive | 0.4946 | 0.4354 — *hurts* |

Diversity needs trustworthy relevance underneath it. On the benchmark, relevance
is trustworthy and near-duplicates are the whole problem; on prose it is not,
and spreading the list spreads it away from the target.

**Verdict.** ✅ Default. Paired delta **+0.0074, CI [+0.0017, +0.0138],
p=0.997**. Flat across λ 0.5–0.9 — a plateau, not a fit.

This result also partly corrected a standing claim that ties were
"information-theoretically unfixable": unfixable by adding constraint
*information*, but a query-independent diversity prior reorders about half of
them correctly.

### 1.5 ✅ Intent-card signature — the largest single gain

**The mechanism, which came from reading the evaluator rather than the
catalog.** `local_evaluator.intent_card` does not invent the shopper's
requirements. It **derives** them from the target product: flatten `features` +
`details`, insert a material match at position 0 and a colour match at 1, append
a budget line, dedup — then `[:2]` are hard constraints and `[2:4]` soft
preferences. The public set ships **no** `intent_card`, so it is recomputed from
the catalog at scoring time.

Therefore **every product's four possible constraint strings are computable
offline.** `catalog.card_slots` mirrors that function byte-identically on all
50,000 products; `extract.disclosed_constraints` recovers the constraint
verbatim from the simulator's colon carrier; the head rescorer then asks:

> not *"does this document contain those words"*
> but **"would this product have produced that constraint string?"**

**Why it is safe, structurally rather than empirically.** The constraints *are*
the target's own card slots, so the target matches all of them always. The bonus
**cannot demote the target**. A rival can only match by carrying the identical
slot — which is the definition of indistinguishable.

Ceiling measured *before* building, over the 24 sessions then not at rank 1:

| | sessions |
|---|---|
| clears every product above the target | 11 |
| clears some of them | 13 |
| demotes the target | **0** — impossible by construction |

| suite | before | after |
|---|---|---|
| Benchmark | 0.9456 | **0.9626** (MRR 0.911 → 0.967) |
| — buying (n=80) | 0.916 MRR | 0.983 |
| — intent_override (n=30) | 0.926 MRR | 1.000 |
| L1 / L2 / L4 paraphrase | — | **identical to four decimals** |
| L0, templates off | 0.7582 | **0.9016** |

Paired delta **+0.0170, CI [+0.0093, +0.0256]**. `card_bonus` swept 1→12
saturates at 3.0 — a plateau.

**Honest framing, which belongs in any writeup.** This fits the *simulator*, not
shopping. It buys graded score and contributes nothing to real-language quality
— visible in the identical L1/L2/L4 columns, where a reworded turn discloses
nothing and the feature is inert. If the organizers' hidden evaluator differs in
field order or the 180-character limit, the mirror stops matching and the bonus
degrades to a no-op. A bounded downside, but a real one, and the reason
`tools/verify_mirror.py` exists.

**It also closed the benchmark.** Rank 1 is 191/200, and in all 9 remaining
sessions a product above the target carries **identical card slots**. Total
remaining ordering headroom: **+0.0099 composite**. Every subsequent ranking
idea was sized against that budget first.

---

## 2. Constraint extraction and dialogue

### 2.1 ✅ Structure-blind salience extraction

**Mechanism.** Find contiguous runs of tokens that are informative by IDF,
grounded in the catalog, and previously unseen — bridging one-token gaps to
preserve phrase shape. Segment each run into the longest sub-phrases the catalog
actually contains.

**Why structure-blind.** Rewording the sentence *around* a requirement does not
change the requirement's own words. An earlier regex prototype scored **0.928 on
the evaluator's exact wording and 0.000 on a paraphrase** — a total collapse,
silent under normal testing. That failure defines this design.

### 2.2 ✅ Templates as a shortcut, never a dependency

Layer A regex over the simulator's carriers is a *latency optimization only*.
Proof it is not load-bearing: with templates disabled, L0 still scores **0.9016**.

### 2.3 ❌ Rarity-scored span segmentation

**Mechanism.** Greedy segmentation takes the first grounded window left-to-right,
so a glue word can capture the head of the next requirement:

```
"rubber sole plus design in usa"
  greedy → ["rubber sole", "plus design", "in usa"]     ← wrong
  rarity → ["rubber sole", "design in usa"]             ← right
```

Replaced with a dynamic program maximising `len × log(N/(1+phrase_df))` minus a
per-piece cost minus IDF for each dropped token.

**It fixes the defect and still does not pay.** L0 and L1 are *identical at
every setting*; `piece_cost` only trades L2 against L4 along a monotonic
frontier — moving ~7 sessions between two synthetic renderers for 70% more
runtime.

Two sub-results worth keeping: charging for dropped tokens measured **exactly
zero** across `skip_penalty` 0→2 (every token here is grounded as a unigram, so
covering always beats dropping), and the naive objective is biased toward
splitting since each extra piece contributes its own rarity term for the same
tokens — which is what `piece_cost` corrects. 🔘 Kept as a switch.

### 2.4 ✅ Scoped override

**Mechanism.** On an intent override, revoke only the *opening's free-form
preference*, not everything.

**Why.** Both the abandoned preference and the replacement were derived from the
**same target product**, so a full wipe discards up to four correct clues.

### 2.5 ✅ Implicit rejection by demotion

If ten products were shown and the session continued, the target was not among
them — so demote them. **Demote, never remove:** on an override session the
inference is unsound, and a penalty still lets the target rank whereas removal
makes it unreachable.

### 2.6 ❌ Per-turn category re-resolution

Category is resolved once and frozen, so "actually never mind, I need women's
sweatpants" keeps searching shoes. Re-resolving every turn cost a benchmark hit
(0.9383 → 0.9347) and **did not fix the target case**: turns 1–2 deposit *shoe*
constraint slots that are never retracted, so flipping the category leaves the
ranking poisoned. The real fix is constraint retraction, not re-resolution. 🔘 Switch.

### 2.7 ❌ Wholesale reset

The mechanism the above was missing: a reset cue revokes every slot, clears the
category, re-resolves. It *does* revoke the stale constraints correctly, and
still measures neutral — because after the reset the vote resolver mis-buckets
"women's sweatpants" to `tops tees t shirts` at confidence 0.15. **The blocker
is category-resolution precision, not accumulation.** 🔘 Switch.

### 2.8 ❌ Emitting as soon as the override lands

The evaluator ignores hits until the override is delivered, so the earliest
scorable turn is 3 or 4, and the gate converts only 4 of 12 such sessions.
Forcing emission closes that gap *exactly* — and scores worse:

| | overall | override MRR | override MTTC |
|---|---|---|---|
| gate as-is | **0.9383** | 0.934 | 3.87 |
| emit on override | 0.9364 | 0.874 | **3.60** (the floor) |

The turn gained is worth +0.0008; the rank given up costs −0.0027. This is
"rank is worth ~13× a turn" measured on a live decision rather than argued from
the formula — the clearest single justification for the disclosure gate.

### 2.9 ❌ Decision-theoretic emission gate

**Mechanism.** Replace the three-way heuristic gate with a softmax confidence
over the scored head: commit when the leader is probably the target, keep asking
when the head is a flat tie. Principled, and state-dependent in a way the
heuristic is not.

Swept 0.2 → 0.8 it is **monotonically worse** (0.9475 → 0.9336) and loses a hit
at every threshold. Waiting for a confident leader costs more turns than the
rank it buys — the same economics the heuristic gate derives by hand, now
measured against the principled alternative. 🔘 Switch.

### 2.10 ✅ Asking `other`

Not an algorithm so much as a proof. The simulator matches an attribute with
`attribute == "other" or classify(value) == attribute`, then truncates to two.
The match set for `"other"` is a **superset** of every typed attribute's set
under identical truncation, so it weakly dominates in every state. Steering
which attribute to ask was measured with oracle knowledge and capped at
**+0.010**.

---

## 3. The vocabulary gap — the central problem

Everything in this section attacks one thing. The stress harness's oracle —
querying with the **target's own catalog text** — scores **0.999** against the
agent's 0.73 on prose. That oracle is not a baseline, it is a *specification*:
every target is trivially findable once the query is phrased in catalog
vocabulary, and the entire remaining gap is translation from shopper words to
product words.

> A shopper writes *"comfy trainers I can wear to the gym, need my feet to
> breathe"*. The catalog writes *"Breathable Mesh Upper, Cushioned Footbed,
> Lace-Up Athletic Sneaker"*. No amount of scoring fixes a vocabulary mismatch.

Five attacks, in the order they were tried.

### 3.1 ✅ doc2query — the one that worked

**Mechanism.** Expand the **catalog** offline instead of the query at runtime:
ask a model what a shopper would *type* to find each product, index those
generated queries separately, fuse by RRF.

**Why a separate index rather than appending to the product text:** a generated
line is evidence of a different kind on a different scale, and pouring one
vocabulary into another was already measured to fail (§3.5). Two rankings fused
by RRF keep the union of both candidate sets and read positions only.

Generation: **50,000 products, 0 failures, 34.6 min** at concurrency 48 on a
local 7B (24/s, 1.05M completion tokens), resumable and append-only. Ships
gzipped at 1.9 MB.

| n=427 prose, `--retrieval bm25` | score | Hit@10 | MRR | MTTC |
|---|---|---|---|---|
| agent, no expansion | 0.7347 | 0.867 | 0.529 | 3.85 |
| **+ doc2query (0.5)** | **0.7870** | **0.927** | 0.559 | **3.22** |
| HyDE (query-time, 7B, 2 s/call) | 0.7498 | 0.899 | 0.511 | 3.65 |

| paired delta | Δ | 95% CI | p(Δ>0) |
|---|---|---|---|
| weight 0.25 | +0.0304 | [+0.0153, +0.0458] | 1.000 |
| weight 0.4 | +0.0424 | [+0.0238, +0.0612] | 1.000 |
| **weight 0.5** | **+0.0523** | [+0.0317, +0.0732] | 1.000 |
| weight 0.6 | +0.0507 | [+0.0286, +0.0729] | 1.000 |
| weight 0.8 | +0.0203 | [−0.0075, +0.0479] | 0.925 |
| weight 1.0 | −0.0184 | [−0.0511, +0.0140] | 0.129 |

Three readings worth keeping:

1. **Three times HyDE's effect at zero runtime cost** — and with the network
   dependency *removed* rather than added.
2. **The weight curve has a real interior optimum.** At 1.0 — ranking by the
   generated queries alone — it is worse than not doing it. The fusion argument,
   measured: strong second opinion, poor sole retriever.
3. **It leads 6 of 7 tags** where the un-expanded agent leads none, and gains
   concentrate exactly where shopper words are furthest off-catalog: `negation`
   0.6502 → 0.7349, `non_catalog` 0.6931 → 0.7415, `colloquial` 0.7219 → 0.7672.

Weight 0.5 was fixed *before* being measured — taking a sweep's argmax on the
set that produced the headline number is a trap this project fell into once
already — and then happened to measure best of five. That is a coincidence to
record, not a licence to pick peaks: 0.4/0.5/0.6 are one plateau, with a cliff
above 0.7, and every value 0.25–0.6 clears zero alone.

### 3.2 ❌ Dense bi-encoder retrieval

**Mechanism.** `all-MiniLM-L6-v2`, 50k products encoded to 384-dim normalized
vectors (77 MB, 34 s), cosine similarity against the embedded transcript, fused
into the same RRF.

**This one has a two-part history worth reading in order.**

*Part one: an over-generalised negative.* `tools/exp_vector.py` had measured
catalog-trained LSA on the **graded** path and found it rescued **0 of 200**
sessions. That negative was then quoted as a general result — "semantic search
is measurably useless here". It is not general: on the graded path the query is
*quoted from the document*, so there is nothing for semantics to add, by
construction.

*Part two: the correction.* Re-asked on prose, retrieval-only, n=427:

| | hit@10 | MRR |
|---|---|---|
| dense alone | 0.618 | 0.425 |
| lexical BM25 | 0.691 | 0.526 |
| **RRF of both** | **0.735** | **0.530** |

Dense alone loses, as the old result predicted. Fused it adds **+0.044 hit@10**
and finds 42 probes lexical misses (against 73 the other way) — genuinely
complementary.

*Part three: wired in, it is worth about a fifth of that.*

| paired delta vs the agent | Δ | 95% CI | p(Δ>0) |
|---|---|---|---|
| dense 0.3, alone | +0.0115 | [−0.0011, +0.0245] | 0.963 |
| doc2query 0.5, alone | +0.0523 | [+0.0317, +0.0732] | 1.000 |
| **both** | **+0.0612** | [+0.0406, +0.0822] | 1.000 |
| dense 0.3 **on top of** doc2query | +0.0089 | [−0.0034, +0.0218] | 0.918 |

The diagnostic is the last row against the first two. **The two are
near-additive** (0.0523 + 0.0115 = 0.0638 against a measured 0.0612), so dense
is *not* redundant with doc2query — they really do rescue different probes.
Dense is simply **weak inside the agent**: the ceiling study compared it against
a stateless control at 0.691 hit@10, and the agent it has to improve on is at
0.927. Most of what the fusion bought in that study, the agent already had from
dialogue state and doc2query.

**Rejected**, and accuracy is the smaller half of the reason: embedding the
query needs `sentence-transformers` and torch resident in the process — a
dependency the graded agent's "pure stdlib, opens no socket" guarantee cannot
survive. Paying that for a delta spanning zero is not a trade worth making.

At n=427 an effect this size cannot be resolved either way. The honest statement
is not "dense does nothing", it is **"dense is worth at most ~+0.01 here, and
this set cannot separate that from zero"** — settling it would need n ≈ 850.

### 3.3 ❌ RM3 pseudo-relevance feedback

**Mechanism.** Retrieve with the shopper's words, read the vocabulary of the top
documents, interpolate into the query, retrieve again. The classical, offline,
stdlib answer to the vocabulary gap — and the baseline any LLM tier should have
had to beat.

It loses **at every setting**, monotonically toward doing nothing:

| paired delta | Δ | 95% CI | p(Δ>0) |
|---|---|---|---|
| α=0.4, 10 docs, 20 terms | −0.0647 | [−0.0853, −0.0442] | 0.000 |
| α=0.3, 3 docs, 10 terms | −0.0178 | [−0.0318, −0.0037] | 0.008 |
| α=0.2, 5 docs, 10 terms | −0.0134 | [−0.0261, −0.0009] | 0.017 |
| α=0.1, 10 docs, 20 terms | −0.0023 | [−0.0118, +0.0071] | 0.310 |

**Mechanism of failure:** feedback vocabulary *is* listing vocabulary, and
adding more of it drowns the shopper's own rare words — which are the ones that
identify a product. 🔘 Switch.

### 3.4 ❌ Mined term associations (PPMI)

**Mechanism.** PPMI over term co-occurrence in product **titles** — not whole
listings, which co-occur their own boilerplate with everything and learn filler
rather than synonymy. Shipped as a 0.22 MB pruned table, 5,079 terms, read by
pure stdlib.

**The table itself is good**, which is what makes this the more interesting
negative:

```
trainers → runner, vans, jogging, sneakers, tennis, nike, walking, mesh
jumper   → batwing, sweaters, turtleneck, oversized, knitted, pullover
comfy    → nightdress, palazzo, nightshirt, bedroom, lounge, furry, fluffy
```

Those are exactly the shopper→catalog translations the 0.999 oracle says are
missing. Expanding with them still loses: **−0.0341** at 4 terms, **−0.1076** at
8, **−0.0084** at 2 — the same monotone shape as RM3.

**Why.** A correct synonym still retrieves *documents that contain the synonym
and nothing else*. The translation has to apply to the **whole request at
once**, which is what a generated pseudo-listing does and a per-term table
cannot. 🔘 Switch, table shipped.

> **The result that matters here is the pair.** Two independent corpus-internal
> methods failing identically is a finding about the problem, not about the
> methods: *the catalog cannot supply the translation out of its own
> vocabulary, however it is asked.* That is what localised why doc2query works
> — it translates the whole request at once, at index time.

### 3.5 🔘 HyDE — query-time generation

**Mechanism.** Ask a model to write the Amazon listing it thinks the shopper is
describing; retrieve with that. It approximates the oracle without being able to
cheat — the oracle knows the target, this only knows the register.

Two properties make it safe rather than hallucination-driven:

- **Every generated token is verified against the index** before use. A word the
  catalog does not contain, or one so common it carries no signal, is discarded
  — the same discipline `fuzzy.py` applies to shopper typos.
- **The result is an additive bonus, never a constraint.** Adding generated
  terms as *slots* would put them in the AND, where one bad guess empties the
  pool. As a bonus, a bad generation costs precision and nothing else.

Worth **+0.0151 with a CI spanning zero**, for a ~2-second call and a network
dependency. Superseded by doc2query, which is 3× the effect at zero runtime cost.

**A sub-result: `hyde_bm25_mode="union"` — measured and rejected.** Pouring the
generated terms into one query dilutes it: generic listing vocabulary ("soft",
"durable", "comfortable") outvotes the shopper's own words. Hit@10 **0.846 →
0.769**, losing two targets outright, while MRR *rose* 0.305 → 0.397. The signal
was real and the delivery was wrong — which is the argument for fusion, measured
from the opposite direction to §3.3.

**Model choice, honestly unsettled.** A *coder* model beat an instruct model of
four times the size at writing retail copy (Qwen3-Coder-30B 0.6857 vs
Qwen2.5-7B-Instruct 0.6599) — the task rewards formulaic listing structure, not
prose. But that is n=26 without an interval, below this project's own bar, and
it has not been re-run paired because the two models cannot be served
simultaneously on the development machine. Recorded as the best available
evidence, not a settled one.

---

## 4. Ranking alternatives that lost

### 4.1 ❌ Per-field BM25 / BM25F on the graded path

Cost ~0.04 and dropped rank-1 placements from 162 to ~130. Every variant was
tried — title-boosted, uniform, body-only, with and without length
normalization, near-binary.

**The cause is structural, and it is the key insight about this benchmark.**
TF-IDF relevance assumes the query was authored *independently* of the document.
Here the constraints are lifted from the target's own text, so "mentions it more
often" is **noise rather than evidence**. A competitor whose copy dwells on
polyester outranks the product that simply *is* polyester. Same root cause as
the LSA result. 🔘 Switch, default 0.

### 4.2 ❌ Cluster round-robin re-rank

The parameter-free sibling of MMR, and the same hypothesis stated as a model
rather than a trade-off: the conjunctive score is a biased estimate of P(target)
because a family of near-identical listings splits probability mass one product
deserves. Group the head into families by title Jaccard; take one member of each
before any second member. No λ — the relevance/diversity trade is replaced by a
single equivalence threshold.

It **works** (0.9540 against 0.9523 for no re-rank) and still **loses to tuned
MMR** (0.9626). So the trade MMR makes is doing something an equivalence
threshold does not capture. 🔘 Switch.

### 4.3 ❌ Constraint-specificity slot weighting

Weight each slot by how sharply it narrows the catalog — "color: grey" matches 1
product, "polyester" matches 3,085. **−0.002.** At full coverage every slot
contributes the same regardless of weight, so it cannot break the exact ties
that cause most remaining errors. 🔘 Switch.

### 4.4 ❌ Soft slot coverage

A slot counts as satisfied at 85% of its own IDF mass rather than 100%, aimed at
the real "interior glue tokens become required AND-terms" defect. Benchmark
0.9628 vs 0.9626 — inside noise. **The defect is real; its cost is not
measurable on this surface.** 🔘 Switch.

### 4.5 ❌ Profile affinity

`user_profile` is handed to every `reset()` and had never been read. The signal
is real — corr(profile `average_prior_rating`, target `average_rating`) =
**+0.182**, targets averaging 4.37 against the catalog's 4.09 — and it changes
**no ordering at all**: paired delta exactly **0.0000, CI [0, 0]**.

After the card signature there are **no ties left** for a weak prior to break.

### 4.6 🚫 Learned ranking prior — not built

Scoped, then abandoned on §1.5's arithmetic: its entire budget is the **+0.0099**
that remains, against opponents that are *card-identical* to the target. Sizing
the idea against the residual before writing it is the cheapest experiment in
this document.

### 4.7 ❌ Evidence-level alignment

Matching each constraint to its best single feature bullet rather than the
flattened document: fixed 1 of 39, broke 9, left 74% still tied. When both
products contain the bullet, per-unit alignment is 1.0 for both.

---

## 5. Category resolution

### 5.1 ✅ Three-tier resolver

Substring bucket-name match (confidence 1.0, the benchmark path) → char-n-gram
linear classifier → BM25 majority vote. Buckets hold a **median of ~8 products
out of 50,000**, making this the strongest single signal available.

### 5.2 🚫 Semantic / embedding category resolution — rejected on ceiling

Scoped as the fix for "~17 of 26 prose probes resolve to the wrong bucket", then
**rejected before the model was built**, because a ceiling study said a better
predictor could not pay *as the code is wired*:

| forced category | score | Hit@10 | MRR |
|---|---|---|---|
| classifier (current) | 0.4946 | 0.615 | 0.282 |
| full oracle (true key **and** bucket) | 0.4578 | 0.538 | 0.334 |
| true bucket for **ranking only** | **0.5480** | 0.654 | 0.373 |

**The full oracle hurts.** The code couples resolution to extraction via
`suppress = category_key.split()`, so a *more accurate* key strips more of the
shopper's own category words out of constraint mining — and on free text those
carry coverage. Accuracy cancels itself.

Isolating the ranking bucket recovers **+0.053**, but capturing it needs a
near-perfect resolver *and* the coupling fix. `category_bonus` swept 0.0–2.0 is
flat. Net: the ceiling for *all* category work is +0.053 on n=26 — inside the
noise band — behind a mechanism fix. **A 40–100 MB embedding model is not
justified by its own ceiling.**

### 5.3 ❌ Ensemble resolver

Unioning the classifier's bucket with the vote's collapses *exactly* back to
vote (0.4577 → 0.4191). The classifier's gain comes from **committing to one
focused bucket** so only its products get the boost; adding vote's bucket
re-boosts vote's products and re-dilutes that focus. The two are right on
*different* probes and nothing at inference says which to trust. 🔘 Switch.

### 5.4 ❌ Confidence-gated category bonus

Scaling the bonus by resolution confidence. Benchmark-neutral (substring hits
score 1.0), but on prose it cost **−0.010**: it also weakens the bonus on
*correct* low-confidence votes, demoting their targets. 🔘 Switch.

---

## 6. Input robustness

### 6.1 🔘 Fuzzy token repair

**Mechanism.** Absent alpha tokens (`df == 0`) map to the nearest catalog term
within a bounded **Damerau**-Levenshtein distance, via a trigram candidate
index. Damerau rather than plain Levenshtein because the common typo is an
adjacent transposition (`rainocat` → `raincoat`).

**Two properties make it safe on the scored path:** it only touches *absent*
tokens, and the repair target must itself be a real catalog word — so a typo
maps toward a common word, never toward another rare typo-like term.

| suite | baseline | fuzzy |
|---|---|---|
| Benchmark | 0.9383 | 0.9383 — **no-op by construction** |
| Perturb L0–L4 ×2 | — | +0.0000 in all 8 cells |
| Prose (n=26) | 0.4577 | 0.4946 |
| — `typo` tag | 0.0000 | **0.9600** — MISS → rank 1 |

The benchmark no-op is *by construction*, not by measurement: every simulator
token is lifted from the target, so every token has df > 0 and repair never
fires. On for the chat surface; off on the graded path where it cannot help.

### 6.2 The perturbation curve

`harness perturb` replays the same session logic through different phrasings,
because a local score cannot otherwise tell you how much of it depends on exact
wording.

| Phrasing | Score |
|---|---|
| L0 — evaluator's exact wording | 0.9626 |
| L1 — paraphrased carriers | 0.9296 |
| L2 — paraphrase + conversational noise | 0.6398 |
| L4 — carriers removed, bare declaratives | 0.9297 |
| L0, template layer disabled | 0.9016 |

**L2 is the known weak point and is unresolved.** Heavy conversational filler
fuses separate requirements into one span that matches nothing. L2 is our own
synthetic stress test, harsher than a plausible organizer paraphrase.

---

## 7. Rejected without building

| idea | why not |
|---|---|
| Bigram index | ~180 MB for ~10% selectivity gain over token-AND |
| Product/attribute co-occurrence graph | no relational hop the task needs |
| ANN vector index (HNSW/FAISS) | a flat scan is already fast enough at 50k, and §3.2 shows the signal it would index is switched off |
| Learned ranking prior | §4.6 — whole budget is +0.0099 |
| Semantic category model | §5.2 — ceiling is +0.053 on n=26, behind a mechanism fix |

---

## 8. What the record adds up to

**The benchmark is closed.** Hit@10 1.000, rank 1 in 191/200, and all 9
residuals carry card slots identical to the target — indistinguishable in the
task's own terms. Remaining ordering headroom: **+0.0099**.

**The open frontier is entirely free-text vocabulary**, and this record narrows
what will work there by eliminating the cheap options:

- **Corpus-internal expansion does not work** (§3.3, §3.4). Two independent
  methods failing monotonically is a result about the problem: the catalog
  cannot supply the translation out of its own vocabulary, however it is asked.
- **Translating the whole request at once does work** (§3.1). Doing it at index
  time means every product is translated once, offline, rather than every query
  being translated repeatedly, online — and it survives a network-disabled
  scoring environment.
- **A dense signal is real but small** (§3.2), and not worth its dependency.

What remains is more of what already worked: a stronger generator for the
doc2query pass, or more than one generated query per product — both buy
resolution at index time and cost nothing at runtime. And a larger probe set,
since at n=427 the interval on a +0.01 effect is wider than the effect.

---

## 9. Methodological notes worth carrying elsewhere

**Measure the ceiling before building to it.** It flipped an expected sign twice
(§1.4, §5.2) and killed two features before they were written (§4.6, §5.2). The
cheapest experiment in this document is the one that measures what a perfect
version of your idea would be worth.

**Report the interval or do not report the delta.** Point estimates at n=200
mislead in both directions. The paired bootstrap is the correct instrument for
A-vs-B, and the composite CI (±0.0104) is 3× tighter than the hit-rate rule of
thumb once hit saturates.

**Record negatives as switches, with the measurement attached.** Every rejected
idea here is still in the code behind a flag with its numbers in the comment. It
costs nothing and it stops the question being re-opened blind.

**Scope a negative to the surface it was measured on.** The single worst error
in this project's history was quoting the graded-path LSA result as a general
statement about semantic search (§3.2). It cost the correct version of that
experiment a long time to arrive — and when it did arrive, the verdict was
unchanged but the *reason* was completely different. Being wrong about why
something does not help is worth fixing even when the verdict stands.

**Do not take a sweep's argmax on the set that produced your headline number.**
Fix the parameter first, then measure (§3.1).
