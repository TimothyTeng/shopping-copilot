# Measurement

How this project decides whether a change is real, and what that machinery has
cost it in conclusions.

The short version: the agent is measured against two different worlds, an
overturned claim is kept visible rather than deleted, and no delta is reported
without the interval that bounds it. Five of the six claims tested in the most
recent round of work were overturned or downgraded by the instrument described
here. That is the point of building it.

---

## 1. Two worlds, never conflated

**The graded world** is the official evaluator: 200 sessions, a simulator that
lifts the shopper's constraints **verbatim from the target product's own
`features` and `details`**. Query and document share vocabulary by construction.
Score: **0.9456**.

**The free-text world** is `tools/stress.py`: a shopper who types what they
actually want, having never seen the page. Score on the prose path: **0.7347**.

The gap between those two numbers is not a bug. It is the measure of how much
the graded score owes to an artifact of its own simulator, and the whole reason
the second harness exists. A system that only matches exact strings scores well
in the first world and can be useless to a person in the second.

**Every result in this document belongs to one world and does not transfer.**
The clearest case: the LLM tier *helps* on free text and *costs* on the graded
path (0.9456 → 0.9390), because a generated listing can only add noise to a
query that is already in catalog vocabulary. Ship with `backend="null"`.

---

## 2. The probe set

### The hand-authored 26 — authoritative, and still the default

Written by a person who had not seen the agent's behaviour, targets chosen
before any wording. No model wrote a word of them. Nothing was ever tuned on
them. They remain `tools/stress.py`'s default and the honest set.

Their limit is arithmetic: **the noise floor at n=26 is ±0.03**, which is wider
than most decisions worth making.

### The generated 427 — resolution, bought at a cost

`tools/genprobes.py` samples the catalog stratified across coarse categories,
shows each product to an LLM, and asks for the turns a shopper would type
*without having seen the page*.

A generated test set is worthless if it flatters the system, so five rejection
guards run on every candidate. All five fired on the real run:

| rejected | n | why it matters |
|---|---|---|
| copied (overlap > 0.75) | 118 | quoting the listing is the official simulator's failure mode, reproduced |
| `title_ngram` | 19 | overlap can sit at 0.6 while still lifting the distinctive part of the title |
| `off_target` (overlap < 0.20) | 14 | the model wrote about something else — unanswerable, not hard |
| `no_negation` | 12 | a mislabelled probe corrupts the per-tag table it exists to produce |
| `restates` | 10 | "men's tshirt" → "black men's tshirt" is a one-turn probe padded to four |

**427 kept from 600. Mean verbatim overlap 0.54**, against the hand-authored
set's 0.68 — meaningfully harder, not easier.

**Leakage control.** Probes are generated with `qwen3-coder` (:30801); HyDE
retrieves with Qwen2.5-7B (:30800). Generating and retrieving with the same
model would let the rewriter invert its own vocabulary priors. `genprobes.py`
warns when they match, and the probe file records the model it was written by.

**The standing caveat.** This set is model-written and inherits an LLM's idea of
how shoppers talk. It buys resolution, not authority. The framing to use is
**n=26 for truth, n=427 for power**, and §5 is the work that would put a number
on the caveat instead of a hedge.

---

## 3. Controls

A score in isolation means nothing. Every probe is run against four reference
points in the same pass.

| control | what it gets | what it proves |
|---|---|---|
| **oracle** | the target's own title and text as the query | the probe set is *valid* — a failure measures the agent, not a broken probe |
| **BM25, whole transcript** | all turns, on turn 1 | can the dialogue beat no dialogue at all? |
| **BM25, turn-matched** | exactly `turns[:t]` at turn t | the same, without the information asymmetry |
| **the agent** | one turn at a time | — |

**The oracle at 0.9990 (hit 1.000) is the validity proof.** Every generated
target is findable with the right words. Without it, a low agent score is
ambiguous between a weak agent and a test of nothing.

**Why two BM25 controls.** The whole-transcript control is generous on purpose —
it never has to ask a question, so it cannot lose efficiency. But that
generosity makes it *unreadable* on any probe whose decisive word arrives late:
it never pays the turns the agent spends waiting to hear the thing that matters.

That is not hypothetical. It produced a false finding, documented in §6.

The turn-matched control was added in response. It has no state, no slots and no
category, and it hears exactly what the agent has heard, when the agent heard
it. **A gap that survives against it is dialogue handling. A gap that appears
only against the one-shot control is the asymmetry.**

The two land at nearly identical composites by opposite routes, which is why
both are kept:

| n=427, `--retrieval bm25` | score | Hit@10 | MRR | MTTC |
|---|---|---|---|---|
| agent | **0.7347** | **0.867** | **0.529** | **3.85** |
| BM25, turn-matched | 0.6409 | 0.756 | 0.414 | 4.07 |
| BM25, whole transcript | 0.6415 | 0.691 | 0.526 | 4.09 |

Turn-matched has better *recall* (0.756 vs 0.691 — ten shots at growing
prefixes) and worse *ranking* (MRR 0.414 vs 0.526).

---

## 4. The paired bootstrap

A point estimate cannot tell a 0.015 effect from a 0.015 resample.
`tools/bootstrap.py` resamples probes with replacement and recomputes the
composite from per-probe hit, reciprocal rank and first-hit turn — the official
metric, not a rounded summary.

**It is paired.** One draw picks a set of probe indices and scores *every*
configuration on that same set, so the probe-to-probe variance that dominates an
unpaired interval cancels and what remains is the effect of the change. This is
not a nicety: an unpaired interval on two configurations that differ only in a
fusion weight would be several times too wide and would call every real effect
noise.

Pairing is guaranteed structurally rather than by convention. `--vs` runs every
configuration over the identical probe list in the identical order, in one
process, so alignment cannot drift:

```bash
python -m tools.stress --track natural --probes data/probes_generated.jsonl \
    --retrieval bm25 \
    --vs "hyde:backend=hyde" \
    --vs "hyde always:backend=hyde,hyde_gate=always" \
    --crosstab
```

`--vs` takes `label:key=value,key=value` for any field of `Settings`. A label
may not contain `=`, and an unknown key is rejected **before the catalog loads**
— a malformed spec discovered after a 427-probe run has already cost the run it
invalidates.

Every run prints a score CI and paired deltas with `p(Δ>0)`. `--crosstab` adds
score-per-tag-per-configuration with the winner marked, and a **"leads no tag at
all"** line. That line is the one that matters: it is what turned "conjunctive
is 0.29 behind" into "conjunctive leads 0 of 7 tags" — a trade-off into a
dead-weight verdict.

---

## 5. Human adjudication — the control a script cannot run

The five guards in §2 check what a regular expression can see. They cannot tell
you whether a kept probe reads like something a person would type, or whether
the target is genuinely the best answer to it. That is the difference between
*the set is clean* and *the set is valid*.

```bash
python -m tools.adjudicate sample -n 40   # blind sheet, stratified by tag
python -m tools.adjudicate score          # agreement + Wilson intervals
```

The sheet shows the probe and the target and **nothing about how the agent
scored on it**. An adjudicator who can see the failures is grading the agent,
not the probe. Stratified by tag so the rare tags — the ones whose per-tag
scores are least stable and most likely to be acted on — are represented.

Three axes, each y/n: *phrasing* (would a shopper type this?), *target* (is it a
reasonable best answer?), *tag* (is the label right?). `score` reports each with
a **Wilson** interval — correct near 0 and 1, where a normal approximation on 40
samples is not — plus the **all-three-hold** rate, because 90% on each axis
separately is closer to 73% sound overall.

**Status: sheet generated at `data/adjudication.jsonl`, not yet filled in.**
This is the outstanding gap in the evidence. Until it is done, the n=427 numbers
carry a qualitative caveat where they could carry a bounded one.

---

## 6. What the instrument has overturned

Kept in full in the README's *Corrections* section. Summarised here because the
pattern is the point.

| claim | drawn at | overturned at |
|---|---|---|
| "HyDE's gain is pure ordering — recall held exactly" | n=26, Hit@10 0.846 both ways | n=427: recall **rises** 0.867 → 0.899, MRR **falls** 0.529 → 0.511. "Held exactly" was 22/26 twice. |
| "HyDE is worth +0.015" | n=427 point estimate | paired: CI **[−0.0030, +0.0334]**, p 0.947. Probable, not demonstrated. |
| "the gate costs ~0.011" | n=26 | n=427: `always` − `unsatisfied` = **0.0008**. No trade exists; the gate is free. |
| "`hyde_rrf_weight` 0.3 looks best" | n=26 sweep | n=427: **flat**. 0.3 is now 0.0008 *below* 0.5. |
| "conjunctive loses to a stateless control" | n=26, −0.079 | **correct but unearned**: that CI was [−0.212, +0.055], p 0.118 — it spanned zero. At n=427: −0.1964, CI [−0.2364, −0.1576]. |
| "the control beating the agent on `brand` is a bug" | one-shot control, 0.147 gap | **artifact.** Turn-matched: +0.051, CI [−0.045, +0.148]. |
| "the hold-back gate discards turn-1 hits" | per-probe trajectories | **disconfirmed.** `gate_enabled=false` = −0.0008, CI [−0.0100, +0.0089]. |

Two of these are worth reading twice.

**The paired test did not rescue the HyDE result, it bounded it.** Wiring it was
expected to promote +0.015 to significance; instead it showed the lower bound
sits below zero. An instrument that only ever confirms is not an instrument.

**A conclusion can be correct and still unearned.** The conjunctive claim was
right all along, and the n=26 evidence used to state it did not support it. This
project records that difference, because the reasoning is what generalises, not
the luck.

---

## 7. What is settled

**Conjunctive is dead weight on free text.** It leads **0 of 7 tags** —
including `multi_attr`, the tag it exists to serve. `rrf` also leads 0 of 7 and
sits between the other two on every single tag: a dominated middle, not a hedge.

| tag | n | conjunctive | `bm25` | `rrf` | turn-matched | one-shot |
|---|---|---|---|---|---|---|
| brand | 57 | 0.5799 | 0.7171 | 0.7630 | 0.7678 | **0.8636** |
| colloquial | 73 | 0.3043 | **0.7219** | 0.6587 | 0.5706 | 0.5053 |
| multi_attr | 48 | 0.5504 | **0.8463** | 0.7770 | 0.7621 | 0.7654 |
| natural | 45 | 0.6079 | **0.8384** | 0.7627 | 0.7577 | 0.7758 |
| negation | 61 | 0.3758 | **0.6502** | 0.6021 | 0.5553 | 0.4587 |
| non_catalog | 73 | 0.3280 | **0.6931** | 0.5956 | 0.5278 | 0.5337 |
| vague | 70 | 0.4878 | **0.7364** | 0.6834 | 0.6449 | 0.7028 |

It remains the default only because the **graded** path is the simulator, where
its premise — the shopper quotes the target — actually holds. Retrieval mode is
a deployment choice, not a tuning problem.

**The agent beats both controls**, +0.0939 against turn-matched, CI [+0.0653,
+0.1222] — the same point estimate as against the one-shot control with a
*tighter* interval, because a control that tracks the agent probe-for-probe
leaves less variance in the differences.

**`brand` is the only tag the agent loses**, and it loses to both controls, so
the residual is not purely asymmetry even though the headline was. At +0.051
spanning zero it is not actionable; re-check it if the probe set grows.

**HyDE is not uniform**, and this is the most useful unexploited finding here:

| tag | no LLM | HyDE | Δ |
|---|---|---|---|
| brand | 0.7171 | **0.7886** | +0.072 |
| vague | 0.7364 | **0.7747** | +0.038 |
| non_catalog | 0.6931 | **0.7303** | +0.037 |
| colloquial | **0.7219** | 0.7186 | −0.003 |
| multi_attr | **0.8463** | 0.8255 | −0.021 |
| natural | **0.8384** | 0.7899 | −0.049 |

It pays exactly where the shopper's vocabulary is *off*-catalog and costs where
it is already good. The aggregate +0.015 is a small net of two much larger
opposing effects — which is *why* it cannot clear zero. A gate that
discriminated on that would plausibly produce an effect that does; the current
`unsatisfied` gate fires on nearly the same turns as `always`, so it is not
discriminating on it at all.

**`negation` is the weakest tag** on the shipping prose path (0.6502 against
`multi_attr`'s 0.8463). At n=26 it was 2 probes scoring 0.000 — a curiosity. At
n=61 it is a ranked, actionable weakness, and `enable_reset` in `config.py` is a
documented "CORRECT mechanism that measures neutral" parked because the vote
resolver mis-buckets the single probe it targeted.

---

## 8. Limits of this instrument

- **The generated set is model-written.** §5 is unfinished, so this is a
  qualitative caveat rather than a bounded one.
- **The n=427 tags are unequal** (45–73 probes). Per-tag deltas are directional;
  only the aggregate carries a trustworthy interval.
- **A "leads no tag" verdict is only as good as the tag taxonomy**, which was
  authored alongside the generator.
- **Nothing here measures latency under load.** HyDE's 6 s timeout is a
  correctness device measured against a local vLLM at ~2 s median; a slower
  server changes the trade and no experiment in this document would notice.
- **The paired bootstrap assumes probes are independent.** They are drawn from
  one catalog by one model, so the true interval is likely a little wider than
  reported.

---

## 9. Commands

```bash
# graded world
python -m tools.harness run
python -m tools.harness ci --compare tie_rerank=mmr

# free-text world — the hand-authored 26, authoritative
python -m tools.stress --track natural
python -m tools.stress --track natural --show          # per-probe, with reasons

# free-text world — the generated 427, for resolution
python -m tools.stress --track natural --probes data/probes_generated.jsonl \
    --vs "label:key=value" --crosstab

# regenerate the probe set (needs a model that is NOT the retrieval model)
python -m tools.genprobes --n 600 --out data/probes_generated.jsonl

# the human control
python -m tools.adjudicate sample -n 40
python -m tools.adjudicate score
```

On Windows, prefix with `PYTHONIOENCODING=utf-8` — the cp1252 console default
crashes on product titles containing emoji or typographic dashes.
