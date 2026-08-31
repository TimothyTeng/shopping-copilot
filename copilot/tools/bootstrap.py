"""Bootstrap confidence intervals for the composite score.

The composite is an aggregate of 200 sessions, and a bare point estimate hides
whether a delta is real or a resample away from zero. This resamples the
sessions with replacement to put an interval around the score, and — the part
that actually settles a design decision — around the *delta* between two configs.

Two things make the delta test trustworthy:

* **It is paired.** Each bootstrap draw picks one set of session indices and
  scores BOTH configs on that same set, so the shared session-to-session
  variance cancels and what remains is the effect of the change. An unpaired
  test on two independent resamples would be far wider and would call real
  effects noise.
* **It reads the sessions, not the rounded summary.** hit, reciprocal rank, and
  first-hit turn are pulled per session and the composite is recomputed from
  their means on each draw, exactly as the official metric defines it.

Pure standard library. `random` is seeded so a reported interval reproduces.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

# A miss counts as turn 11 (MAX_TURNS + 1), per the specification.
_MISS_TURN = 11


@dataclass(slots=True)
class Vectors:
    """Per-session components, the only things the composite needs."""

    hit: list[int]
    rr: list[float]
    turn: list[int]

    def __len__(self) -> int:
        """Number of sessions carried."""
        return len(self.hit)


def to_vectors(sessions: list[dict]) -> Vectors:
    """Reduce sessions to the per-session hit/reciprocal-rank/turn vectors the
    resampler needs, so a resample is an index draw rather than a re-scoring."""
    return Vectors(
        hit=[int(s["hit"]) for s in sessions],
        rr=[float(s["reciprocal_rank"]) for s in sessions],
        turn=[s["first_hit_turn"] if s["first_hit_turn"] is not None else _MISS_TURN
              for s in sessions],
    )


def _composite_on(idx: list[int], v: Vectors) -> float:
    """The official composite over a chosen (possibly resampled) index set."""
    n = len(idx)
    hit = sum(v.hit[i] for i in idx) / n
    mrr = sum(v.rr[i] for i in idx) / n
    mttc = sum(v.turn[i] for i in idx) / n
    eff = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return 0.50 * hit + 0.30 * mrr + 0.20 * eff


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 1]. Input must be sorted."""
    if not sorted_vals:
        return 0.0
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 >= len(sorted_vals):
        return sorted_vals[lo]
    return sorted_vals[lo] * (1 - frac) + sorted_vals[lo + 1] * frac


@dataclass(slots=True)
class CI:
    point: float
    lo: float
    hi: float
    mean: float


def score_ci(sessions: list[dict], iters: int = 10000, alpha: float = 0.05,
             seed: int = 0) -> CI:
    """Percentile CI for one config's absolute composite score."""
    v = to_vectors(sessions)
    n = len(v)
    point = _composite_on(list(range(n)), v)
    rng = random.Random(seed)
    draws = [_composite_on(rng.choices(range(n), k=n), v) for _ in range(iters)]
    draws.sort()
    return CI(point=point,
              lo=_percentile(draws, alpha / 2),
              hi=_percentile(draws, 1 - alpha / 2),
              mean=sum(draws) / len(draws))


@dataclass(slots=True)
class DeltaCI:
    delta: float          # point estimate, b - a
    lo: float
    hi: float
    mean: float
    p_positive: float     # share of resamples with delta > 0
    a: float              # point score of config a
    b: float              # point score of config b


def paired_delta_ci(sessions_a: list[dict], sessions_b: list[dict],
                    iters: int = 10000, alpha: float = 0.05,
                    seed: int = 0) -> DeltaCI:
    """Paired bootstrap CI for (b - a). The two session lists must be aligned
    position-for-position (same samples, same order)."""
    if len(sessions_a) != len(sessions_b):
        raise ValueError("paired bootstrap needs equal-length, aligned runs")
    va, vb = to_vectors(sessions_a), to_vectors(sessions_b)
    n = len(va)
    a_point = _composite_on(list(range(n)), va)
    b_point = _composite_on(list(range(n)), vb)
    rng = random.Random(seed)
    deltas: list[float] = []
    positive = 0
    for _ in range(iters):
        idx = rng.choices(range(n), k=n)      # one draw, scored on both
        d = _composite_on(idx, vb) - _composite_on(idx, va)
        deltas.append(d)
        if d > 0:
            positive += 1
    deltas.sort()
    return DeltaCI(
        delta=b_point - a_point,
        lo=_percentile(deltas, alpha / 2),
        hi=_percentile(deltas, 1 - alpha / 2),
        mean=sum(deltas) / len(deltas),
        p_positive=positive / iters,
        a=a_point,
        b=b_point,
    )
