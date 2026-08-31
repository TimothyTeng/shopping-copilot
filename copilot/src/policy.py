"""What to ask, and when to speak.

**What to ask.** The simulator matches a requested attribute with
`attribute == "other" or classify(value) == attribute`, then returns the first
two matches. The match set for `"other"` is therefore a superset of every typed
attribute's match set under identical truncation, so asking `"other"` weakly
dominates every typed ask in every state. It is not a trick; it is the
open-ended question a good assistant asks when it does not yet know which
dimension matters.

**When to speak.** A session ends at the first hit, so surfacing the target at
rank 8 locks that rank in permanently. Moving 8 -> 1 is worth ~0.26 of score
while waiting a turn costs ~0.02, so speaking early with a weak list is a bad
trade. We hold recommendations until the evidence is strong enough — but only up
to a point, because never speaking scores zero.
"""
from __future__ import annotations

import math

from .state import DialogueState

# Order used only if "other" ever stops dominating. Derived from the measured
# yield of each bucket weighted by how sharply it narrows the catalog.
TYPED_FALLBACK = ("feature", "material", "color", "style", "size", "use_case")


def choose_attribute(state: DialogueState) -> str | None:
    """Pick the question with the highest expected yield."""
    if "other" not in state.exhausted:
        return "other"
    for attribute in TYPED_FALLBACK:
        if attribute not in state.exhausted and attribute not in state.declined:
            return attribute
    return None


def _confidence(scored: list[tuple[int, float, float]]) -> float:
    """P(the leader is the target), from the score gap to the rest of the head.

    A softmax over the head's scores. It is a calibration, not a probability —
    but it is the quantity the emission decision actually needs, and it is
    *state-dependent* in a way the three-way heuristic below is not: two
    sessions with the same slot count can have a runaway leader or a flat tie,
    and only one of them is worth committing to.
    """
    if not scored:
        return 0.0
    if len(scored) == 1:
        return 1.0
    peak = scored[0][1]
    weights = [math.exp(total - peak) for _, total, _ in scored]
    return weights[0] / sum(weights)


def should_emit(state: DialogueState, pool: int, turn: int, cfg,
                scored: list[tuple[int, float, float]] | None = None) -> bool:
    """Is the evidence strong enough that this list is worth committing to?"""
    if not cfg.gate_enabled:
        return True
    if turn >= cfg.gate_force_turn:
        return True
    if cfg.gate_mode == "margin":
        # Decision-theoretic form. A session ends at the first hit, so emitting
        # locks the rank in: commit when the leader is probably the target, and
        # keep asking when the head is a flat tie. Rank is worth ~13x a turn, so
        # the threshold sits high.
        return _confidence(scored or []) >= cfg.gate_confidence_min
    if len(state.active_slots()) >= cfg.gate_min_slots:
        return True
    return 0 < pool <= cfg.gate_pool_max


# The structured `ask_attribute` carries the machine signal and the simulator
# ignores this text entirely, so the wording is free to vary. It should: a
# shopper asked the identical question five times running would walk away.
_ACKS = ("Got it", "Noted", "Thanks", "Right", "Okay")
_ASKS = (
    "What else matters — fabric, colour, or how you'll use it?",
    "Anything else I should hold to? Material, colour, or occasion?",
    "What else should I match on — the fabric, the colour, or the use?",
    "Is there anything else that matters — style, colour, or fit?",
    "Anything else worth pinning down before I narrow this further?",
)
_OPENING = (
    "Happy to help — what matters most to you here?",
    "Let's narrow it down. What's the most important thing for you?",
)
_NOTHING_NEW = (
    "No problem. Is there anything else you'd like me to match on?",
    "That's fine. Anything else that would help me narrow it?",
)
# A correction is not a shrug. When the shopper replaces a preference we may
# learn no *new* token — the replacement is often something they already told
# us — but answering "No problem, anything else?" reads as though we missed
# that they changed their mind.
_OVERRIDE_ACK = (
    "Understood — I've dropped the earlier preference and I'm working from {kept} instead.",
    "Got it, that changes things — ignoring the earlier preference and going with {kept}.",
)


def _phrase_of(slot) -> str:
    """A slot rendered for the shopper: at most six words, elided after.

    Cosmetic only. The simulator never reads `message` (it branches on
    `ask_attribute` alone), so wording changes must be score-neutral."""
    text = slot.phrase or slot.key
    words = text.split()
    return " ".join(words[:6]) + ("…" if len(words) > 6 else "")


def compose(attribute: str | None, state: DialogueState, pool: int) -> str:
    """A question that reflects this turn, not a fixed string."""
    if attribute is None:
        return "Here are the closest matches I found."

    turn = max(state.turn, 1)
    fresh = [s for s in state.active_slots() if s.turn == state.turn]
    active = state.active_slots()

    if not active:
        return _OPENING[turn % len(_OPENING)]

    # Don't echo the shopper's own category back as though we learned it.
    category_tokens = set(state.category_key.split()) if state.category_key else set()
    fresh = [s for s in fresh if not set(s.tokens) <= category_tokens]

    if state.override_seen and state.turn == state.override_turn:
        kept = ", ".join(_phrase_of(s) for s in active[:2]) or "what's left"
        return _OVERRIDE_ACK[turn % len(_OVERRIDE_ACK)].format(kept=kept)

    parts: list[str] = []
    if fresh:
        learned = ", ".join(_phrase_of(s) for s in fresh[:2])
        parts.append(f"{_ACKS[turn % len(_ACKS)]} — {learned}.")
    elif turn > 1:
        return _NOTHING_NEW[turn % len(_NOTHING_NEW)]

    if 0 < pool <= 40:
        parts.append(f"That leaves about {pool} that fit.")

    parts.append(_ASKS[turn % len(_ASKS)] if attribute == "other"
                 else f"Do you have a preference on {attribute.replace('_', ' ')}?")
    return " ".join(parts)
