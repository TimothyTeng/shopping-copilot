"""Session simulator with pluggable phrasing.

The official evaluator always words the customer's replies the same way, so a
local score cannot see how much of our performance depends on that wording. This
module replays the *same* session logic through different renderers, which is the
only way to observe the failure before the private run does.

Session semantics (disclosure rules, override timing, hit detection) are imported
from the official evaluator, never reimplemented — only the surface phrasing
changes.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

KIT_ROOT = Path(__file__).resolve().parents[2] / "techjam-conversational-search"
if str(KIT_ROOT) not in sys.path:
    sys.path.insert(0, str(KIT_ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    ALLOWED_ATTRIBUTES,
    MAX_TURNS,
    TOP_K,
    classify_constraint,
    coarse_category,
    materialize_hidden_fields,
    normalize_recommendations,
)


# --------------------------------------------------------------------------
# Renderers: identical information, different words.
# --------------------------------------------------------------------------
class L0:
    """The evaluator's own wording. Must reproduce the official score exactly."""
    name = "L0 exact wording"

    def open_buy(self, cat, c):  return f"I'm looking for {cat}. A key requirement is: {c}."
    def open_browse(self, cat):  return f"I'm looking for {cat}, but I'm still exploring."
    def open_override(self, cat, old): return f"I'm looking for {cat}. {old}"
    def reply(self, matches):    return "For that, what matters is: " + "; ".join(matches) + "."
    def none_left(self, attr):   return f"I don't have an additional preference for {attr}."
    def no_pref(self, attr):     return f"I don't have a preference for {attr}; please use your judgment."
    def override(self, new):     return f"Actually, ignore my earlier preference. What I need is: {new}."
    def nudge(self):             return "Those options are not quite right yet. Ask me about one specific attribute."


class L1(L0):
    """A plausible rewrite: same facts, different carrier sentences."""
    name = "L1 paraphrased"

    def open_buy(self, cat, c):  return f"After some {cat} — it really has to be {c}."
    def open_browse(self, cat):  return f"Just browsing {cat} for now, nothing fixed yet."
    def open_override(self, cat, old): return f"After some {cat}. {old}"
    def reply(self, matches):    return "Main thing for me would be " + ", and also ".join(matches) + "."
    def none_left(self, attr):   return f"Nothing else springs to mind on {attr}."
    def no_pref(self, attr):     return f"No strong feelings about {attr} — your call."
    def override(self, new):     return f"Hmm, scratch that. Really what I'm after is {new}."
    def nudge(self):             return "Not quite. Try asking me about one thing at a time."


class L2(L1):
    """Paraphrase plus conversational noise, casing and punctuation drift."""
    name = "L2 noisy"

    def reply(self, matches):
        return "so yeah i'd say " + " and " + " plus ".join(matches).lower() + " if that helps"

    def open_buy(self, cat, c):
        return f"hey — after some {cat} i guess, main thing is {c} really".lower()


class L4(L0):
    """Carriers removed entirely: bare declaratives, no signalling phrase."""
    name = "L4 no carriers"

    def open_buy(self, cat, c):  return f"{cat}. {c}."
    def open_browse(self, cat):  return f"{cat}."
    def open_override(self, cat, old): return f"{cat}. {old}"
    def reply(self, matches):    return ". ".join(matches) + "."
    def none_left(self, attr):   return "Nothing else."
    def no_pref(self, attr):     return f"No preference for {attr}."
    def override(self, new):     return f"Changed my mind. {new}."
    def nudge(self):             return "Ask me one thing."


RENDERERS = {"L0": L0, "L1": L1, "L2": L2, "L4": L4}


# --------------------------------------------------------------------------
def run_session(agent, sample, products, categories, renderer, rng=None):
    """Mirror of the official session loop, with phrasing delegated."""
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}
    target = str(sample["ground_truth"]["parent_asin"])
    scenario = sample["scenario_type"]
    category = coarse_category(categories.get(target, []))

    disclosed: set[str] = set()
    boundary_used = False
    override_applied = scenario != "intent_override"

    if scenario == "buying" and card.get("hard_constraints"):
        first = str(card["hard_constraints"][0])
        disclosed.add(first)
        message = renderer.open_buy(category, first)
    elif scenario == "intent_override":
        message = renderer.open_override(category, str(behavior["override"]["old_value"]))
    else:
        message = renderer.open_browse(category)

    session_id = f"sim_{sample['sample_id']}"
    agent.reset(session_id, sample["user_profile"])
    hit_turn = best_rank = None

    for turn in range(1, MAX_TURNS + 1):
        try:
            response = agent.respond(session_id, message, turn, TOP_K)
        except Exception:
            response = {"message": "", "ask_attribute": None, "recommendations": []}
        ranked = normalize_recommendations(response.get("recommendations"), set(products))

        if override_applied and target in ranked:
            best_rank, hit_turn = ranked.index(target) + 1, turn
            break
        if turn == MAX_TURNS:
            break

        override = (effective.get("behavior") or {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            message = renderer.override(new_value)
            continue

        attribute = response.get("ask_attribute")
        attribute = attribute if isinstance(attribute, str) else None
        if scenario == "boundary" and not boundary_used and attribute:
            message, boundary_used = renderer.no_pref(attribute), True
        elif not attribute:
            message = renderer.nudge()
        else:
            if attribute not in ALLOWED_ATTRIBUTES:
                attribute = "other"
            pool = [str(v) for v in card.get("hard_constraints", [])]
            pool += [str(v) for v in card.get("soft_preferences", [])]
            matches = [
                v for v in pool
                if v not in disclosed
                and (attribute == "other" or classify_constraint(v) == attribute)
            ][:2]
            if not matches:
                message = renderer.none_left(attribute)
            else:
                disclosed.update(matches)
                message = renderer.reply(matches)

    return {
        "sample_id": sample["sample_id"],
        "scenario_type": scenario,
        "hit": hit_turn is not None,
        "first_hit_turn": hit_turn,
        "best_rank": best_rank,
        "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
    }
