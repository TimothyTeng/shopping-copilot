"""Text normalization shared by the catalog and the dialogue.

Both sides must normalize identically or phrase grounding silently fails.
"""
from __future__ import annotations

import re

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_TOKEN = re.compile(r"[a-z0-9]+")


def norm(value: object) -> str:
    """Lowercase, collapse every non-alphanumeric run to a single space."""
    return _NON_ALNUM.sub(" ", str(value).lower()).strip()


def tokens(text: str) -> list[str]:
    """Tokens of an already-normalized string."""
    return _TOKEN.findall(text)


def padded(text: str) -> str:
    """Space-padded form so `" phrase "` substring tests respect word bounds."""
    return f" {text} "


# Words that carry conversation rather than product meaning. Catalog grounding
# rejects most filler on its own, but these appear in product text often enough
# to survive it, so they are suppressed explicitly.
#
# Deliberately NOT a list of the simulator's template phrases: it is generic
# English plus shopping-dialogue verbs, so paraphrases are covered too.
DIALOGUE_STOP: frozenset[str] = frozenset("""
a about actually additional after all also am an and any anything are around as ask
at be been being browsing but buy buying by call can care changed come could day
do does doesn dont down each either else even ever every exploring feel feelings few
find fine first fixed for forget from get give go going good got had has have having
he her here hers him his how hmm i if in instead into is it its just keep key kind
know like ll look looking lot m main make many matter matters may me mind mine more
most much must my need needs no nope not nothing now of off on one only open or other
our out over own particular please prefer preference preferences pretty quite rather
re really requirement requirements right s said same say scratch see seem she should
side so some something specific springs still strong stuff such sure t take tell than
that thats the their them then there these they thing things think this those though
thought through to too top two up us use usually very want wanted wants was way we
well were what whatever when where which while who why will with within would yeah
yes yet you your yours judgment
""".split())
