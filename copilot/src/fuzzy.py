"""Fuzzy token repair — typo tolerance on the shopper's side only.

Every layer below this keys on exact tokens, so a single mistyped character
drops a whole constraint: `rainocat`, `waterprrof`, `hoood` are absent from the
catalog entirely and ground to nothing. This maps such a token to the nearest
catalog term within a bounded edit distance, before extraction and category
resolution ever see it.

Two properties make it safe to bolt onto the scored path:

* **It only touches ABSENT alpha tokens** (``df == 0``). On the official
  simulator every shopper token is lifted verbatim from the target, so every
  token has ``df > 0`` and repair is a no-op — the benchmark is unaffected by
  construction, not by measurement.
* **The repair target must itself be a real catalog word** (``df >= floor``),
  so a typo maps toward a common word, never toward another rare typo-like term.

Damerau over plain Levenshtein because the common typo is a transposition:
`rainocat -> raincoat`, `lightwieght -> lightweight` are each one adjacent swap.

Pure standard library: a trigram inverted index for candidate generation, then a
bounded Optimal String Alignment distance to verify. No network, no deps.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from .index import InvertedIndex
from .normalize import DIALOGUE_STOP

# Alpha words only. Numbers and codes (`8929`, `NM-M`) are never typos worth
# repairing, and mapping them toward a word would inject pure noise.
_WORD = re.compile(r"[A-Za-z]{2,}")


def _trigrams(word: str) -> set[str]:
    """Boundary-padded character trigrams, so prefixes/suffixes get their own."""
    s = f"${word}$"
    return {s[i:i + 3] for i in range(len(s) - 2)}


def _osa_distance(a: str, b: str, max_dist: int) -> int | None:
    """Optimal String Alignment (restricted Damerau-Levenshtein), bounded.

    Returns the distance if it is <= ``max_dist``, else None. Bounded so a whole
    row exceeding the budget aborts early instead of filling the matrix.
    """
    la, lb = len(a), len(b)
    if abs(la - lb) > max_dist:
        return None
    prev2: list[int] = []
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        row_min = cur[0]
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if (i > 1 and j > 1 and ai == b[j - 2] and a[i - 2] == b[j - 1]):
                v = min(v, prev2[j - 2] + 1)   # adjacent transposition
            cur[j] = v
            if v < row_min:
                row_min = v
        if row_min > max_dist:
            return None
        prev2, prev = prev, cur
    return prev[lb] if prev[lb] <= max_dist else None


def _max_dist(length: int) -> int:
    """Edit budget by word length: short words get 1, longer words get 2.

    A budget of 2 on a 5-letter word would let it reach unrelated words; a
    budget of 1 on a 12-letter word would miss two-typo cases. This tracks the
    rate at which real typos accumulate with length.
    """
    return 1 if length < 7 else 2


class FuzzyRepair:
    """Maps an out-of-catalog shopper token to the nearest real catalog term."""

    __slots__ = ("index", "cfg", "terms", "tri")

    def __init__(self, index: InvertedIndex, cfg) -> None:
        """Build the trigram index of candidate correction targets."""
        self.index = index
        self.cfg = cfg
        # The pool of words a typo may map TO: alpha, long enough to be worth
        # correcting, and common enough to be a genuine word rather than noise.
        self.terms: list[str] = []
        self.tri: dict[str, list[int]] = defaultdict(list)
        floor = cfg.fuzzy_df_floor
        min_len = cfg.fuzzy_min_len
        for term, posting in index.postings.items():
            if len(term) < min_len or len(posting) < floor or not term.isalpha():
                continue
            tid = len(self.terms)
            self.terms.append(term)
            for gram in _trigrams(term):
                self.tri[gram].append(tid)

    def _repair_token(self, low: str) -> str:
        """Return a corrected token, or the input unchanged."""
        cfg, index = self.cfg, self.index
        if len(low) < cfg.fuzzy_min_len or low in DIALOGUE_STOP:
            return low
        df = index.df(low)
        if df > 0 and not cfg.fuzzy_repair_present:
            return low                      # already a catalog word: leave it
        max_dist = _max_dist(len(low))

        counts: Counter[int] = Counter()
        for gram in _trigrams(low):
            for tid in self.tri.get(gram, ()):
                counts[tid] += 1
        if not counts:
            return low

        best: str | None = None
        best_key: tuple[int, int] | None = None
        for tid, _shared in counts.most_common(cfg.fuzzy_candidates):
            cand = self.terms[tid]
            if cand == low or abs(len(cand) - len(low)) > max_dist:
                continue
            dist = _osa_distance(low, cand, max_dist)
            if dist is None:
                continue
            # Prefer the smallest edit distance, then the more common word — a
            # tie between two equally-close candidates goes to the real word a
            # shopper more likely meant.
            key = (dist, -index.df(cand))
            if best_key is None or key < best_key:
                best_key, best = key, cand
        return best if best is not None else low

    def repair_message(self, message: str) -> str:
        """Correct alpha words in place, preserving all punctuation and spacing.

        The extraction cue regexes (override, negation, no-preference) run on the
        surrounding structure — commas mark negation-clause scope — so only the
        word spans are rewritten, never the glue between them.
        """
        def sub(match: re.Match) -> str:
            """Rewrite one word in place, leaving its original case if unchanged."""
            word = match.group(0)
            repaired = self._repair_token(word.lower())
            return repaired if repaired != word.lower() else word
        return _WORD.sub(sub, message)
