"""Term associations mined offline from the catalog (see tools/train_assoc.py).

The semantic sibling of `src/fuzzy.py`. Fuzzy repair closes the *spelling*
distance between what the shopper typed and what the catalog contains; this
closes the *vocabulary* distance — "jumper" is not a misspelling of "sweater",
and no edit distance will ever connect them.

Runtime is pure stdlib reading a 0.2 MB pruned PPMI table, the same shape as
`category_clf`: the model is a build artefact, the inference is a dict lookup.
Absent file means absent feature, never an error.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "models" / "assoc.json.gz"


class Associations:
    """token -> ranked catalog terms that share its titles."""

    __slots__ = ("table",)

    def __init__(self, table: dict[str, list]) -> None:
        """Wrap a mined PPMI table (built offline by `tools/train_assoc.py`)."""
        self.table = table

    @classmethod
    def try_load(cls, path: Path = MODEL_PATH) -> "Associations | None":
        """Load the shipped table, or None if it was never built. Callers must
        handle None: the table is optional and `assoc_expand` is off by default."""
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                return cls(json.load(handle))
        except Exception:
            return None

    def neighbours(self, term: str, limit: int) -> list[tuple[str, float]]:
        """The `limit` catalog terms most associated with `term`, strongest first."""
        row = self.table.get(term)
        if not row:
            return []
        return [(other, float(value)) for other, value in row[:limit]]

    def expand(self, query_terms: set[str], per_term: int,
               weight: float) -> dict[str, float]:
        """Weights for terms the shopper did not say but the catalog uses.

        Weighted by normalized PPMI and capped at `weight`, so an expansion term
        can never outvote a term the shopper actually typed — the failure mode
        `hyde_bm25_mode="union"` was rejected for.
        """
        out: dict[str, float] = {}
        for term in query_terms:
            row = self.neighbours(term, per_term)
            if not row:
                continue
            peak = row[0][1] or 1.0
            for other, value in row:
                if other in query_terms:
                    continue
                out[other] = max(out.get(other, 0.0), weight * value / peak)
        return out
