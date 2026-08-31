"""Runtime inference for the character-n-gram category classifier.

Pure standard library: it reads the pruned weights produced offline by
tools/train_category.py and scores a message as a plain sum of postings over its
n-grams. No numpy, no sklearn, no network — safe on the scored path.

The model is linear, so a shopper phrase P scores against bucket c as

    score(c) = intercept[c] + Σ_{g in P} tf(g) · weight[g][c]

where weight already folds in the idf used at training time. A softmax over the
scores gives a calibrated confidence the resolver can gate on.
"""
from __future__ import annotations

import gzip
import json
import math
from array import array
from pathlib import Path

from .normalize import norm

DEFAULT_PATH = Path(__file__).resolve().parent / "models" / "category_clf.json.gz"


class CategoryClassifier:
    """Trained bucket classifier, loaded from a shipped gzip table.

    Off the graded path: `category_resolver` selects it, and the default is
    the substring bucket match, which survives rephrasing better."""
    __slots__ = ("lo", "hi", "classes", "intercept", "postings")

    def __init__(self, path: str | Path = DEFAULT_PATH) -> None:
        """Load the model: class list, intercepts, and ngram -> (class, weight)."""
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            model = json.load(fh)
        self.lo = model["lo"]
        self.hi = model["hi"]
        self.classes: list[str] = model["classes"]
        self.intercept: list[float] = model["intercept"]
        # ngram -> list of (class_id, weight)
        self.postings: dict[str, list] = model["postings"]

    @classmethod
    def try_load(cls, path: str | Path = DEFAULT_PATH):
        """Return a classifier, or None if the weights have not been trained."""
        return cls(path) if Path(path).exists() else None

    def _char_ngrams(self, text: str) -> dict[str, int]:
        """char_wb 3–5-grams with counts. Must match tools/train_category.py."""
        counts: dict[str, int] = {}
        for word in text.split():
            w = f" {word} "
            for n in range(self.lo, self.hi + 1):
                for i in range(len(w) - n + 1):
                    gram = w[i:i + n]
                    counts[gram] = counts.get(gram, 0) + 1
        return counts

    def predict(self, message: str) -> tuple[str | None, float]:
        """Return (bucket_key, confidence). confidence is the softmax mass on the
        winner; None only if the message has no known n-gram at all."""
        grams = self._char_ngrams(norm(message))
        scores = dict(enumerate(self.intercept))
        touched = False
        for gram, tf in grams.items():
            for cid, weight in self.postings.get(gram, ()):
                scores[cid] = scores[cid] + tf * weight
                touched = True
        if not touched:
            return None, 0.0
        best_id = max(scores, key=scores.__getitem__)
        top = scores[best_id]
        total = sum(math.exp(s - top) for s in scores.values())
        return self.classes[best_id], 1.0 / total
