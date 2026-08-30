"""BM25 over the flattened catalog.

Kept separate from `InvertedIndex` because it answers a different question.
`InvertedIndex` asks *"does this document contain the term"* and is the right
tool when the shopper's words were lifted from the document — which is exactly
what the official simulator does. BM25 asks *"how well does this document
explain the query"*, which is the right tool when the shopper is describing the
product in their own words.

The measured history matters here. BM25F as a *rescorer bolted onto the
conjunctive score* cost 0.04 on the benchmark and was rejected. That is not a
contradiction: on the benchmark the query is quoted from the target, so term
frequency is noise. On natural language a stateless BM25 over the whole
transcript beat the entire conjunctive pipeline (0.4775 vs 0.3749). Both results
are real, which is why this is a mode rather than a replacement.

Term frequencies are stored as parallel arrays per term to keep the index near
the size of the boolean one.
"""
from __future__ import annotations

import math
from array import array
from collections import Counter

from .catalog import CatalogStore
from .normalize import norm, tokens


class Bm25Index:
    """Postings with term frequencies, plus document lengths."""

    __slots__ = ("docs", "freqs", "lengths", "n_docs", "avg_len", "_idf")

    def __init__(self, store: CatalogStore) -> None:
        self.n_docs = len(store)
        docs: dict[str, array] = {}
        freqs: dict[str, array] = {}
        lengths = array("i")
        for doc, text in enumerate(store.text):
            counts = Counter(tokens(text))
            lengths.append(sum(counts.values()))
            for term, freq in counts.items():
                posting = docs.get(term)
                if posting is None:
                    posting = docs[term] = array("i")
                    freqs[term] = array("i")
                posting.append(doc)
                freqs[term].append(freq)
        self.docs = docs
        self.freqs = freqs
        self.lengths = lengths
        self.avg_len = (sum(lengths) / len(lengths)) if len(lengths) else 1.0
        self._idf: dict[str, float] = {}

    def df(self, term: str) -> int:
        posting = self.docs.get(term)
        return len(posting) if posting is not None else 0

    def idf(self, term: str) -> float:
        cached = self._idf.get(term)
        if cached is None:
            df = self.df(term)
            cached = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
            self._idf[term] = cached
        return cached

    def score(self, query: str, cfg, limit: int) -> list[tuple[int, float]]:
        """Rank documents for a free-text query. Returns (doc, score) pairs.

        Repeated query terms are counted once. A shopper who says "leather"
        three times across a conversation does not mean it three times as much,
        and letting the count through makes the ranking depend on how chatty
        they were.
        """
        k1, b = cfg.bm25_k1, cfg.bm25_b
        max_df = self.n_docs * cfg.max_token_df_ratio
        scores: dict[int, float] = {}
        for term in set(tokens(norm(query))):
            posting = self.docs.get(term)
            if posting is None or len(posting) > max_df:
                continue
            idf = self.idf(term)
            for doc, freq in zip(posting, self.freqs[term]):
                norm_len = 1.0 - b + b * (self.lengths[doc] / self.avg_len)
                scores[doc] = scores.get(doc, 0.0) + idf * freq / (freq + k1 * norm_len)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return ranked[:limit]
