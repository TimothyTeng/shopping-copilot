"""Inverted index over the catalog.

Unigram postings only. A bigram index was measured at ~180 MB for roughly a 10%
selectivity gain over unordered token-AND, so phrase matching is instead a
*verified bonus* on a narrowed candidate set rather than an index.
"""
from __future__ import annotations

import math
from array import array
from collections import defaultdict

from .catalog import CatalogStore
from .normalize import tokens


class InvertedIndex:
    """token -> sorted document ordinals, plus IDF and phrase verification."""

    __slots__ = ("postings", "n_docs", "_idf", "_phrase_cache", "_store")

    def __init__(self, store: CatalogStore) -> None:
        self._store = store
        self.n_docs = len(store)
        buckets: dict[str, list[int]] = defaultdict(list)
        for doc, text in enumerate(store.text):
            for term in set(tokens(text)):
                buckets[term].append(doc)
        self.postings: dict[str, array] = {
            term: array("i", docs) for term, docs in buckets.items()
        }
        self._idf: dict[str, float] = {}
        self._phrase_cache: dict[str, frozenset[int]] = {}

    # -- term statistics ---------------------------------------------------
    def df(self, term: str) -> int:
        """How many products contain `term`. 0 if the token is not in the catalog
        — which is how a shopper word that no product uses is detected."""
        posting = self.postings.get(term)
        return len(posting) if posting is not None else 0

    def idf(self, term: str) -> float:
        """BM25 probabilistic IDF, memoised: how much evidence one token carries.

        Every coverage score in `rank.py` is a ratio of these, so a rare token
        dominates a common one rather than counting as one match of many."""
        cached = self._idf.get(term)
        if cached is None:
            df = self.df(term)
            cached = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
            self._idf[term] = cached
        return cached

    def docs_for(self, term: str) -> array:
        """Sorted document ordinals containing `term`; an empty array if unseen."""
        return self.postings.get(term, array("i"))

    # -- phrase verification ----------------------------------------------
    def phrase_docs(self, phrase: str) -> frozenset[int]:
        """Documents containing `phrase` contiguously. Narrow, then verify."""
        cached = self._phrase_cache.get(phrase)
        if cached is not None:
            return cached
        terms = tokens(phrase)
        if not terms:
            result: frozenset[int] = frozenset()
        else:
            rarest = min(terms, key=self.df)
            candidates = self.docs_for(rarest)
            store = self._store
            result = frozenset(d for d in candidates if store.contains(d, phrase))
        if len(self._phrase_cache) < 8192:
            self._phrase_cache[phrase] = result
        return result

    def phrase_df(self, phrase: str) -> int:
        """How many products contain `phrase` contiguously (verified, not indexed)."""
        return len(self.phrase_docs(phrase))

    def informative(self, term: str, max_df_ratio: float) -> bool:
        """True when a term is present in the catalog but not so common it says
        nothing. Both halves matter: df==0 cannot narrow anything, and a token in
        a third of the catalog ("women") is category noise, not a requirement."""
        df = self.df(term)
        return 0 < df <= self.n_docs * max_df_ratio
