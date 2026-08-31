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
from .normalize import DIALOGUE_STOP, norm, tokens


class Bm25Index:
    """Postings with term frequencies, plus document lengths."""

    __slots__ = ("docs", "freqs", "lengths", "n_docs", "avg_len", "_idf", "_store")

    def __init__(self, store: CatalogStore) -> None:
        """Build term-frequency postings. ~2x the cost of the unigram index, so
        the agent only constructs this when a prose mode or resolver reads it."""
        self._store = store          # feedback terms are read back off the text
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
        """Document frequency of `term` in the BM25 postings."""
        posting = self.docs.get(term)
        return len(posting) if posting is not None else 0

    def idf(self, term: str) -> float:
        """BM25 probabilistic IDF, memoised (same form as `index.idf`)."""
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

    # -- pseudo-relevance feedback ----------------------------------------
    def _score_terms(self, weights: dict[str, float], cfg,
                     limit: int) -> list[tuple[int, float]]:
        """BM25 over an explicitly weighted term set, rather than a query string."""
        k1, b = cfg.bm25_k1, cfg.bm25_b
        max_df = self.n_docs * cfg.max_token_df_ratio
        scores: dict[int, float] = {}
        for term, weight in weights.items():
            posting = self.docs.get(term)
            if posting is None or len(posting) > max_df or weight <= 0.0:
                continue
            idf = self.idf(term)
            for doc, freq in zip(posting, self.freqs[term]):
                norm_len = 1.0 - b + b * (self.lengths[doc] / self.avg_len)
                scores[doc] = scores.get(doc, 0.0) + weight * idf * freq / (freq + k1 * norm_len)
        return sorted(scores.items(), key=lambda kv: -kv[1])[:limit]

    def expanded(self, query: str, cfg, limit: int,
                 assoc) -> list[tuple[int, float]]:
        """Rank with the shopper's words plus their mined catalog neighbours.

        Where `rm3` learns its expansion from the documents THIS query happened
        to retrieve — and so inherits any mistake the first retrieval made —
        this reads a global association table built offline from the catalog's
        own titles. The shopper's terms keep full weight; neighbours are capped
        below them (`assoc_weight`), so an expansion can add a document but not
        outvote the rare word that identifies one.
        """
        original = {t for t in tokens(norm(query))
                    if t not in DIALOGUE_STOP and self.df(t) > 0}
        if not original or assoc is None:
            return self.score(query, cfg, limit)
        weights: dict[str, float] = {t: 1.0 for t in original}
        for term, weight in assoc.expand(original, cfg.assoc_terms,
                                         cfg.assoc_weight).items():
            weights.setdefault(term, weight)
        return self._score_terms(weights, cfg, limit)

    def rm3(self, query: str, cfg, limit: int) -> list[tuple[int, float]]:
        """Rank with an RM3-expanded query: retrieve, learn, retrieve again.

        The documented free-text bottleneck is vocabulary — the shopper says
        "comfy trainers", the catalog says "cushioned athletic sneaker". HyDE
        attacks that by asking a 7B model to write the listing. Relevance
        modelling attacks the same gap with the corpus itself: take the top few
        documents the shopper's own words retrieve, read the vocabulary those
        documents actually use, and search again with it.

        This is the classical baseline HyDE should have to beat. It is offline,
        stdlib, adds no network clause to the submission, and costs one extra
        retrieval instead of a 2-second generation.

        The expansion is *interpolated*, never substituted (`rm3_alpha`), which
        is the same discipline `hyde_bm25_mode="fuse"` arrived at empirically:
        pouring generic vocabulary into one flat query drowns the shopper's own
        rare words, which are the ones that identify the product.
        """
        base = self.score(query, cfg, max(limit, cfg.rm3_fb_docs))
        if not base:
            return base
        original = {t for t in tokens(norm(query)) if t not in DIALOGUE_STOP}
        feedback = base[: cfg.rm3_fb_docs]
        mass = sum(score for _, score in feedback) or 1.0
        max_df = self.n_docs * cfg.max_token_df_ratio

        expansion: dict[str, float] = {}
        for doc, score in feedback:
            share = score / mass                      # P(doc | query), roughly
            counts = Counter(tokens(self._store.text[doc]))
            length = sum(counts.values()) or 1
            for term, freq in counts.items():
                if term in DIALOGUE_STOP or term in original:
                    continue
                df = self.df(term)
                if df <= 0 or df > max_df:
                    continue
                expansion[term] = expansion.get(term, 0.0) + share * (freq / length)

        if not expansion:
            return base[:limit]
        top = sorted(expansion, key=expansion.__getitem__, reverse=True)[: cfg.rm3_fb_terms]
        peak = expansion[top[0]] or 1.0
        weights = {term: (1.0 - cfg.rm3_alpha) for term in original}
        for term in top:
            weights[term] = weights.get(term, 0.0) + cfg.rm3_alpha * expansion[term] / peak
        return self._score_terms(weights, cfg, limit)
