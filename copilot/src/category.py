"""Category resolution.

The opening message contains the target's coarse category verbatim, and those
buckets hold a median of ~8 products out of 50,000 — the single strongest signal
available. Resolution works by finding a known bucket name *inside* the message,
so it survives any rephrasing of the sentence wrapped around it.
"""
from __future__ import annotations

from array import array
from collections import defaultdict

from .catalog import CatalogStore, coarse_category
from .normalize import norm, padded, tokens


class CategoryIndex:
    """Coarse-category buckets, plus ancestor prefixes for widening."""

    __slots__ = ("buckets", "_keys_by_len", "_key_tokens", "prefixes",
                 "_doc_prefix", "doc_key")

    def __init__(self, store: CatalogStore) -> None:
        buckets: dict[str, list[int]] = defaultdict(list)
        prefixes: dict[tuple[str, ...], list[int]] = defaultdict(list)
        doc_prefix: list[tuple[str, ...]] = []
        doc_key: list[str] = []

        for doc, segments in enumerate(store.cat_path):
            key = norm(coarse_category(segments))
            buckets[key].append(doc)
            doc_key.append(key)
            path = tuple(norm(s) for s in segments if s)
            doc_prefix.append(path)
            for depth in range(1, len(path) + 1):
                prefixes[path[:depth]].append(doc)

        self.doc_key = doc_key                 # doc -> its coarse bucket key
        self.buckets: dict[str, array] = {k: array("i", v) for k, v in buckets.items()}
        self.prefixes: dict[tuple[str, ...], array] = {
            k: array("i", v) for k, v in prefixes.items()
        }
        self._doc_prefix = doc_prefix
        # Longest keys first so "jewelry necklaces" wins over "necklaces".
        self._keys_by_len = sorted(self.buckets, key=len, reverse=True)
        self._key_tokens = {k: frozenset(tokens(k)) for k in self.buckets}

    def resolve(self, message: str, cfg=None, bm25=None, clf=None) -> tuple[str | None, array, float]:
        """Find the best category bucket for the message.

        Returns (key, docs, confidence). Confidence is 1.0 for an exact bucket
        name found in the message, and in [0, 1) for either fallback — the
        caller can scale the category bonus by it so a shaky bucket does not
        rank as hard as a certain one.
        """
        text = padded(norm(message))
        for key in self._keys_by_len:
            if key and padded(key) in text:
                return key, self.buckets[key], 1.0

        mode = getattr(cfg, "category_resolver", "overlap") if cfg else "overlap"
        if mode == "ensemble" and clf is not None and bm25 is not None:
            return self._resolve_ensemble(message, cfg, bm25, clf)
        if mode == "classifier" and clf is not None:
            key, conf = clf.predict(message)
            if (key is not None and key in self.buckets
                    and conf >= getattr(cfg, "category_min_confidence", 0.0)):
                return key, self.buckets[key], conf
            # Weak or unknown prediction: fall back to the vote resolver.
            if bm25 is not None:
                return self._resolve_by_vote(message, cfg, bm25)
            return self._resolve_by_overlap(text)
        if mode == "vote" and bm25 is not None:
            return self._resolve_by_vote(message, cfg, bm25)
        return self._resolve_by_overlap(text)

    def _resolve_by_overlap(self, text: str) -> tuple[str | None, array, float]:
        """Legacy fallback: token overlap over bucket labels."""
        msg_tokens = set(tokens(text))
        best_key, best_score = None, 0.0
        for key, key_tokens in self._key_tokens.items():
            if not key_tokens:
                continue
            overlap = len(key_tokens & msg_tokens)
            if not overlap:
                continue
            score = overlap / len(key_tokens)
            if score > best_score:
                best_key, best_score = key, score
        if best_key is not None and best_score >= 0.5:
            return best_key, self.buckets[best_key], best_score
        return None, array("i"), 0.0

    def _resolve_ensemble(self, message: str, cfg, bm25, clf) -> tuple[str | None, array, float]:
        """Union the classifier's bucket with the vote's bucket.

        The two resolvers err on different products: the classifier lifts recall
        (it finds the right coarse bucket on more paraphrases) but occasionally
        lands in a *sibling* sub-bucket, demoting a target the vote had placed at
        rank 1. Taking the union keeps a target in-category when EITHER resolver
        is right, recovering the vote's ranking wins without giving up the
        classifier's extra hits. `key` follows the more confident resolver, only
        to suppress its tokens in extraction; the union is what the ranker reads.
        """
        ckey, cconf = clf.predict(message)
        cdocs = self.buckets.get(ckey, array("i")) if ckey else array("i")
        vkey, vdocs, vconf = self._resolve_by_vote(message, cfg, bm25)
        merged = array("i", sorted(set(cdocs) | set(vdocs)))
        if not len(merged):
            return None, array("i"), 0.0
        key = ckey if cconf >= vconf else vkey
        return key, merged, max(cconf, vconf)

    def _resolve_by_vote(self, message: str, cfg, bm25) -> tuple[str | None, array, float]:
        """Fallback: the majority coarse category of the products the message
        retrieves, weighted by BM25 match mass.

        Matching against whole product text rather than the bucket label is what
        makes this survive renaming: "men's dive watch" retrieves dive watches,
        whose bucket wins on the products themselves, not on sharing the token
        "men" with `men hoodies`. Confidence is the winner's share of the total
        retrieved mass, so a message that pulls one clean category scores high
        and one smeared across many scores low.
        """
        ranked = bm25.score(message, cfg, cfg.category_vote_top)
        if not ranked:
            return None, array("i"), 0.0
        mass: dict[str, float] = {}
        for doc, score in ranked:
            key = self.doc_key[doc]
            mass[key] = mass.get(key, 0.0) + score
        total = sum(mass.values())
        if total <= 0.0:
            return None, array("i"), 0.0
        best_key = max(mass, key=mass.__getitem__)
        confidence = mass[best_key] / total
        if confidence < getattr(cfg, "category_min_confidence", 0.0):
            return None, array("i"), 0.0
        return best_key, self.buckets[best_key], confidence

    def widen(self, docs: array, min_docs: int) -> array:
        """Climb to an ancestor category until the pool is big enough."""
        if len(docs) >= min_docs or not len(docs):
            return docs
        path = self._doc_prefix[docs[0]]
        for depth in range(len(path) - 1, 0, -1):
            wider = self.prefixes.get(path[:depth])
            if wider is not None and len(wider) >= min_docs:
                return wider
        return docs
