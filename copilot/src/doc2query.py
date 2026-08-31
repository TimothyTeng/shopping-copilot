"""Runtime side of doc2query: a second index over generated shopper queries.

`tools/doc2query.py` asks a model, offline, what a shopper would *type* to find
each product, and stores those queries. This loads them into a BM25 index of
their own, so a turn can be scored twice: once against what the catalog says
about itself, and once against how a shopper would ask for it.

Why a separate index rather than appending the queries to the product text: a
generated line is evidence of a different kind and on a different scale, and
`hyde_bm25_mode="union"` already measured what happens when you pour one
vocabulary into another — generic words outvote the shopper's own. Two rankings
fused by RRF keep the union of both candidate sets and read positions only.

The whole cost is paid at build time. At scoring time this is a dict, an array
and stdlib arithmetic — no socket, no model, nothing for a network-disabled
environment to fail on, which is the property the HyDE backend can never have.
"""
from __future__ import annotations

import gzip
import json
from array import array
from collections import Counter
from pathlib import Path

from .catalog import CatalogStore
from .normalize import norm, tokens

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "doc2query.jsonl"


class Doc2QueryIndex:
    """BM25 over the generated queries, addressed by the same doc ordinals."""

    __slots__ = ("docs", "freqs", "lengths", "n_docs", "avg_len", "_idf", "covered")

    def __init__(self, store: CatalogStore, expansions: dict[str, list[str]]) -> None:
        self.n_docs = len(store)
        docs: dict[str, array] = {}
        freqs: dict[str, array] = {}
        lengths = array("i", [0]) * self.n_docs
        covered = 0
        for asin, queries in expansions.items():
            doc = store.ord_of.get(asin)
            if doc is None:
                continue
            counts = Counter(tokens(norm(" ".join(queries))))
            if not counts:
                continue
            covered += 1
            lengths[doc] = sum(counts.values())
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
        self.covered = covered
        filled = [n for n in lengths if n]
        self.avg_len = (sum(filled) / len(filled)) if filled else 1.0
        self._idf: dict[str, float] = {}

    @classmethod
    def try_load(cls, store: CatalogStore,
                 path: Path = DATA_PATH) -> "Doc2QueryIndex | None":
        """Absent or unreadable file means the feature is simply off.

        Prefers the shipped `.jsonl.gz` (6.6 MB of generations compress to 1.9)
        and falls back to the raw `.jsonl` the generator appends to, so a run in
        progress is readable without a repack.
        """
        path = Path(path)
        packed = path.with_suffix(path.suffix + ".gz")
        if packed.exists():
            path = packed
        opener = gzip.open if path.suffix == ".gz" else open
        try:
            expansions: dict[str, list[str]] = {}
            with opener(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    queries = row.get("queries") or []
                    if queries:
                        expansions[str(row["asin"])] = [str(q) for q in queries]
        except Exception:
            return None
        if not expansions:
            return None
        return cls(store, expansions)

    def idf(self, term: str) -> float:
        cached = self._idf.get(term)
        if cached is None:
            import math
            df = len(self.docs.get(term, ()))
            cached = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
            self._idf[term] = cached
        return cached

    def score(self, query: str, cfg, limit: int) -> list[tuple[int, float]]:
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
        return sorted(scores.items(), key=lambda kv: -kv[1])[:limit]
