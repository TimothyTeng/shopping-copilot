"""Runtime side of dense retrieval: a semantic second opinion on the prose path.

The project's earlier verdict — "semantic retrieval buys nothing here", from
`tools/exp_vector.py`'s 0-of-200 LSA result — was measured on the GRADED path,
where the simulator quotes the target's own words back at us and lexical
matching is near-optimal by construction. `tools/exp_dense.py` re-asked it on
free text and got the opposite answer: dense alone loses to BM25 (hit@10 0.618
vs 0.691), dense FUSED with BM25 beats it (0.735), and the two disagree in both
directions — 42 probes dense finds that lexical misses, 73 the other way. Two
retrievers that fail on different documents are worth fusing; two that fail on
the same ones are not.

So this is deliberately not a replacement for `Bm25Index`. It produces one more
ranking for `Ranker._fuse`, on the same footing as the doc2query index — RRF
reads positions only, so a semantic ranking on a cosine scale and a lexical one
on a BM25 scale never need to be normalized against each other.

Two properties bound the risk:

* **Positional alignment is verified, not assumed.** Vectors are addressed by
  doc ordinal, so a file built against a different catalog would score
  confidently against the wrong products, silently. `fingerprint()` refuses.
* **It is not on the scored path and cannot be.** Embedding the query needs the
  encoder in-process — torch, and a model load. That is a dependency the graded
  agent does not have and must never acquire (see submission_rules.md: scoring
  may run network-disabled, and the stdlib guarantee is what makes that safe).
  `dense_weight` is 0.0 in Settings; `demo chat` is where this belongs.

Absent or mismatched files mean the feature is simply off, exactly like
doc2query — never an exception on a live turn.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .catalog import CatalogStore
from .normalize import norm

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
VECTORS_PATH = DATA_DIR / "dense.npy"
META_PATH = DATA_DIR / "dense.json"


def fingerprint(store: CatalogStore) -> str:
    """Identify the catalog these vectors were built against.

    Cheap and positional: the length plus the first, middle and last ids. Doc
    ordinals are what the vector file is indexed by, so what has to match is the
    ORDER, and a different order shows up in the sampled ids.
    """
    n = len(store)
    sample = "|".join(store.ids[i] for i in (0, n // 2, n - 1)) if n else ""
    return hashlib.sha256(f"{n}\x00{sample}".encode("utf-8")).hexdigest()[:16]


class DenseIndex:
    """Cosine similarity against pre-encoded product vectors."""

    __slots__ = ("vectors", "model", "meta", "_encoder")

    def __init__(self, vectors, meta: dict) -> None:
        self.vectors = vectors
        self.meta = meta
        self.model = str(meta.get("model", ""))
        self._encoder = None

    @classmethod
    def try_load(cls, store: CatalogStore,
                 vectors_path: Path = VECTORS_PATH,
                 meta_path: Path = META_PATH) -> "DenseIndex | None":
        """Load the vectors, or return None and leave the agent untouched.

        Every failure mode is the same failure mode: the feature is off. Missing
        numpy, missing files, a catalog that has moved on since the build — none
        of them may raise into a turn.
        """
        try:
            import numpy as np
        except Exception:
            return None
        if not vectors_path.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            # mmap: 77 MB that the OS can evict, rather than 77 MB of heap in a
            # process whose whole point is a 36 ms turn.
            vectors = np.load(vectors_path, mmap_mode="r")
        except Exception:
            return None
        if len(vectors) != len(store) or meta.get("fingerprint") != fingerprint(store):
            # Loud, because this one is a bug rather than an absent feature: the
            # artefact exists and does not match, and scoring anyway would be
            # confidently wrong.
            print(f"[dense] vectors do not match the catalog "
                  f"({len(vectors)} vs {len(store)}); dense retrieval is off")
            return None
        return cls(vectors, meta)

    # -- encoder -----------------------------------------------------------
    def _encode(self, text: str):
        """Embed one query. Loads the encoder on first use, never at import."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.model)
        return self._encoder.encode([text], convert_to_numpy=True,
                                    normalize_embeddings=True)[0]

    def score(self, query: str, cfg, limit: int) -> list[tuple[int, float]]:
        """Top `limit` products by cosine similarity to `query`.

        Returns [] on any failure — a missing encoder, a model that will not
        load, a query that embeds to nothing — so a dense outage degrades to the
        lexical ranking rather than to an error.
        """
        query = norm(query)
        if not query:
            return []
        try:
            import numpy as np
            vector = self._encode(query)
            sims = self.vectors @ vector
            limit = min(limit, len(sims))
            top = np.argpartition(-sims, limit - 1)[:limit]
            top = top[np.argsort(-sims[top])]
        except Exception:
            return []
        return [(int(doc), float(sims[doc])) for doc in top]
