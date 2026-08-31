"""Ceiling study: is dense retrieval worth anything on the PROSE path?

`tools/exp_vector.py` already answered this for the graded path and answered it
no — catalog-trained LSA rescued 0 of 200 sessions. But that experiment queried
with the *benchmark's* constraints, which the simulator lifts verbatim from the
target, so lexical matching is near-optimal there by construction. The negative
was sound for that surface and was then quoted as a general one.

This runs the same question on the surface where the project's own oracle says
the bottleneck lives: free text, `data/probes_generated.jsonl`, where the
shopper's words are their own and the "product's own words" oracle scores 0.999
against the agent's 0.73.

Retrieval power only — no dialogue, no gate, no category. Every system gets the
identical query (the whole transcript) and returns 10 products, so the numbers
are comparable to each other and to nothing else.

    python -m tools.exp_dense
    python -m tools.exp_dense --model sentence-transformers/all-MiniLM-L6-v2

Needs numpy + sentence-transformers. This is an experiment, never a runtime
path: nothing in `src/` imports it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src import config  # noqa: E402
from src.bm25 import Bm25Index  # noqa: E402
from src.catalog import CatalogStore  # noqa: E402

TOP_K = 10


def load_probes(path: Path) -> list[dict]:
    rows = []
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        if "target" in row and row.get("turns"):
            rows.append(row)
    return rows


def metrics(ranks: list[int | None]) -> dict:
    hit = sum(1 for r in ranks if r is not None) / len(ranks)
    mrr = sum((1.0 / r) if r else 0.0 for r in ranks) / len(ranks)
    return {"hit": hit, "mrr": mrr}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--probes", default="data/probes_generated.jsonl")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--title-only", action="store_true",
                        help="encode titles instead of the full flattened text")
    args = parser.parse_args()

    print("loading catalog…")
    store = CatalogStore.load(config.CATALOG_PATH)
    bm25 = Bm25Index(store)
    probes = load_probes(ROOT / args.probes)
    print(f"  {len(store):,} products, {len(probes)} probes")

    from sentence_transformers import SentenceTransformer

    print(f"encoding with {args.model}…")
    model = SentenceTransformer(args.model)
    started = time.perf_counter()
    # Dense encoders truncate hard (256 word-pieces here), and a flattened Amazon
    # listing is far longer than that — so the tail of a long description is
    # invisible to the encoder either way. Titles are the honest comparison.
    texts = ([store.raw_title[d] for d in range(len(store))] if args.title_only
             else [store.raw_title[d] + ". " + store.text[d][:800]
                   for d in range(len(store))])
    doc_vecs = model.encode(texts, batch_size=args.batch, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=True)
    print(f"  {doc_vecs.shape} in {time.perf_counter() - started:.1f}s "
          f"({doc_vecs.nbytes / 1e6:.0f} MB)")

    queries = [" ".join(p["turns"]) for p in probes]
    query_vecs = model.encode(queries, batch_size=args.batch, convert_to_numpy=True,
                              normalize_embeddings=True)

    dense_ranks: list[int | None] = []
    lexical_ranks: list[int | None] = []
    fused_ranks: list[int | None] = []
    started = time.perf_counter()
    for probe, qv in zip(probes, query_vecs):
        target = store.ord_of.get(probe["target"])
        if target is None:
            continue
        sims = doc_vecs @ qv
        dense = np.argpartition(-sims, TOP_K)[:TOP_K]
        dense = dense[np.argsort(-sims[dense])].tolist()
        lexical = [d for d, _ in bm25.score(" ".join(probe["turns"]),
                                            config.DEFAULT, 200)]

        def rank_of(ordered: list[int]) -> int | None:
            return ordered.index(target) + 1 if target in ordered[:TOP_K] else None

        dense_ranks.append(rank_of(dense))
        lexical_ranks.append(rank_of(lexical))
        # Reciprocal-rank fusion, the same instrument the agent already uses to
        # combine two rankings on incomparable scales.
        points: dict[int, float] = {}
        for ranking in (dense, lexical[:200]):
            for position, doc in enumerate(ranking):
                points[doc] = points.get(doc, 0.0) + 1.0 / (60.0 + position + 1)
        fused_ranks.append(rank_of(sorted(points, key=points.__getitem__, reverse=True)))
    print(f"  retrieval in {time.perf_counter() - started:.1f}s")

    print("\nRETRIEVAL ONLY, whole transcript as the query, top 10")
    for name, ranks in (("dense (bi-encoder)", dense_ranks),
                        ("lexical BM25", lexical_ranks),
                        ("RRF of both", fused_ranks)):
        m = metrics(ranks)
        print(f"  {name:<20} hit@10 {m['hit']:.3f}   mrr {m['mrr']:.3f}")

    rescued = sum(1 for d, l in zip(dense_ranks, lexical_ranks) if d and not l)
    lost = sum(1 for d, l in zip(dense_ranks, lexical_ranks) if l and not d)
    print(f"\n  probes dense finds that lexical misses: {rescued}")
    print(f"  probes lexical finds that dense misses: {lost}")


if __name__ == "__main__":
    main()
