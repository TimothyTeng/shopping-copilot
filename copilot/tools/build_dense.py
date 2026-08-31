"""Encode the catalog once, so a turn can be scored semantically at runtime.

`tools/exp_dense.py` measured that a dense ranking, FUSED with the lexical one,
beats lexical alone on the prose set (+0.044 hit@10) while losing to it alone.
That is a ceiling study on a stateless control; this builds the artefact the
agent needs to find out what the signal is worth inside the real pipeline.

Output is two files beside the catalog:

    data/dense.npy    float32 [n_products, dim], L2-normalized, doc-ordinal order
    data/dense.json   the model name, dim, text mode and the catalog fingerprint

The fingerprint is what makes this safe: `src/dense.py` refuses to load vectors
built against a different catalog, because doc ordinals are positional and a
silent misalignment would score confidently against the wrong products.

    python -m tools.build_dense                    # title + 800 chars of text
    python -m tools.build_dense --title-only

Needs numpy + sentence-transformers. Build time only — but note that the
RUNTIME still needs the encoder to embed the query, which is why the feature is
off by default and lives on the prose surface, never the scored path.
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
from src.catalog import CatalogStore  # noqa: E402
from src.dense import VECTORS_PATH, META_PATH, fingerprint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--title-only", action="store_true",
                        help="encode titles instead of title + leading text")
    args = parser.parse_args()

    print("loading catalog…")
    store = CatalogStore.load(config.CATALOG_PATH)
    print(f"  {len(store):,} products")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)
    # The encoder truncates at 256 word-pieces; a flattened Amazon listing is far
    # longer, so the tail is invisible either way. Cutting at 800 characters is
    # explicit about that rather than letting the tokenizer decide silently.
    texts = ([store.raw_title[d] for d in range(len(store))] if args.title_only
             else [store.raw_title[d] + ". " + store.text[d][:800]
                   for d in range(len(store))])

    started = time.perf_counter()
    vectors = model.encode(texts, batch_size=args.batch, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=True)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    elapsed = time.perf_counter() - started
    print(f"  {vectors.shape} in {elapsed:.1f}s ({vectors.nbytes / 1e6:.0f} MB)")

    np.save(VECTORS_PATH, vectors)
    META_PATH.write_text(json.dumps({
        "model": args.model,
        "dim": int(vectors.shape[1]),
        "count": int(vectors.shape[0]),
        "text": "title" if args.title_only else "title+800",
        "fingerprint": fingerprint(store),
    }, indent=2), encoding="utf-8")
    print(f"  written {VECTORS_PATH.name} and {META_PATH.name}")


if __name__ == "__main__":
    main()
