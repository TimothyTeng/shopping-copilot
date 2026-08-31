"""Mine a term-association table from the catalog, offline.

The documented free-text bottleneck is vocabulary: the shopper says "cosy
slippers", the catalog says "plush indoor house shoes". `src/fuzzy.py` already
closes the *spelling* distance between a shopper token and a catalog token. This
closes the *semantic* distance, with the same shape of solution — mined offline,
shipped as a static table, read at runtime by pure stdlib code, exactly as
`category_clf` is.

Method: positive pointwise mutual information over term co-occurrence inside
product titles.

    PPMI(x, y) = max(0, log( P(x, y) / (P(x) P(y)) ))

Titles rather than whole documents, deliberately. A full listing co-occurs its
own boilerplate ("machine wash", "imported") with everything, so document-level
co-occurrence learns the corpus's filler rather than its synonymy. A title is
the compressed name of one product, so two terms that share titles are usually
two names for the same thing.

    python -m tools.train_assoc            # writes src/models/assoc.json.gz
    python -m tools.train_assoc --probe cosy trainers waterproof

Nothing here runs on the scored path; this is a build step.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.catalog import CatalogStore  # noqa: E402
from src.normalize import DIALOGUE_STOP, tokens  # noqa: E402

MODEL_PATH = ROOT / "src" / "models" / "assoc.json.gz"


def build(store: CatalogStore, min_df: int, max_df_ratio: float,
          neighbours: int, min_pair: int) -> dict[str, list[list]]:
    n_docs = len(store)
    max_df = n_docs * max_df_ratio

    unigram: Counter[str] = Counter()
    title_tokens: list[list[str]] = []
    for title in store.title:
        terms = sorted({t for t in tokens(title) if t not in DIALOGUE_STOP and len(t) > 2})
        title_tokens.append(terms)
        unigram.update(terms)

    # Only mine terms that are common enough to have a stable neighbourhood and
    # rare enough to carry meaning — the same band `informative()` uses.
    vocab = {t for t, c in unigram.items() if min_df <= c <= max_df}
    print(f"  vocabulary {len(vocab):,} of {len(unigram):,} title terms")

    pairs: dict[str, Counter[str]] = defaultdict(Counter)
    for terms in title_tokens:
        kept = [t for t in terms if t in vocab]
        if len(kept) < 2 or len(kept) > 40:
            continue
        for i, left in enumerate(kept):
            for right in kept[i + 1:]:
                pairs[left][right] += 1
                pairs[right][left] += 1

    total = sum(unigram[t] for t in vocab) or 1
    table: dict[str, list[list]] = {}
    for term, counts in pairs.items():
        p_term = unigram[term] / total
        scored: list[tuple[float, str]] = []
        for other, count in counts.items():
            if count < min_pair:
                continue
            joint = count / total
            ppmi = math.log(joint / (p_term * (unigram[other] / total)))
            if ppmi > 0.0:
                scored.append((ppmi, other))
        if not scored:
            continue
        scored.sort(reverse=True)
        table[term] = [[other, round(value, 3)] for value, other in scored[:neighbours]]
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-df", type=int, default=20,
                        help="a term needs this many titles to be mined")
    parser.add_argument("--max-df-ratio", type=float, default=0.05,
                        help="terms above this share of titles are filler")
    parser.add_argument("--neighbours", type=int, default=12)
    parser.add_argument("--min-pair", type=int, default=8,
                        help="co-occurrences needed before a pair is trusted")
    parser.add_argument("--probe", nargs="*", default=[],
                        help="print the neighbours of these terms and exit")
    parser.add_argument("--out", default=str(MODEL_PATH))
    args = parser.parse_args()

    started = time.perf_counter()
    print("loading catalog…")
    store = CatalogStore.load(config.CATALOG_PATH)
    print(f"  {len(store):,} products in {time.perf_counter() - started:.1f}s")

    print("mining associations…")
    started = time.perf_counter()
    table = build(store, args.min_df, args.max_df_ratio,
                  args.neighbours, args.min_pair)
    print(f"  {len(table):,} terms in {time.perf_counter() - started:.1f}s")

    if args.probe:
        for term in args.probe:
            row = table.get(term)
            print(f"  {term:>14} -> " +
                  (", ".join(f"{o}({v})" for o, v in row) if row else "(no entry)"))
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as handle:
        json.dump(table, handle)
    print(f"  wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
