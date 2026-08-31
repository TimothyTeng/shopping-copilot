"""doc2query: expand the CATALOG offline, instead of the query at runtime.

HyDE translates shopper words into catalog words at request time, and pays for
it — a 2-second generation on the scored path, and a network dependency the
submission rules warn may not exist (`submission_rules.md:59`). The translation
can run in the other direction and at build time instead: ask the model what a
shopper would *type* to find each product, store those queries as an extra
field, and index them.

Same vocabulary bridge, opposite direction, and the runtime stays pure stdlib
with no socket open — the property `backend="hyde"` can never have.

    python -m tools.doc2query --limit 200            # pilot, measures throughput
    python -m tools.doc2query                        # the whole catalog
    python -m tools.doc2query --resume               # continue an interrupted run

Output is JSONL, one `{"asin": ..., "queries": [...]}` per product, appended as
it goes so an interrupted run loses nothing. `--resume` skips what is already
there, reading the shipped `data/doc2query.jsonl.gz` as well as the raw file, so
a repacked tree resumes rather than regenerating. Nothing in `src/` reads
this file unless `doc2query_expansions` is on.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.backends import client  # noqa: E402
from src.backends.hyde import resolve_model  # noqa: E402

OUT_PATH = ROOT / "data" / "doc2query.jsonl"

SYSTEM = (
    "You predict what a shopper types into a search box. Given an Amazon "
    "clothing listing, write the three different searches a real person would "
    "type to end up at THIS product. Use everyday shopper words, not the "
    "listing's marketing words - if the listing says 'cushioned footbed "
    "athletic trainer', a shopper might type 'comfy gym shoes'. One search per "
    "line, three lines, no numbering, no explanation."
)


def prompt_for(product: dict) -> str:
    title = str(product.get("title") or "")[:200]
    features = [str(f) for f in (product.get("features") or [])][:3]
    category = " > ".join(str(c) for c in (product.get("categories") or [])[-3:])
    body = "\n".join(f"- {f[:120]}" for f in features)
    return f"Category: {category}\nTitle: {title}\n{body}"


def load_done(path: Path) -> set[str]:
    """What is already expanded, across both the raw file and the shipped pack.

    Only the `.jsonl.gz` is kept in the tree — the plaintext it was packed from
    is 6.4 MB of the same bytes. Reading both means `--resume` still means
    "finish what is missing" after a repack, instead of silently regenerating
    all 50,000 products.
    """
    done: set[str] = set()
    for source in (path, path.with_suffix(path.suffix + ".gz")):
        if not source.exists():
            continue
        opener = gzip.open if source.suffix == ".gz" else open
        with opener(source, "rt", encoding="utf-8") as handle:
            for line in handle:
                try:
                    done.add(json.loads(line)["asin"])
                except Exception:
                    continue
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="0 = whole catalog")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=60)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--base", default="http://localhost:30800/v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    model = args.model or resolve_model(args.base)
    url = args.base.rstrip("/") + "/chat/completions"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out) if args.resume else set()
    print(f"model {model}\n  {len(done):,} products already expanded")

    products: list[dict] = []
    with Path(config.CATALOG_PATH).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            if str(product["parent_asin"]) in done:
                continue
            products.append(product)
            if args.limit and len(products) >= args.limit:
                break
    print(f"  {len(products):,} to generate")

    lock = threading.Lock()
    state = {"ok": 0, "fail": 0, "tokens": 0}
    started = time.perf_counter()
    handle = out.open("a", encoding="utf-8")

    def work(product: dict) -> None:
        try:
            text, _pt, ct = client.chat(
                url, model,
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": prompt_for(product)}],
                max_tokens=args.max_tokens, timeout=args.timeout,
            )
        except Exception:
            with lock:
                state["fail"] += 1
            return
        queries = [line.strip(" -*0123456789.") for line in text.splitlines()]
        queries = [q for q in queries if len(q) > 2][:3]
        if not queries:
            with lock:
                state["fail"] += 1
            return
        row = json.dumps({"asin": str(product["parent_asin"]), "queries": queries})
        with lock:
            handle.write(row + "\n")
            state["ok"] += 1
            state["tokens"] += ct
            total = state["ok"] + state["fail"]
            if total % 200 == 0:
                rate = total / (time.perf_counter() - started)
                remaining = (len(products) - total) / rate if rate else 0
                handle.flush()
                print(f"  {total:,}/{len(products):,}  {rate:.1f}/s  "
                      f"{state['fail']} failed  eta {remaining / 60:.0f} min",
                      flush=True)

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(work, products))
    finally:
        handle.close()

    elapsed = time.perf_counter() - started
    rate = (state["ok"] + state["fail"]) / elapsed if elapsed else 0
    print(f"\n  {state['ok']:,} expanded, {state['fail']:,} failed in "
          f"{elapsed / 60:.1f} min ({rate:.1f}/s, {state['tokens']:,} completion tokens)")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
