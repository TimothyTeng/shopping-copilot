"""Generate natural-language stress probes from the catalog, offline.

**Why this exists.** `tools/probes.py` holds 26 hand-authored probes, and n=26 is
too small to decide anything. The record is full of results the project could not
act on for exactly that reason — "read the direction, not the third decimal",
"inside the noise band", "tuning on 26 self-authored probes would be fitting the
test". The measurement, not the agent, is the bottleneck: a +0.05 effect cannot
be separated from noise at n=26, so every remaining design question stays open.
This tool trades a fixed offline cost for the resolution to close them.

**It runs offline and ships nothing.** The output is a static JSONL file read by
`tools/stress.py`. No scored path imports this module, and the agent never calls
a model because of it.

**The direction is what makes a probe honest.** The target is chosen FIRST, from
the catalog, and the shopper's words are written afterwards from the product —
never the other way round. That is the same discipline the hand-authored set
follows, and it is the whole reason this suite measures something the official
evaluator cannot: the official simulator lifts its constraints verbatim from the
target's own fields, so query and document share vocabulary by construction.

**Two-sided validity guard.** Every generated probe must land inside a verbatim
overlap band and must not lift a contiguous 4-gram from the product title:

  * too much overlap  -> the probe is quoting the product page. That is the
    official simulator's failure mode, and it would flatter the agent.
  * too little        -> the model wrote about something else, and the probe is
    unanswerable rather than hard.

Rejected samples are counted and reported, never silently dropped.

**Leakage warning.** Generating probes with the same model that `backend="hyde"`
uses at runtime lets the rewriter invert its own vocabulary priors, which would
flatter HyDE specifically. Use a different model for generation than for
retrieval, and the tool warns when they match.

    python -m tools.genprobes --n 500 --model qwen3-coder \\
        --base http://localhost:30801/v1 --out data/probes_generated.jsonl
    python -m tools.stress --probes data/probes_generated.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.backends import client  # noqa: E402
from src.catalog import coarse_category  # noqa: E402
from src.normalize import DIALOGUE_STOP, norm, tokens  # noqa: E402

# Styles mirror the hand-authored tag vocabulary, so a generated set is
# stratified the same way and the two are directly comparable per tag.
STYLES: dict[str, str] = {
    "natural": "Write plainly, the way an ordinary shopper types.",
    "colloquial": (
        "Use informal or regional wording (trainers, jumper, cosy, sneakers, "
        "pumps) rather than the words on the product page."
    ),
    "vague": (
        "Open vaguely, with only a rough need, then narrow over the later turns."
    ),
    "multi_attr": "Pack two or three requirements into a single turn.",
    "non_catalog": (
        "Deliberately describe the product using everyday words that a product "
        "listing would NOT use."
    ),
    "brand": "Mention the brand somewhere, the way shoppers often do.",
    "negation": "State at least one thing they do NOT want.",
}

SYSTEM = (
    "You write realistic shopping-assistant test cases. Given a product, you "
    "write what a shopper would type when looking for it WITHOUT having seen "
    "the product page. Never copy phrases from the listing: use the shopper's "
    "own everyday words. Reply with JSON only: "
    '{"turns": ["...", "...", "..."]}'
)


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
def load_products(path: Path) -> list[dict]:
    """Read the catalog, keeping only products with enough substance to describe.

    A product whose listing is a bare title cannot produce a fair multi-turn
    probe — there is nothing for a shopper to narrow on, so a miss would measure
    the catalog rather than the agent.
    """
    kept: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            product = json.loads(raw)
            features = product.get("features") or []
            details = product.get("details") or {}
            if len(features) < 2 and len(details) < 2:
                continue
            if not (product.get("title") or "").strip():
                continue
            kept.append(product)
    return kept


def stratified(products: list[dict], n: int, seed: int) -> list[dict]:
    """Sample across coarse categories, so the set is not all sneakers.

    The hand-authored set is shoe-heavy by accident of authorship. Round-robin
    over category buckets removes that bias, which matters because the category
    resolver is the component most likely to be flattered by a narrow set.
    """
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for product in products:
        buckets[coarse_category(product.get("categories") or [])].append(product)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    keys = sorted(buckets)
    rng.shuffle(keys)

    chosen: list[dict] = []
    depth = 0
    while len(chosen) < n:
        added = False
        for key in keys:
            if depth < len(buckets[key]):
                chosen.append(buckets[key][depth])
                added = True
                if len(chosen) >= n:
                    break
        if not added:
            break
        depth += 1
    return chosen


# ---------------------------------------------------------------------------
# validity
# ---------------------------------------------------------------------------
def content_words(text: str) -> set[str]:
    return {t for t in tokens(norm(text)) if t not in DIALOGUE_STOP and len(t) > 2}


def overlap_with(turns: list[str], product_text: str) -> float:
    said = content_words(" ".join(turns))
    if not said:
        return 0.0
    return len(said & set(tokens(norm(product_text)))) / len(said)


def lifts_title_ngram(turns: list[str], title: str, n: int = 4) -> bool:
    """Did the model copy a contiguous run of the title verbatim?

    Overlap alone cannot catch this: a probe can sit at a respectable 0.6 while
    still quoting the distinctive part of the title, which is the only part that
    matters for retrieval.
    """
    title_tokens = tokens(norm(title))
    if len(title_tokens) < n:
        return False
    grams = {tuple(title_tokens[i:i + n]) for i in range(len(title_tokens) - n + 1)}
    said = tokens(norm(" ".join(turns)))
    return any(tuple(said[i:i + n]) in grams for i in range(max(0, len(said) - n + 1)))


NEGATION_CUES = ("not ", "no ", "don't", "dont", "without", "rather than",
                 "instead of", "avoid", "nothing ", "isn't", "aren't", "never")


def restates(turns: list[str], threshold: float = 0.7) -> bool:
    """Do two turns say the same thing?

    Observed failure: "men's tshirt" / "black men's tshirt" / "cotton men's
    tshirt". Each turn restates its predecessor instead of adding a requirement,
    so the probe measures nothing about multi-turn narrowing — it is a one-turn
    probe padded to four.
    """
    sets = [content_words(t) for t in turns]
    for i, left in enumerate(sets):
        for right in sets[i + 1:]:
            if not left or not right:
                continue
            union = len(left | right)
            if union and len(left & right) / union > threshold:
                return True
    return False


def parse_turns(text: str) -> list[str] | None:
    """Pull the turn list out of a model response. Tolerant, but not credulous."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start:end + 1])
    except Exception:
        return None
    turns = payload.get("turns")
    if not isinstance(turns, list):
        return None
    cleaned = [" ".join(str(t).split()) for t in turns if str(t).strip()]
    return cleaned or None


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
def describe(product: dict, limit: int = 900) -> str:
    parts = [f"Title: {product.get('title', '')}"]
    features = product.get("features") or []
    if features:
        parts.append("Features: " + " | ".join(str(f) for f in features[:6]))
    details = product.get("details") or {}
    if isinstance(details, dict) and details:
        parts.append("Details: " + " | ".join(
            f"{k}: {v}" for k, v in list(details.items())[:8]))
    categories = product.get("categories") or []
    if categories:
        parts.append("Category: " + " > ".join(str(c) for c in categories[-3:]))
    return "\n".join(parts)[:limit]


def make_probe(args, product: dict, style: str, index: int) -> dict:
    prompt = (
        f"{describe(product)}\n\n"
        f"Style: {STYLES[style]}\n"
        f"Write {args.turns} turns a shopper would type across a conversation, "
        f"looking for this product without having seen the page above. Each turn "
        f"is one short sentence or phrase. The first turn is the opening request; "
        f"later turns add one more requirement each. JSON only."
    )
    text, prompt_tokens, completion_tokens = client.chat(
        args.base.rstrip("/") + "/chat/completions",
        args.model,
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": prompt}],
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        key=os.environ.get("COPILOT_LLM_KEY"),
        # A fixed seed per sample keeps the whole set reproducible while still
        # varying between samples.
        seed=index,
        json_object=True,
    )
    turns = parse_turns(text)
    if turns is None:
        return {"ok": False, "reason": "unparseable"}
    if not (2 <= len(turns) <= 6):
        return {"ok": False, "reason": "turn_count"}
    if any(not (2 <= len(t.split()) <= 30) for t in turns):
        return {"ok": False, "reason": "turn_length"}
    if restates(turns):
        return {"ok": False, "reason": "restates"}
    # The style label is a claim about the probe, and a mislabelled probe
    # corrupts the per-tag table that the label exists to produce. Only checked
    # where the style has an unambiguous lexical signature.
    if style == "negation":
        joined = " ".join(turns).lower()
        if not any(cue in joined for cue in NEGATION_CUES):
            return {"ok": False, "reason": "no_negation"}

    product_text = " ".join([
        str(product.get("title", "")),
        " ".join(str(f) for f in (product.get("features") or [])),
        " ".join(f"{k} {v}" for k, v in (product.get("details") or {}).items()),
        str(product.get("description") or ""),
        " ".join(str(c) for c in (product.get("categories") or [])),
    ])
    ratio = overlap_with(turns, product_text)
    if ratio > args.max_overlap:
        return {"ok": False, "reason": "copied"}
    if ratio < args.min_overlap:
        return {"ok": False, "reason": "off_target"}
    if lifts_title_ngram(turns, str(product.get("title", ""))):
        return {"ok": False, "reason": "title_ngram"}

    return {
        "ok": True,
        "probe": {
            "id": f"gen_{index:04d}_{style}",
            "target": product["parent_asin"],
            "turns": turns,
            "tags": [style],
            "overlap": round(ratio, 3),
        },
        "tokens": prompt_tokens + completion_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stress probes offline")
    parser.add_argument("--n", type=int, default=200, help="probes to attempt")
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--base", default=os.environ.get(
        "COPILOT_LLM_BASE", client.DEFAULT_BASE))
    parser.add_argument("--model", default=os.environ.get(
        "COPILOT_LLM_MODEL", client.DEFAULT_MODEL))
    parser.add_argument("--out", default="data/probes_generated.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--min-overlap", type=float, default=0.20,
                        help="below this the probe describes something else")
    # The hand-authored set averages 0.68 verbatim overlap. A generated probe
    # allowed past 0.75 is quoting the listing, which is the official
    # simulator's failure mode and would flatter the agent.
    parser.add_argument("--max-overlap", type=float, default=0.75,
                        help="above this the probe is quoting the product page")
    args = parser.parse_args()

    runtime_model = os.environ.get("COPILOT_LLM_MODEL", client.DEFAULT_MODEL)
    if args.model == runtime_model:
        print(f"  WARNING: generating with {args.model!r}, the same model the "
              f"runtime backend uses.\n  A HyDE rewriter can invert its own "
              f"vocabulary priors, which flatters it. Prefer a different model.")

    print(f"  loading catalog…")
    products = load_products(config.CATALOG_PATH)
    print(f"  {len(products):,} products with enough text to describe")
    sample = stratified(products, args.n, args.seed)
    print(f"  sampled {len(sample):,} across "
          f"{len({coarse_category(p.get('categories') or []) for p in sample}):,} "
          f"categories\n  generating with {args.model} at {args.base}…")

    styles = sorted(STYLES)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(
            lambda pair: _safe(args, pair[1], styles[pair[0] % len(styles)], pair[0]),
            enumerate(sample),
        ))
    elapsed = time.perf_counter() - started

    kept = [r["probe"] for r in results if r.get("ok")]
    rejected = Counter(r["reason"] for r in results if not r.get("ok"))
    tokens_used = sum(r.get("tokens", 0) for r in results if r.get("ok"))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "_meta": {
                "generated_by": "tools/genprobes.py",
                "model": args.model,
                "base": args.base,
                "seed": args.seed,
                "turns": args.turns,
                "attempted": len(sample),
                "kept": len(kept),
                "min_overlap": args.min_overlap,
                "max_overlap": args.max_overlap,
                "warning": ("Generated by an LLM. Do not tune the retrieval "
                            "model against a set written by the same model."),
            }
        }) + "\n")
        for probe in kept:
            handle.write(json.dumps(probe) + "\n")

    mean_overlap = sum(p["overlap"] for p in kept) / len(kept) if kept else 0.0
    print(f"\n  kept {len(kept)}/{len(sample)}   mean overlap {mean_overlap:.2f}   "
          f"{elapsed:.0f}s   {tokens_used:,} tokens")
    if rejected:
        print("  rejected: " + "  ".join(f"{reason}={count}"
                                         for reason, count in rejected.most_common()))
    print(f"  written {out.relative_to(ROOT)}")
    print(f"\n  python -m tools.stress --probes {args.out}")


def _safe(args, product: dict, style: str, index: int) -> dict:
    """A single failed sample must not abandon the batch."""
    try:
        return make_probe(args, product, style, index)
    except Exception as error:
        return {"ok": False, "reason": type(error).__name__}


if __name__ == "__main__":
    main()
