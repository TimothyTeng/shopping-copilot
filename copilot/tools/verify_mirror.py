"""Verify that our mirrors of the evaluator still agree with the evaluator.

Three functions in `src/catalog.py` are *copies* of evaluator logic, not
abstractions over it:

    searchable_text   -> local_evaluator.searchable_text   (phrase grounding)
    coarse_category   -> local_evaluator.coarse_category   (category buckets)
    card_slots        -> local_evaluator.intent_card       (the card signature)

Each fails silently when it diverges. A `searchable_text` drift breaks phrase
grounding against the real scorer; a `card_slots` drift turns the card-signature
bonus into a no-op and quietly gives back +0.017. Neither raises, and neither
shows up as an error in any other tool — which is exactly why this exists.

    python -m tools.verify_mirror            # all 50,000 products
    python -m tools.verify_mirror -n 2000    # quick check

Run it after touching `src/catalog.py`, and after any update of the kit.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.sim import KIT_ROOT  # noqa: E402  (puts the kit on sys.path)

from evaluator import local_evaluator as kit  # noqa: E402

from src import catalog, config  # noqa: E402


def main() -> int:
    """Compare our mirrored evaluator functions against the kit's own, over all
    50,000 products. They fail silently on divergence, so this is the guard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--limit", type=int, default=0,
                        help="products to check; 0 = the whole catalog")
    parser.add_argument("--show", type=int, default=3,
                        help="how many divergences to print")
    args = parser.parse_args()

    checks = {"searchable_text": 0, "coarse_category": 0, "card_slots": 0}
    shown = {name: 0 for name in checks}
    total = 0

    with Path(config.CATALOG_PATH).open(encoding="utf-8") as handle:
        stream = itertools.islice(handle, args.limit) if args.limit else handle
        for line in stream:
            if not line.strip():
                continue
            product = json.loads(line)
            total += 1

            card = kit.intent_card(product)
            expected = {
                "searchable_text": kit.searchable_text(product),
                "coarse_category": kit.coarse_category(
                    [str(v) for v in (product.get("categories") or [])]),
                "card_slots": list(dict.fromkeys(
                    list(card["hard_constraints"]) + list(card["soft_preferences"]))),
            }
            actual = {
                "searchable_text": catalog.searchable_text(product),
                "coarse_category": catalog.coarse_category(
                    [str(v) for v in (product.get("categories") or [])]),
                "card_slots": catalog.card_slots(product),
            }
            for name in checks:
                if expected[name] != actual[name]:
                    checks[name] += 1
                    if shown[name] < args.show:
                        shown[name] += 1
                        print(f"\nDIVERGENCE  {name}  {product['parent_asin']}")
                        print(f"  evaluator: {expected[name]!r:.300}")
                        print(f"  ours:      {actual[name]!r:.300}")

    print(f"\nchecked {total:,} products against {KIT_ROOT.name}")
    failed = False
    for name, count in checks.items():
        status = "OK" if not count else f"DIVERGED on {count:,}"
        print(f"  {name:<18} {status}")
        failed |= bool(count)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
