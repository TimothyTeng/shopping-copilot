"""Human adjudication of the generated probe set.

`tools/genprobes.py` rejects a probe on five mechanical grounds — copying,
title n-grams, off-target drift, a missing negation, a restatement. Those
guards are cheap and they fire, but they can only check what a regular
expression can see. They cannot tell you whether a kept probe reads like
something a person would actually type, or whether the target is genuinely the
best answer to it in a catalog of thousands.

That is the difference between "the set is clean" and "the set is valid", and
only a person can close it. This tool does the two halves a person should not
have to do by hand:

    python -m tools.adjudicate sample   # draw a blind sample into a sheet
    python -m tools.adjudicate score    # read the filled sheet back

The sample is *blind* in the sense that matters: the sheet shows the probe and
the target, and nothing about how the agent performed on it. An adjudicator who
can see which probes the agent failed is scoring the agent, not the probe.

Three verdicts per probe, each a plain y/n:

    phrasing   would a shopper plausibly type this?
    target     is the stated target a reasonable best answer to it?
    tag        is the tag correct? (a mislabelled probe corrupts the per-tag
               table it exists to produce)

`score` reports the agreement rate with a Wilson interval, because the number
this produces is only worth having if its own uncertainty is attached. A rate
computed from 40 draws is not a point.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.catalog import CatalogStore  # noqa: E402

DEFAULT_PROBES = ROOT / "data" / "probes_generated.jsonl"
DEFAULT_SHEET = ROOT / "data" / "adjudication.jsonl"

FIELDS = ("phrasing", "target", "tag")


def load_probes(path: Path) -> list[dict]:
    probes = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw and "_meta" not in raw[:12]:
                probes.append(json.loads(raw))
    return probes


def cmd_sample(args) -> None:
    probes = load_probes(Path(args.probes))
    # Stratified by tag, so the rare tags — the ones whose per-tag scores are
    # least stable and most likely to be acted on — are actually represented.
    # A uniform draw of 40 from 427 would give `negation` four probes on a good
    # day and zero on a bad one.
    by_tag: dict[str, list[dict]] = {}
    for probe in probes:
        by_tag.setdefault(probe["tags"][0] if probe["tags"] else "untagged", []).append(probe)
    rng = random.Random(args.seed)
    per_tag = max(1, round(args.n / max(1, len(by_tag))))
    drawn: list[dict] = []
    for tag in sorted(by_tag):
        pool = sorted(by_tag[tag], key=lambda p: p["id"])
        drawn.extend(rng.sample(pool, min(per_tag, len(pool))))
    rng.shuffle(drawn)
    drawn = drawn[: args.n]

    store = CatalogStore.load(config.CATALOG_PATH)
    out = Path(args.out)
    with out.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"_meta": {
            "tool": "tools/adjudicate.py", "source": Path(args.probes).name,
            "n": len(drawn), "seed": args.seed,
            "instructions": "Set each of phrasing/target/tag to \"y\" or \"n\". "
                            "Leave \"\" to skip. Add a note if you want the "
                            "reason recorded.",
        }}) + "\n")
        for probe in drawn:
            doc = store.ord_of.get(probe["target"])
            handle.write(json.dumps({
                "id": probe["id"],
                "tags": probe["tags"],
                "turns": probe["turns"],
                "target": probe["target"],
                "target_title": store.raw_title[doc] if doc is not None else "?",
                "target_text": (store.text[doc][:400] if doc is not None else "?"),
                # deliberately absent: anything about how the agent did on it
                "phrasing": "", "target_ok": "", "tag_ok": "", "note": "",
            }) + "\n")
    print(f"wrote {len(drawn)} probes to {out}")
    print(f"  tags covered: {len(by_tag)}  ({per_tag} drawn per tag, capped at n)")
    print("  fill in phrasing / target_ok / tag_ok, then: "
          "python -m tools.adjudicate score")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — correct near 0 and 1, where a normal
    approximation on 40 samples is not."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cmd_score(args) -> None:
    rows = [json.loads(l) for l in Path(args.sheet).read_text(
        encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if "_meta" not in r]
    judged = [r for r in rows if any(r.get(f) for f in
                                     ("phrasing", "target_ok", "tag_ok"))]
    if not judged:
        print(f"nothing filled in yet in {args.sheet}")
        return
    print(f"\nADJUDICATION  {len(judged)} of {len(rows)} probes judged\n")
    print(f"  {'axis':<12}{'agree':<10}{'rate':<10}{'95% Wilson'}")
    for field, key in (("phrasing", "phrasing"), ("target", "target_ok"),
                       ("tag", "tag_ok")):
        vals = [r[key].strip().lower() for r in judged if r.get(key, "").strip()]
        k = sum(1 for v in vals if v.startswith("y"))
        lo, hi = wilson(k, len(vals))
        rate = k / len(vals) if vals else 0.0
        print(f"  {field:<12}{k}/{len(vals):<7}{rate:<10.3f}[{lo:.3f}, {hi:.3f}]")
    # The number that actually bounds the set: a probe is sound only if all
    # three hold. Reporting the axes separately would let a set with 90% on
    # each look like a 90% set when it is closer to 73%.
    whole = [r for r in judged if all(r.get(k, "").strip() for k in
                                      ("phrasing", "target_ok", "tag_ok"))]
    k = sum(1 for r in whole if all(r[key].strip().lower().startswith("y")
                                    for key in ("phrasing", "target_ok", "tag_ok")))
    if whole:
        lo, hi = wilson(k, len(whole))
        print(f"\n  sound on all three   {k}/{len(whole)}   {k/len(whole):.3f}   "
              f"95% Wilson [{lo:.3f}, {hi:.3f}]")
        print(f"\n  Read this as the ceiling on what the n=427 set can claim: a "
              f"score\n  measured on it is measured on a set roughly "
              f"{k/len(whole):.0%} of which is sound.")
    notes = [r for r in judged if r.get("note", "").strip()]
    if notes:
        print(f"\n  NOTES")
        for r in notes:
            print(f"    {r['id']}: {r['note']}")
    bad_tags = Counter(r["tags"][0] for r in judged
                       if r.get("tag_ok", "").strip().lower().startswith("n"))
    if bad_tags:
        print(f"\n  mislabelled by tag: "
              + ", ".join(f"{t} {c}" for t, c in bad_tags.most_common()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="draw a blind sample into a sheet")
    s.add_argument("--probes", default=str(DEFAULT_PROBES))
    s.add_argument("--out", default=str(DEFAULT_SHEET))
    s.add_argument("-n", type=int, default=40)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_sample)
    c = sub.add_parser("score", help="read the filled sheet back")
    c.add_argument("--sheet", default=str(DEFAULT_SHEET))
    c.set_defaults(func=cmd_score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
