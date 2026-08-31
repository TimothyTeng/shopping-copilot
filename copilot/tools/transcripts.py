"""Dump full conversation transcripts, with the top-10 ranking on every turn.

Two worlds, never conflated (see CLAUDE.md "The two scores"):

  benchmark   the official simulator, whose constraints are lifted verbatim
              from the target's own catalog text. Session semantics come from
              the evaluator itself, exactly as `tools/demo.py replay` drives
              them -- disclosure rules, override timing, hit detection.
  stress      `tools/probes.py`, hand-authored shopper wording written without
              looking at the product page.

    python -m tools.transcripts --n 50 --out results/transcripts.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.probes import FILLER, PROBES  # noqa: E402
from tools.sim import KIT_ROOT, RENDERERS  # noqa: E402

from evaluator.local_evaluator import (  # noqa: E402
    ALLOWED_ATTRIBUTES,
    MAX_TURNS,
    TOP_K,
    classify_constraint,
    coarse_category,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)

from src import config  # noqa: E402
from src.agent import Agent  # noqa: E402


def listing(agent: Agent, asins: list[str], target: str) -> list[dict]:
    rows = []
    for rank, asin in enumerate(asins, 1):
        d = agent.store.ord_of.get(asin)
        rows.append({
            "rank": rank,
            "asin": asin,
            "title": agent.store.raw_title[d] if d is not None else "(unknown)",
            "is_target": asin == target,
        })
    return rows


# ---------------------------------------------------------------------------
# the official simulator -- same loop as tools/demo.py replay
# ---------------------------------------------------------------------------
def run_benchmark(agent: Agent, sample, products, categories, renderer) -> dict:
    card, behavior = materialize_hidden_fields(sample, products)
    target = str(sample["ground_truth"]["parent_asin"])
    scenario = sample["scenario_type"]
    category = coarse_category(categories.get(target, []))

    disclosed: set[str] = set()
    boundary_used = False
    override_applied = scenario != "intent_override"

    if scenario == "buying" and card.get("hard_constraints"):
        first = str(card["hard_constraints"][0])
        disclosed.add(first)
        message = renderer.open_buy(category, first)
    elif scenario == "intent_override":
        message = renderer.open_override(category, str(behavior["override"]["old_value"]))
    else:
        message = renderer.open_browse(category)

    sid = f"transcript::{sample['sample_id']}"
    agent.reset(sid, sample["user_profile"])
    turns: list[dict] = []
    hit_rank = hit_turn = None

    for turn in range(1, MAX_TURNS + 1):
        response = agent.respond(sid, message, turn, TOP_K)
        ranked = normalize_recommendations(response.get("recommendations"), set(products))
        state = agent._sessions[sid]
        row = {
            "turn": turn,
            "shopper": message,
            "agent": response.get("message", ""),
            "ask_attribute": response.get("ask_attribute"),
            "category_key": state.category_key,
            "clues": [s.key for s in state.active_slots()],
            "results": listing(agent, ranked, target),
            "held_back": not ranked,
            "override_active": override_applied,
        }
        turns.append(row)

        if override_applied and target in ranked:
            hit_rank, hit_turn = ranked.index(target) + 1, turn
            row["hit"] = True
            break
        if turn == MAX_TURNS:
            break

        override = (behavior or {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            message = renderer.override(new_value)
            continue

        attribute = response.get("ask_attribute")
        attribute = attribute if isinstance(attribute, str) else None
        if scenario == "boundary" and not boundary_used and attribute:
            message, boundary_used = renderer.no_pref(attribute), True
        elif not attribute:
            message = renderer.nudge()
        else:
            if attribute not in ALLOWED_ATTRIBUTES:
                attribute = "other"
            pool = [str(v) for v in card.get("hard_constraints", [])]
            pool += [str(v) for v in card.get("soft_preferences", [])]
            matches = [v for v in pool
                       if v not in disclosed
                       and (attribute == "other" or classify_constraint(v) == attribute)][:2]
            if not matches:
                message = renderer.none_left(attribute)
            else:
                disclosed.update(matches)
                message = renderer.reply(matches)

    d = agent.store.ord_of.get(target)
    return {
        "world": "benchmark",
        "id": sample["sample_id"],
        "scenario": scenario,
        "category": category,
        "target": target,
        "target_title": agent.store.raw_title[d] if d is not None else "(unknown)",
        "hard": [str(c) for c in card.get("hard_constraints", [])],
        "soft": [str(c) for c in card.get("soft_preferences", [])],
        "override": (behavior or {}).get("override") or {},
        "rank": hit_rank,
        "turn": hit_turn if hit_rank else 11,
        "turns": turns,
    }


# ---------------------------------------------------------------------------
# the hand-authored prose probes -- same loop as tools/stress.py
# ---------------------------------------------------------------------------
def run_probe(agent: Agent, probe: dict) -> dict:
    sid = f"transcript::stress::{probe['id']}"
    agent.reset(sid, {})
    script = list(probe["turns"])
    target = probe["target"]
    turns: list[dict] = []
    hit_rank = hit_turn = None

    for turn in range(1, MAX_TURNS + 1):
        message = script.pop(0) if script else FILLER[(turn - 1) % len(FILLER)]
        response = agent.respond(sid, message, turn, TOP_K)
        ranked = [str(r.get("parent_asin")) for r in (response.get("recommendations") or [])]
        state = agent._sessions[sid]
        row = {
            "turn": turn,
            "shopper": message,
            "agent": response.get("message", ""),
            "ask_attribute": response.get("ask_attribute"),
            "category_key": state.category_key,
            "clues": [s.key for s in state.active_slots()],
            "results": listing(agent, ranked, target),
            "held_back": not ranked,
            "scripted": bool(turn <= len(probe["turns"])),
        }
        turns.append(row)
        if target in ranked:
            hit_rank, hit_turn = ranked.index(target) + 1, turn
            row["hit"] = True
            break

    d = agent.store.ord_of.get(target)
    return {
        "world": "stress",
        "id": probe["id"],
        "scenario": ",".join(probe.get("tags") or []),
        "target": target,
        "target_title": agent.store.raw_title[d] if d is not None else "(unknown)",
        "rank": hit_rank,
        "turn": hit_turn if hit_rank else 11,
        "turns": turns,
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows) or 1
    hits = [r for r in rows if r["rank"]]
    mttc = sum(r["turn"] for r in rows) / n
    hit = len(hits) / n
    mrr = sum(1.0 / r["rank"] for r in hits) / n
    eff = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return {"n": len(rows), "hit": hit, "mrr": mrr, "mttc": mttc,
            "score": 0.50 * hit + 0.30 * mrr + 0.20 * eff}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="benchmark sessions to dump")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", default="L0", choices=sorted(RENDERERS))
    ap.add_argument("--out", default="results/transcripts.json")
    args = ap.parse_args()

    print("loading catalog…", file=sys.stderr)
    agent = Agent(config.CATALOG_PATH, config.DEFAULT)

    products, categories = {}, {}
    with (KIT_ROOT / "data" / "catalog.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            p = json.loads(line)
            asin = str(p["parent_asin"])
            products[asin] = p
            categories[asin] = [str(v) for v in p.get("categories") or []]

    samples = load_jsonl(KIT_ROOT / "data" / "public_set.jsonl")
    # Stratify so the dump mirrors the public set's scenario mix rather than
    # whichever types a flat sample happens to draw.
    by_type: dict[str, list] = {}
    for s in samples:
        by_type.setdefault(s["scenario_type"], []).append(s)
    rng = random.Random(args.seed)
    picked = []
    for kind, pool in sorted(by_type.items()):
        take = max(1, round(args.n * len(pool) / len(samples)))
        picked += rng.sample(pool, min(take, len(pool)))
    picked = picked[: args.n]

    renderer = RENDERERS[args.level]()
    bench = [run_benchmark(agent, s, products, categories, renderer) for s in picked]
    print(f"benchmark: {len(bench)} sessions", file=sys.stderr)
    stress = [run_probe(agent, p) for p in PROBES]
    print(f"stress: {len(stress)} probes", file=sys.stderr)

    out = {
        "config": {"level": args.level, "seed": args.seed,
                   "retrieval": config.DEFAULT.retrieval, "top_k": TOP_K},
        "benchmark": {"summary": summarize(bench), "sessions": bench},
        "stress": {"summary": summarize(stress), "sessions": stress},
    }
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}", file=sys.stderr)
    for name in ("benchmark", "stress"):
        s = out[name]["summary"]
        print(f"  {name:<10} n={s['n']:<4} score {s['score']:.4f}  hit {s['hit']:.3f}  "
              f"mrr {s['mrr']:.3f}  mttc {s['mttc']:.2f}", file=sys.stderr)


if __name__ == "__main__":
    main()
