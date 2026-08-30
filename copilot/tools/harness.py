"""Evaluation harness.

Scores our agent with the **official** evaluator by importing it and passing our
agent in — `evaluate()` takes the agent as an argument, so nothing in the
competition kit is ever modified.

    python -m tools.harness run                 # official score + per scenario
    python -m tools.harness perturb             # robustness curve L0..L4
    python -m tools.harness ablate              # switch-by-switch contribution

Run from the `copilot/` directory.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.sim import KIT_ROOT, RENDERERS, run_session  # noqa: E402

from evaluator.local_evaluator import (  # noqa: E402
    catalog_index,
    evaluate,
    load_jsonl,
    metric_summary,
)

from src import config  # noqa: E402
from src.agent import Agent  # noqa: E402


def composite(summary: dict) -> tuple[float, float]:
    efficiency = max(0.0, min(1.0, (11.0 - float(summary["mttc"])) / 10.0))
    score = (
        0.50 * summary["hit_rate_at_10"]
        + 0.30 * summary["mrr"]
        + 0.20 * efficiency
    )
    return score, efficiency


def show(label: str, sessions: list[dict]) -> dict:
    overall = metric_summary(sessions)
    score, efficiency = composite(overall)
    print(f"\n{label}")
    print(f"  score {score:.4f}   hit {overall['hit_rate_at_10']:.3f}   "
          f"mrr {overall['mrr']:.3f}   mttc {overall['mttc']:.2f}   eff {efficiency:.3f}")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in sessions:
        grouped[item["scenario_type"]].append(item)
    for name in sorted(grouped):
        part = metric_summary(grouped[name])
        print(f"    {name:<16} n={part['sample_count']:<4} "
              f"hit {part['hit_rate_at_10']:.3f}  mrr {part['mrr']:.3f}  "
              f"mttc {part['mttc']:.2f}")
    return {"score": score, **overall}


def load_world():
    samples = load_jsonl(KIT_ROOT / "data" / "public_set.jsonl")
    ids, categories, products = catalog_index(KIT_ROOT / "data" / "catalog.jsonl")
    return samples, ids, categories, products


# Set from common CLI flags so every subcommand honours them without threading
# an argument through each one.
RETRIEVAL: str | None = None
RESOLVER: str | None = None
GATE_CATEGORY_BONUS: bool = False
RERESOLVE: bool = False
SETS: dict = {}


def _coerce(field: str, value: str):
    """Coerce a --set string to the type of the matching Settings field."""
    current = getattr(config.DEFAULT, field)
    if isinstance(current, bool):
        return value.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def build_agent(**overrides) -> Agent:
    if RETRIEVAL is not None:
        overrides.setdefault("retrieval", RETRIEVAL)
    if RESOLVER is not None:
        overrides.setdefault("category_resolver", RESOLVER)
    if GATE_CATEGORY_BONUS:
        overrides.setdefault("gate_category_bonus", True)
    if RERESOLVE:
        overrides.setdefault("reresolve_category", True)
    for field, value in SETS.items():
        overrides.setdefault(field, _coerce(field, value))
    settings = config.DEFAULT.replace(**overrides) if overrides else config.DEFAULT
    started = time.perf_counter()
    agent = Agent(config.CATALOG_PATH, settings)
    print(f"  index built in {time.perf_counter() - started:.1f}s "
          f"({len(agent.store):,} products, {len(agent.index.postings):,} terms)")
    return agent


def cmd_run(args) -> None:
    samples, ids, categories, products = load_world()
    agent = build_agent()
    started = time.perf_counter()
    result = evaluate(agent, samples, ids, categories, products)
    elapsed = time.perf_counter() - started
    show("OFFICIAL EVALUATOR", result["sessions"])
    print(f"\n  {elapsed:.1f}s for {len(samples)} sessions "
          f"({elapsed / len(samples) * 1000:.0f} ms/session)")
    out = ROOT / "results" / "official.json"
    out.write_text(json.dumps({k: v for k, v in result.items() if k != "sessions"},
                              indent=2), encoding="utf-8")
    print(f"  written {out.relative_to(ROOT)}")


def cmd_perturb(args) -> None:
    """The point of the whole exercise: does the score survive rewording?"""
    samples, ids, categories, products = load_world()
    rows = []
    for level in args.levels.split(","):
        renderer = RENDERERS[level]()
        for templates in (True, False):
            agent = build_agent(use_templates=templates)
            sessions = [run_session(agent, s, products, categories, renderer)
                        for s in samples]
            label = f"{renderer.name}  templates={'on ' if templates else 'off'}"
            rows.append((label, show(label, sessions)["score"]))
    print("\n" + "=" * 58)
    print("ROBUSTNESS SUMMARY")
    for label, score in rows:
        print(f"  {label:<40} {score:.4f}")


def cmd_ci(args) -> None:
    """Bootstrap a confidence interval on the score, or on the delta to another
    config.

        python -m tools.harness ci                      # CI on the current score
        python -m tools.harness ci --compare tie_rerank=mmr   # is the delta real?
        python -m tools.harness ci --set fuzzy_repair=true --compare tie_rerank=mmr

    `--set` fixes the baseline; `--compare` adds overrides on top of it for the
    B run, and the two are compared with a paired bootstrap.
    """
    from tools.bootstrap import paired_delta_ci, score_ci

    samples, ids, categories, products = load_world()

    print("baseline:")
    agent_a = build_agent()
    sessions_a = evaluate(agent_a, samples, ids, categories, products)["sessions"]

    if not args.compare:
        ci = score_ci(sessions_a, iters=args.iters, alpha=args.alpha, seed=args.seed)
        pct = int(round((1 - args.alpha) * 100))
        print(f"\n  score {ci.point:.4f}   {pct}% CI "
              f"[{ci.lo:.4f}, {ci.hi:.4f}]   (± {(ci.hi - ci.lo) / 2:.4f}, "
              f"{args.iters:,} resamples)")
        return

    overrides = dict(pair.split("=", 1) for pair in args.compare)
    coerced = {f: _coerce(f, v) for f, v in overrides.items()}
    print(f"\ncompare (baseline + {coerced}):")
    agent_b = build_agent(**coerced)
    sessions_b = evaluate(agent_b, samples, ids, categories, products)["sessions"]

    d = paired_delta_ci(sessions_a, sessions_b, iters=args.iters,
                        alpha=args.alpha, seed=args.seed)
    pct = int(round((1 - args.alpha) * 100))
    excludes_zero = d.lo > 0 or d.hi < 0
    verdict = (f"significant at {pct}% (CI excludes 0)" if excludes_zero
               else f"NOT significant at {pct}% (CI spans 0)")
    print("\n" + "=" * 58)
    print("PAIRED BOOTSTRAP")
    print(f"  A  score {d.a:.4f}")
    print(f"  B  score {d.b:.4f}")
    print(f"  delta (B - A)  {d.delta:+.4f}   {pct}% CI "
          f"[{d.lo:+.4f}, {d.hi:+.4f}]")
    print(f"  P(delta > 0)   {d.p_positive:.3f}   ({args.iters:,} paired resamples)")
    print(f"  verdict: {verdict}")


def cmd_ablate(args) -> None:
    samples, ids, categories, products = load_world()
    switches = {
        "baseline": {},
        "no gate": {"gate_enabled": False},
        "no exclusion": {"demote_shown": False},
        "no phrase bonus": {"phrase_bonus": 0.0},
        "no category bonus": {"category_bonus": 0.0},
        "sum not log-product": {"log_epsilon": 1.0},
        "override: erase all": {"override_mode": "erase"},
        "override: keep all": {"override_mode": "keep"},
        "no templates": {"use_templates": False},
        "no salience": {"use_salience": False},
    }
    results = {}
    for name, overrides in switches.items():
        agent = build_agent(**overrides)
        result = evaluate(agent, samples, ids, categories, products)
        results[name] = show(name, result["sessions"])["score"]
    base = results["baseline"]
    print("\n" + "=" * 58)
    print("ABLATION (delta vs baseline)")
    for name, score in results.items():
        marker = "" if name == "baseline" else f"  {score - base:+.4f}"
        print(f"  {name:<24} {score:.4f}{marker}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TechJam agent harness")
    sub = parser.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--retrieval",
                        choices=["conjunctive", "bm25", "rrf", "auto"],
                        help="override the retrieval mode for this run")
    common.add_argument("--resolver", choices=["overlap", "vote", "classifier", "ensemble"],
                        help="override the category-resolution fallback")
    common.add_argument("--gate-category-bonus", action="store_true",
                        help="scale the category bonus by resolution confidence")
    common.add_argument("--reresolve", action="store_true",
                        help="re-resolve the category on an override turn")
    common.add_argument("--set", dest="sets", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="override any Settings field (repeatable)")
    sub.add_parser("run", parents=[common]).set_defaults(func=cmd_run)
    perturb = sub.add_parser("perturb", parents=[common])
    perturb.add_argument("--levels", default="L0,L1,L2,L4")
    perturb.set_defaults(func=cmd_perturb)
    sub.add_parser("ablate", parents=[common]).set_defaults(func=cmd_ablate)
    ci = sub.add_parser("ci", parents=[common])
    ci.add_argument("--compare", action="append", default=[], metavar="KEY=VALUE",
                    help="overrides for the B config; paired-compared to baseline "
                         "(repeatable)")
    ci.add_argument("--iters", type=int, default=10000,
                    help="bootstrap resamples (default 10000)")
    ci.add_argument("--alpha", type=float, default=0.05,
                    help="1 - confidence (default 0.05 -> 95%% CI)")
    ci.add_argument("--seed", type=int, default=0, help="resampling seed")
    ci.set_defaults(func=cmd_ci)
    args = parser.parse_args()
    global RETRIEVAL, RESOLVER, GATE_CATEGORY_BONUS, RERESOLVE, SETS
    RETRIEVAL = getattr(args, "retrieval", None)
    RESOLVER = getattr(args, "resolver", None)
    GATE_CATEGORY_BONUS = getattr(args, "gate_category_bonus", False)
    RERESOLVE = getattr(args, "reresolve", False)
    SETS = dict(pair.split("=", 1) for pair in getattr(args, "sets", []))
    args.func(args)


if __name__ == "__main__":
    main()
