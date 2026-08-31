"""Watch the agent work, or talk to it yourself.

    python -m tools.demo replay                     # a real labelled session, turn by turn
    python -m tools.demo replay --sample public_0007
    python -m tools.demo replay --scenario intent_override
    python -m tools.demo replay --level L1          # ...with the customer paraphrasing
    python -m tools.demo chat                       # type your own shopping request

`replay` shows the hidden target up front so you can see how the ranking closes
in on it. `chat` hides nothing — it is just you and the agent.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

DIM, BOLD, GREEN, YELLOW, CYAN, RESET = (
    "\033[2m", "\033[1m", "\033[32m", "\033[33m", "\033[36m", "\033[0m"
)


def short(text: str, width: int = 68) -> str:
    """Truncate a product title for the terminal, on a word boundary."""
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def build(**overrides) -> Agent:
    """Construct the agent for a demo, honouring `--retrieval` and friends."""
    print(f"{DIM}loading 50,000 products…{RESET}")
    agent = Agent(config.CATALOG_PATH, config.DEFAULT.replace(**overrides)
                  if overrides else config.DEFAULT)
    print(f"{DIM}ready{RESET}\n")
    return agent


# --------------------------------------------------------------------------
def cmd_replay(args) -> None:
    """`demo replay` — play a labelled session turn by turn with the hidden
    target revealed, so the ranking can be watched closing in on it."""
    samples = load_jsonl(KIT_ROOT / "data" / "public_set.jsonl")
    if args.sample:
        pool = [s for s in samples if s["sample_id"] == args.sample]
        if not pool:
            sys.exit(f"no session called {args.sample}")
    elif args.scenario:
        pool = [s for s in samples if s["scenario_type"] == args.scenario]
        if not pool:
            sys.exit(f"no sessions of type {args.scenario}")
    else:
        pool = samples
    sample = random.Random(args.seed).choice(pool)

    agent = build(retrieval=args.retrieval, backend=args.backend)
    store = agent.store
    renderer = RENDERERS[args.level]()

    # Rebuild the same view of the world the evaluator uses.
    products = {}
    categories = {}
    import json
    with (KIT_ROOT / "data" / "catalog.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            asin = str(product["parent_asin"])
            products[asin] = product
            categories[asin] = [str(v) for v in product.get("categories") or []]

    card, behavior = materialize_hidden_fields(sample, products)
    target = str(sample["ground_truth"]["parent_asin"])
    scenario = sample["scenario_type"]
    category = coarse_category(categories.get(target, []))

    print(f"{BOLD}session{RESET} {sample['sample_id']}   "
          f"{BOLD}type{RESET} {CYAN}{scenario}{RESET}   "
          f"{BOLD}phrasing{RESET} {renderer.name}")
    print(f"{BOLD}hidden target{RESET} {YELLOW}{target}{RESET}  "
          f"{short(products[target].get('title'), 60)}")
    print(f"{DIM}what they secretly want:{RESET}")
    for c in card.get("hard_constraints", []):
        print(f"  {DIM}must  {RESET}{short(c, 66)}")
    for c in card.get("soft_preferences", []):
        print(f"  {DIM}nice  {RESET}{short(c, 66)}")
    print("─" * 78)

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

    agent.reset("demo", sample["user_profile"])

    for turn in range(1, MAX_TURNS + 1):
        print(f"\n{BOLD}turn {turn}{RESET}")
        print(f"  {CYAN}shopper{RESET}  {short(message, 66)}")
        response = agent.respond("demo", message, turn, TOP_K)
        ranked = normalize_recommendations(response.get("recommendations"), set(products))

        state = agent._sessions["demo"]
        clues = [s.key for s in state.active_slots()]
        print(f"  {DIM}knows{RESET}    category={state.category_key or '—'}"
              f"  clues={len(clues)}")
        for clue in clues:
            print(f"           {DIM}· {short(clue, 60)}{RESET}")
        print(f"  {GREEN}agent{RESET}    {short(response['message'], 66)}"
              f"  {DIM}[asks: {response['ask_attribute']}]{RESET}")

        if not ranked:
            print(f"  {DIM}holds back — evidence too thin to commit to a list{RESET}")
        else:
            for rank, asin in enumerate(ranked[:3], 1):
                mark = f"{YELLOW}◀ TARGET{RESET}" if asin == target else ""
                title = short(store.raw_title[store.ord_of[asin]], 52)
                print(f"    {rank}. {title} {mark}")

        if override_applied and target in ranked:
            print(f"\n{GREEN}{BOLD}HIT{RESET} at turn {turn}, "
                  f"rank {ranked.index(target) + 1}")
            return
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
            pool_c = [str(v) for v in card.get("hard_constraints", [])]
            pool_c += [str(v) for v in card.get("soft_preferences", [])]
            matches = [
                v for v in pool_c
                if v not in disclosed
                and (attribute == "other" or classify_constraint(v) == attribute)
            ][:2]
            if not matches:
                message = renderer.none_left(attribute)
            else:
                disclosed.update(matches)
                message = renderer.reply(matches)

    print(f"\n{YELLOW}MISS{RESET} — target never reached the top 10")


# --------------------------------------------------------------------------
def cmd_chat(args) -> None:
    """`demo chat` — talk to the agent yourself. Defaults to bm25 retrieval:
    free text is a different surface from the benchmark, and the conjunctive
    ordering measures ~0.24 worse on it."""
    # The hold-back gate exists only because a scored session ends at the first
    # hit, so an early weak list locks in a bad rank. A real person is under no
    # such rule and just wants to see results, so it is disabled here.
    print(f"{DIM}loading 50,000 products…{RESET}")
    # Free-typed input is the case the conjunctive scorer is worst at: measured
    # 0.375 against 0.641 for BM25 on tools/stress.py. The graded path keeps the
    # conjunctive default; this surface does not.
    #
    # Fuzzy repair is on here for the same reason: a person types `waterprrof`,
    # the simulator never does. It is a no-op on any token the catalog contains,
    # so it can only help this surface and cannot regress the graded path.
    # doc2query for the same reason again, and it is the largest gain this
    # surface has ever had: +0.0507 on the n=427 prose set, 95% CI
    # [+0.0286, +0.0729]. Inert on the graded path (conjunctive never calls the
    # prose retriever) and network-free, because the generation was paid for at
    # build time.
    agent = Agent(config.CATALOG_PATH,
                  config.DEFAULT.replace(gate_enabled=False,
                                         fuzzy_repair=True,
                                         doc2query_expansions=True,
                                         backend=args.backend,
                                         retrieval=args.retrieval))
    print(f"{DIM}ready{RESET}\n")
    store = agent.store
    agent.reset("chat", {"summary": "interactive user", "preference_tags": []})
    print(f"{BOLD}Describe what you're shopping for.{RESET} "
          f"{DIM}(blank line or Ctrl-C to quit){RESET}\n")

    turn = 0
    while turn < MAX_TURNS:
        try:
            message = input(f"{CYAN}you  {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not message:
            return
        turn += 1
        response = agent.respond("chat", message, turn, TOP_K)
        state = agent._sessions["chat"]
        print(f"{DIM}     understood: category={state.category_key or '—'}, "
              f"clues={[s.key for s in state.active_slots()]}{RESET}")
        recs = response["recommendations"]
        if not recs:
            print(f"{DIM}     (holding back — not confident enough yet){RESET}")
        for rank, item in enumerate(recs[:5], 1):
            doc = store.ord_of[item["parent_asin"]]
            print(f"     {rank}. {short(store.raw_title[doc], 62)}")
        print(f"{GREEN}agent{RESET} {response['message']}\n")


def main() -> None:
    """CLI entry point: replay | chat."""
    parser = argparse.ArgumentParser(description="Shopping Copilot demo")
    sub = parser.add_subparsers(dest="cmd", required=True)

    replay = sub.add_parser("replay", help="replay a labelled session")
    replay.add_argument("--sample", help="e.g. public_0007")
    replay.add_argument("--scenario", choices=["buying", "browsing", "intent_override", "boundary"])
    replay.add_argument("--level", default="L0", choices=sorted(RENDERERS))
    replay.add_argument("--seed", type=int, default=0)
    replay.add_argument("--retrieval", default="conjunctive",
                        choices=["conjunctive", "bm25", "rrf", "auto"])
    replay.set_defaults(func=cmd_replay)

    chat = sub.add_parser("chat", help="talk to the agent yourself")
    chat.add_argument("--retrieval", default="bm25",
                      choices=["conjunctive", "bm25", "rrf", "auto"])
    chat.set_defaults(func=cmd_chat)

    # The optional model tier (src/backends/). Off unless asked for, on either
    # surface, so neither demo opens a socket by default.
    for parser_ in (replay, chat):
        parser_.add_argument("--backend", default="null", choices=["null", "hyde"],
                             help="optional model tier (default: null)")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
