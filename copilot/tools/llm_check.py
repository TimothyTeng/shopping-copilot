"""Smoke-test the optional model tier, and measure what it costs.

Disclosure is a submission requirement — `competition_specification.md:91` asks
for model choice, approximate cost, token usage, latency, and fallback
behaviour. This prints all five.

    PYTHONIOENCODING=utf-8 python -m tools.llm_check
    PYTHONIOENCODING=utf-8 python -m tools.llm_check --offline   # fallback proof
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import backends, config          # noqa: E402
from src.agent import Agent               # noqa: E402

PROBES = [
    "comfy trainers I can wear to the gym, need my feet to breathe",
    "looking for a wristwatch for my wife, something classic with a metal strap",
    "cosy slippers for my wife, knitted, sturdy enough to step outside to the bins",
    "a rainocat that is waterprrof with a hoood, lightwieght",
    "I need a jumper, merino wool, with a zip",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Model-tier smoke test")
    parser.add_argument("--gate", default="always", choices=["always", "unsatisfied"])
    parser.add_argument("--offline", action="store_true",
                        help="point at a dead endpoint to prove the fallback")
    args = parser.parse_args()

    if args.offline:
        os.environ["COPILOT_LLM_BASE"] = "http://127.0.0.1:9/v1"
        os.environ["COPILOT_LLM_CACHE"] = os.devnull

    cfg = config.DEFAULT.replace(backend="hyde", hyde_gate=args.gate)
    backend = backends.build(cfg)
    print(f"endpoint  {backend.url}")
    print(f"model     {backend.model}")
    print(f"timeout   {cfg.hyde_timeout_s}s   gate={cfg.hyde_gate}\n")

    print("loading catalog…")
    agent = Agent(config.CATALOG_PATH, cfg)
    print(f"ready  ({len(agent.store):,} products)\n")

    latencies = []
    for probe in PROBES:
        started = time.perf_counter()
        expansion = agent.backend.expand(probe, cfg.hyde_timeout_s)
        elapsed = (time.perf_counter() - started) * 1000
        latencies.append(elapsed)
        terms = agent._ground(expansion.text) if expansion.ok else []
        status = "OK" if expansion.ok else "EMPTY -> Tier-0 ranking is kept"
        tag = " (cached)" if expansion.cached else ""
        print(f"  shopper   {probe}")
        print(f"  {status}{tag}  {elapsed:.0f} ms  "
              f"{expansion.prompt_tokens}+{expansion.completion_tokens} tok")
        if expansion.ok:
            listing = " / ".join(line.strip() for line in expansion.text.splitlines()
                                 if line.strip())
            print(f"  listing   {listing[:150]}")
            print(f"  grounded  {' '.join(terms[:18])}")
            if not terms:
                print("  grounded  (nothing survived index verification)")
        print()

    stats = agent.backend.stats()
    total = stats["prompt_tokens"] + stats["completion_tokens"]
    print(f"calls {stats['calls']}   failures {stats['failures']}   "
          f"tokens {total}   median {sorted(latencies)[len(latencies)//2]:.0f} ms")
    if args.offline and stats["failures"] == len(PROBES):
        print("\nfallback verified: every call failed and none raised.")


if __name__ == "__main__":
    main()
