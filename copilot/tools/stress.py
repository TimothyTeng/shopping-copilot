"""An independent stress test for natural language.

The official evaluator is not a natural-language test, and its score should not
be read as one. Its simulator lifts the shopper's constraints **verbatim from
the target product's own `features` and `details`**, so the query and the
document share vocabulary by construction. A system that only matches exact
strings scores well there and can still be useless to a person typing what they
actually want.

This suite is built to find that gap rather than hide it.

Rules followed, from `docs/competition_specification.md`:

  * `top_k` = 10, at most 10 turns, session ends at the first hit
  * a miss counts as turn 11
  * TechnicalScore = 0.50·HitRate@10 + 0.30·MRR + 0.20·Efficiency
  * Efficiency = clip((11 − MTTC) / 10, 0, 1)
  * the agent is driven through the unmodified `reset` / `respond` contract

Three things keep the result honest:

1. **Verbatim overlap is measured and printed.** For every probe we report the
   share of the shopper's content words that appear in the target's catalog
   text. A high number means an easy probe, not a good agent.
2. **Every probe is scored against reference points**, not in isolation:
   an oracle that queries with the target's own catalog text (the advantage the
   official simulator hands us for free), a textbook BM25 run over the whole
   message with no dialogue state at all, and a category-plus-popularity floor.
   If the agent cannot beat plain BM25 on natural language, that is the finding.
3. **Nothing was tuned on this set**, and the probe targets were chosen before
   any shopper wording was written.

    python -m tools.stress            # everything
    python -m tools.stress --track natural
    python -m tools.stress --track vocab
    python -m tools.stress --show     # per-probe detail and failure reasons
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from array import array
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bootstrap import paired_delta_ci, score_ci  # noqa: E402
from tools.probes import FILLER, PROBES  # noqa: E402
from tools.sim import KIT_ROOT  # noqa: E402  (puts the kit on sys.path)

from evaluator.local_evaluator import MAX_TURNS, TOP_K  # noqa: E402

from src import config  # noqa: E402
from src.agent import Agent  # noqa: E402
from src.bm25 import Bm25Index
from src.catalog import CatalogStore  # noqa: E402
from src.index import InvertedIndex  # noqa: E402
from src.normalize import DIALOGUE_STOP, norm, tokens  # noqa: E402

DIM, BOLD, GREEN, RED, YELLOW, CYAN, RESET = (
    "\033[2m", "\033[1m", "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[0m"
)


# ---------------------------------------------------------------------------
# metrics, exactly as the organizers define them
# ---------------------------------------------------------------------------
def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    hits = [r for r in rows if r["rank"] is not None]
    hit_rate = len(hits) / n if n else 0.0
    mrr = sum(1.0 / r["rank"] for r in hits) / n if n else 0.0
    # A miss counts as turn 11, per the specification.
    mttc = sum(r["turn"] if r["rank"] is not None else 11 for r in rows) / n if n else 0.0
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return {
        "n": n,
        "hit": hit_rate,
        "mrr": mrr,
        "mttc": mttc,
        "score": 0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency,
    }


def line(label: str, s: dict, width: int = 26) -> str:
    return (f"  {label:<{width}} n={s['n']:<4} score {s['score']:.4f}   "
            f"hit {s['hit']:.3f}   mrr {s['mrr']:.3f}   mttc {s['mttc']:.2f}")


# ---------------------------------------------------------------------------
# comparing configurations
# ---------------------------------------------------------------------------
# Row labels, named once so the pairing code and the printing code cannot
# disagree about which run is which.
BASE = "agent (ours)"
CONTROL = "BM25, whole transcript"
MATCHED = "BM25, turn-matched"
ORACLE = "oracle: product's own words"


def sessions_from(rows: list[dict]) -> list[dict]:
    """Stress results in the shape `tools.bootstrap` already consumes.

    The bootstrap was written against the official evaluator's per-session
    records and recomputes the composite from hit, reciprocal rank and
    first-hit turn. Those are exactly the three things a stress probe produces,
    so this is a rename, not a second implementation of the metric.
    """
    return [{"hit": 1 if r["rank"] is not None else 0,
             "reciprocal_rank": (1.0 / r["rank"]) if r["rank"] is not None else 0.0,
             "first_hit_turn": r["turn"] if r["rank"] is not None else None}
            for r in rows]


def delta_line(label: str, d) -> str:
    """One paired delta, with the sign kept visible."""
    sig = GREEN if d.lo > 0 else RED if d.hi < 0 else YELLOW
    verdict = ("clears zero" if d.lo > 0 else
               "below zero" if d.hi < 0 else "spans zero")
    return (f"{label:<34} Δ {d.delta:+.4f}   95% CI [{d.lo:+.4f}, {d.hi:+.4f}]   "
            f"p(Δ>0) {d.p_positive:.3f}   {sig}{verdict}{RESET}")


def settings_from(args, extra: dict | None = None) -> config.Settings:
    """The command line's overrides, plus a variant's own, over the default."""
    overrides: dict = {"retrieval": args.retrieval}
    if args.resolver is not None:
        overrides["category_resolver"] = args.resolver
    if args.gate_category_bonus:
        overrides["gate_category_bonus"] = True
    if args.reresolve:
        overrides["reresolve_category"] = True
    for pair in getattr(args, "sets", []):
        field, value = pair.split("=", 1)
        overrides[field] = _coerce(field, value)
    for field, value in (extra or {}).items():
        overrides[field] = _coerce(field, value)
    return config.DEFAULT.replace(**overrides)


def _coerce(field: str, value: str):
    """Parse a KEY=VALUE string against the declared type of that setting."""
    current = getattr(config.DEFAULT, field)
    if isinstance(current, bool):
        return value.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def parse_variant(spec: str) -> tuple[str, dict]:
    """`label:key=value,key=value` — the label is optional and defaults to the
    overrides themselves, which is usually the more honest name anyway.

    Every key is checked against `Settings` here rather than at construction
    time. A mistyped override that silently becomes a label, or a label that
    silently becomes an override, would run for minutes and then report a
    number for a configuration nobody asked for — which is worse than a crash,
    because the number looks fine.
    """
    label = ""
    if ":" in spec and "=" not in spec.split(":", 1)[0]:
        label, spec = spec.split(":", 1)
    overrides = {}
    for pair in spec.split(","):
        if not pair:
            continue
        if "=" not in pair:
            raise SystemExit(
                f"--vs {spec!r}: '{pair}' is not key=value. Expected "
                f"'label:key=value,key=value' — and a label may not contain '='.")
        field, value = pair.split("=", 1)
        field = field.strip()
        if not hasattr(config.DEFAULT, field):
            raise SystemExit(
                f"--vs {spec!r}: '{field}' is not a Settings field. "
                f"(If it was meant as a label, labels may not contain '='.)")
        overrides[field] = value
    if not overrides:
        raise SystemExit(f"--vs {spec!r}: no overrides given.")
    return (label or spec), overrides


def crosstab(probes: list[dict], runs: dict[str, list[dict]],
             labels: list[str]) -> None:
    """Score per tag per configuration, with the per-tag winner marked.

    The point of this table is not the totals — those are already printed. It
    is whether a configuration that loses overall wins *anywhere*. A mode that
    never leads a single tag is not a trade-off, it is dead weight.
    """
    tags: dict[str, list[int]] = defaultdict(list)
    for i, probe in enumerate(probes):
        for tag in probe["tags"]:
            tags[tag].append(i)

    width = max(len(t) for t in tags) + 2 if tags else 10
    print(f"\n{BOLD}TAG × CONFIGURATION{RESET}  {DIM}composite score; "
          f"{GREEN}green{RESET}{DIM} is the winner for that tag{RESET}")
    header = f"  {'tag':<{width}}{'n':<5}" + "".join(f"{l[:20]:<22}" for l in labels)
    print(f"{DIM}{header}{RESET}")
    wins = Counter()
    for tag in sorted(tags):
        idx = tags[tag]
        scores = [summarize([runs[l][i] for i in idx])["score"] for l in labels]
        # A tag every configuration scores identically on — usually all-zero —
        # has no winner. Crediting one to whichever column sorted first would
        # put fake leads in the very count this table exists to produce.
        best = (max(range(len(scores)), key=lambda k: scores[k])
                if max(scores) > min(scores) else None)
        if best is not None:
            wins[labels[best]] += 1
        cells = "".join(
            (f"{GREEN}{scores[k]:<22.4f}{RESET}" if k == best
             else f"{scores[k]:<22.4f}")
            for k in range(len(scores)))
        print(f"  {tag:<{width}}{len(idx):<5}{cells}")
    print(f"\n{DIM}  tags led: "
          + ", ".join(f"{l} {wins[l]}" for l in labels) + f"{RESET}")
    dead = [l for l in labels if wins[l] == 0]
    if dead:
        print(f"  {YELLOW}leads no tag at all: {', '.join(dead)}{RESET}")


# ---------------------------------------------------------------------------
# reference retrievers — deliberately simple, and none of them use our agent
# ---------------------------------------------------------------------------
class Bm25:
    """The stateless control: BM25 over the whole transcript, nothing else.

    Deliberately shares `src.bm25.Bm25Index` with the agent's own `bm25`
    retrieval mode rather than reimplementing it. A control that drifts from the
    thing it controls for is worse than no control — this way the only
    difference measured is dialogue handling, not two people's BM25.
    """

    def __init__(self, store: CatalogStore) -> None:
        self.index = Bm25Index(store)

    def search(self, query: str, top_k: int) -> list[int]:
        return [doc for doc, _ in self.index.score(query, config.DEFAULT, top_k)]


# ---------------------------------------------------------------------------
# running one probe
# ---------------------------------------------------------------------------
def run_agent_probe(agent: Agent, probe: dict, target_doc: int) -> dict:
    """Drive the agent through a scripted shopper. Official stopping rules."""
    session = f"stress::{probe['id']}"
    agent.reset(session, {})
    script = list(probe["turns"])
    transcript: list[tuple[str, list[int]]] = []

    for turn in range(1, MAX_TURNS + 1):
        if script:
            message = script.pop(0)
        else:
            message = FILLER[(turn - 1) % len(FILLER)]
        try:
            response = agent.respond(session, message, turn, TOP_K)
        except Exception as exc:                       # a crash is a miss
            return {"rank": None, "turn": 11, "why": f"exception: {exc}",
                    "transcript": transcript}
        ids = [r.get("parent_asin") for r in (response.get("recommendations") or [])]
        docs = [agent.store.ord_of.get(i, -1) for i in ids]
        transcript.append((message, docs))
        if target_doc in docs:
            return {"rank": docs.index(target_doc) + 1, "turn": turn,
                    "why": "", "transcript": transcript}
    return {"rank": None, "turn": 11, "why": "", "transcript": transcript}


def run_turnmatched(search, probe: dict, target_doc: int) -> dict:
    """A stateless retriever that hears the same words on the same turn.

    The one-shot control below is handed the WHOLE transcript on turn 1, which
    is generous on purpose — but it makes that control unreadable on any probe
    whose decisive word arrives late. It cannot lose the turns the agent spends
    waiting for the shopper to say the thing that matters, so part of its lead
    is information the agent did not have yet, not retrieval the agent got
    wrong.

    This one is the honest comparison: at turn t it sees exactly `turns[:t]`,
    and nothing else. No state, no slots, no category. A gap that survives
    against THIS is dialogue handling; a gap that only appears against the
    one-shot control is the asymmetry.

    Measured worth: on the `brand` tag the one-shot control scores 0.8636
    against the agent's 0.7171, which reads as a large defect. The turn-matched
    control scores 0.7678 — two thirds of that gap was the asymmetry, and the
    remainder does not clear zero (+0.051, 95% CI [-0.045, +0.148]).
    """
    for turn in range(1, MAX_TURNS + 1):
        said = probe["turns"][:turn] or [FILLER[0]]
        docs = search(" ".join(said), TOP_K)
        if target_doc in docs:
            return {"rank": docs.index(target_doc) + 1, "turn": turn, "why": "",
                    "transcript": []}
    return {"rank": None, "turn": 11, "why": "", "transcript": []}


def run_oneshot(search, probe: dict, target_doc: int) -> dict:
    """A stateless retriever gets the whole transcript at once, on turn 1.

    Generous on purpose: it never has to ask a question, so it cannot lose
    efficiency. If it still beats the agent, the dialogue is not helping.
    """
    query = " ".join(probe["turns"])
    docs = search(query, TOP_K)
    if target_doc in docs:
        return {"rank": docs.index(target_doc) + 1, "turn": 1, "why": "",
                "transcript": []}
    return {"rank": None, "turn": 11, "why": "", "transcript": []}


# ---------------------------------------------------------------------------
# honesty controls
# ---------------------------------------------------------------------------
def load_probes(path: str | None) -> list[dict]:
    """The hand-authored 26, or a generated set from `tools/genprobes.py`.

    The hand-authored set stays the default and stays authoritative: it was
    written by a person who had not seen the agent's behaviour, and no model
    wrote a word of it. A generated set buys statistical resolution that n=26
    cannot give, at the cost of being written by a model — so it is opt-in, and
    its provenance is printed rather than assumed.
    """
    if not path:
        return PROBES
    file = Path(path)
    if not file.is_absolute():
        file = ROOT / file
    probes: list[dict] = []
    meta: dict = {}
    with file.open(encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if "_meta" in row:
                meta = row["_meta"]
                continue
            probes.append(row)
    print(f"{YELLOW}  generated probe set: {file.name}  n={len(probes)}  "
          f"model={meta.get('model', '?')}{RESET}")
    if meta.get("warning"):
        print(f"{DIM}  {meta['warning']}{RESET}")
    return probes


def verbatim_overlap(probe: dict, store: CatalogStore, doc: int) -> float:
    """Share of the shopper's content words that occur in the target's text.

    1.0 means the probe is quoting the product page, which is what the official
    simulator does. Low values are what makes a probe a real test.
    """
    said = {t for t in tokens(norm(" ".join(probe["turns"])))
            if t not in DIALOGUE_STOP and len(t) > 2}
    if not said:
        return 0.0
    have = set(tokens(store.text[doc]))
    return len(said & have) / len(said)


def diagnose(agent: Agent, probe: dict, store: CatalogStore, index: InvertedIndex,
             doc: int, result: dict) -> str:
    """Why did this probe fail? Cheap, mechanical, no guessing."""
    if result["rank"] == 1:
        return ""
    said = [t for t in tokens(norm(" ".join(probe["turns"])))
            if t not in DIALOGUE_STOP and len(t) > 2]
    have = set(tokens(store.text[doc]))
    missing = [t for t in said if t not in have and index.df(t) > 0]
    unknown = [t for t in said if index.df(t) == 0]
    state = agent._sessions.get(f"stress::{probe['id']}")
    reasons = []
    if state is not None:
        target_cat = norm(store.category[doc])
        if state.category_key and state.category_key not in target_cat:
            reasons.append(f"category resolved to '{state.category_key}', "
                           f"target is in '{target_cat[:40]}'")
        if state.category_docs is not None and doc not in set(state.category_docs):
            reasons.append("target not in the resolved category bucket")
    if unknown:
        reasons.append(f"words absent from the catalog entirely: {unknown[:4]}")
    if missing:
        reasons.append(f"words in the catalog but not on the target: {missing[:5]}")
    if not reasons:
        reasons.append("target matched but outranked")
    return "; ".join(reasons)


# ---------------------------------------------------------------------------
def cmd_natural(args) -> None:
    # Parsed before anything expensive happens. A malformed --vs discovered
    # after a 427-probe base run has already cost the run it invalidates.
    specs = [parse_variant(s) for s in getattr(args, "vs", [])]
    print(f"{DIM}loading catalog…{RESET}")
    base_settings = settings_from(args)
    agent = Agent(config.CATALOG_PATH, base_settings)
    store, index = agent.store, agent.index
    print(f"{DIM}building independent BM25 control…{RESET}")
    bm25 = Bm25(store)

    source = load_probes(getattr(args, "probes", None))
    missing = [p for p in source if p["target"] not in store.ord_of]
    if missing:
        print(f"{RED}targets not in catalog: {[p['id'] for p in missing]}{RESET}")
    probes = [p for p in source if p["target"] in store.ord_of]
    docs_of = [store.ord_of[p["target"]] for p in probes]

    # Every run below walks `probes` in this one order, so results line up
    # position-for-position and the paired bootstrap is valid by construction.
    runs: dict[str, list[dict]] = {}
    runs[BASE] = [run_agent_probe(agent, p, d) for p, d in zip(probes, docs_of)]
    runs[CONTROL] = [run_oneshot(bm25.search, p, d) for p, d in zip(probes, docs_of)]
    runs[MATCHED] = [run_turnmatched(bm25.search, p, d)
                     for p, d in zip(probes, docs_of)]
    runs[ORACLE] = [
        run_oneshot(bm25.search,
                    # The ceiling: query with the product's own words, which is
                    # the advantage the official simulator hands us for free.
                    {"turns": [" ".join((store.raw_title[d], store.text[d][:600]))]},
                    d)
        for d in docs_of
    ]

    # Extra configurations, each a full agent over the identical probes. Built
    # and dropped one at a time: two catalogs in memory is the price of one
    # comparison, and there is no reason to pay it N times over.
    variants: list[str] = []
    for label, overrides in specs:
        print(f"{DIM}running variant {label}: {overrides}…{RESET}")
        other = Agent(config.CATALOG_PATH, settings_from(args, overrides))
        runs[label] = [run_agent_probe(other, p, d) for p, d in zip(probes, docs_of)]
        variants.append(label)
        del other

    overlaps = [verbatim_overlap(p, store, d) for p, d in zip(probes, docs_of)]

    print(f"\n{BOLD}NATURAL-LANGUAGE STRESS TEST{RESET}   "
          f"{len(probes)} probes, official metric")
    print(f"{DIM}  targets chosen before any wording was written; nothing tuned "
          f"on this set{RESET}\n")
    for label in [BASE] + variants + [MATCHED, CONTROL, ORACLE]:
        print(line(label, summarize(runs[label])))

    mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
    print(f"\n{DIM}  mean verbatim overlap with the target: {mean_overlap:.2f}"
          f"  (1.00 would mean the probes quote the product page){RESET}")

    # ---- is the difference real? ----------------------------------------
    # A point estimate cannot tell a 0.015 effect from a 0.015 resample. These
    # are paired: one bootstrap draw picks a set of probe indices and scores
    # both configs on that same set, so the probe-to-probe variance that
    # dominates an unpaired interval cancels out.
    if not args.no_ci:
        iters, seed = args.boot_iters, args.boot_seed
        ci = score_ci(sessions_from(runs[BASE]), iters=iters, seed=seed)
        print(f"\n{BOLD}PAIRED BOOTSTRAP{RESET}  {DIM}{iters} resamples over the "
              f"{len(probes)} probes, seed {seed}{RESET}")
        print(f"  {BASE:<26} {ci.point:.4f}   95% CI "
              f"[{ci.lo:.4f}, {ci.hi:.4f}]")
        pairs = [(CONTROL, BASE), (MATCHED, BASE)] + [(BASE, v) for v in variants]
        for left, right in pairs:
            d = paired_delta_ci(sessions_from(runs[left]), sessions_from(runs[right]),
                                iters=iters, seed=seed)
            print("  " + delta_line(f"{right} − {left}", d))

    print(f"\n{BOLD}BY TAG{RESET}  {DIM}(agent only; small n, read as direction "
          f"not measurement){RESET}")
    by_tag: dict[str, list[dict]] = defaultdict(list)
    for probe, result in zip(probes, runs[BASE]):
        for tag in probe["tags"]:
            by_tag[tag].append(result)
    for tag in sorted(by_tag):
        print(line(tag, summarize(by_tag[tag]), width=20))

    if args.crosstab:
        crosstab(probes, runs, [BASE] + variants + [MATCHED, CONTROL])

    if args.show:
        print(f"\n{BOLD}PER PROBE{RESET}")
        for i, probe in enumerate(probes):
            def mark(r):
                if r["rank"] is None:
                    return f"{RED}miss{RESET}"
                colour = GREEN if r["rank"] == 1 else YELLOW
                return f"{colour}r{r['rank']}@t{r['turn']}{RESET}"
            doc, ares = docs_of[i], runs[BASE][i]
            print(f"\n  {BOLD}{probe['id']}{RESET}  "
                  f"{DIM}[{','.join(probe['tags'])}]  overlap {overlaps[i]:.2f}{RESET}")
            print(f"    {DIM}target {store.raw_title[doc][:66]}{RESET}")
            print(f"    agent {mark(ares)}    bm25 {mark(runs[CONTROL][i])}    "
                  f"oracle {mark(runs[ORACLE][i])}")
            why = diagnose(agent, probe, store, index, doc, ares)
            if why:
                print(f"    {DIM}why: {why}{RESET}")


def cmd_vocab(args) -> None:
    """Track V: how much of the score rests on rare verbatim tokens?

    A hand-written synonym map was tried first and **measured nothing**
    (0.9383 → 0.9261 at nominally 100% substitution). The map was the problem,
    not the finding: it rewrites `polyester` but leaves the rest of
    "44% Acrylic, 28% Cotton, 20% Merino Wool, 8% Polyester" standing, and the
    bare numbers are already unique. Any test whose result depends on which
    words the author happened to map is not measuring the system.

    This replaces it with something the author cannot tilt. Rank each
    constraint's tokens by IDF and delete the rarest `q` share of them. A real
    shopper does not say `8929`, `NM-M`, or `44`; they say the common words. The
    curve is therefore a direct measure of how much of the score is carried by
    tokens no person would ever type.

    Deletion, not substitution: what replaces a word is a judgement call, and
    what remains after deleting it is not.
    """
    from evaluator.local_evaluator import metric_summary
    from tools.harness import composite, load_world
    from tools.sim import RENDERERS, run_session

    samples, ids, categories, products = load_world()
    agent = Agent(config.CATALOG_PATH, config.DEFAULT)
    index = agent.index

    def strip_rarest(value: str, quantile: float) -> str:
        """Drop the `quantile` share of this constraint's rarest tokens."""
        words = value.split()
        if quantile <= 0.0 or len(words) < 2:
            return value
        ranked = sorted(range(len(words)),
                        key=lambda i: index.idf(norm(words[i]) or " "),
                        reverse=True)
        drop = set(ranked[: max(1, round(len(words) * quantile))])
        kept = [w for i, w in enumerate(words) if i not in drop]
        return " ".join(kept) or words[-1]

    print(f"\n{BOLD}RARE-TOKEN ABLATION{RESET}  "
          f"{DIM}official public set; the rarest words are deleted from every "
          f"constraint{RESET}")
    print(f"{DIM}  measures how much of the score depends on words a shopper "
          f"would never type{RESET}\n")
    print(f"  {'dropped':<10}{'score':<10}{'hit@10':<10}{'mrr':<10}{'mttc'}")

    base_renderer = RENDERERS["L0"]
    for quantile in (0.0, 0.2, 0.4, 0.6, 0.8):
        class Ablated(base_renderer):                    # type: ignore[misc]
            name = f"rare-token ablation {quantile:.0%}"

            def open_buy(self, cat, c):
                return super().open_buy(cat, strip_rarest(c, quantile))

            def reply(self, matches):
                return super().reply([strip_rarest(m, quantile) for m in matches])

            def override(self, new):
                return super().override(strip_rarest(new, quantile))

        rows = [run_session(agent, s, products, categories, Ablated())
                for s in samples]
        m = metric_summary(rows)
        score, _ = composite(m)
        print(f"  {quantile:<10.0%}{score:<10.4f}{m['hit_rate_at_10']:<10.3f}"
              f"{m['mrr']:<10.3f}{m['mttc']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent stress test")
    parser.add_argument("--track", default="all",
                        choices=["all", "natural", "vocab"])
    parser.add_argument("--retrieval", default="conjunctive",
                        choices=["conjunctive", "bm25", "rrf", "auto"])
    parser.add_argument("--resolver", default=None,
                        choices=["overlap", "vote", "classifier", "ensemble"],
                        help="category-resolution fallback (default: config)")
    parser.add_argument("--gate-category-bonus", action="store_true",
                        help="scale the category bonus by resolution confidence")
    parser.add_argument("--reresolve", action="store_true",
                        help="re-resolve the category on an override turn")
    parser.add_argument("--set", dest="sets", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="override any Settings field (repeatable)")
    parser.add_argument("--probes", default=None, metavar="PATH",
                        help="a generated probe set from tools.genprobes "
                             "(default: the 26 hand-authored probes)")
    parser.add_argument("--show", action="store_true",
                        help="per-probe detail and failure reasons")
    parser.add_argument("--vs", action="append", default=[], metavar="SPEC",
                        help="a second configuration to run over the SAME probes, "
                             "as 'label:key=value,key=value' (repeatable). Each is "
                             "reported with a paired bootstrap delta against the "
                             "base run.")
    parser.add_argument("--crosstab", action="store_true",
                        help="score per tag per configuration, winner marked")
    parser.add_argument("--boot-iters", type=int, default=10000,
                        help="bootstrap resamples (default 10000)")
    parser.add_argument("--boot-seed", type=int, default=0,
                        help="bootstrap seed, so a reported interval reproduces")
    parser.add_argument("--no-ci", action="store_true",
                        help="skip the bootstrap block")
    args = parser.parse_args()
    if args.track in ("all", "natural"):
        cmd_natural(args)
    if args.track in ("all", "vocab"):
        cmd_vocab(args)


if __name__ == "__main__":
    main()
