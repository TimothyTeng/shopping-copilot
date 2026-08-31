"""A local web UI for the agent: shop by typing, or watch the suites run.

    python3 -m tools.webapp            # then open http://127.0.0.1:8000

Standard library only, like everything on the scored path — `http.server` plus
one embedded HTML page. Nothing here is imported by `src/`; this is a viewer for
the same code the harness scores, not a second implementation of it.

Two surfaces, and they are deliberately not the same agent:

* **Shop** is the free-text surface. It runs the `demo chat` configuration —
  `retrieval=bm25`, fuzzy repair and doc2query on, hold-back gate off — because
  a person types their own words and just wants to see products. The conjunctive
  ordering measures ~0.24 worse on exactly this input.
* **Tests** build a *fresh* agent per run at the graded defaults, so a benchmark
  run here reproduces `tools.harness run` rather than scoring the chat config by
  accident. That costs a ~10 s index build per run and is worth it; the agent is
  dropped again when the run finishes.

Per-item pass/fail, the questions asked and the answers produced all come from
`tools.transcripts`, which drives the official session loop turn by turn.
"""
from __future__ import annotations

import json
import random
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.agent import Agent  # noqa: E402

TOP_K = 10
MAX_TURNS = 10

# --------------------------------------------------------------------------
# The chat agent: built once, lazily, and shared by every browser session.
# --------------------------------------------------------------------------
_CHAT_LOCK = threading.Lock()
_CHAT_AGENT: Agent | None = None
_CHAT_STATE = {"loading": False, "error": ""}

# Browser sessions. `target` is set only when the shopper pressed "surprise me",
# in which case we can tell them whether the product they picked was the one.
SESSIONS: dict[str, dict] = {}

# Test runs, keyed by id. Each holds rows as they complete so the page can poll.
RUNS: dict[str, dict] = {}
RUNS_LOCK = threading.Lock()


def chat_agent() -> Agent:
    """The interactive agent, built on first use.

    Mirrors `tools.demo chat` exactly. The hold-back gate is off: it exists
    because a *scored* session ends at the first hit, so an early weak list
    locks in a bad rank — a person is under no such rule and wants results.
    """
    global _CHAT_AGENT
    with _CHAT_LOCK:
        if _CHAT_AGENT is None:
            _CHAT_STATE["loading"] = True
            try:
                _CHAT_AGENT = Agent(
                    config.CATALOG_PATH,
                    config.DEFAULT.replace(
                        gate_enabled=False,
                        fuzzy_repair=True,
                        doc2query_expansions=True,
                        retrieval="bm25",
                    ),
                )
            except Exception as exc:                      # surfaced in the UI
                _CHAT_STATE["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                _CHAT_STATE["loading"] = False
        return _CHAT_AGENT


def warm_up() -> None:
    """Start building the chat agent in the background at server start, so the
    first person to type is not waiting on a 10 s index build."""
    threading.Thread(target=chat_agent, daemon=True).start()


# --------------------------------------------------------------------------
# Shop
# --------------------------------------------------------------------------
def api_random() -> dict:
    """A random catalog product for the shopper to describe in their own words.

    This is the honest version of the exercise: they see the product page, not
    its search text, and have to say what they want without quoting it — which
    is the whole difference between the benchmark and free text.
    """
    agent = chat_agent()
    store = agent.store
    doc = random.randrange(len(store))
    price = store.price[doc]
    return {
        "asin": store.ids[doc],
        "title": store.raw_title[doc],
        "category": " › ".join(store.cat_path[doc][-3:]),
        "price": None if price != price else round(float(price), 2),   # NaN check
        "rating": round(float(store.rating_avg[doc]), 2) or None,
        "ratings": int(store.rating_n[doc]),
    }


def api_new_session(target: str | None) -> dict:
    """Open a browser session, optionally against a known target product."""
    agent = chat_agent()
    sid = f"web::{uuid.uuid4().hex[:12]}"
    agent.reset(sid, {"summary": "interactive user", "preference_tags": []})
    SESSIONS[sid] = {"turn": 0, "target": target, "done": False}
    return {"session": sid}


def api_say(sid: str, text: str) -> dict:
    """One turn of conversation: the agent's question plus its current top 10."""
    agent = chat_agent()
    session = SESSIONS.get(sid)
    if session is None:
        raise KeyError("unknown session — reload the page")
    if session["turn"] >= MAX_TURNS:
        return {"turn": session["turn"], "exhausted": True, "question": "",
                "results": [], "clues": [], "category": None}

    session["turn"] += 1
    turn = session["turn"]
    response = agent.respond(sid, text, turn, TOP_K)
    state = agent._sessions[sid]
    store = agent.store

    results = []
    for rank, item in enumerate(response.get("recommendations") or [], 1):
        asin = str(item["parent_asin"])
        doc = store.ord_of.get(asin)
        results.append({
            "rank": rank,
            "asin": asin,
            "title": store.raw_title[doc] if doc is not None else "(unknown)",
            "category": " › ".join(store.cat_path[doc][-2:]) if doc is not None else "",
            "is_target": bool(session["target"]) and asin == session["target"],
        })

    return {
        "turn": turn,
        "question": response.get("message", ""),
        "ask_attribute": response.get("ask_attribute"),
        # What the agent actually understood — the useful part when it is wrong.
        "category": state.category_key,
        "clues": [s.key for s in state.active_slots()],
        "results": results,
        "holding_back": not results,
        "exhausted": turn >= MAX_TURNS,
    }


def api_pick(sid: str, asin: str) -> dict:
    """The shopper says "that one". Ends the session and scores it if we know
    what they were sent to find."""
    agent = chat_agent()
    session = SESSIONS.get(sid)
    if session is None:
        raise KeyError("unknown session — reload the page")
    session["done"] = True
    store = agent.store
    doc = store.ord_of.get(asin)
    target = session.get("target")
    return {
        "turns": session["turn"],
        "picked": store.raw_title[doc] if doc is not None else asin,
        # None when the shopper typed their own request: there is no ground
        # truth then, and claiming one would be a lie.
        "correct": None if not target else bool(asin == target),
        "target_title": (store.raw_title[store.ord_of[target]]
                         if target and target in store.ord_of else None),
    }


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def _score(rows: list[dict]) -> dict:
    """The official composite over completed rows (a miss counts as turn 11)."""
    n = len(rows) or 1
    hits = [r for r in rows if r["rank"]]
    mttc = sum(r["turn"] for r in rows) / n
    hit = len(hits) / n
    mrr = sum(1.0 / r["rank"] for r in hits) / n
    eff = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return {"n": len(rows), "hit": hit, "mrr": mrr, "mttc": mttc,
            "score": 0.50 * hit + 0.30 * mrr + 0.20 * eff,
            "passed": len(hits), "failed": len(rows) - len(hits)}


def _run_tests(run_id: str, suite: str, n: int, mode: str) -> None:
    """Background worker: play sessions one at a time, publishing each result.

    Uses `tools.transcripts`, which drives the official session loop — including
    the rule that a hit before an `intent_override` lands does not count — so a
    row that passes here is a row the evaluator would have scored.
    """
    from tools import transcripts as T
    from tools.probes import PROBES
    from tools.sim import KIT_ROOT, RENDERERS
    from evaluator.local_evaluator import coarse_category, load_jsonl

    run = RUNS[run_id]
    try:
        run["stage"] = "building the index"
        # A fresh agent at the graded defaults, with ONLY the retrieval mode
        # varied — never the chat configuration. Folding the chat extras
        # (fuzzy repair, doc2query) in here would quietly inflate the stress
        # score by ~0.09 against what `tools.stress --retrieval bm25` reports,
        # and a viewer that disagrees with the suite it claims to run is worse
        # than no viewer.
        agent = Agent(config.CATALOG_PATH, config.DEFAULT.replace(retrieval=mode))

        if suite == "benchmark":
            run["stage"] = "loading the public set"
            products, categories = {}, {}
            with (KIT_ROOT / "data" / "catalog.jsonl").open(encoding="utf-8") as fh:
                for line in fh:
                    p = json.loads(line)
                    asin = str(p["parent_asin"])
                    products[asin] = p
                    categories[asin] = [str(v) for v in p.get("categories") or []]
            samples = load_jsonl(KIT_ROOT / "data" / "public_set.jsonl")
            # Stratified so a short run still mirrors the real scenario mix
            # rather than whichever types a flat sample happens to draw.
            by_type: dict[str, list] = {}
            for s in samples:
                by_type.setdefault(s["scenario_type"], []).append(s)
            rng = random.Random(0)
            picked: list = []
            for kind, pool in sorted(by_type.items()):
                take = max(1, round(n * len(pool) / len(samples)))
                picked += rng.sample(pool, min(take, len(pool)))
            picked = picked[:n]
            renderer = RENDERERS["L0"]()

            run["total"] = len(picked)
            run["stage"] = "running sessions"
            for sample in picked:
                row = T.run_benchmark(agent, sample, products, categories, renderer)
                row["passed"] = row["rank"] is not None
                _publish(run_id, row)
        else:
            run["total"] = len(PROBES)
            run["stage"] = "running probes"
            for probe in PROBES:
                row = T.run_probe(agent, probe)
                row["passed"] = row["rank"] is not None
                _publish(run_id, row)

        run["summary"] = _score(run["rows"])
        run["stage"] = "done"
    except Exception:
        run["error"] = traceback.format_exc(limit=4)
        run["stage"] = "failed"
    finally:
        run["done"] = True


def _publish(run_id: str, row: dict) -> None:
    """Append one finished session so the page can show it immediately."""
    with RUNS_LOCK:
        RUNS[run_id]["rows"].append(row)
        RUNS[run_id]["summary"] = _score(RUNS[run_id]["rows"])


def api_tests_start(suite: str, n: int, mode: str) -> dict:
    run_id = uuid.uuid4().hex[:12]
    RUNS[run_id] = {"suite": suite, "mode": mode, "rows": [], "total": n,
                    "stage": "queued", "done": False, "error": "",
                    "summary": None, "started": time.time()}
    threading.Thread(target=_run_tests, args=(run_id, suite, n, mode),
                     daemon=True).start()
    return {"run": run_id}


def api_tests_status(run_id: str, since: int) -> dict:
    """Poll a run. `since` is how many rows the page already has, so a long run
    does not re-send its whole history on every poll."""
    run = RUNS.get(run_id)
    if run is None:
        raise KeyError("unknown run")
    with RUNS_LOCK:
        rows = run["rows"][since:]
        return {"stage": run["stage"], "done": run["done"], "error": run["error"],
                "total": run["total"], "count": len(run["rows"]),
                "summary": run["summary"], "rows": rows,
                "elapsed": round(time.time() - run["started"], 1)}


# --------------------------------------------------------------------------
# HTTP plumbing
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "ShoppingCopilot/1.0"

    def log_message(self, fmt, *args):     # quieter than the default access log
        pass

    # -- helpers ----------------------------------------------------------
    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length) or b"{}")

    def _guard(self, fn) -> None:
        """Run a handler, turning any failure into JSON the page can display."""
        try:
            self._send(fn())
        except KeyError as exc:
            self._send({"error": str(exc)}, 400)
        except Exception as exc:
            self._send({"error": f"{type(exc).__name__}: {exc}"}, 500)

    # -- routes -----------------------------------------------------------
    def do_GET(self) -> None:
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if url.path in ("/", "/index.html"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/api/status":
            ready = _CHAT_AGENT is not None
            self._send({"ready": ready, "loading": _CHAT_STATE["loading"],
                        "error": _CHAT_STATE["error"],
                        "products": len(_CHAT_AGENT.store) if ready else 0})
        elif url.path == "/api/random":
            self._guard(api_random)
        elif url.path == "/api/tests/status":
            run_id = (query.get("run") or [""])[0]
            since = int((query.get("since") or ["0"])[0])
            self._guard(lambda: api_tests_status(run_id, since))
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self) -> None:
        url = urlparse(self.path)
        body = self._body()
        if url.path == "/api/session":
            self._guard(lambda: api_new_session(body.get("target")))
        elif url.path == "/api/say":
            self._guard(lambda: api_say(body.get("session", ""),
                                        str(body.get("text", ""))))
        elif url.path == "/api/pick":
            self._guard(lambda: api_pick(body.get("session", ""),
                                         str(body.get("asin", ""))))
        elif url.path == "/api/tests/start":
            self._guard(lambda: api_tests_start(
                str(body.get("suite", "stress")),
                max(1, min(200, int(body.get("n", 20)))),
                str(body.get("mode", "conjunctive"))))
        else:
            self._send({"error": "not found"}, 404)


# --------------------------------------------------------------------------
# The page. One file, no build step, no CDN — it has to work offline, like the
# rest of the scored path.
# --------------------------------------------------------------------------
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Shopping Copilot</title>
<style>
  :root {
    --ground:#eff2f3; --card:#fff; --card-2:#f7f9fa; --ink:#10171c; --muted:#5c6a74;
    --faint:#8797a2; --line:#d8e0e4; --accent:#0d5e63; --accent-soft:#d9ecec;
    --target:#9a5209; --target-soft:#fbeedc; --good:#1c6a46; --good-soft:#dff0e6;
    --miss:#9a2f2b; --miss-soft:#f9e3e2;
    --shadow:0 1px 2px rgba(16,23,28,.06), 0 8px 24px -18px rgba(16,23,28,.5);
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--ground); color:var(--ink);
    font:15px/1.55 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif; }
  header { padding:20px 24px 0; max-width:1080px; margin:0 auto; }
  h1 { font-size:20px; margin:0 0 2px; letter-spacing:-.01em; }
  .sub { color:var(--muted); font-size:13px; margin:0 0 16px; }
  main { max-width:1080px; margin:0 auto; padding:0 24px 64px; }
  .tabs { display:flex; gap:6px; border-bottom:1px solid var(--line); margin-bottom:20px; }
  .tab { padding:9px 16px; border:0; background:none; cursor:pointer; font:inherit;
    color:var(--muted); border-bottom:2px solid transparent; margin-bottom:-1px; }
  .tab.on { color:var(--accent); border-bottom-color:var(--accent); font-weight:600; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px;
    padding:16px 18px; box-shadow:var(--shadow); margin-bottom:16px; }
  button.act { font:inherit; padding:8px 14px; border-radius:8px; cursor:pointer;
    border:1px solid var(--accent); background:var(--accent); color:#fff; }
  button.act.ghost { background:#fff; color:var(--accent); }
  button.act:disabled { opacity:.45; cursor:default; }
  .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  input[type=text], select { font:inherit; padding:9px 12px; border:1px solid var(--line);
    border-radius:8px; background:#fff; color:inherit; }
  input[type=text] { flex:1; min-width:240px; }
  .muted { color:var(--muted); } .faint { color:var(--faint); }
  .small { font-size:13px; } .tiny { font-size:12px; }
  .mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
  .brief { background:var(--target-soft); border:1px solid #edd6b6; border-radius:10px;
    padding:14px 16px; margin-bottom:16px; }
  .brief h3 { margin:0 0 4px; font-size:15px; }
  .qa { border-left:3px solid var(--accent-soft); padding:2px 0 2px 12px; margin:14px 0 10px; }
  .you { font-weight:600; }
  ol.results { list-style:none; margin:8px 0 0; padding:0; }
  ol.results li { display:flex; gap:10px; align-items:baseline; padding:7px 8px;
    border-radius:8px; cursor:pointer; border:1px solid transparent; }
  ol.results li:hover { background:var(--card-2); border-color:var(--line); }
  ol.results li .n { color:var(--faint); width:22px; flex:none; font-size:12px; }
  ol.results li.hit { background:var(--target-soft); border-color:#edd6b6; }
  .pill { display:inline-block; padding:1px 8px; border-radius:999px; font-size:12px;
    background:var(--card-2); border:1px solid var(--line); color:var(--muted); }
  .pill.pass { background:var(--good-soft); border-color:#bfe0cd; color:var(--good); }
  .pill.fail { background:var(--miss-soft); border-color:#eec7c5; color:var(--miss); }
  table { border-collapse:collapse; width:100%; font-size:13px; }
  th, td { text-align:left; padding:7px 8px; border-bottom:1px solid var(--line);
    vertical-align:top; }
  th { color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase;
    letter-spacing:.04em; }
  tr.clickable { cursor:pointer; } tr.clickable:hover td { background:var(--card-2); }
  .stat { display:inline-block; margin-right:20px; }
  .stat b { display:block; font-size:20px; letter-spacing:-.01em; }
  .stat span { color:var(--muted); font-size:12px; }
  .bar { height:4px; background:var(--line); border-radius:2px; overflow:hidden; margin:10px 0 0; }
  .bar i { display:block; height:100%; background:var(--accent); transition:width .2s; }
  details.turns { margin-top:8px; } details.turns summary { cursor:pointer; color:var(--accent); }
  .hidden { display:none; }
  @media (prefers-color-scheme: dark) {
    :root { --ground:#0f1417; --card:#161d21; --card-2:#1b2429; --ink:#e8eef0;
      --muted:#9bacb5; --faint:#71838d; --line:#2a353b; --accent:#66c7c9;
      --accent-soft:#1d3a3c; --target:#e2a663; --target-soft:#33260f;
      --good:#79d4a4; --good-soft:#14301f; --miss:#e78d88; --miss-soft:#33191a; }
    input[type=text], select { background:#0f1417; }
    button.act.ghost { background:#0f1417; }
    .brief { border-color:#4a3a1c; }
  }
</style>
</head>
<body>
<header>
  <h1>Shopping Copilot</h1>
  <p class="sub">Conversational search over 50,000 products —
    <span id="status" class="faint">starting…</span></p>
</header>
<main>
  <div class="tabs">
    <button class="tab on" data-tab="shop">Shop</button>
    <button class="tab" data-tab="tests">Tests</button>
  </div>

  <!-- ------------------------------------------------------------------ -->
  <section id="shop">
    <div class="row" style="margin-bottom:14px">
      <button class="act ghost" id="surprise">Give me a random item to describe</button>
      <button class="act ghost" id="restart">Start over</button>
      <span class="tiny faint">bm25 retrieval · fuzzy repair on · hold-back off</span>
    </div>

    <div id="brief" class="brief hidden">
      <h3>Describe this, in your own words</h3>
      <div id="brief-title"></div>
      <div class="small muted" id="brief-meta"></div>
      <div class="tiny faint" style="margin-top:6px">
        Don't copy the title — say it the way you'd say it out loud. The agent
        has to bridge your words to the catalog's, which is the hard part.
      </div>
    </div>

    <div class="card">
      <form id="say-form" class="row" autocomplete="off">
        <input type="text" id="say" placeholder="e.g. a warm waterproof jacket for hiking"
               autofocus>
        <button class="act" id="send" type="submit">Send</button>
      </form>
      <div class="tiny faint" style="margin-top:8px">
        Answer the agent's question to narrow it down. Click any suggestion when
        it's the one you meant.
      </div>
    </div>

    <div id="thread"></div>
  </section>

  <!-- ------------------------------------------------------------------ -->
  <section id="tests" class="hidden">
    <div class="card">
      <div class="row">
        <select id="suite">
          <option value="stress">Stress test — 26 hand-written shopper probes</option>
          <option value="benchmark">Benchmark — official simulator sessions</option>
        </select>
        <label class="small muted">sessions
          <input type="text" id="n" value="20" style="width:64px" class="mono"></label>
        <select id="mode">
          <option value="conjunctive">conjunctive (graded default)</option>
          <option value="bm25">bm25 (prose default)</option>
        </select>
        <button class="act" id="run">Run</button>
      </div>
      <div class="tiny faint" style="margin-top:8px">
        Each run builds a fresh agent at the graded defaults with only the
        retrieval mode varied (~10 s), then plays real sessions turn by turn, so
        these numbers match <span class="mono">tools.harness run</span> and
        <span class="mono">tools.stress</span> rather than the chat surface. A row
        passes when the hidden target reaches the top 10 within its ten turns.
        Session count applies to the benchmark; the stress set is always all 26.
      </div>
      <div class="bar hidden" id="bar"><i style="width:0"></i></div>
      <div class="small muted hidden" id="stage" style="margin-top:8px"></div>
    </div>

    <div class="card hidden" id="summary"></div>
    <div class="card hidden" id="rows-card">
      <table><thead><tr>
        <th style="width:80px">Result</th><th>Case</th><th>Target</th>
        <th style="width:60px">Rank</th><th style="width:60px">Turn</th>
      </tr></thead><tbody id="rows"></tbody></table>
    </div>
    <div id="detail"></div>
  </section>
</main>

<script>
const $ = s => document.querySelector(s);
const el = (tag, cls, text) => { const n = document.createElement(tag);
  if (cls) n.className = cls; if (text !== undefined) n.textContent = text; return n; };
const esc = s => (s == null ? "" : String(s));

async function api(path, body) {
  const opts = body ? {method:"POST", headers:{"Content-Type":"application/json"},
                       body: JSON.stringify(body)} : {};
  const r = await fetch(path, opts);
  const data = await r.json();
  if (data.error) throw new Error(data.error);
  return data;
}

/* -- tabs ------------------------------------------------------------- */
document.querySelectorAll(".tab").forEach(t => t.onclick = () => {
  document.querySelectorAll(".tab").forEach(x => x.classList.toggle("on", x === t));
  $("#shop").classList.toggle("hidden", t.dataset.tab !== "shop");
  $("#tests").classList.toggle("hidden", t.dataset.tab !== "tests");
});

/* -- status ----------------------------------------------------------- */
async function poll_status() {
  try {
    const s = await api("/api/status");
    if (s.error) { $("#status").textContent = "failed to load: " + s.error; return; }
    if (s.ready) {
      $("#status").textContent = s.products.toLocaleString() + " products indexed, ready";
      return;
    }
    $("#status").textContent = "building the index (about 10 seconds)…";
  } catch (e) { $("#status").textContent = "server not responding"; }
  setTimeout(poll_status, 1200);
}
poll_status();

/* -- shop ------------------------------------------------------------- */
let session = null, target = null, finished = false;

async function new_session(t) {
  const r = await api("/api/session", {target: t || null});
  session = r.session; target = t || null; finished = false;
  $("#thread").innerHTML = "";
  $("#say").value = ""; $("#say").focus();
}

$("#surprise").onclick = async () => {
  $("#surprise").disabled = true;
  try {
    const p = await api("/api/random");
    await new_session(p.asin);
    $("#brief").classList.remove("hidden");
    $("#brief-title").textContent = p.title;
    const bits = [p.category];
    if (p.price) bits.push("$" + p.price.toFixed(2));
    if (p.ratings) bits.push(p.rating + "★ (" + p.ratings.toLocaleString() + ")");
    $("#brief-meta").textContent = bits.filter(Boolean).join(" · ");
  } catch (e) { alert(e.message); }
  $("#surprise").disabled = false;
};

$("#restart").onclick = async () => {
  $("#brief").classList.add("hidden");
  await new_session(null);
};

$("#say-form").onsubmit = async ev => {
  ev.preventDefault();
  const text = $("#say").value.trim();
  if (!text) return;
  if (!session) await new_session(target);
  $("#send").disabled = true; $("#say").value = "";
  try {
    const r = await api("/api/say", {session, text});
    render_turn(text, r);
  } catch (e) { alert(e.message); }
  $("#send").disabled = false; $("#say").focus();
};

function render_turn(said, r) {
  const card = el("div", "card");
  const qa = el("div", "qa");
  qa.appendChild(el("div", "you", "You · turn " + r.turn));
  qa.appendChild(el("div", "small", said));
  card.appendChild(qa);

  if (r.question) {
    card.appendChild(el("div", null, r.question));
    const meta = [];
    if (r.ask_attribute) meta.push("asking about: " + r.ask_attribute);
    if (r.category) meta.push("category: " + r.category);
    if (r.clues && r.clues.length) meta.push("clues: " + r.clues.join(", "));
    if (meta.length) card.appendChild(el("div", "tiny faint", meta.join("  ·  ")));
  }

  if (r.holding_back) {
    card.appendChild(el("div", "small muted", "(nothing worth showing yet — answer the question)"));
  } else {
    const ol = el("ol", "results");
    r.results.forEach(item => {
      const li = el("li", item.is_target ? "hit" : null);
      li.appendChild(el("span", "n", "#" + item.rank));
      const box = el("div");
      box.appendChild(el("div", null, item.title));
      box.appendChild(el("div", "tiny faint", item.category + " · " + item.asin));
      li.appendChild(box);
      li.title = "Click if this is the one you meant";
      li.onclick = () => pick(item.asin);
      ol.appendChild(li);
    });
    card.appendChild(ol);
  }
  if (r.exhausted) card.appendChild(el("div", "small muted", "Ten turns used — that is the limit a scored session gets."));
  $("#thread").prepend(card);
}

async function pick(asin) {
  if (finished) return;
  finished = true;
  const r = await api("/api/pick", {session, asin});
  const card = el("div", "card");
  let verdict = "Session closed after " + r.turns + (r.turns === 1 ? " turn." : " turns.");
  if (r.correct === true) verdict = "Correct — found in " + r.turns +
      (r.turns === 1 ? " turn." : " turns.");
  if (r.correct === false) verdict = "That is not the item you were given. The target was: " +
      r.target_title;
  card.appendChild(el("div", r.correct === false ? "pill fail" : "pill pass", verdict));
  card.appendChild(el("div", "small", "You picked: " + r.picked));
  $("#thread").prepend(card);
}

/* -- tests ------------------------------------------------------------ */
let run_id = null, have = 0, rows = [];

$("#run").onclick = async () => {
  $("#run").disabled = true;
  rows = []; have = 0;
  $("#rows").innerHTML = ""; $("#detail").innerHTML = "";
  $("#summary").classList.add("hidden");
  $("#rows-card").classList.remove("hidden");
  $("#bar").classList.remove("hidden"); $("#bar i").style.width = "0";
  $("#stage").classList.remove("hidden"); $("#stage").textContent = "starting…";
  try {
    const r = await api("/api/tests/start", {
      suite: $("#suite").value, n: parseInt($("#n").value || "20", 10),
      mode: $("#mode").value });
    run_id = r.run;
    poll_run();
  } catch (e) { alert(e.message); $("#run").disabled = false; }
};

async function poll_run() {
  let s;
  try { s = await api("/api/tests/status?run=" + run_id + "&since=" + have); }
  catch (e) { $("#stage").textContent = e.message; $("#run").disabled = false; return; }

  s.rows.forEach(add_row);
  have = s.count;
  const pct = s.total ? Math.round(100 * s.count / s.total) : 0;
  $("#bar i").style.width = pct + "%";
  $("#stage").textContent = s.stage + " — " + s.count + "/" + s.total +
                            " · " + s.elapsed + "s";
  if (s.summary) show_summary(s.summary);
  if (s.error) {
    $("#stage").textContent = "failed";
    const pre = el("pre", "small mono"); pre.textContent = s.error;
    $("#detail").appendChild(pre);
  }
  if (s.done) { $("#run").disabled = false; return; }
  setTimeout(poll_run, 600);
}

function show_summary(m) {
  const c = $("#summary");
  c.classList.remove("hidden");
  c.innerHTML = "";
  const mk = (v, label) => { const d = el("div", "stat");
    d.appendChild(el("b", null, v)); d.appendChild(el("span", null, label)); return d; };
  c.appendChild(mk(m.score.toFixed(4), "composite score"));
  c.appendChild(mk(m.passed + " / " + m.n, "passed"));
  c.appendChild(mk(m.failed, "failed"));
  c.appendChild(mk(m.hit.toFixed(3), "hit@10"));
  c.appendChild(mk(m.mrr.toFixed(3), "MRR"));
  c.appendChild(mk(m.mttc.toFixed(2), "MTTC"));
}

function add_row(row) {
  rows.push(row);
  const tr = el("tr", "clickable");
  const verdict = el("td");
  verdict.appendChild(el("span", "pill " + (row.passed ? "pass" : "fail"),
                         row.passed ? "PASS" : "FAIL"));
  tr.appendChild(verdict);
  const who = el("td");
  who.appendChild(el("div", "mono tiny", row.id));
  who.appendChild(el("div", "tiny faint", row.scenario || ""));
  tr.appendChild(who);
  tr.appendChild(el("td", "small", row.target_title || row.target));
  tr.appendChild(el("td", "mono", row.rank == null ? "—" : String(row.rank)));
  tr.appendChild(el("td", "mono", row.rank == null ? "miss" : String(row.turn)));
  tr.onclick = () => show_detail(row);
  $("#rows").appendChild(tr);
}

function show_detail(row) {
  const box = $("#detail");
  box.innerHTML = "";
  const card = el("div", "card");
  const head = el("div");
  head.appendChild(el("span", "pill " + (row.passed ? "pass" : "fail"),
                      row.passed ? "PASS" : "FAIL"));
  head.appendChild(el("span", "small", "  " + row.id + " · " + (row.scenario || "")));
  card.appendChild(head);
  card.appendChild(el("div", "small muted", "Target: " + (row.target_title || row.target)));
  if (row.hard && row.hard.length)
    card.appendChild(el("div", "tiny faint", "hidden requirements: " + row.hard.join(" · ")));
  if (row.soft && row.soft.length)
    card.appendChild(el("div", "tiny faint", "soft preferences: " + row.soft.join(" · ")));

  row.turns.forEach(t => {
    const qa = el("div", "qa");
    qa.appendChild(el("div", "you", "shopper · turn " + t.turn));
    qa.appendChild(el("div", "small", t.shopper));
    if (t.agent) qa.appendChild(el("div", "small muted", "agent: " + t.agent));
    const meta = [];
    if (t.ask_attribute) meta.push("asks: " + t.ask_attribute);
    if (t.category_key) meta.push("category: " + t.category_key);
    if (t.clues && t.clues.length) meta.push("clues: " + t.clues.join(", "));
    if (meta.length) qa.appendChild(el("div", "tiny faint", meta.join("  ·  ")));

    if (t.held_back) {
      qa.appendChild(el("div", "tiny faint", "(held back — no list emitted this turn)"));
    } else {
      const ol = el("ol", "results");
      t.results.slice(0, 10).forEach(r => {
        const li = el("li", r.is_target ? "hit" : null);
        li.appendChild(el("span", "n", "#" + r.rank));
        const d = el("div");
        d.appendChild(el("div", "small", r.title));
        if (r.is_target) d.appendChild(el("div", "tiny", "← the hidden target"));
        li.appendChild(d);
        li.style.cursor = "default";
        ol.appendChild(li);
      });
      qa.appendChild(ol);
    }
    card.appendChild(qa);
  });
  box.appendChild(card);
  card.scrollIntoView({behavior:"smooth", block:"start"});
}
</script>
</body>
</html>
"""


def main() -> None:
    """Serve the page on localhost and start warming the index."""
    import argparse
    ap = argparse.ArgumentParser(description="Shopping Copilot web UI")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    warm_up()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"  serving on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    print("  building the catalog index in the background…")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")


if __name__ == "__main__":
    main()
