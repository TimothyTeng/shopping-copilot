"""HyDE: answer the query with a fabricated product, then retrieve with that.

The idea is not speculative here — `tools/stress.py` already measures its
ceiling. Three systems run on the same 26 natural-language probes:

    agent (lexical)                    0.3985
    BM25 over the raw transcript       0.4775
    oracle: the product's OWN words    0.9827   hit@10 1.000

The oracle is not a baseline, it is a specification. It says every target is
trivially findable *once the query is phrased in catalog vocabulary*, and that
the entire remaining gap is the translation from shopper words to product words.
A shopper writes "comfy trainers I can wear to the gym, need my feet to
breathe"; the catalog writes "Breathable Mesh Upper, Cushioned Footbed,
Lace-Up Athletic Sneaker". No amount of scoring fixes a vocabulary mismatch.

So: ask the model to write the Amazon listing it thinks the shopper is
describing, and search with that instead. It approximates the oracle without
being able to cheat — the oracle knows the target, this only knows the register.

Two properties make it safe rather than hallucination-driven:

* **Every generated token is verified against the index** before it is used
  (`Agent._ground`). A word the catalog does not contain, or one so common it
  carries no signal, is discarded. This is the same discipline `src/fuzzy.py`
  applies to shopper typos.
* **The result is an additive bonus, never a constraint.** See `config.hyde_bonus`
  and `Ranker.rank` — it is wired exactly like `category_bonus`, so a bad
  generation costs ranking precision and can never empty the conjunctive pool.
  Adding generated terms as *slots* would put them in the AND, which is the
  documented "constraints only accumulate" failure mode.

Transport is `urllib` against any OpenAI-compatible endpoint, so there is no
third-party runtime dependency to declare. Configuration is environment-only:

    COPILOT_LLM_BASE    default http://localhost:30800/v1
    COPILOT_LLM_MODEL   default: ask the server what it serves and take the
                        best-measured one (see PREFERRED_MODELS)
    COPILOT_LLM_KEY     optional; omitted for a local vLLM server
    COPILOT_LLM_CACHE   default copilot/.hyde_cache.json
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import urllib.request
from pathlib import Path

from . import Expansion, client

COPILOT_ROOT = Path(__file__).resolve().parents[2]

SYSTEM = (
    "You write Amazon product listings for the Clothing, Shoes & Jewelry "
    "department. Given what a shopper said, write the listing for the single "
    "product they are most likely describing. Reply with exactly four lines: "
    "line 1 is the product title, lines 2-4 are short feature phrases. Use "
    "ordinary listing vocabulary - department, material, closure, sole, fit, "
    "lining. No preamble, no markdown, no bullet characters, no explanation."
)


# Preference order for picking a model when the environment does not name one,
# best measured first. A *coder* model beat an instruct model of four times the
# size at writing retail copy — the task rewards formulaic listing structure, not
# prose — so the ordering is by measurement, not by parameter count:
#
#   Qwen3-Coder-30B-A3B   0.6857  hit 0.885  865 ms   (n=26 hand-authored set)
#   Qwen2.5-7B-Instruct   0.6599  hit 0.846  1993 ms
#
# Matching is a case-insensitive substring against whatever `/v1/models` reports,
# so it works against vLLM ids ("Qwen/Qwen3-Coder-30B-A3B-Instruct") and ollama
# tags ("qwen3-coder:30b") alike.
#
# CAVEAT ON THAT ORDERING: the coder-beats-instruct result is n=26 on the
# hand-authored set, which is below this project's own bar (see CLAUDE.md,
# "report the interval or do not report the delta"). It has NOT been re-run
# against the n=427 generated set with a paired bootstrap, because the two
# models cannot be served simultaneously on the development machine — a 30B at
# ~46 GB does not fit beside a running vLLM 7B in 121 GB of unified memory. The
# preference order is therefore the best available evidence, not a settled one.
# To settle it, serve both and run:
#
#   python -m tools.stress --track natural --probes data/probes_generated.jsonl \
#       --retrieval bm25 \
#       --vs "7b:backend=hyde,hyde_base=http://localhost:30800/v1" \
#       --vs "30b:backend=hyde,hyde_base=http://localhost:11434/v1"
#
# `hyde_base` / `hyde_model` exist so both run in ONE process over the identical
# probes, which is what makes the bootstrap paired.
PREFERRED_MODELS = (
    "qwen3-coder",
    "qwen2.5-coder",
    "qwen2.5-32b",
    "qwen2.5-7b",
)
FALLBACK_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def resolve_model(base: str, timeout: float = 2.0) -> str:
    """Which model to talk to: the environment's, else the best one served.

    Asking the server what it has is what makes "use the better model" a
    property of the deployment rather than a constant somebody has to remember
    to edit. Never raises and never blocks for long — an unreachable server
    yields the documented default, and the backend then fails its calls
    silently, exactly as it does today.
    """
    named = os.environ.get("COPILOT_LLM_MODEL")
    if named:
        return named
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/models", timeout=timeout) as response:
            served = [str(row.get("id", "")) for row in
                      (json.loads(response.read().decode("utf-8")).get("data") or [])]
    except Exception:
        return FALLBACK_MODEL
    served = [name for name in served if name]
    if not served:
        return FALLBACK_MODEL
    for preferred in PREFERRED_MODELS:
        for name in served:
            if preferred in name.lower():
                return name
    return served[0]


class HydeBackend:
    """Generate a pseudo-listing for the transcript. Never raises."""

    name = "hyde"

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        # Settings win over the environment, so an experiment can put two models
        # in ONE process and let the paired bootstrap compare them over the same
        # probes. Empty (the default) means "ask the environment, then the
        # server", which is what a deployment wants.
        base = (getattr(cfg, "hyde_base", "")
                or os.environ.get("COPILOT_LLM_BASE", "http://localhost:30800/v1"))
        self.url = base.rstrip("/") + "/chat/completions"
        self.model = getattr(cfg, "hyde_model", "") or resolve_model(base)
        self.key = os.environ.get("COPILOT_LLM_KEY")
        self.calls = 0
        self.failures = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        cache_path = os.environ.get("COPILOT_LLM_CACHE")
        self.cache_path = Path(cache_path) if cache_path else COPILOT_ROOT / ".hyde_cache.json"
        self._cache: dict[str, str] = self._load_cache()
        self._dirty = 0
        atexit.register(self._flush)

    # -- cache -------------------------------------------------------------
    def _load_cache(self) -> dict[str, str]:
        """A warm cache makes replaying the same deterministic sessions free.

        Keyed on model + prompt, so changing either invalidates cleanly.
        """
        if not self.cfg.hyde_cache:
            return {}
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _flush(self) -> None:
        if not self.cfg.hyde_cache or not self._dirty:
            return
        try:
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._cache), encoding="utf-8")
            tmp.replace(self.cache_path)
            self._dirty = 0
        except Exception:
            pass

    # -- transport ---------------------------------------------------------
    def _post(self, prompt: str, deadline_s: float) -> tuple[str, int, int]:
        # Deterministic by default (temperature 0, fixed seed), so a measurement
        # is reproducible and the response cache is sound. A sampled rewriter
        # would make every score unrepeatable.
        return client.chat(
            self.url,
            self.model,
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": prompt}],
            max_tokens=self.cfg.hyde_max_tokens,
            timeout=deadline_s,
            key=self.key,
        )

    # -- Backend -----------------------------------------------------------
    def expand(self, transcript: str, deadline_s: float) -> Expansion:
        """Return a pseudo-listing for `transcript`, or an empty Expansion.

        Catches everything. A timeout, a refused connection, a malformed body,
        a model that returns nothing — all degrade to "no augmentation", which
        leaves the Tier-0 ranking untouched.
        """
        transcript = " ".join(transcript.split())[: self.cfg.hyde_max_chars]
        if not transcript:
            return Expansion()
        prompt = f'Shopper said: "{transcript}"'

        if self.cfg.hyde_cache:
            digest = hashlib.sha256(
                f"{self.model}\x00{self.cfg.hyde_max_tokens}\x00{prompt}".encode("utf-8")
            ).hexdigest()
            hit = self._cache.get(digest)
            if hit is not None:
                # Cached tokens were paid for once. Reporting them again would
                # inflate the disclosed usage figure.
                return Expansion(text=hit, cached=True)

        try:
            text, prompt_tokens, completion_tokens = self._post(prompt, deadline_s)
        except Exception:
            self.failures += 1
            return Expansion()

        self.calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        if self.cfg.hyde_cache:
            self._cache[digest] = text
            self._dirty += 1
            if self._dirty >= 25:
                self._flush()
        return Expansion(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def stats(self) -> dict:
        return {
            "backend": self.name,
            "model": self.model,
            "calls": self.calls,
            "failures": self.failures,
            "cached": len(self._cache),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }
