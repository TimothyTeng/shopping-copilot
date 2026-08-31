"""The optional model tier: a seam, not a dependency.

Every scored path must produce a complete, valid answer with `NullBackend`, which
is the default. A backend may only *augment* a finished Tier-0 result — it is
never a step the answer depends on. Three rules follow from the competition
rules and are enforced by the callers, not by convention:

* **Tier-0 first.** `Agent.respond` ranks with the stdlib pipeline, and only then
  offers the result to a backend. If the backend is absent, slow, broken, or
  returns nonsense, the Tier-0 ranking is what ships.
* **Failure is silent and total.** `competition_specification.md:65` — "Exceptions,
  invalid output, and timeouts may count as a miss." A backend that raises turns
  a rank-1 hit into a zero, so `expand` catches everything and returns an empty
  `Expansion` rather than propagating.
* **Offline by default.** `submission_rules.md:59` — final scoring may run with
  network access disabled. `backend="null"` is the default, so the answer to
  "does this submission require the network?" is no.

Credentials come from the environment only (`submission_rules.md:49`, no secrets
in the tree).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class Expansion:
    """What a backend returns. Empty means 'nothing to add' — never an error."""

    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False

    @property
    def ok(self) -> bool:
        """True when the expansion carries usable text. Every caller must check:
        a timeout, a refusal or an unreachable endpoint all return an empty one."""
        return bool(self.text.strip())


class Backend(Protocol):
    """Rewrite what the shopper said into text a lexical retriever can use."""

    name: str

    def expand(self, transcript: str, deadline_s: float) -> Expansion:
        ...


class NullBackend:
    """The default. No network, no model, no latency, no failure mode."""

    name = "null"

    def expand(self, transcript: str, deadline_s: float) -> Expansion:
        return Expansion()


def build(cfg) -> Backend | None:
    """Construct the configured backend, or None when the tier is off.

    Import of a live backend is deferred so that the stdlib path never imports
    anything a network-disabled environment could choke on.
    """
    name = getattr(cfg, "backend", "null")
    if name in (None, "", "null", "none"):
        return None
    if name == "hyde":
        from .hyde import HydeBackend
        return HydeBackend(cfg)
    raise ValueError(f"unknown backend: {name!r}")
