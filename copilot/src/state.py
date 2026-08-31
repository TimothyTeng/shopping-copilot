"""Per-session dialogue state.

Slots accumulate; they are never rebuilt. Two behaviours matter most:

* **Override is scoped, not global.** The simulator states the abandoned
  preference in the *opening* and later replaces it. Both descriptions came from
  the same target product, so wiping everything discards correct information.
  Only the opening's free-form preference is revoked.

* **Implicit rejection.** If the session did not end after we showed ten
  products, the target was not among them. Those products are *demoted* on later
  turns rather than removed, so a wrong inference costs rank instead of making
  the target unreachable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .extract import Span
from .index import InvertedIndex


class Provenance:
    """Where a slot came from. This is what makes an override *scoped*: only
    OPENING_FREEFORM is revoked, because only that was replaced."""
    OPENING_REQUIREMENT = "opening_requirement"   # a stated hard requirement
    OPENING_FREEFORM = "opening_freeform"         # the preference an override replaces
    ASK_REPLY = "ask_reply"
    OVERRIDE = "override"


@dataclass(slots=True)
class Slot:
    """One accumulated constraint: its tokens, their IDF weights, the contiguous
    phrase it came from if any, and enough provenance to revoke it later."""
    tokens: tuple[str, ...]
    idfs: tuple[float, ...]
    phrase: str | None
    provenance: str
    layer: str
    turn: int
    weight: float = 1.0
    status: str = "active"

    @property
    def idf_total(self) -> float:
        """Total IDF mass of this slot — the denominator of its coverage ratio.
        Floored at 1.0 so an all-stopword slot cannot divide by zero."""
        return sum(self.idfs) or 1.0

    @property
    def key(self) -> str:
        """Canonical identity, used to deduplicate repeats of the same constraint."""
        return " ".join(self.tokens)


@dataclass
class DialogueState:
    """Everything remembered about one shopper session.

    Constructed per `session_id` by `Agent.reset` and mutated in place by
    `Agent._observe`. Slots only ever accumulate or get revoked; nothing here
    is rebuilt from scratch mid-session."""
    session_id: str
    profile: dict
    turn: int = 0
    slots: list[Slot] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)          # tokens already told to us
    asked: list[str] = field(default_factory=list)
    exhausted: set[str] = field(default_factory=set)
    declined: set[str] = field(default_factory=set)
    emitted: list[list[int]] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)   # raw turns, for BM25
    # Constraint strings recovered verbatim from the simulator's colon carrier.
    # These are intent-card slots of the target, so a candidate that carries the
    # same slots in ITS card is a far stronger match than one that merely
    # contains the words. Empty on free text. See extract.disclosed_constraints.
    disclosed: set[str] = field(default_factory=set)
    category_key: str | None = None
    category_docs: object = None
    category_confidence: float = 1.0     # 1.0 for an exact bucket-name match
    override_seen: bool = False
    override_turn: int = 0
    # --- optional model tier (src/backends/) ------------------------------
    # A generated pseudo-listing, reduced to the terms the catalog actually
    # contains, plus the products it retrieves. Both are empty unless a backend
    # is configured AND its gate fired AND the call succeeded, so every path
    # that reads them must treat "absent" as the normal case.
    hyde_text: str = ""
    hyde_ranking: tuple = ()             # BM25 order over the generated listing
    hyde_turn: int = 0                   # last turn an expansion was attempted

    def transcript(self) -> str:
        """Everything the shopper has said, for retrieval modes that read prose.

        The slot pipeline discards anything it cannot ground in the catalog,
        which is right when constraints are quoted from the product and wrong
        when they are the shopper's own words.
        """
        return " ".join(self.messages)

    # -- slots -------------------------------------------------------------
    @staticmethod
    def _specificity(span: Span, index: InvertedIndex, cfg) -> float:
        """How sharply this constraint narrows the catalog, in [floor, 1].

        A constraint matching one product should outweigh one matching 3,000.
        In the log-product this scales the penalty for failing to satisfy it.
        """
        if cfg is None or not getattr(cfg, "specificity_weighting", False):
            return 1.0
        phrase = span.phrase or span.key
        df = index.phrase_df(phrase) if phrase else 0
        if df <= 0:
            return 1.0
        weight = math.log(index.n_docs / (1 + df)) / math.log(index.n_docs)
        return max(cfg.specificity_floor, min(1.0, weight))

    def add_spans(self, spans: list[Span], index: InvertedIndex, provenance: str,
                  cfg=None) -> int:
        """Add newly extracted spans as slots, skipping ones already held.

        Deduplication is by `Slot.key`, so the shopper repeating a requirement
        in different words costs nothing. Returns how many were genuinely new,
        which is what `policy` reads to decide whether the turn made progress.
        """
        added = 0
        existing = {slot.key for slot in self.slots}
        for span in spans:
            if span.key in existing:
                continue
            slot = Slot(
                weight=self._specificity(span, index, cfg),
                tokens=span.tokens,
                idfs=tuple(index.idf(t) for t in span.tokens),
                phrase=span.phrase,
                provenance=provenance,
                layer=span.layer,
                turn=self.turn,
            )
            self.slots.append(slot)
            existing.add(span.key)
            self.seen.update(span.tokens)
            added += 1
        return added

    def active_slots(self) -> list[Slot]:
        """Slots still in force: not revoked by an override or a retraction, and
        not zeroed by specificity weighting. This is the conjunction we rank on."""
        return [s for s in self.slots if s.status == "active" and s.weight > 0]

    def retract(self, phrases: list[str]) -> int:
        """Revoke active slots the shopper just took back.

        A slot is revoked when it and the retracted phrase name the same thing —
        one token-set contains the other — so "not leather" clears both a bare
        "leather" slot and a "leather upper" one without touching "canvas".
        """
        revoked = 0
        for phrase in phrases:
            pt = set(phrase.split())
            if not pt:
                continue
            for slot in self.slots:
                if slot.status != "active":
                    continue
                st = set(slot.tokens)
                if pt <= st or st <= pt:
                    slot.status, slot.weight = "revoked", 0.0
                    revoked += 1
        return revoked

    def reset_constraints(self) -> None:
        """Abandon the whole line of search: revoke every slot and forget what we
        have shown or seen, so a fresh category starts clean."""
        for slot in self.slots:
            slot.status, slot.weight = "revoked", 0.0
        self.seen.clear()
        self.emitted.clear()
        self.category_key = None
        self.category_docs = None

    def apply_override(self, mode: str) -> None:
        """Revoke only what the override actually replaced."""
        self.override_seen = True
        self.override_turn = self.turn
        self.maybe_override = False
        # Anything shown before the override could not have converted, so it is
        # not evidence of rejection. Forget it rather than penalize it.
        self.emitted.clear()
        if mode == "keep":
            return
        for slot in self.slots:
            if mode == "erase":
                slot.status, slot.weight = "revoked", 0.0
            elif slot.provenance == Provenance.OPENING_FREEFORM:
                slot.status, slot.weight = "revoked", 0.0

    # -- implicit rejection ------------------------------------------------
    def record_emission(self, docs: list[int]) -> None:
        """Remember what we showed. If the session did not end, the target was
        not among them."""
        if docs:
            self.emitted.append(docs)

    def shown(self) -> set[int]:
        """Products already offered and implicitly rejected.

        These are *demoted*, never removed: on an override session the rejection
        inference is unsound, and a penalty still lets the target rank whereas
        removal would make it unreachable.
        """
        out: set[int] = set()
        for group in self.emitted:
            out.update(group)
        return out
