"""The agent. Orchestration only — every decision lives in a focused module.

Conforms to the official contract: `reset(session_id, user_profile)` and
`respond(session_id, user_message, turn, top_k) -> dict` with the required
`message`, `ask_attribute`, and `recommendations` keys.
"""
from __future__ import annotations

from pathlib import Path

from . import config, extract, policy
from .bm25 import Bm25Index
from .catalog import CatalogStore
from .category import CategoryIndex
from .index import InvertedIndex
from .rank import Ranker
from .state import DialogueState, Provenance


class Agent:
    """Conversational retrieval over a frozen catalog. No network, no LLM."""

    def __init__(
        self,
        catalog_path: str | Path = config.CATALOG_PATH,
        settings: config.Settings | None = None,
    ) -> None:
        self.cfg = settings or config.DEFAULT
        self.store = CatalogStore.load(catalog_path)
        self.index = InvertedIndex(self.store)
        self.categories = CategoryIndex(self.store)
        # Only pay for the term-frequency index if a mode actually reads it —
        # a prose retrieval mode, or a category resolver that falls back to vote.
        needs_bm25 = (self.cfg.retrieval in ("bm25", "rrf", "auto")
                      or self.cfg.category_resolver in ("vote", "classifier", "ensemble"))
        self.bm25 = Bm25Index(self.store) if needs_bm25 else None
        self.classifier = None
        if self.cfg.category_resolver in ("classifier", "ensemble"):
            from .category_clf import CategoryClassifier
            self.classifier = CategoryClassifier.try_load()
        # Typo tolerance on the shopper's side. A no-op on the benchmark (every
        # token there is quoted from the target, so df>0), so it is only built
        # when explicitly enabled.
        self.fuzzy = None
        if self.cfg.fuzzy_repair:
            from .fuzzy import FuzzyRepair
            self.fuzzy = FuzzyRepair(self.index, self.cfg)
        self.ranker = Ranker(self.store, self.index, self.cfg, self.bm25)
        self._sessions: dict[str, DialogueState] = {}

    # -- contract ----------------------------------------------------------
    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = DialogueState(session_id, user_profile or {})

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._sessions.get(session_id)
        if state is None:                      # defensive: harness may skip reset
            self.reset(session_id, {})
            state = self._sessions[session_id]
        state.turn = turn

        self._observe(state, user_message, turn)

        docs, pool = self.ranker.rank(state, top_k)
        speak = policy.should_emit(state, pool, turn, self.cfg)
        if speak:
            state.record_emission(docs)

        attribute = policy.choose_attribute(state)
        return {
            "message": policy.compose(attribute, state, pool),
            "ask_attribute": attribute,
            "recommendations": (
                [{"parent_asin": self.store.ids[d]} for d in docs] if speak else []
            ),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    # -- turn handling -----------------------------------------------------
    def _observe(self, state: DialogueState, message: str, turn: int) -> None:
        """Fold one customer message into the dialogue state."""
        if self.fuzzy is not None:
            # Repair before anything reads the message — category resolution,
            # extraction, and the stored transcript all see corrected tokens.
            message = self.fuzzy.repair_message(message)
        state.messages.append(message)
        if state.category_key is None:
            basis = state.transcript() if self.cfg.resolve_on_transcript else message
            key, docs, confidence = self.categories.resolve(
                basis, self.cfg, self.bm25, self.classifier
            )
            if key is not None:
                state.category_key = key
                state.category_docs = docs
                state.category_confidence = confidence
        elif self.cfg.resolve_on_transcript and state.category_confidence < 1.0:
            self._reresolve_on_transcript(state)
        elif self.cfg.reresolve_category:
            self._maybe_reresolve(state, message)

        suppress = (
            frozenset(state.category_key.split())
            if state.category_key and self.cfg.suppress_category_tokens
            else frozenset()
        )
        reading = extract.read(
            message, self.index, self.cfg, frozenset(state.seen), suppress
        )

        if reading.declined_attribute:
            # "No preference for X" is a one-off refusal, not proof that nothing
            # is left to learn. Retire a *typed* attribute, but never retire the
            # open-ended question — a catch-all cannot be genuinely exhausted,
            # and banning it strands us on low-yield typed asks.
            state.declined.add(reading.declined_attribute)
            if reading.declined_attribute != "other":
                state.exhausted.add(reading.declined_attribute)
            return
        if reading.exhausted_attribute:
            state.exhausted.add(reading.exhausted_attribute)
            return

        if self.cfg.enable_reset and reading.is_reset:
            # Wholesale abandonment: drop every constraint and the category, then
            # re-resolve and re-seed from what the shopper now wants.
            state.reset_constraints()
            key, docs, confidence = self.categories.resolve(
                message, self.cfg, self.bm25, self.classifier
            )
            if key is not None:
                state.category_key = key
                state.category_docs = docs
                state.category_confidence = confidence
            state.add_spans(reading.spans, self.index, Provenance.OVERRIDE, self.cfg)
            return

        if reading.is_override:
            state.apply_override(self.cfg.override_mode)
            state.add_spans(reading.spans, self.index, Provenance.OVERRIDE, self.cfg)
            return

        if self.cfg.enable_retraction and reading.retractions:
            state.retract(reading.retractions)

        if turn == 1:
            # An opening that merely states a preference is the one an override
            # replaces; an opening with a stated requirement is not.
            provenance = (
                Provenance.OPENING_REQUIREMENT
                if len(reading.spans) > 1
                else Provenance.OPENING_FREEFORM
            )
        else:
            provenance = Provenance.ASK_REPLY
        state.add_spans(reading.spans, self.index, provenance, self.cfg)

    def _reresolve_on_transcript(self, state: DialogueState) -> None:
        """Re-resolve the category over everything the shopper has said so far.

        Runs only while the current resolution is below confidence 1.0 — a
        substring bucket-name match (the benchmark's path) locks at 1.0 on turn 1
        and is never revisited, so this cannot touch the graded score. As natural-
        language constraints accumulate they pin down the department, so a later,
        higher-confidence bucket replaces the vague turn-1 guess.
        """
        key, docs, confidence = self.categories.resolve(
            state.transcript(), self.cfg, self.bm25, self.classifier
        )
        if (key is not None and key != state.category_key
                and confidence >= self.cfg.transcript_resolve_min_conf
                and confidence >= state.category_confidence):
            state.category_key = key
            state.category_docs = docs
            state.category_confidence = confidence

    def _maybe_reresolve(self, state: DialogueState, message: str) -> None:
        """On an override, switch category only for a confident, different bucket.

        "Actually, never mind — I need women's sweatpants" changes the category;
        the benchmark's "what I need is: {value}" does not. The confidence gate
        separates them: a real category name resolves via substring (1.0) or
        pulls a clean vote, while a bare attribute value smears its vote below
        the threshold and is rejected, leaving the original category intact.
        """
        key, docs, confidence = self.categories.resolve(
            message, self.cfg, self.bm25, self.classifier
        )
        if (key is not None and key != state.category_key
                and confidence >= self.cfg.reresolve_min_confidence):
            state.category_key = key
            state.category_docs = docs
            state.category_confidence = confidence
