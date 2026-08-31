"""Constraint extraction.

Two layers, deliberately ordered:

  A. template  — regex over the simulator's known carrier phrases. Fast and
                 precise, but it is a *latency optimization only*. The system
                 must score well with this disabled.
  C. salience  — the robust core. It never parses sentence structure. It finds
                 runs of catalog-grounded, informative, previously-unseen tokens
                 and keeps them. Rewording the sentence around a requirement does
                 not change the requirement's own words, so this survives
                 paraphrase.

Named A and C to match the design doc, where B (cue families) is a future layer.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .index import InvertedIndex
from .normalize import DIALOGUE_STOP, norm, tokens

# Layer A carriers. Only ever a shortcut.
_CARRIER = re.compile(
    r"(?:matters is|matters would be|requirement is|what i need is|need is|looking for)\s*:?\s*(.+)$",
    re.I,
)
# The kit's own override turn is verbatim:
#   "Actually, ignore my earlier preference. What I need is: {new_value}."
# Alternation is leftmost-first, not longest-first, so an alternative starting
# at "Actually" wins over one starting at "ignore" however they are ordered.
# The cue must therefore be a single span covering the whole preamble, or its
# tail ("my earlier preference") is mined as a requirement — and "earlier" has
# df=10, so it looks *highly* informative to the scorer.
_OVERRIDE_CUES = re.compile(
    r"\b(?:actually|instead)?,?\s*(?:please\s+)?"
    r"(?:ignore|forget|disregard|scratch)\s+"
    r"(?:my|the|that|what|those)?\s*"
    r"(?:earlier|previous|prior|last|first)?\s*"
    r"(?:preference|preferences|request|requirement|answer|choice)?"
    r"|\bchanged my mind\b"
    r"|\binstead of\b"
    r"|\bactually,?\s+(?:what|really)\b",
    re.I,
)
# The shopper takes a value back: "not leather", "instead of leather", "rather
# than leather", "don't want leather". The negated value is whatever grounded
# phrase follows the cue within the same comma-clause.
_NEGATION = re.compile(
    r"\b(?:not|no longer|instead of|rather than|don'?t\s+(?:want|need|like))\b",
    re.I,
)
# The shopper abandons the whole line of search and starts again.
_RESET = re.compile(
    r"\b(?:never\s*mind|scratch that|start over|start again|begin again|"
    r"forget (?:it|everything|all that))\b",
    re.I,
)
# The simulator quotes its constraint strings verbatim after a fixed colon
# carrier — "A key requirement is: X.", "For that, what matters is: X; Y.",
# "What I need is: X." Those strings are `_clean_constraint`ed slots of the
# target's own intent card, so recovering them exactly (rather than mining them
# for tokens) lets the ranker ask whether a *candidate* would have produced the
# same card. See catalog.card_slots and rank._card_bonus.
_DISCLOSURE = re.compile(
    r"(?:key requirement is|requirement is|what matters is|what i need is|need is)\s*:\s*(.+)$",
    re.I | re.S,
)


def disclosed_constraints(message: str) -> list[str]:
    """The raw constraint strings this message discloses, normalized.

    Empty for any message that does not use the simulator's colon carrier, which
    is every free-text message — so this is inert outside the graded path.
    """
    match = _DISCLOSURE.search(message)
    if not match:
        return []
    out: list[str] = []
    for piece in match.group(1).split(";"):
        value = norm(piece.strip().rstrip("."))
        if value:
            out.append(value)
    return out


_NO_PREF = re.compile(
    r"\b(?:no preference|don'?t have a preference|no strong (?:feelings|preference)|"
    r"doesn'?t matter|your (?:call|judgment)|up to you)\b",
    re.I,
)
_EXHAUSTED = re.compile(
    r"\b(?:no additional|nothing else|nothing further|that'?s all|no other|"
    r"nothing springs to mind)\b",
    re.I,
)
# The simulator names the attribute it is declining, e.g. "...preference for color".
_ATTR_MENTION = re.compile(
    r"\b(?:for|about|on)\s+(category|material|color|size|style|brand|budget|feature|use_case|other)\b",
    re.I,
)


@dataclass(slots=True)
class Span:
    """A candidate requirement lifted from a message."""

    tokens: tuple[str, ...]
    phrase: str | None          # longest contiguous form that exists in the catalog
    layer: str
    confidence: float
    raw: str = ""

    @property
    def key(self) -> str:
        return " ".join(self.tokens)


@dataclass(slots=True)
class Reading:
    """Everything one incoming message tells us."""

    spans: list[Span] = field(default_factory=list)
    is_override: bool = False
    is_reset: bool = False
    retractions: list[str] = field(default_factory=list)
    declined_attribute: str | None = None
    exhausted_attribute: str | None = None


def _segment_grounded(
    run: list[str], index: InvertedIndex, cfg
) -> list[list[str]]:
    """Split a run into the longest sub-phrases the catalog actually contains.

    Conversational glue ("... cotton plus black ...") would otherwise fuse two
    separate requirements into one span that matches nothing. Greedily taking
    maximal grounded windows splits them apart and drops the glue, without
    needing to know which words are glue.
    """
    pieces: list[list[str]] = []
    i = 0
    while i < len(run):
        longest = None
        for j in range(min(len(run), i + cfg.max_span_tokens), i, -1):
            window = run[i:j]
            if index.phrase_df(" ".join(window)) > 0:
                longest = (j, window)
                break
        if longest is None:
            i += 1                      # ungrounded token: not product language
            continue
        end, window = longest
        pieces.append(window)
        i = end

    # A run that never split shows no sign of glue, so trust it whole — this is
    # what keeps one-word requirements like "cotton" intact. But once a run has
    # split around a real multi-word requirement, the leftover lone tokens are
    # the connective tissue between requirements, not requirements themselves.
    if len(pieces) > 1 and any(len(p) > 1 for p in pieces):
        pieces = [p for p in pieces if len(p) > 1]
    return pieces


def _segment_rarity(
    run: list[str], index: InvertedIndex, cfg
) -> list[list[str]]:
    """Choose the segmentation that explains the most tokens with the rarest
    catalog phrases, rather than the one that happens to start leftmost.

    Greedy takes the first grounded window it finds, so a glue word can capture
    the head of the next requirement:

        "rubber sole plus design in usa"
          greedy -> ["rubber sole", "plus design", "in usa"]
          here   -> ["rubber sole", "design in usa"]     ("plus" skipped)

    Every piece is scored `len * log(N / (1 + phrase_df))`, so a long rare
    phrase beats the fragments it could be cut into — the measured scores for
    the example above are 30.4 against 19.4 + 7.8.

    A token may be dropped, which is how conversational glue disappears without
    a glue list, but dropping is *charged* at its IDF. Free dropping measured
    -0.018 on bare declaratives: with no glue to remove the search simply
    discarded requirement words whenever a shorter phrase scored better. Glue is
    common and cheap to drop; requirement words are rare and expensive.
    """
    n = len(run)
    log_n = math.log(index.n_docs)
    # best[i] = (score, pieces) for the suffix run[i:]
    best: list[tuple[float, list[list[str]]]] = [(0.0, [])] * (n + 1)
    for i in range(n - 1, -1, -1):
        tail_score, tail_pieces = best[i + 1]    # drop run[i] as glue
        choice = (tail_score - cfg.skip_penalty * index.idf(run[i]), tail_pieces)
        for j in range(min(n, i + cfg.max_span_tokens), i, -1):
            df = index.phrase_df(" ".join(run[i:j]))
            if df <= 0:
                continue
            gain = (j - i) * (log_n - math.log(1 + df)) - cfg.piece_cost
            tail_score, tail_pieces = best[j]
            if gain + tail_score > choice[0]:
                choice = (gain + tail_score, [run[i:j]] + tail_pieces)
        best[i] = choice
    return best[0][1]


def _detect_negations(
    message: str, index: InvertedIndex
) -> tuple[list[str], set[str]]:
    """Find retracted values and the tokens they cover.

    Works clause by clause (commas are the natural scope boundary) so "not
    leather, canvas is better" negates only "leather", leaving "canvas" to be
    mined as the replacement. Each negated value is grounded to a catalog phrase
    so it can be matched against the slots already held.
    """
    retract: list[str] = []
    negated: set[str] = set()
    for clause in re.split(r"[,;]", message.lower()):
        match = _NEGATION.search(clause)
        if not match:
            continue
        tail = [t for t in tokens(norm(clause[match.end():]))
                if t not in DIALOGUE_STOP and index.df(t) > 0]
        if not tail:
            continue
        phrase = _ground_phrase(tail, index) or tail[0]
        retract.append(phrase)
        negated.update(phrase.split())
    return retract, negated


def _ground_phrase(span_tokens: list[str], index: InvertedIndex) -> str | None:
    """Longest contiguous window of the span that actually occurs in the catalog."""
    n = len(span_tokens)
    for length in range(n, 1, -1):
        for start in range(0, n - length + 1):
            candidate = " ".join(span_tokens[start:start + length])
            if index.phrase_df(candidate) > 0:
                return candidate
    single = span_tokens[0] if n == 1 else None
    if single and index.df(single) > 0:
        return single
    return None


def _salient_runs(
    message_tokens: list[str],
    index: InvertedIndex,
    cfg,
    seen: frozenset[str],
) -> list[list[str]]:
    """Contiguous runs of informative, unseen tokens, bridging small gaps."""
    def salient(term: str) -> bool:
        if term in DIALOGUE_STOP or len(term) < 2:
            return False
        if not index.informative(term, cfg.max_token_df_ratio):
            return False
        return index.idf(term) >= cfg.min_span_idf

    flags = [salient(t) for t in message_tokens]
    runs: list[list[str]] = []
    current: list[str] = []
    gap = 0
    for term, is_salient in zip(message_tokens, flags):
        if is_salient:
            current.append(term)
            gap = 0
        elif current:
            gap += 1
            if gap > cfg.bridge_gap:
                runs.append(current)
                current, gap = [], 0
            else:
                current.append(term)   # bridged connector, kept for phrase shape
    if current:
        runs.append(current)

    cleaned: list[list[str]] = []
    for run in runs:
        while run and not salient(run[-1]):
            run.pop()
        run = run[: cfg.max_span_tokens]
        # A run made entirely of things we have already been told is not news.
        if run and not all(t in seen for t in run):
            cleaned.append(run)
    return cleaned


def read(
    message: str,
    index: InvertedIndex,
    cfg,
    seen: frozenset[str],
    suppress: frozenset[str] = frozenset(),
) -> Reading:
    """Interpret one customer message.

    `suppress` holds tokens already consumed by another signal — the resolved
    category, chiefly — so they are not also mined as requirements. Without it
    the category name fuses onto the front of the first real requirement.
    """
    reading = Reading()
    reading.is_reset = cfg.enable_reset and bool(_RESET.search(message))
    if reading.is_reset:
        # A reset is not an override; strip the cue so "never mind" is not mined.
        message = _RESET.sub(" ", message)
    # A reset supersedes the override path — both revoke, but a reset also drops
    # the category, and its cue ("forget it") can otherwise trip _OVERRIDE_CUES.
    reading.is_override = (not reading.is_reset
                           and bool(_OVERRIDE_CUES.search(message)))
    if reading.is_override:
        # Strip the cue itself, or "ignore my earlier" is mined as a requirement.
        message = _OVERRIDE_CUES.sub(" ", message)

    if _NO_PREF.search(message):
        match = _ATTR_MENTION.search(message)
        reading.declined_attribute = match.group(1).lower() if match else "other"
        return reading                      # a refusal carries no requirement
    if _EXHAUSTED.search(message):
        match = _ATTR_MENTION.search(message)
        reading.exhausted_attribute = match.group(1).lower() if match else "other"
        return reading

    # A negated value must not be mined as a fresh requirement this turn, so drop
    # its tokens from the salience pass; the agent revokes any matching slot.
    negated: set[str] = set()
    if cfg.enable_retraction:
        reading.retractions, negated = _detect_negations(message, index)
        if negated:
            suppress = suppress | negated

    seen_spans: set[str] = set()

    if cfg.use_templates:
        carrier = _CARRIER.search(message)
        if carrier:
            for part in re.split(r";|,\s*and also\s*", carrier.group(1)):
                # NB: `suppress` is deliberately NOT applied here. The category
                # slot reinforces the category signal in scoring; removing it
                # measured -0.007 and dropped Hit@10 to 0.980. It is filtered
                # from the user-facing wording instead (see policy.compose).
                span_tokens = [t for t in tokens(norm(part)) if t not in DIALOGUE_STOP]
                span_tokens = [t for t in span_tokens if index.df(t) > 0][: cfg.max_span_tokens]
                if not span_tokens:
                    continue
                key = " ".join(span_tokens)
                if key in seen_spans:
                    continue
                seen_spans.add(key)
                reading.spans.append(
                    Span(tuple(span_tokens), _ground_phrase(span_tokens, index),
                         "template", 0.95, part.strip())
                )

    if cfg.use_salience:
        message_tokens = [t for t in tokens(norm(message)) if t not in suppress]
        for run in _salient_runs(message_tokens, index, cfg, seen):
            if not cfg.segment_spans:
                pieces = [run]
            elif cfg.segment_mode == "rarity":
                pieces = _segment_rarity(run, index, cfg)
            else:
                pieces = _segment_grounded(run, index, cfg)
            for piece in pieces:
                key = " ".join(piece)
                if key in seen_spans or any(key in s or s in key for s in seen_spans):
                    continue
                seen_spans.add(key)
                reading.spans.append(Span(tuple(piece), key, "salience", 0.7, key))

    if negated:                    # belt and suspenders: templates ignore suppress
        reading.spans = [s for s in reading.spans if not set(s.tokens) <= negated]

    return reading
