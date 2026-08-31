"""In-memory product catalog.

Text is flattened exactly the way the official evaluator flattens it
(`SEARCH_FIELDS`, dicts rendered as "key value"), because the hidden constraints
are drawn from that same rendering. Any divergence here breaks phrase grounding.
"""
from __future__ import annotations

import json
import re
from array import array
from dataclasses import dataclass
from pathlib import Path

from .normalize import norm, padded, tokens

# Field order mirrors evaluator.local_evaluator.SEARCH_FIELDS.
SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")

# Category segments the evaluator strips when forming a coarse category.
_EXCLUDED_SEGMENTS = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}


def flatten(value: object) -> list[str]:
    """Render one product field the way the evaluator does."""
    if value is None:
        return []
    if isinstance(value, dict):
        return [f"{key} {item}" for key, item in value.items()]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def searchable_text(product: dict) -> str:
    """Flatten a product exactly the way the evaluator does.

    MIRRORS THE KIT. The hidden constraints are drawn from this same rendering,
    so any divergence stops matching silently — `tools/verify_mirror.py`
    compares all 50,000 products against the kit's own function."""
    parts: list[str] = []
    for field in SEARCH_FIELDS:
        parts.extend(flatten(product.get(field)))
    return " ".join(parts).strip()


def parse_price(value: object) -> float:
    """Price is float | None | dirty string (em-dash, 'from 12.99'). Never raises."""
    if value is None or value == "":
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace("$", "").replace(",", "").strip()
    for chunk in text.replace("—", " ").replace("–", " ").split():
        try:
            return float(chunk)
        except ValueError:
            continue
    return float("nan")


# --- intent-card mirror ---------------------------------------------------
# The simulator does not invent the shopper's requirements: it *derives* them
# from the target product, deterministically, in `local_evaluator.intent_card`.
# The public set ships no `intent_card`, so `materialize_hidden_fields` rebuilds
# one from the catalog at scoring time — which means every product's four
# possible constraint strings are computable offline, here.
#
# This mirrors that function exactly, the same way `coarse_category` and
# `searchable_text` do. If it diverges, the signature silently stops matching
# and the bonus becomes a no-op — a degradation, never a wrong answer.
_MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I)
_COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I)
_WS_RE = re.compile(r"\s+")

CARD_LIMIT = 180        # local_evaluator.intent_card's default `limit`
CARD_SLOTS = 4          # hard_constraints[:2] + soft_preferences[2:4]


def clean_constraint(value: str, limit: int = CARD_LIMIT) -> str:
    """Mirror of `local_evaluator._clean_constraint`."""
    return _WS_RE.sub(" ", value).strip(" -;,.\t\n")[:limit].rstrip()


def card_values(value: object) -> list[str]:
    """Mirror of `local_evaluator._flatten_values`.

    Note this is NOT `flatten`: the card renders a dict as "key: item", while the
    searchable text renders it as "key item". Two different renderings of the
    same field, and the constraint strings come from this one.
    """
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items()
                if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def card_slots(product: dict) -> list[str]:
    """The constraint strings the simulator can draw from this product.

    `intent_card` takes `cleaned[:2]` as hard constraints and `cleaned[2:4]` as
    soft preferences, so the first four are everything a session can disclose.
    """
    candidates = [*card_values(product.get("features")),
                  *card_values(product.get("details"))]
    corpus = searchable_text(product)
    material = _MATERIAL_RE.search(corpus)
    color = _COLOR_RE.search(corpus)
    if material:
        candidates.insert(0, material.group(1).lower())
    if color:
        candidates.insert(1, f"color: {color.group(1).lower()}")
    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")
    cleaned = list(dict.fromkeys(
        clean_constraint(item) for item in candidates if clean_constraint(item)
    ))
    if not cleaned:
        cleaned = [clean_constraint(str(product.get("title") or "product"))]
    return cleaned[:CARD_SLOTS]


def coarse_category(values: list[str]) -> str:
    """Mirror of the evaluator's coarse_category, used to key category buckets."""
    cleaned: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in _EXCLUDED_SEGMENTS:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


@dataclass(slots=True)
class CatalogStore:
    """Column-oriented catalog. Documents are addressed by integer ordinal."""

    ids: list[str]
    ord_of: dict[str, int]
    text: list[str]           # normalized, space-padded full searchable text
    title: list[str]          # normalized title (field weighting)
    raw_title: list[str]      # original title, for display
    category: list[str]       # normalized category path
    cat_path: list[list[str]]  # raw breadcrumb segments
    card: list[frozenset[str]]  # normalized intent-card slots (see card_slots)
    rating_n: array
    rating_avg: array
    price: array
    # Per-field token counts, for BM25 length normalization.
    len_title: array
    len_cat: array
    len_body: array
    avg_title: float
    avg_cat: float
    avg_body: float

    def __len__(self) -> int:
        """Number of products in the catalog."""
        return len(self.ids)

    def contains(self, doc: int, phrase: str) -> bool:
        """Word-bounded substring test against the normalized document."""
        return padded(phrase) in self.text[doc]

    @classmethod
    def load(cls, path: str | Path) -> "CatalogStore":
        """Read the catalog once into parallel arrays keyed by document ordinal.

        Columnar rather than a list of dicts: the ranker touches one field at a
        time over many documents, and arrays keep the whole store near 90 MB."""
        ids: list[str] = []
        text: list[str] = []
        title: list[str] = []
        raw_title: list[str] = []
        category: list[str] = []
        cat_path: list[list[str]] = []
        card: list[frozenset[str]] = []
        rating_n = array("i")
        rating_avg = array("f")
        price = array("f")
        len_title = array("i")
        len_cat = array("i")
        len_body = array("i")

        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                ids.append(str(product["parent_asin"]))
                text.append(padded(norm(searchable_text(product))))
                raw_title.append(str(product.get("title") or ""))
                title.append(norm(product.get("title") or ""))
                segments = [str(v) for v in (product.get("categories") or [])]
                cat_path.append(segments)
                category.append(norm(" ".join(segments)))
                # Normalized on both sides so a punctuation difference between
                # the message and the stored slot cannot break the match.
                card.append(frozenset(norm(slot) for slot in card_slots(product)))
                n_all = len(tokens(text[-1]))
                n_ttl = len(tokens(title[-1]))
                n_cat = len(tokens(category[-1]))
                len_title.append(n_ttl)
                len_cat.append(n_cat)
                len_body.append(max(n_all - n_ttl - n_cat, 0))
                rating_n.append(int(product.get("rating_number") or 0))
                rating_avg.append(float(product.get("average_rating") or 0.0))
                price.append(parse_price(product.get("price")))

        return cls(
            ids=ids,
            ord_of={pid: i for i, pid in enumerate(ids)},
            text=text,
            title=title,
            raw_title=raw_title,
            category=category,
            cat_path=cat_path,
            card=card,
            rating_n=rating_n,
            rating_avg=rating_avg,
            price=price,
            len_title=len_title,
            len_cat=len_cat,
            len_body=len_body,
            avg_title=(sum(len_title) / len(len_title)) or 1.0,
            avg_cat=(sum(len_cat) / len(len_cat)) or 1.0,
            avg_body=(sum(len_body) / len(len_body)) or 1.0,
        )
