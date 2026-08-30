"""In-memory product catalog.

Text is flattened exactly the way the official evaluator flattens it
(`SEARCH_FIELDS`, dicts rendered as "key value"), because the hidden constraints
are drawn from that same rendering. Any divergence here breaks phrase grounding.
"""
from __future__ import annotations

import json
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
        return len(self.ids)

    def contains(self, doc: int, phrase: str) -> bool:
        """Word-bounded substring test against the normalized document."""
        return padded(phrase) in self.text[doc]

    @classmethod
    def load(cls, path: str | Path) -> "CatalogStore":
        ids: list[str] = []
        text: list[str] = []
        title: list[str] = []
        raw_title: list[str] = []
        category: list[str] = []
        cat_path: list[list[str]] = []
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
