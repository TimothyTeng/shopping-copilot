"""Offline trainer for the character-n-gram category classifier.

Trains a linear (log-loss) classifier that maps a short product phrase to its
coarse catalog bucket, then exports pruned weights the runtime resolver reads
with nothing but the standard library. sklearn/numpy are used HERE ONLY — the
scored path never imports them.

Why char n-grams over a product's *title* (not its full text): the title is the
closest thing in the catalog to how a shopper phrases a request, and character
n-grams give morphology for free — "watch"/"watches", "sweatpant"/"sweatpants",
and typos like "waterprrof" all share most of their 3–5-grams. The `categories`
field is deliberately excluded from the input: it is the label, and training on
it would leak.

    PYTHONIOENCODING=utf-8 python -m tools.train_category

Writes src/models/category_clf.json.gz.
"""
from __future__ import annotations

import gzip
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier

from src import config
from src.catalog import CatalogStore, coarse_category
from src.normalize import DIALOGUE_STOP, norm, tokens

LO, HI = 3, 5
MIN_DF = 5
MAX_FEATURES = 30000
TOP_PER_CLASS = 140
AUGMENT = 3          # short word-subset rows synthesised per title
AUG_MIN, AUG_MAX = 1, 3   # words per synthetic short "query"


def content_tokens(title: str) -> list[str]:
    return [t for t in tokens(norm(title))
            if t not in DIALOGUE_STOP and len(t) > 1]


def short_variants(words: list[str], rng) -> list[str]:
    """A few 1–3-word contiguous slices, to mimic how a shopper actually types.

    The classifier is queried with 1–3 words but a title is a dozen; training on
    the full title alone teaches it to lean on signal a real query never carries.
    Synthetic short rows close that train/inference gap.
    """
    out: list[str] = []
    if not words:
        return out
    for _ in range(AUGMENT):
        k = min(len(words), rng.randint(AUG_MIN, AUG_MAX))
        start = rng.randint(0, len(words) - k)
        out.append(" ".join(words[start:start + k]))
    return out
OUT = Path(__file__).resolve().parents[1] / "src" / "models" / "category_clf.json.gz"


def char_ngrams(text: str) -> list[str]:
    """char_wb-style n-grams: pad each word, take 3–5-grams within it.

    IDENTICAL to the runtime extractor in src/category_clf.py — the two must not
    drift or the shipped weights address different features than inference does.
    """
    out: list[str] = []
    for word in text.split():
        w = f" {word} "
        for n in range(LO, HI + 1):
            for i in range(len(w) - n + 1):
                out.append(w[i:i + n])
    return out


def main() -> None:
    t0 = time.perf_counter()
    import random
    rng = random.Random(0)
    store = CatalogStore.load(config.CATALOG_PATH)
    # X = content words of the title (stopwords dropped) PLUS short word-subsets,
    # so the model sees inputs the length of a real query, not just full titles.
    docs: list[str] = []
    labels: list[str] = []
    for title, seg in zip(store.raw_title, store.cat_path):
        bucket = norm(coarse_category(seg))
        words = content_tokens(title)
        if not words:
            continue
        docs.append(" ".join(words))
        labels.append(bucket)
        for variant in short_variants(words, rng):
            docs.append(variant)
            labels.append(bucket)
    print(f"  {len(docs):,} training rows ({len(set(labels)):,} buckets, "
          f"{AUGMENT}x short-query augmentation)")

    # Build the n-gram vocabulary with a document-frequency floor.
    df: dict[str, int] = {}
    for text in docs:
        for gram in set(char_ngrams(text)):
            df[gram] = df.get(gram, 0) + 1
    vocab_items = [(g, c) for g, c in df.items() if c >= MIN_DF]
    vocab_items.sort(key=lambda gc: -gc[1])
    vocab_items = vocab_items[:MAX_FEATURES]
    vocab = {g: i for i, (g, _) in enumerate(vocab_items)}
    n_feat = len(vocab)
    idf = np.zeros(n_feat, dtype=np.float64)
    for g, c in vocab_items:
        idf[vocab[g]] = math.log((len(docs) + 1) / (c + 1)) + 1.0
    print(f"  {n_feat:,} n-gram features (min_df={MIN_DF})")

    # Sparse TF-IDF rows (no row norm, so the score stays linear in the weights).
    rows, cols, vals = [], [], []
    for r, text in enumerate(docs):
        counts: dict[int, int] = {}
        for gram in char_ngrams(text):
            j = vocab.get(gram)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        for j, tf in counts.items():
            rows.append(r)
            cols.append(j)
            vals.append(tf * idf[j])
    X = csr_matrix((vals, (rows, cols)), shape=(len(docs), n_feat))

    classes = sorted(set(labels))
    cls_id = {c: i for i, c in enumerate(classes)}
    y = np.array([cls_id[c] for c in labels])
    print(f"  training {len(classes):,}-way classifier on {X.shape}…")
    clf = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=40,
                        tol=1e-4, n_jobs=-1, random_state=0)
    clf.fit(X, y)
    acc = (clf.predict(X) == y).mean()
    print(f"  train accuracy {acc:.3f}")

    # Prune to the top features per class and fold idf into the shipped weight,
    # so runtime inference is a plain sum of postings over the message n-grams.
    coef = clf.coef_                      # (n_classes, n_feat)
    inv_vocab = {i: g for g, i in vocab.items()}
    postings: dict[str, list] = {}
    for c in range(coef.shape[0]):
        row = coef[c]
        top = np.argsort(np.abs(row))[-TOP_PER_CLASS:]
        for j in top:
            w = float(row[j] * idf[j])
            if w == 0.0:
                continue
            postings.setdefault(inv_vocab[int(j)], []).append([c, round(w, 4)])

    model = {
        "lo": LO, "hi": HI,
        "classes": classes,
        "intercept": [round(float(v), 4) for v in clf.intercept_],
        "postings": postings,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8") as fh:
        json.dump(model, fh)
    size = OUT.stat().st_size / 1e6
    print(f"  wrote {OUT.relative_to(OUT.parents[2])}  ({size:.1f} MB, "
          f"{len(postings):,} n-grams)  in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
