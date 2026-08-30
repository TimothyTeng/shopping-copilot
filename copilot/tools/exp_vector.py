"""Feasibility + value test for vector search over the 50k catalog.

Builds catalog-trained LSA vectors (TF-IDF -> truncated SVD), then compares
pure vector retrieval against the existing lexical ranker on an identical
"oracle" query: all four of a session's hidden constraints, no dialogue, no gate.
That isolates retrieval power from everything else.
"""
from __future__ import annotations
import sys, time, tracemalloc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from tools.sim import KIT_ROOT
from evaluator.local_evaluator import load_jsonl, catalog_index, materialize_hidden_fields
from src import config
from src.agent import Agent
from src.extract import Span
from src.state import DialogueState, Provenance
from src.normalize import norm, tokens

DIMS = 256

print("loading agent (lexical index)...")
t0 = time.perf_counter()
agent = Agent(config.CATALOG_PATH, config.DEFAULT)
print(f"  lexical index: {time.perf_counter()-t0:.1f}s")

store = agent.store
texts = [t.strip() for t in store.text]

print(f"\nbuilding vector index ({DIMS} dims)...")
tracemalloc.start()
t0 = time.perf_counter()
vec = TfidfVectorizer(min_df=3, max_df=0.5, sublinear_tf=True, token_pattern=r"[a-z0-9]+")
X = vec.fit_transform(texts)
t_tfidf = time.perf_counter() - t0
print(f"  tf-idf matrix   {X.shape[0]:,} x {X.shape[1]:,}  nnz={X.nnz:,}  {t_tfidf:.1f}s")

t0 = time.perf_counter()
svd = TruncatedSVD(n_components=DIMS, random_state=0)
D = normalize(svd.fit_transform(X)).astype(np.float32)
t_svd = time.perf_counter() - t0
_, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
print(f"  SVD             {t_svd:.1f}s   explained variance {svd.explained_variance_ratio_.sum():.1%}")
print(f"  doc vectors     {D.nbytes/1e6:.0f} MB   peak build RAM {peak/1e6:.0f} MB")

# query latency
q = D[:1]
t0 = time.perf_counter()
for _ in range(100):
    sims = D @ q.T
    np.argpartition(-sims[:, 0], 10)[:10]
print(f"  query latency   {(time.perf_counter()-t0)/100*1000:.2f} ms  (brute force, all 50k)")

# ---- value test ----------------------------------------------------------
samples = load_jsonl(KIT_ROOT / "data" / "public_set.jsonl")
ids, cats, products = catalog_index(KIT_ROOT / "data" / "catalog.jsonl")

def lexical_rank(constraints, target_doc):
    st = DialogueState("x", {})
    st.turn = 1
    spans = []
    for c in constraints:
        tk = [t for t in tokens(norm(c)) if agent.index.df(t) > 0][:12]
        if tk:
            spans.append(Span(tuple(tk), " ".join(tk), "test", 1.0, c))
    st.add_spans(spans, agent.index, Provenance.ASK_REPLY)
    docs, _ = agent.ranker.rank(st, 1000)
    return docs.index(target_doc) + 1 if target_doc in docs else None

print("\nrank of the true target given all 4 constraints (200 sessions):")
lex, vecr, hyb = [], [], []
for s in samples:
    card, _ = materialize_hidden_fields(s, products)
    cons = list(dict.fromkeys(card["hard_constraints"] + card["soft_preferences"]))
    tgt = store.ord_of[str(s["ground_truth"]["parent_asin"])]

    lex.append(lexical_rank(cons, tgt))

    qv = normalize(svd.transform(vec.transform([norm(" ".join(cons))]))).astype(np.float32)
    sims = (D @ qv.T)[:, 0]
    order = np.argsort(-sims)
    pos = int(np.where(order == tgt)[0][0]) + 1
    vecr.append(pos if pos <= 1000 else None)

    lr = lex[-1] or 10_000
    hyb.append(min(lr, pos) if pos <= 1000 else lr)

def report(name, ranks):
    hit10 = sum(1 for r in ranks if r and r <= 10) / len(ranks)
    mrr = sum(1.0 / r for r in ranks if r) / len(ranks)
    top1 = sum(1 for r in ranks if r == 1) / len(ranks)
    print(f"  {name:<22} hit@10 {hit10:.3f}   MRR {mrr:.3f}   rank-1 {top1:.3f}")

report("lexical (current)", lex)
report("vector only", vecr)
report("best-of-both (oracle)", [r if r != 10_000 else None for r in hyb])

only_vec = sum(1 for l, v in zip(lex, vecr) if (l is None or l > 10) and v and v <= 10)
print(f"\n  sessions vector rescues that lexical misses at top-10: {only_vec}/200")
