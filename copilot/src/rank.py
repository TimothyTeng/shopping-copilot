"""Candidate generation and scoring.

Scoring is a **log-product** over constraints, not a sum. The hidden requirements
are a conjunction, and an additive score lets a product that nails one constraint
outrank a product that satisfies all of them — which is exactly the tie that
costs precision at the top of the list.

Exact phrase matches are a *bonus*, never a filter, so a paraphrased requirement
still competes on token coverage.
"""
from __future__ import annotations

import math
from array import array
from collections import Counter

from .catalog import CatalogStore
from .index import InvertedIndex
from .normalize import DIALOGUE_STOP, tokens
from .state import DialogueState, Slot


class Ranker:
    def __init__(self, store: CatalogStore, index: InvertedIndex, cfg,
                 bm25=None) -> None:
        self.store = store
        self.index = index
        self.cfg = cfg
        self.bm25 = bm25          # built only when a mode needs it
        self._title_toks: dict[int, frozenset[str]] = {}   # lazy, for MMR only

    def _title_set(self, doc: int) -> frozenset[str]:
        cached = self._title_toks.get(doc)
        if cached is None:
            cached = frozenset(tokens(self.store.title[doc]))
            self._title_toks[doc] = cached
        return cached

    def _similarity(self, a: int, b: int) -> float:
        """Title-token Jaccard: a proxy for 'these are the same product'."""
        ta, tb = self._title_set(a), self._title_set(b)
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        return inter / (len(ta) + len(tb) - inter)

    def _mmr_rerank(self, scored: list[tuple[float, float, int]],
                    top_k: int) -> list[int]:
        """Reorder the head by Maximal Marginal Relevance, leave the tail as-is.

        `scored` is (total, prior, doc) sorted best-first. Only the top
        `mmr_pool` compete for reordering; everything below keeps its order, so
        the change is confined to the region where ties actually occur.
        """
        cfg = self.cfg
        pool = scored[: cfg.mmr_pool]
        if len(pool) <= 1:
            return [d for _, _, d in scored[:top_k]]
        totals = [t for t, _, _ in pool]
        lo, hi = min(totals), max(totals)
        span = (hi - lo) or 1.0
        lam = cfg.mmr_lambda

        remaining = list(pool)
        picked: list[int] = []
        # The first pick has no selected neighbour, so it is the pure-relevance
        # leader — this is what keeps a clean rank-1 target at rank 1.
        while remaining and len(picked) < top_k:
            best_i, best_val = 0, None
            for i, (total, _prior, doc) in enumerate(remaining):
                rel = (total - lo) / span
                sim = max((self._similarity(doc, p) for p in picked), default=0.0)
                val = lam * rel - (1.0 - lam) * sim
                if best_val is None or val > best_val:
                    best_val, best_i = val, i
            picked.append(remaining.pop(best_i)[2])
        # Fill from the untouched tail if the pool was smaller than top_k.
        if len(picked) < top_k:
            picked += [d for _, _, d in scored[cfg.mmr_pool:]][: top_k - len(picked)]
        return picked[:top_k]

    def _coverage(self, slot: Slot) -> dict[int, float]:
        """doc -> IDF mass of this slot's tokens present in that doc."""
        acc: dict[int, float] = {}
        for term, weight in zip(slot.tokens, slot.idfs):
            if not self.index.informative(term, self.cfg.max_token_df_ratio):
                continue
            for doc in self.index.docs_for(term):
                acc[doc] = acc.get(doc, 0.0) + weight
        return acc

    def _prior(self, doc: int) -> float:
        """Query-independent evidence for a product the constraints cannot rank.

        With a median tied-group size of 13, this decides ~19% of sessions.
        """
        mode = self.cfg.tiebreak
        if mode == "none":
            return 0.0
        s = self.store
        count, avg = s.rating_n[doc], s.rating_avg[doc]
        popularity = math.log1p(count)
        if mode == "popularity":
            return popularity
        bayes = (avg * count + 4.0 * 50.0) / (count + 50.0)   # smoothed quality
        if mode == "bayes":
            return bayes
        price = s.price[doc]
        has_price = 0.0 if price != price else 1.0            # NaN means unknown
        if mode == "price_pop":
            return 10.0 * has_price + popularity
        if mode == "price_bayes":
            return 10.0 * has_price + bayes
        return popularity

    def _bm25f(self, doc: int, coverages: list) -> float:
        """BM25F over title / categories / body.

        Coverage saturates at 1.0 for every product containing all the query
        terms, which is what makes near-identical products tie. BM25F separates
        them: it counts how *often* a term occurs and penalizes long documents,
        and it weights a title match above one buried in a long description.
        """
        s, cfg = self.store, self.cfg
        tf_title = Counter(tokens(s.title[doc]))
        tf_cat = Counter(tokens(s.category[doc]))
        tf_all = Counter(tokens(s.text[doc]))

        def norm_len(length: int, avg: float) -> float:
            return 1.0 - cfg.b_norm + cfg.b_norm * (length / avg if avg else 1.0)

        d_title = norm_len(s.len_title[doc], s.avg_title)
        d_cat = norm_len(s.len_cat[doc], s.avg_cat)
        d_body = norm_len(s.len_body[doc], s.avg_body)

        total = 0.0
        for _, slot in coverages:
            acc = 0.0
            for term, idf in zip(slot.tokens, slot.idfs):
                n_t = tf_title.get(term, 0)
                n_c = tf_cat.get(term, 0)
                n_b = max(tf_all.get(term, 0) - n_t - n_c, 0)
                ntf = (cfg.boost_title * n_t / d_title
                       + cfg.boost_cat * n_c / d_cat
                       + cfg.boost_body * n_b / d_body)
                if ntf > 0.0:
                    acc += idf * ntf / (ntf + cfg.k1)
            total += slot.weight * acc / slot.idf_total
        return total

    def _bm25_ranking(self, state: DialogueState) -> list[int]:
        """Rank by BM25 over everything the shopper actually said."""
        if self.bm25 is None:
            return []
        query = state.transcript()
        if self.cfg.bm25_drop_stopwords:
            query = " ".join(t for t in tokens(query) if t not in DIALOGUE_STOP)
        return [doc for doc, _ in
                self.bm25.score(query, self.cfg, self.cfg.rrf_depth)]

    @staticmethod
    def _fuse(rankings: list[list[int]], k: float,
              weights: list[float] | None = None) -> list[int]:
        """Reciprocal-rank fusion.

        Chosen over a weighted score sum because the two rankings are on
        incomparable scales — one is a log-product of coverage ratios, the other
        a sum of saturating BM25 terms — and normalizing them would introduce a
        tuning knob per turn. RRF only reads positions.
        """
        points: dict[int, float] = {}
        weights = weights or [1.0] * len(rankings)
        for ranking, weight in zip(rankings, weights):
            for position, doc in enumerate(ranking):
                points[doc] = points.get(doc, 0.0) + weight / (k + position + 1)
        return sorted(points, key=points.__getitem__, reverse=True)

    def rank(self, state: DialogueState, top_k: int) -> tuple[list[int], int]:
        """Return (ranked doc ordinals, size of the fully-satisfying pool)."""
        cfg = self.cfg
        slots = state.active_slots()
        category: array = state.category_docs if state.category_docs is not None else array("i")
        shown = state.shown() if cfg.demote_shown else set()

        if not slots:
            # No grounded constraint. On the benchmark that means the shopper
            # has not disclosed anything yet; on real input it usually means
            # nothing they said could be grounded — which is precisely when a
            # prose retriever has something to offer and the slot path does not.
            ranked = self._bm25_ranking(state)
            if ranked:
                self.last_scored = [(d, 0.0, 0.0) for d in ranked[:20]]
                return ranked[:top_k], 0
            pool = list(category)
            pool.sort(key=lambda d: (d in shown, -self._prior(d)))
            self.last_scored = [(d, 0.0, float(self.store.rating_n[d])) for d in pool[:20]]
            return pool[:top_k], len(pool)

        coverages = [(self._coverage(s), s) for s in slots]

        # Prefilter on raw coverage mass so full scoring runs on a bounded set.
        raw: dict[int, float] = {}
        for acc, slot in coverages:
            w = slot.weight
            for doc, mass in acc.items():
                raw[doc] = raw.get(doc, 0.0) + w * mass
        for doc in category:
            raw[doc] = raw.get(doc, 0.0) + 0.5
        if not raw:
            return [], 0

        candidates = sorted(raw, key=raw.__getitem__, reverse=True)[: cfg.candidate_cap]
        category_set = set(category)
        category_bonus = cfg.category_bonus
        if cfg.gate_category_bonus:
            category_bonus *= state.category_confidence

        scored: list[tuple[float, float, int]] = []
        satisfied = 0
        for doc in candidates:
            total = 0.0
            full = True
            for acc, slot in coverages:
                cover = acc.get(doc, 0.0) / slot.idf_total
                if cover < 0.999:
                    full = False
                total += slot.weight * math.log(cfg.log_epsilon + cover ** cfg.coverage_gamma)
            if full:
                satisfied += 1
            if doc in category_set:
                total += category_bonus
            if doc in shown:
                total -= cfg.shown_penalty
            prior = self._prior(doc)
            scored.append((total, prior, doc))

        scored.sort(key=lambda row: (-row[0], -row[1]))

        # Rescore only the head, where ordering actually decides the outcome.
        head = scored[: cfg.phrase_verify_top]
        phrases = [s.phrase for _, s in coverages if s.phrase]
        if head:
            rescored: list[tuple[float, float, int]] = []
            for total, prior, doc in head:
                bonus = sum(
                    cfg.phrase_bonus for phrase in phrases if self.store.contains(doc, phrase)
                )
                if cfg.bm25f_weight:
                    bonus += cfg.bm25f_weight * self._bm25f(doc, coverages)
                rescored.append((total + bonus, prior, doc))
            rescored.sort(key=lambda row: (-row[0], -row[1]))
            scored = rescored + scored[cfg.phrase_verify_top:]

        # Diagnostics: keep the head with scores so the harness can attribute
        # ranking losses without re-deriving them.
        self.last_scored = [(d, tot, pri) for tot, pri, d in scored[:20]]
        conjunctive = [doc for _, _, doc in scored]

        # `satisfied` is computed above in every mode, because the disclosure
        # gate reads it. Only the ordering changes below.
        mode = cfg.retrieval
        if mode == "auto":
            # Which retriever suits this input is not a guess — the input says
            # so. When the shopper's words were lifted from a real product, at
            # least one product satisfies every constraint by construction, and
            # the conjunctive score is exact. When they are the shopper's own
            # words, nothing satisfies all of them and the conjunction is
            # scoring a query that no document can answer, so prose retrieval
            # is the honest fallback.
            mode = "conjunctive" if satisfied > 0 else cfg.auto_fallback

        if mode == "conjunctive":
            if cfg.tie_rerank == "mmr":
                return self._mmr_rerank(scored, top_k), satisfied
            return conjunctive[:top_k], satisfied
        if mode == "bm25":
            ordered = self._bm25_ranking(state) or conjunctive
        else:
            ordered = self._fuse(
                [conjunctive[: cfg.rrf_depth], self._bm25_ranking(state)],
                cfg.rrf_k,
                [cfg.rrf_weight, 1.0 - cfg.rrf_weight],
            )
        if shown:
            ordered = [d for d in ordered if d not in shown] + \
                      [d for d in ordered if d in shown]
        return ordered[:top_k], satisfied
