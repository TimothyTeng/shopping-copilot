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
    """Turns dialogue state into an ordered top-k.

    Everything optional (BM25, associations, doc2query, dense vectors) arrives
    as a constructor argument that may be None, so the default graded path is
    the stdlib conjunctive scorer and nothing else.
    """

    def __init__(self, store: CatalogStore, index: InvertedIndex, cfg,
                 bm25=None, assoc=None, d2q=None, dense=None) -> None:
        """Wire in whichever signals the config actually asked for."""
        self.assoc = assoc        # mined term associations; None unless enabled
        self.d2q = d2q            # BM25 over generated shopper queries, or None
        self.dense = dense        # bi-encoder vectors, or None
        self.store = store
        self.index = index
        self.cfg = cfg
        self.bm25 = bm25          # built only when a mode needs it
        self._title_toks: dict[int, frozenset[str]] = {}   # lazy, for MMR only

    def _title_set(self, doc: int) -> frozenset[str]:
        """Memoised title tokens. Only MMR needs these, so they are built lazily
        rather than for all 50,000 products up front."""
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

    def _card_bonus(self, doc: int, disclosed: set[str]) -> float:
        """How many disclosed constraints are slots of *this product's own card*.

        Coverage asks "does this document contain the words". This asks the
        sharper question the simulator's own construction allows: "would this
        product have produced that constraint string in the first place?" The
        constraints are `intent_card` slots of the target, and `catalog.card`
        mirrors that function exactly, so the target matches every disclosed
        constraint BY CONSTRUCTION — the bonus can raise a tie-mate above the
        target only if that mate would have generated the identical card slot,
        which is the definition of indistinguishable.

        Measured before building (docs/algorithm-audit.md §2.5): of the 24
        sessions not at rank 1, this clears every product above the target in 11
        and some of them in 13 more, and it demotes the target in none.
        """
        if not disclosed:
            return 0.0
        card = self.store.card[doc]
        return float(sum(1 for value in disclosed if value in card))

    def _cluster_rerank(self, scored: list[tuple[float, float, int]],
                        top_k: int) -> list[int]:
        """Round-robin the head over near-duplicate clusters.

        The parameter-free sibling of `_mmr_rerank`, and the same hypothesis
        stated as a model rather than as a trade-off: the conjunctive score is a
        biased estimate of P(this is the target), because a family of
        near-identical listings splits the probability mass that one product
        deserves. Grouping the family and taking one member before any second
        member spends the top-10 on distinct products instead of on one family.

        No lambda: the relevance/diversity trade MMR tunes is replaced by a
        single equivalence threshold, and ordering within and across clusters is
        pure relevance. Like MMR it cannot demote the leader — the first cluster
        emitted is the leader's.
        """
        pool = scored[: self.cfg.mmr_pool]
        if len(pool) <= 1:
            return [d for _, _, d in scored[:top_k]]
        threshold = self.cfg.cluster_threshold
        clusters: list[list[int]] = []          # each best-first, by construction
        for _total, _prior, doc in pool:
            for members in clusters:
                if self._similarity(doc, members[0]) >= threshold:
                    members.append(doc)
                    break
            else:
                clusters.append([doc])
        picked: list[int] = []
        depth = 0
        while len(picked) < top_k and any(len(c) > depth for c in clusters):
            for members in clusters:
                if len(members) > depth:
                    picked.append(members[depth])
                    if len(picked) >= top_k:
                        break
            depth += 1
        if len(picked) < top_k:
            picked += [d for _, _, d in scored[self.cfg.mmr_pool:]][: top_k - len(picked)]
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

    def _profile_affinity(self, state: DialogueState) -> float | None:
        """The rating the shopper's history says they end up buying.

        `user_profile` is handed to `reset()` on every session and has never
        been read. Measured on the public set: corr(average_prior_rating,
        the target's own average_rating) = +0.182, and targets sit above the
        catalog mean (4.37 vs 4.09). Weak, but it is free signal, and it lands
        exactly where the remaining benchmark points are — inside tied groups
        the constraints cannot separate.
        """
        if not self.cfg.profile_affinity:
            return None
        value = (state.profile or {}).get("average_prior_rating")
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

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
            """BM25 length normalisation: how much to discount a long field."""
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
        cfg = self.cfg
        query = state.transcript()
        if cfg.bm25_drop_stopwords:
            query = " ".join(t for t in tokens(query) if t not in DIALOGUE_STOP)
        if cfg.rm3:
            base = [doc for doc, _ in self.bm25.rm3(query, cfg, cfg.rrf_depth)]
        elif self.assoc is not None:
            base = [doc for doc, _ in
                    self.bm25.expanded(query, cfg, cfg.rrf_depth, self.assoc)]
        else:
            base = [doc for doc, _ in self.bm25.score(query, cfg, cfg.rrf_depth)]

        if self.d2q is not None:
            # A second opinion in the shopper's own register. Fused, not merged:
            # the two rankings are on incomparable scales and RRF reads only
            # positions, so the generated side can add a candidate but cannot
            # dilute the shopper's rare words (the `union` failure, recorded in
            # config.hyde_bm25_mode).
            base = self._fuse(
                [base, [doc for doc, _ in self.d2q.score(query, cfg, cfg.rrf_depth)]],
                cfg.rrf_k,
                [1.0 - cfg.doc2query_weight, cfg.doc2query_weight],
            )

        if self.dense is not None and cfg.dense_weight > 0.0:
            # The third opinion, and the only one that is not lexical at all.
            # Same instrument and the same argument as doc2query above: dense
            # loses to BM25 as a sole retriever and beats it fused, because the
            # two miss different documents. RRF is what lets a cosine and a
            # BM25 sum be combined without inventing a scale between them.
            dense = [doc for doc, _ in
                     self.dense.score(query, cfg, cfg.dense_depth)]
            if dense:
                base = self._fuse([base, dense], cfg.rrf_k,
                                  [1.0 - cfg.dense_weight, cfg.dense_weight])

        mode = cfg.hyde_bm25_mode
        if not state.hyde_text or mode == "off":
            return base
        if mode == "union":
            # MEASURED AND REJECTED as the default. Pouring the generated terms
            # into one query dilutes it: generic listing vocabulary ("soft",
            # "durable", "comfortable") outvotes the shopper's own words and
            # pulls in the wrong products. Stress hit@10 0.846 -> 0.769, losing
            # two targets outright, while MRR rose 0.305 -> 0.397. The signal was
            # real and the delivery was wrong. Kept as a switch.
            return [doc for doc, _ in
                    self.bm25.score(f"{query} {state.hyde_text}", cfg, cfg.rrf_depth)]
        # `fuse` — keep the transcript ranking intact and let the generated
        # listing contribute a SECOND opinion. Fusion is the right instrument
        # because a fused list contains the union of both inputs, so the
        # generation can only add recall, never spend it. RRF also reads
        # positions only, which matters here: the two queries are on
        # incomparable scales (one is the shopper, one is a fabrication).
        return self._fuse(
            [base, list(state.hyde_ranking)],
            cfg.rrf_k,
            [1.0 - cfg.hyde_rrf_weight, cfg.hyde_rrf_weight],
        )

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
        # The generated listing contributes CANDIDATES, not just an ordering.
        # Half the natural-language loss is recall (stress hit@10 0.500), and a
        # product the constraints never reached cannot be rescued by reranking.
        hyde = frozenset(state.hyde_ranking)
        if hyde:
            for doc in hyde:
                raw[doc] = raw.get(doc, 0.0) + cfg.hyde_prefilter_mass
        if not raw:
            return [], 0

        candidates = sorted(raw, key=raw.__getitem__, reverse=True)[: cfg.candidate_cap]
        category_set = set(category)
        category_bonus = cfg.category_bonus
        if cfg.gate_category_bonus:
            category_bonus *= state.category_confidence

        scored: list[tuple[float, float, int]] = []
        satisfied = 0
        floor = cfg.slot_cover_floor
        affinity = self._profile_affinity(state)
        for doc in candidates:
            total = 0.0
            full = True
            for acc, slot in coverages:
                cover = acc.get(doc, 0.0) / slot.idf_total
                if floor < 1.0:
                    # A slot is "met" once it covers `slot_cover_floor` of its own
                    # IDF mass, not all of it. `_salient_runs` bridges connectors
                    # into a span ("rubber sole plus design"), and `_coverage`
                    # then requires every token of it — so a glue word becomes a
                    # mandatory conjunction term. Charging only for the mass that
                    # matters lets the requirement survive its own connective
                    # tissue. Exactly 1.0 restores the original hard behaviour.
                    cover = min(1.0, cover / floor)
                if cover < 0.999:
                    full = False
                total += slot.weight * math.log(cfg.log_epsilon + cover ** cfg.coverage_gamma)
            if full:
                satisfied += 1
            if doc in category_set:
                total += category_bonus
            if hyde and doc in hyde:
                # Additive, exactly like the category bonus. Deliberately NOT a
                # slot: a generated term inside the conjunction would have to be
                # satisfied by every candidate, and one bad guess would empty the
                # pool — the documented "constraints only accumulate" failure.
                # As a bonus, a bad generation costs precision and nothing else.
                total += cfg.hyde_bonus
            if doc in shown:
                total -= cfg.shown_penalty
            prior = self._prior(doc)
            if affinity is not None:
                # Closer to the shopper's habitual rating is better; the weight
                # keeps this strictly a tie-break, never a ranking signal.
                prior -= self.cfg.profile_affinity_weight * abs(
                    self.store.rating_avg[doc] - affinity)
            scored.append((total, prior, doc))

        scored.sort(key=lambda row: (-row[0], -row[1]))

        # Rescore only the head, where ordering actually decides the outcome.
        head = scored[: cfg.phrase_verify_top]
        phrases = [s.phrase for _, s in coverages if s.phrase]
        disclosed = state.disclosed if cfg.card_signature else set()
        if head:
            rescored: list[tuple[float, float, int]] = []
            for total, prior, doc in head:
                bonus = sum(
                    cfg.phrase_bonus for phrase in phrases if self.store.contains(doc, phrase)
                )
                if disclosed:
                    bonus += cfg.card_bonus * self._card_bonus(doc, disclosed)
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
            if cfg.tie_rerank == "cluster":
                return self._cluster_rerank(scored, top_k), satisfied
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
