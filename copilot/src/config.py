"""Tunables and ablation switches.

Every knob that changes behaviour lives here so experiments are a one-line diff
and the harness can sweep them without touching logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# The official competition kit. Read-only: we import its evaluator and read its
# catalog, but never write to it.
KIT_ROOT = Path(__file__).resolve().parents[2] / "techjam-conversational-search"
CATALOG_PATH = KIT_ROOT / "data" / "catalog.jsonl"
PUBLIC_SET_PATH = KIT_ROOT / "data" / "public_set.jsonl"


@dataclass
class Settings:
    # --- extraction -------------------------------------------------------
    use_templates: bool = True      # Layer A: regex fast path
    use_salience: bool = True       # Layer C: catalog-grounded (the robust core)
    # IDF is log((N-df+.5)/(df+.5)+1): ~1.0 admits terms in <=30% of the catalog,
    # which keeps real attribute words like "material" while dropping filler.
    min_span_idf: float = 1.0       # a token must be this informative to seed a span
    max_span_tokens: int = 12
    # Splitting spans at catalog-grounding seams removes conversational glue that
    # would otherwise fuse two requirements into one span matching nothing.
    # Originally off: it cost ~0.03 on the evaluator's own phrasing. After the
    # category-suppression and override-cue fixes that cost vanished, and it is
    # now worth ~+0.20 on heavily noised input at no cost on L0. Re-measured
    # rather than assumed — see the robustness table.
    segment_spans: bool = True
    # How to cut a run into requirements once we have decided to cut it.
    #   greedy — first grounded window wins, left to right
    #   rarity — the segmentation explaining the most tokens with the rarest
    #            catalog phrases (see extract._segment_rarity)
    #
    # MEASURED AND REJECTED as a default. `rarity` fixes the specific defect it
    # was built for ("rubber sole plus design in usa" no longer splits as
    # "plus design" + "in usa") but does not pay: L0 and L1 are *identical* at
    # every setting, and piece_cost only trades L2 against L4 along a monotonic
    # frontier — 0.6755/0.9124 at one end, 0.6186/0.9309 at the other, against
    # greedy's 0.6493/0.9300. It moves ~7 sessions between two synthetic
    # renderers for 70% more runtime. Kept as a switch.
    segment_mode: str = "greedy"
    # Fixed cost per piece, for `rarity` only. Without it the search is biased
    # toward splitting, because every extra piece contributes its own rarity
    # term for the same tokens. Dropping a token is charged at its IDF.
    piece_cost: float = 3.0
    skip_penalty: float = 1.0
    bridge_gap: int = 1             # low-salience tokens a span may bridge

    # --- category resolution ---------------------------------------------
    # How the coarse category is recovered from the shopper's message. The
    # substring path (exact bucket name found in the message) is ALWAYS tried
    # first and is what the benchmark rides on, so these switches only change
    # the fallback — the natural-language path.
    #   overlap — token overlap over bucket labels at a 0.5 threshold (legacy).
    #             Counts tokens equally, so "men's dive watch" resolves to
    #             `men hoodies` on the shared generic token. Fired wrong on
    #             17/26 stress probes.
    #   vote    — poll the products the message actually retrieves (BM25) and
    #             take their majority coarse category, weighted by match mass.
    #             Uses whole product text, so it survives category renaming.
    #
    # MEASURED: `vote` is benchmark-neutral (0.9383 -> 0.9383, the substring path
    # still wins there at confidence 1.0) and worth +0.037 on the natural-language
    # stress set (0.3749 -> 0.4118, Hit@10 0.462 -> 0.500 — a recall gain, not a
    # reorder; the `colloquial` tag went 0.000 -> 0.320). n=26, so read the
    # direction, not the third decimal. Flipped to `vote` as the default since it
    # dominates `overlap` — equal on the benchmark, better on natural language.
    #   classifier — an offline-trained char-n-gram linear classifier (see
    #                tools/train_category.py); falls back to `vote` when its
    #                prediction is below category_min_confidence.
    #
    # MEASURED: `classifier` is benchmark-neutral (0.9383) and the best natural-
    # language result yet — stress 0.4191 -> 0.4577 (+0.039), Hit@10 0.500 ->
    # 0.577, above the stateless BM25 control's 0.538. It trades a little ranking
    # precision for recall (MRR 0.289 -> 0.244) and is noisy per-probe. Costs a
    # 0.7 MB shipped model + a training step; runtime stays pure stdlib. n=26.
    # Default, since the composite (the graded metric) is what rises. If the
    # model file is absent the resolver degrades gracefully to `vote`.
    #
    #   ensemble   — MEASURED AND REJECTED. Unioning the classifier and vote
    #                buckets to "keep the best of both" collapses exactly back to
    #                vote (0.4577 -> 0.4191): adding vote's bucket re-boosts its
    #                products, which re-dilute the focused boost the classifier's
    #                gain depends on. The two are right on different probes and
    #                nothing at inference says which to trust. Kept as a switch.
    category_resolver: str = "classifier"
    category_vote_top: int = 40         # products polled for the majority vote
    category_min_confidence: float = 0.0  # reject a resolution weaker than this
    # Whether the resolved category's tokens are suppressed from being mined as
    # constraints. Measured (oracle ceiling study): giving the ranker the TRUE
    # bucket is worth +0.053 on natural language, but forcing the true *key*
    # nets -0.037 because it also suppresses more of the shopper's own category
    # words from extraction, and on free text those words carry coverage signal.
    # The two are the same knob today. This switch decouples them so the ranking
    # bucket can improve without the extraction penalty. Benchmark expectation:
    # the category slot reinforces the category signal there (removing it from
    # the template path measured -0.007), so this is measured on both suites.
    suppress_category_tokens: bool = True
    # Category is resolved once (turn 1) and frozen. Re-resolving on every later
    # turn (switch only to a *confident, different* bucket) was meant to catch
    # "actually never mind, I need women's sweatpants", which carries no override
    # cue.
    #
    # MEASURED AND REJECTED: it cost the benchmark a hit (0.9383 -> 0.9347,
    # Hit@10 1.000 -> 0.995) because some mid-session answers voted confidently
    # enough to switch to a wrong bucket, and it did NOT fix the target case
    # (category_switch stayed 0.000). The switch probe deposits shoe constraint
    # slots on turns 1-2 that are never retracted, so flipping the category still
    # leaves the ranking poisoned by the old constraints. The real fix is
    # constraint retraction, not re-resolution. Kept as a switch. A confidence
    # threshold cannot cleanly separate a genuine switch from an attribute answer.
    reresolve_category: bool = False
    reresolve_min_confidence: float = 0.5
    # Re-resolve the category over the ACCUMULATED transcript (every message so
    # far), not the single latest message, and only while the current resolution
    # is a low-confidence fallback — a substring hit locks at confidence 1.0 and
    # is never revisited. The benchmark opens with the exact bucket name, so it
    # locks on turn 1 and this is a no-op there BY CONSTRUCTION. Natural-language
    # openers are vague ("cosy slippers for my wife" -> wrong bucket); later
    # turns ("real leather not synthetic") disambiguate the department, and a
    # vote over the full transcript ranks the true bucket #1 for 5 of 6 stuck
    # stress probes. Distinct from `reresolve_category`, which re-resolved
    # per-message with no lock gate and flip-flopped into wrong buckets on the
    # benchmark (measured -0.0036).
    #
    # MEASURED AND REJECTED. Benchmark-neutral BY CONSTRUCTION and verified so:
    # 0.9456 -> 0.9456, byte-identical across every sub-scenario, because the
    # opener's substring bucket-name locks confidence at 1.0 on turn 1 and this
    # never fires. But it is stress-NEGATIVE, because the vote-over-full-
    # transcript diagnostic was misleading: mid-session re-votes are not
    # monotonic toward truth, so accumulating unrelated constraints drag a
    # correct turn-1 bucket off target. With the vote resolver it cost the best
    # static combo -0.023 (none+vote+suppress=False 0.4992 -> 0.4765) and vote
    # alone -0.066 (0.4577 -> 0.3921, losing ~3 hits); with the classifier
    # resolver it is a no-op (the classifier never falls through to vote). It
    # DOES fix the one "vague opener, sharp later turns" probe it was aimed at
    # (contradiction_leather_canvas rank 8 -> 1), but the 4 hardest probes
    # (loafer/flat/switch/turtleneck) miss regardless, because their failure is
    # over-determined: wrong bucket AND colloquial words ("cosy slippers") mined
    # as poisoning constraint slots. Fixing the bucket alone is insufficient; the
    # residual is the vocabulary bottleneck. Kept as a switch.
    resolve_on_transcript: bool = False
    transcript_resolve_min_conf: float = 0.25  # vote mass share to adopt a re-resolution
    # Constraint retraction. Slots otherwise only accumulate, so "actually not
    # leather, canvas is better" ends up requiring BOTH and the conjunction
    # matches nothing.
    #   enable_retraction — a negation cue ("not X", "instead of X", "rather than
    #                       X") revokes the active slot(s) matching X; the rest of
    #                       the turn is added normally.
    #   enable_reset      — a reset cue ("never mind", "scratch that", "start
    #                       over") revokes ALL slots, clears the category, and
    #                       re-resolves from this message.
    # The benchmark uses neither cue (its override is "ignore ... what I need is:
    # X"), so both are benchmark-neutral (measured: 0.9383 -> 0.9383 for each).
    #
    # MEASURED: retraction is a clean win — on the stress set it took the
    # `contradiction` probe ("not leather, canvas is better") from 0.000 to 0.940
    # (rank 1) and lifted the whole set +0.007 (0.4118 -> 0.4191). Default ON.
    #
    # `enable_reset` is a CORRECT mechanism that measures neutral (0.4118, no
    # harm, no gain) and is left OFF: the one probe it targets, category_switch,
    # stays 0.000 because the *vote* category resolver mis-buckets "women's
    # sweatpants" (conf 0.15) — the blocker is resolution precision, not
    # accumulation, so reset cannot pay until that is fixed. Kept as a switch.
    enable_retraction: bool = True
    enable_reset: bool = False
    # Scale category_bonus by resolution confidence instead of applying it flat,
    # so a low-confidence fallback bucket cannot poison the ranking the way a
    # wrong bucket at the full 2.0 does. Substring hits score confidence 1.0, so
    # the benchmark is unaffected.
    #
    # MEASURED AND REJECTED as a default: benchmark-neutral, but on the stress set
    # it cost -0.010 against `vote` alone (0.4118 -> 0.4022) — scaling the bonus
    # down on correct-but-low-confidence votes demoted their targets (MRR
    # 0.257 -> 0.225, `brand` tag 0.485 -> 0.425). Kept as a switch.
    gate_category_bonus: bool = False

    # --- fuzzy token repair (natural-language typo tolerance) ------------
    # Shopper tokens ABSENT from the catalog (df==0) are mapped to the nearest
    # catalog term within a bounded Damerau-Levenshtein distance, via a trigram
    # candidate index (see fuzzy.py). Only absent alpha tokens are touched, so
    # every benchmark token — lifted verbatim from the target, df>0 — is a
    # no-op and the graded path is unaffected BY CONSTRUCTION, not by luck.
    # Targets the documented typo weakness (rainocat->raincoat,
    # waterprrof->waterproof, lightwieght->lightweight, hoood->hood).
    fuzzy_repair: bool = False
    fuzzy_min_len: int = 4          # never repair a token shorter than this
    fuzzy_df_floor: int = 3         # a repair target must be at least this common
    fuzzy_candidates: int = 40      # trigram-ranked candidates verified per token
    fuzzy_repair_present: bool = False  # also re-map present-but-rare tokens (risky)

    # --- retrieval --------------------------------------------------------
    candidate_cap: int = 4000
    phrase_verify_top: int = 400    # verify exact phrases on this many candidates
    max_token_df_ratio: float = 0.35  # ignore tokens appearing in >35% of catalog

    # --- scoring ----------------------------------------------------------
    # BM25F rescoring of the head. MEASURED AND REJECTED: every configuration
    # tried (title-boosted, uniform, body-only, with and without length
    # normalization) cost ~0.04 and dropped rank-1 placements from 162 to ~130.
    # Term frequency assumes the query was authored independently of the
    # document; here the constraints are lifted from the target's own text, so
    # "mentions it more often" is noise, not evidence. Kept as a switch.
    bm25f_weight: float = 0.0
    k1: float = 1.2
    b_norm: float = 0.6
    boost_title: float = 3.0
    boost_cat: float = 2.0
    boost_body: float = 1.0

    # How the final ordering is produced.
    #   conjunctive — log-product over extracted constraint slots (see rank.py)
    #   bm25        — BM25 over the raw transcript, no slots, no category
    #   rrf         — reciprocal-rank fusion of the two
    #   auto        — conjunctive when some product satisfies every constraint,
    #                 `auto_fallback` when none does
    # The two disagree because they suit different inputs: conjunctive wins when
    # the shopper's words were lifted from the target (the official simulator),
    # BM25 wins when they are the shopper's own (tools/stress.py). Measure both
    # before changing this.
    retrieval: str = "conjunctive"
    auto_fallback: str = "bm25"    # bm25 | rrf
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    rrf_weight: float = 0.5         # share of fusion mass on the conjunctive side
    rrf_k: float = 60.0             # standard RRF damping
    rrf_depth: int = 200            # how deep each ranking contributes
    bm25_drop_stopwords: bool = True

    coverage_gamma: float = 1.5     # concavity on per-slot coverage
    log_epsilon: float = 0.02       # floor inside log-product
    phrase_bonus: float = 1.6
    category_bonus: float = 2.0
    soft_slot_weight: float = 0.6   # weight for revoked/soft slots
    # An oracle that asks for the sharpest constraint first is worth +0.010.
    # We cannot choose which constraint arrives, but we can weight the ones we
    # have: "color: grey" matches 1 product, "polyester" matches 3,085.
    # MEASURED AND REJECTED (-0.002): at full coverage every slot contributes
    # the same regardless of weight, so this cannot break the exact ties that
    # cause most remaining errors. Kept as a switch.
    specificity_weighting: bool = False
    specificity_floor: float = 0.2
    # How to order products the constraint score cannot separate. Measured: the
    # target sits at the ~84th percentile of its tied group by review count, so
    # a product-side prior genuinely carries signal here.
    tiebreak: str = "price_pop"     # popularity | bayes | price_pop | price_bayes | none
    # Diversity re-rank of the head, to spread a top-k list across a tied cluster
    # instead of filling it with near-duplicate siblings. Maximal Marginal
    # Relevance: greedily pick the item maximizing
    #   lambda * relevance - (1 - lambda) * max_similarity_to_already_picked
    # over a bounded pool, with similarity = title-token Jaccard.
    #   none — off.
    #   mmr  — on. Structurally cannot demote the single top-scored item (its
    #          first pick has no competitor to be similar to), so a clean rank-1
    #          target is safe; it only reshuffles items *below* the leader that
    #          are close enough in score to be reachable within the pool.
    # MEASURED, and the reverse of the intuition. It HELPS the benchmark and
    # HURTS conjunctive natural language:
    #   benchmark  0.9383 -> 0.9456 (MRR 0.883 -> 0.911, hit stays 1.000). The
    #     gain is concentrated in `browsing` (MRR 0.819 -> 0.889, +0.070) — the
    #     exploratory sessions with the weakest constraints and thus the most
    #     exact-score ties. `buying` barely moves (+0.004), `intent_override`
    #     dips -0.008 (n=30). Flat across lambda 0.5-0.9 (0.9445-0.9459), so it
    #     is a plateau, not a fit. The +0.007 looks small against the ±0.03
    #     noise floor, but that floor is the UNPAIRED absolute-score band; a
    #     paired bootstrap is the right instrument (see below).
    #   stress     0.4946 -> 0.4354 on the conjunctive path: when relevance is
    #     already shaky, trading it for diversity pushes a barely-ranking target
    #     out. So MMR pays only where relevance is trustworthy enough that the
    #     near-ties are genuine equals — which is the benchmark, not free text.
    # Only wired into the conjunctive return, so it is INERT on the bm25 NL
    # surface (demo chat) and cannot harm it.
    #
    # NOW THE DEFAULT, settled on evidence. A paired bootstrap
    # (`harness ci --compare tie_rerank=mmr`, 10k resamples) puts the +0.0074
    # delta at a 95% CI of [+0.0017, +0.0138], P(delta>0)=0.997 — significant.
    # With hit fixed at 1.000 the paired interval is far tighter than the
    # unpaired ±0.03, so the earlier "inside the noise band" caveat was the wrong
    # test, not the wrong result.
    #
    # This partly corrects the "~18 ties are information-theoretically
    # unfixable" claim: they are unfixable by adding constraint *information*,
    # but a query-independent diversity prior over the tied set correlates with
    # the target often enough to reorder about half of them the right way.
    tie_rerank: str = "mmr"         # none | mmr
    mmr_lambda: float = 0.7         # 1.0 = pure relevance, 0.0 = pure diversity
    mmr_pool: int = 30              # head depth eligible for reordering

    # --- dialogue ---------------------------------------------------------
    override_mode: str = "scoped"   # scoped | erase | keep
    demote_shown: bool = True       # implicit rejection, as a penalty
    shown_penalty: float = 1.0      # subtracted from products already offered

    # --- disclosure gate --------------------------------------------------
    gate_enabled: bool = True
    gate_pool_max: int = 5          # emit if the AND-pool is this small
    gate_min_slots: int = 3         # ...or we have this many constraints
    gate_force_turn: int = 5        # ...or we've waited long enough

    def replace(self, **kw) -> "Settings":
        from dataclasses import replace as _r
        return _r(self, **kw)


DEFAULT = Settings()
