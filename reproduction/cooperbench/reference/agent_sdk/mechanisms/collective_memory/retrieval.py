"""
Event-driven retrieval engine for Gene+Event collective memory.

Schema 1.6.0 redesign (2026-04-20):

  retrieve_score = BM25Okapi(query_tokens, gene_doc_tokens)   when corpus >= 4
                 | set-overlap TF fallback                    when corpus <  4
  combined       = (kw * w_kw + vitality * w_vit + confidence * w_conf)
                 * spatial_multiplier

Query tokens are built by expanding each query field by its configured
weight (default planner_task=3x, signals_evidence=3x, inventory=1.5x,
physical_state=1.5x). Doc tokens expand similarly by field weight
(signals_match=3x, summary=2x, body=1x). Because BM25 counts token
frequency, duplication acts as a native per-field weight knob without
having to touch the library internals.

Spatial multiplier = ``floor + (1-floor) * exp(-d/tau)`` with d the
Chebyshev distance between ``gene.origin_coordinate`` and the caller's
agent xy. ``None`` coord or ``enabled=false`` → 1.0 (no penalty) so
non-spatial envs and legacy pre-1.6 stores aren't punished.

Vitality: linear time decay + capped upvote bonus (transient, never
mutates ``gene.confidence``). Confidence: credibility-weighted Laplace
posterior from votes (see FeedbackLedger).

A fraction (``explore_ratio``) of ``top_k`` slots is reserved for
underexplored genes (<2 referencing events) to avoid permanent
starvation of new uploads. Set ``explore_ratio=0`` for deterministic
top-k (used by the synthetic frame tests).

Back-compat: the old ``query(store, keywords=[...], intent_tokens=[...])``
signature used by ``agent_sdk/core/percept.py`` and ``decide.py`` still
works — the new kwargs (planner_task / inventory_tokens /
physical_state_tokens / origin_agent_xy) are all optional. The nature
env harness view-builder is the only caller threading them through.
"""
from __future__ import annotations
import math
import re
import random
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from .types import Gene, RankedGene, AssetStatus

_TOKEN_RE = re.compile(r"[a-z0-9_]{2,}")
_FP_STOPWORDS = frozenset({
    "at", "in", "to", "the", "and", "by", "for", "on", "is", "was",
    "with", "from", "using", "successfully", "spring", "summer", "fall", "winter",
})


def _summary_fingerprint(summary: str) -> str:
    """Stable fingerprint for near-duplicate dedup.

    Strips coordinates, entity IDs, and stopwords so that
    'Stone Axe recipe crafted at (87,42)' and
    'Stone Axe recipe crafted at (78,26)' collapse to the same key.
    """
    text = re.sub(r"\([0-9, ]+\)", "", summary.lower())
    text = re.sub(r"_\d{3,}", "", text)  # entity instance IDs like campfire_40729
    words = sorted(w for w in _TOKEN_RE.findall(text) if w not in _FP_STOPWORDS)
    return " ".join(words[:12])


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(str(text).lower())


def _expand_tokens(tokens: List[str], weight: float) -> List[str]:
    """Repeat tokens max(1, round(weight)) times for BM25 weighting.

    Empty input short-circuits to [] so an empty query field doesn't
    become a phantom zero-token stream. ``round(1.5) == 2`` (Python 3
    banker's rounding), ``round(0.3) == 0`` → clamped to 1 so a weight
    below 1.0 still keeps the token once (a dimension worth <1x is still
    worth keeping, just not amplifying).
    """
    if not tokens:
        return []
    repeats = max(1, int(round(weight)))
    return list(tokens) * repeats


def _spatial_multiplier(
    gene_xy: Optional[Tuple[int, int]],
    agent_xy: Optional[Tuple[int, int]],
    enabled: bool = True,
    tau: float = 30.0,
    floor: float = 0.6,
) -> float:
    """Soft-clamp retrieval score by gene↔agent Chebyshev distance.

    ``mult = floor + (1-floor) * exp(-d/tau)``. Any missing coord or
    ``enabled=False`` → 1.0 so non-spatial envs and legacy stores with
    no origin coord don't get artificially down-weighted. Malformed
    tuples fall through the try/except as neutral.
    """
    if not enabled:
        return 1.0
    if gene_xy is None or agent_xy is None:
        return 1.0
    try:
        dx = abs(int(gene_xy[0]) - int(agent_xy[0]))
        dy = abs(int(gene_xy[1]) - int(agent_xy[1]))
    except (ValueError, TypeError, IndexError):
        return 1.0
    d = max(dx, dy)  # Chebyshev matches view_builder's distance idiom
    span = 1.0 - floor
    return floor + span * math.exp(-d / tau)


class RetrievalTrigger(str, Enum):
    INTENT_UPDATED = "intent_updated"
    PROGRESS_STATE_CHANGED = "progress_state_changed"
    TOOL_RESULT_RECEIVED = "tool_result_received"
    ENV_FEEDBACK_RECEIVED = "env_feedback_received"
    TASK_CHECKPOINT_REACHED = "task_checkpoint_reached"


class RetrievalEngine:
    """Event-driven retrieval with BM25 backend + TF fallback for tiny corpora."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        retrieval_cfg = self.config.get("retrieval", {}) if isinstance(self.config, dict) else {}
        self.default_top_k = int(retrieval_cfg.get("top_k", 5) or 5)
        self._explore_ratio = float(retrieval_cfg.get("explore_ratio", 0.2) or 0.0)
        self._keyword_augmenters: List[Callable] = []

    def register_keyword_augmenter(
        self, fn: Callable[[List[str], Any], List[str]],
    ) -> None:
        """Register env-specific keyword augmenter.
        fn(base_keywords, world_signals) -> augmented_keywords
        """
        self._keyword_augmenters.append(fn)

    def query(
        self,
        store: Any,  # AssetStore
        keywords: Optional[List[str]] = None,
        trigger: str = RetrievalTrigger.ENV_FEEDBACK_RECEIVED,
        agent_credibilities: Optional[Dict[str, float]] = None,
        world_signals: Any = None,
        top_k: Optional[int] = None,
        current_turn: int = 0,
        intent_tokens: Optional[List[str]] = None,
        # 1.6.0 redesign kwargs — all optional for back-compat with
        # trade_env / coding_env callers that only pass keywords+intent_tokens.
        planner_task: Optional[str] = None,
        physical_state_tokens: Optional[List[str]] = None,
        inventory_tokens: Optional[List[str]] = None,
        origin_agent_xy: Optional[Tuple[int, int]] = None,
    ) -> List[RankedGene]:
        k = int(top_k or self.default_top_k)
        if k <= 0 or store is None:
            return []

        genes_dict = getattr(store, "genes", {})
        if not genes_dict:
            return []

        active_genes: List[Gene] = [
            g for g in genes_dict.values()
            if g.status not in (AssetStatus.DEPRECATED, AssetStatus.SUPERSEDED)
        ]
        if not active_genes:
            return []

        # ── config ──────────────────────────────────────────────────────
        retrieval_cfg = self.config.get("retrieval", {}) if isinstance(self.config, dict) else {}

        query_weights = retrieval_cfg.get("query_weights", {}) or {}
        w_planner = float(query_weights.get("planner_task", 3.0) or 3.0)
        w_evidence = float(query_weights.get("signals_evidence", 3.0) or 3.0)
        w_inventory = float(query_weights.get("inventory", 1.5) or 1.5)
        w_physical = float(query_weights.get("physical_state", 1.5) or 1.5)

        doc_weights = retrieval_cfg.get("doc_weights", {}) or {}
        w_doc_signals = float(doc_weights.get("signals_match", 3.0) or 3.0)
        w_doc_summary = float(doc_weights.get("summary", 2.0) or 2.0)
        w_doc_body = float(doc_weights.get("body", 1.0) or 1.0)

        spatial_cfg = retrieval_cfg.get("spatial", {}) or {}
        spatial_enabled = bool(spatial_cfg.get("enabled", True))
        spatial_tau = float(spatial_cfg.get("tau", 30.0) or 30.0)
        spatial_floor = float(spatial_cfg.get("floor", 0.6) or 0.6)

        score_weights = retrieval_cfg.get("score_weights", {}) or {}
        w_kw = float(score_weights.get("keyword", 0.5) or 0.5)
        w_vit = float(score_weights.get("vitality", 0.3) or 0.3)
        w_conf = float(score_weights.get("confidence", 0.2) or 0.2)

        decay_per_turn = float(retrieval_cfg.get("vitality_decay_per_turn", 0.017) or 0.017)
        vote_bonus = float(retrieval_cfg.get("vitality_vote_bonus", 0.5) or 0.5)
        max_vote_bonus = float(retrieval_cfg.get("max_vote_vitality_bonus", 0.8) or 0.8)

        backend_name = str(retrieval_cfg.get("backend", "bm25")).lower()

        # ── query tokens ────────────────────────────────────────────────
        # Legacy intent_tokens merge into evidence (same 3x weight). Old
        # callers that only pass keywords+intent_tokens still produce a
        # working query stream.
        evidence_input: List[str] = []
        if keywords:
            evidence_input.extend(str(x) for x in keywords)
        if intent_tokens:
            evidence_input.extend(str(x) for x in intent_tokens)

        augmented_evidence = list(evidence_input)
        for augmenter in self._keyword_augmenters:
            try:
                augmented_evidence = augmenter(augmented_evidence, world_signals)
            except Exception:
                pass

        query_tokens: List[str] = []
        evidence_tokens = _tokenize(" ".join(augmented_evidence))
        query_tokens.extend(_expand_tokens(evidence_tokens, w_evidence))

        if planner_task:
            planner_tokens = _tokenize(str(planner_task))
            query_tokens.extend(_expand_tokens(planner_tokens, w_planner))

        if inventory_tokens:
            inv_flat: List[str] = []
            for t in inventory_tokens:
                inv_flat.extend(_tokenize(str(t)))
            query_tokens.extend(_expand_tokens(inv_flat, w_inventory))

        if physical_state_tokens:
            phy_flat: List[str] = []
            for t in physical_state_tokens:
                phy_flat.extend(_tokenize(str(t)))
            query_tokens.extend(_expand_tokens(phy_flat, w_physical))

        if not query_tokens:
            return []

        # ── corpus ──────────────────────────────────────────────────────
        corpus: List[List[str]] = []
        for gene in active_genes:
            doc: List[str] = []
            signals_flat: List[str] = []
            for s in gene.signals_match or ():
                signals_flat.extend(_tokenize(str(s)))
            doc.extend(_expand_tokens(signals_flat, w_doc_signals))
            doc.extend(_expand_tokens(_tokenize(gene.summary or ""), w_doc_summary))
            doc.extend(_expand_tokens(_tokenize(gene.body or ""), w_doc_body))
            if not doc:
                # BM25Okapi.get_scores() divides by doc length; empty docs
                # can cause nan. Sentinel token keeps the slot scoreable.
                doc = ["__empty__"]
            corpus.append(doc)

        # ── score ───────────────────────────────────────────────────────
        if backend_name == "bm25" and len(corpus) >= 4:
            raw_scores = self._bm25_scores(corpus, query_tokens)
        else:
            raw_scores = self._tf_scores(corpus, query_tokens)

        # Max-normalize so kw/vit/conf live on a common [0,1] scale.
        max_raw = max(raw_scores) if raw_scores else 0.0
        if max_raw > 0:
            normalized_kw = [s / max_raw for s in raw_scores]
        else:
            normalized_kw = [0.0] * len(raw_scores)

        feedback = getattr(store, "feedback", None)
        all_scored: List[RankedGene] = []
        _seen_fingerprints: Dict[str, RankedGene] = {}

        for gene, raw_score, norm_kw in zip(active_genes, raw_scores, normalized_kw):
            conf = float(gene.confidence or 0.0)

            age = max(0, int(current_turn) - int(getattr(gene, "turn_created", 0) or 0))
            base_vitality = max(0.0, 1.0 - decay_per_turn * age)
            like_count = 0
            if feedback is not None:
                try:
                    stats = feedback.get_asset_stats(gene.asset_id)
                    like_count = int(getattr(stats, "like_count", 0) or 0)
                except Exception:
                    pass
            vote_vitality = min(max_vote_bonus, like_count * vote_bonus)
            vitality = min(1.0, base_vitality + vote_vitality)

            spatial_mult = _spatial_multiplier(
                gene.origin_coordinate, origin_agent_xy,
                enabled=spatial_enabled,
                tau=spatial_tau,
                floor=spatial_floor,
            )

            combined = (norm_kw * w_kw + vitality * w_vit + conf * w_conf) * spatial_mult

            # Drop noise: a gene with zero retrieval signal AND almost
            # zero combined score. Genes with no match but healthy
            # vitality+confidence (e.g. a new upload on turn 0 with
            # no text overlap) would still score ≈ 0.3 from vit+conf
            # alone — keep those for the exploration quota below.
            if norm_kw <= 0 and combined <= 0.001:
                continue

            rg = RankedGene(gene=gene, score=combined, vitality=vitality)
            all_scored.append(rg)
            fp = _summary_fingerprint(gene.summary or "")
            existing = _seen_fingerprints.get(fp)
            if existing is None or combined > existing.score:
                _seen_fingerprints[fp] = rg

        winner_ids = {rg.gene.asset_id for rg in _seen_fingerprints.values()}
        candidates = [r for r in all_scored if r.gene.asset_id in winner_ids]
        candidates.sort(key=lambda r: -r.score)

        # ── top-k with optional exploration quota ───────────────────────
        if self._explore_ratio <= 0.0:
            explore_slots = 0
        else:
            explore_slots = max(1, int(k * self._explore_ratio))
        main_slots = max(0, k - explore_slots)
        main = candidates[:main_slots]
        main_ids = {r.gene.asset_id for r in main}

        if explore_slots > 0:
            unexplored = [
                r for r in candidates
                if r.gene.asset_id not in main_ids and self._is_underexplored(r.gene, store)
            ]
            if unexplored:
                explore = random.sample(unexplored, min(explore_slots, len(unexplored)))
                main.extend(explore)
            elif len(main) < k and len(candidates) > len(main):
                remaining = [r for r in candidates if r.gene.asset_id not in main_ids]
                main.extend(remaining[: k - len(main)])

        return main[:k]

    # ── backends ────────────────────────────────────────────────────────

    @staticmethod
    def _bm25_scores(corpus: List[List[str]], query_tokens: List[str]) -> List[float]:
        """BM25Okapi raw scores. Expects corpus to have >= 4 non-empty docs."""
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi(corpus)
        raw = bm25.get_scores(query_tokens)
        try:
            return [float(x) for x in raw.tolist()]
        except AttributeError:
            return [float(x) for x in raw]

    @staticmethod
    def _tf_scores(corpus: List[List[str]], query_tokens: List[str]) -> List[float]:
        """Set-overlap TF for tiny corpora.

        BM25Okapi's IDF ``log((N - df + 0.5) / (df + 0.5))`` collapses
        (or even goes negative) when N < 4 — matching docs can score 0
        while non-matching docs score positive. Fall back to a stable
        query-set overlap so a 2-gene store still ranks the matching
        gene first.
        """
        if not query_tokens:
            return [0.0] * len(corpus)
        q_set = set(query_tokens)
        denom = max(1, len(q_set))
        return [len(q_set & set(doc)) / denom for doc in corpus]

    @staticmethod
    def _is_underexplored(gene: Gene, store: Any) -> bool:
        """Gene with few referencing events is underexplored."""
        events = getattr(store, "events", {})
        ref_count = sum(1 for c in events.values() if c.gene_id == gene.asset_id)
        return ref_count < 2


# ──────────────────────────────────────────────────────────────────────
# Back-compat helpers used by the legacy v2 path and frontend replay.
# ──────────────────────────────────────────────────────────────────────

def extract_keywords(intent: str, world_signals: Any = None):
    """Extract retrieval keywords from intent + domain signals.

    Returns ``(intent_tokens, context_keywords)`` — intent tokens are
    weighted higher in scoring to prioritize action-relevant genes.
    Still used by ``agent_sdk/core/percept.py`` and ``decide.py``.
    """
    intent_tokens = _tokenize(intent)
    context: List[str] = []
    if world_signals is not None:
        context.extend(str(k) for k in getattr(world_signals, "intent_keywords", []) or [])
        context.extend(str(k) for k in getattr(world_signals, "evidence_keywords", []) or [])
        context.extend(str(k) for k in getattr(world_signals, "context_keywords", []) or [])
    return intent_tokens, list(set(context))


def render_events(events: List[Any], max_items: int = 5) -> str:
    """Render recent events as compact prompt text."""
    if not events:
        return ""
    lines = []
    for c in events[:max_items]:
        prefix = c.asset_id
        gene_ref = f", gene: {c.gene_id}" if c.gene_id else ""
        lines.append(f"- [{prefix}] {c.summary} (turn {c.turn_created}{gene_ref})")
    return "\n".join(lines)


def render_knowledge_index(
    ranked_genes: List[RankedGene],
    recent_events: List[Any],
    current_turn: int = 0,
) -> str:
    """Render combined genes + events for perception injection."""
    parts: List[str] = []
    if ranked_genes:
        parts.append(render_genes(ranked_genes, current_turn))
    if recent_events:
        parts.append("## Recent Events")
        parts.append(render_events(recent_events))
    return "\n\n".join(parts) if parts else ""


def render_genes(ranked: List[RankedGene], current_turn: int = 0) -> str:
    """Render ranked Genes as prompt text."""
    if not ranked:
        return ""
    lines = ["# Shared Knowledge (Top Matches)", ""]
    for item in ranked:
        g = item.gene
        prefix = g.asset_id
        vit_pct = f", vitality: {item.vitality:.0%}" if item.vitality is not None else ""
        lines.append(
            f"- [{prefix}] {g.summary} "
            f"(type: {g.category}, confidence: {g.confidence:.2f}{vit_pct}, score: {item.score:.2f})"
        )
    lines.append("")
    # Stronger vote nudge (2026-04-20): observational voting is OK. The
    # team needs consensus signal on whether a gene is useful even
    # before anyone has formally "tested" it — confidence comes from
    # seeing the advice applied in a real situation by anyone,
    # including yourself. Low vote rate is the #1 reason genes drift
    # stale and never get culled.
    lines.append(
        "If one of the genes above matches your just-committed situation"
        " — even loosely — call `vote_asset(uid=[asset_id], vote_type=...)` "
        "in THIS Reflect phase. Upvote when the advice aligns with what you "
        "observed/tried; downvote when it misled or is demonstrably wrong."
    )
    lines.append(
        "Do NOT re-upload knowledge that overlaps with what you see here — "
        "vote instead so the team's confidence signal converges."
    )
    return "\n".join(lines)
