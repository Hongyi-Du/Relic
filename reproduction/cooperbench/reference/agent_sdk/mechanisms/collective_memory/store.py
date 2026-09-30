"""
AssetStore — Generic Gene+Event collective memory store.

Pure in-memory with JSON serialization. Zero env-specific logic.
Content-addressable via sha256 hash. No LLM audit.
"""
from __future__ import annotations
import hashlib
import json
import re
import string
from typing import Any, Dict, List, Optional, Tuple
from .types import (
    Gene, Event, AssetStatus, AssetCategory, SCHEMA_VERSION,
    DisputeThread, DisputeStatus, RankedGene,
    FeedbackRecord, _coerce_coordinate,
    canonicalize_category,
)
from .feedback import FeedbackLedger


_PUNCT_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))
_WS_RE = re.compile(r"\s+")


def _normalize_summary(summary: str) -> str:
    """Canonical form for dedup hashing only.

    Lowercase, underscore→space, strip punctuation, collapse whitespace.
    Does NOT stem, sort tokens, or drop stopwords. Solves the class of
    false duplicates like "Raw Beef Harvesting" vs "raw_beef_harvest" —
    the hash converges while gene.summary preserves the human-readable
    original form for display.
    """
    if not summary:
        return ""
    s = str(summary).lower().replace("_", " ")
    s = s.translate(_PUNCT_TABLE)
    return _WS_RE.sub(" ", s).strip()


def _compute_gene_id(summary: str, category: str, signals_match: List[str]) -> str:
    payload = json.dumps(
        {
            "summary": _normalize_summary(summary),
            "category": canonicalize_category(category),
            "signals": sorted(signals_match),
        },
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()[:6]


def _compute_event_id(trace_hash: str, gene_id: str) -> str:
    payload = json.dumps(
        {"trace_hash": trace_hash, "gene_id": gene_id},
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()[:6]


class AssetStore:
    """
    Generic Gene+Event asset store.
    No env-specific logic. No LLM dependency. No hard-coded dedup patterns.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.genes: Dict[str, Gene] = {}
        self.events: Dict[str, Event] = {}
        self.feedback: FeedbackLedger = FeedbackLedger(self.config.get("feedback", {}))
        self.dispute_threads: Dict[str, DisputeThread] = {}
        self._initial_confidence = float(self.config.get("initial_confidence", 0.3))
        self._vote_deprecate_threshold = int(self.config.get("vote_deprecate_consecutive_dislikes", 3))

    # ── Upload ──

    def upload_event(
        self, agent_id: str, event_data: Dict[str, Any], turn: int,
    ) -> Event:
        """Upload an Event. Schema validation + hash dedup. Direct store, no LLM audit."""
        trace_hash = str(event_data.get("trace_hash", ""))
        gene_id = str(event_data.get("gene_id", ""))
        asset_id = _compute_event_id(trace_hash, gene_id)

        # Hash dedup
        if asset_id in self.events:
            return self.events[asset_id]

        evt = Event(
            asset_id=asset_id,
            gene_id=gene_id,
            trigger=event_data.get("trigger", []),
            summary=event_data.get("summary", ""),
            contributor_id=agent_id,
            turn_created=turn,
            confidence=self._initial_confidence,
            action_related=event_data.get("action_related", False),
            domain_signals=event_data.get("domain_signals", {}),
            intent_action=event_data.get("intent_action", ""),
            env_result=event_data.get("env_result", ""),
            env_fingerprint=event_data.get("env_fingerprint", {}),
            trace_hash=trace_hash,
            outcome=event_data.get("outcome", {"status": "unknown", "score": 0.0}),
            blast_radius=event_data.get("blast_radius", {}),
            success_streak=event_data.get("success_streak", 0),
            metadata=event_data.get("metadata", {}),
        )
        self.events[asset_id] = evt
        self.feedback.records.append(FeedbackRecord(
            voter_id=agent_id, asset_id=asset_id,
            feedback_type="upload_event", turn=turn,
            reason=event_data.get("summary", ""),
        ))

        # Link event to gene (for display/retrieval) + update contributor credibility.
        # Events no longer update gene.confidence — votes are the sole authority.
        auto_attributed = event_data.get("auto_attributed", True)
        if gene_id and gene_id in self.genes:
            gene = self.genes[gene_id]
            if asset_id not in gene.event_ids:
                gene.event_ids.append(asset_id)
            if auto_attributed:
                cred_event = "event_success" if evt.outcome.get("status") == "success" else "event_failure"
                self.feedback.update_contributor_credibility(gene.contributor_id, cred_event)

        # Uploader social credibility
        self.feedback.update_uploader_credibility(agent_id)
        return evt

    def upload_gene(
        self, agent_id: str, gene_data: Dict[str, Any], turn: int,
    ) -> Gene:
        """Upload a Gene. Event evidence optional but strengthens credibility. Direct store.

        Schema 1.6 accepts:
          - summary (str, required)
          - category (str, coerced to 4-value canonical set via canonicalize_category)
          - signals_match (list[str], 1-5 stigmergy tokens preferred)
          - body (str, detailed when/where/why/how — Schema 1.6 addition)
          - origin_coordinate (tuple[int,int] or None — spatial envs only)
          - event_ids / action_related / metadata (optional)
        """
        summary = str(gene_data.get("summary", ""))
        raw_category = str(gene_data.get("category", AssetCategory.OBSERVATION))
        category = canonicalize_category(raw_category)
        signals = list(gene_data.get("signals_match") or [])
        body = str(gene_data.get("body", ""))
        event_ids = gene_data.get("event_ids") or []
        # Optional spatial provenance (1.6.0). None in non-spatial envs
        # causes spatial_multiplier at retrieve time to decay to 1.0.
        origin_coordinate = _coerce_coordinate(gene_data.get("origin_coordinate"))

        # origin_coordinate coercion (accept list, tuple, or None)
        coord_raw = gene_data.get("origin_coordinate")
        origin_coordinate: Optional[Tuple[int, int]] = None
        if isinstance(coord_raw, (list, tuple)) and len(coord_raw) == 2:
            try:
                origin_coordinate = (int(coord_raw[0]), int(coord_raw[1]))
            except (TypeError, ValueError):
                origin_coordinate = None

        # Validate event references (prefix match: agents may send truncated IDs)
        valid_events = self._resolve_event_ids(event_ids)

        # ── Jaccard signal dedup ──
        # If an ACTIVE gene with highly overlapping signals already exists,
        # return it instead of creating a duplicate.
        # Guard: require union >= 3 tokens before applying 0.6 threshold so
        # a single shared small-set token (e.g. "food") doesn't falsely
        # collapse two unrelated genes.
        sig_set = set(signals)
        if sig_set:
            for existing in self.genes.values():
                if existing.status != AssetStatus.ACTIVE:
                    continue
                e_set = set(existing.signals_match)
                if not e_set:
                    continue
                intersection = len(sig_set & e_set)
                union = len(sig_set | e_set)
                if union >= 3 and intersection / union >= 0.6:
                    return existing

        asset_id = _compute_gene_id(summary, category, signals)

        # Hash dedup
        if asset_id in self.genes:
            return self.genes[asset_id]

        gene = Gene(
            asset_id=asset_id,
            category=category,
            signals_match=signals,
            summary=summary,
            body=body,
            contributor_id=agent_id,
            turn_created=turn,
            event_ids=valid_events if valid_events else event_ids,
            confidence=self._initial_confidence,
            action_related=gene_data.get("action_related", False),
            status=AssetStatus.ACTIVE,
            revision=1,
            origin_coordinate=origin_coordinate,
            metadata=gene_data.get("metadata", {}),
        )
        self.genes[asset_id] = gene
        self.feedback.update_uploader_credibility(agent_id)
        self.feedback.records.append(FeedbackRecord(
            voter_id=agent_id, asset_id=asset_id,
            feedback_type="upload_gene", turn=turn, reason=summary,
        ))

        # Event evidence is linked but no longer drives confidence.
        # Confidence starts at initial_confidence and is updated by votes only.

        return gene

    # ── Maintenance ──

    def decay_genes(self, factor: float = 0.85, floor: float = 0.05) -> int:
        """Multiplicatively decay confidence of all non-DEPRECATED genes.

        Env state is mutable (forest depleted, season flipped) — gene
        historical confidence should sink unless reinforced by new votes.
        Called periodically by HarnessRunner.tick_collective_memory_maintenance.

        Returns the number of genes decayed. See
        docs/superpowers/plans/2026-04-19/harness-cm-bug-fix.md §3.5.
        """
        decayed = 0
        for gene in self.genes.values():
            if gene.status == AssetStatus.DEPRECATED:
                continue
            gene.confidence = max(floor, gene.confidence * factor)
            decayed += 1
        return decayed

    # ── Retrieval (delegates to RetrievalEngine, but also provides direct access) ──

    def get_gene(self, asset_id: str) -> Optional[Gene]:
        return self.genes.get(asset_id)

    def find_similar_gene(self, summary: str, threshold: float = 0.4):
        """Find an existing gene with similar summary using Jaccard keyword overlap.
        Threshold=0.4 catches paraphrases while avoiding false matches."""
        if not summary:
            return None
        query_words = set(summary.lower().split())
        best_match = None
        best_score = 0.0
        for gene in self.genes.values():
            if gene.status != "active":
                continue
            gene_words = set(gene.summary.lower().split())
            if not gene_words:
                continue
            overlap = len(query_words & gene_words)
            score = overlap / max(len(query_words | gene_words), 1)
            if score > threshold and score > best_score:
                best_score = score
                best_match = gene
        return best_match

    def link_event_to_gene(self, gene_id: str, agent_id: str, turn: int):
        """Link a new replication event to an existing gene (marks it as replicated).
        A gene with len(event_ids) > 1 is considered 'replicated' by view_builder.py."""
        gene = self.genes.get(gene_id)
        if gene is None:
            return
        event_id = f"repl_{agent_id}_{turn}"
        if event_id not in gene.event_ids:
            gene.event_ids.append(event_id)
        # Boost confidence slightly for replication
        gene.confidence = min(1.0, gene.confidence + 0.05)

    def get_event(self, asset_id: str) -> Optional[Event]:
        return self.events.get(asset_id)

    def get_events_for_gene(self, gene_id: str) -> List[Event]:
        return [c for c in self.events.values() if c.gene_id == gene_id]

    # ── Feedback ──

    def _resolve_asset_id(self, raw_id: str) -> Optional[str]:
        """Resolve a possibly-truncated asset ID to its full key via prefix match.

        Handles multiple formats LLMs may produce:
        - Full ID: "sha256:abc123"
        - Bare hex: "abc123" (without sha256: prefix)
        - Truncated: "sha256:abc1" or "abc1"
        """
        if not raw_id:
            return None
        # Exact match first
        if raw_id in self.genes or raw_id in self.events:
            return raw_id
        # If bare hex (no prefix), try with prefix
        if not raw_id.startswith("sha256:"):
            prefixed = "sha256:" + raw_id
            if prefixed in self.genes or prefixed in self.events:
                return prefixed
            # Prefix match with sha256: prepended
            matches = [k for k in self.genes if k.startswith(prefixed)]
            matches += [k for k in self.events if k.startswith(prefixed)]
            if len(matches) == 1:
                return matches[0]
        # Prefix match across genes and events (original ID as-is)
        matches = [k for k in self.genes if k.startswith(raw_id)]
        matches += [k for k in self.events if k.startswith(raw_id)]
        if len(matches) == 1:
            return matches[0]
        return None

    def apply_vote(
        self,
        asset_id: str,
        vote_type: str,
        voter_id: str,
        turn: int,
        reason: str = "",
    ) -> float:
        """Apply a vote (like/dislike). Returns reward value.

        Always updates gene confidence (credibility-weighted).
        Checks consecutive-dislike deprecation after each vote.
        """
        resolved_id = self._resolve_asset_id(asset_id)
        if resolved_id is None:
            return 0.0
        asset = self.genes.get(resolved_id) or self.events.get(resolved_id)
        if asset is None:
            return 0.0

        if self.feedback.agent_already_voted(resolved_id, voter_id):
            return 0.0

        gdp = self.feedback.record_vote(resolved_id, vote_type, voter_id, turn, reason)

        asset.confidence = self.feedback.compute_weighted_vote_confidence(
            resolved_id, self._initial_confidence,
        )

        # Update contributor credibility
        event = "liked" if vote_type == "like" else "disliked"
        self.feedback.update_contributor_credibility(asset.contributor_id, event)

        # Vote-based deprecation: N consecutive unique dislikes → deprecated
        if (vote_type == "dislike"
                and isinstance(asset, Gene)
                and asset.status == AssetStatus.ACTIVE):
            consecutive = self.feedback.count_consecutive_dislikes(resolved_id)
            if consecutive >= self._vote_deprecate_threshold:
                asset.status = AssetStatus.DEPRECATED

        return gdp

    # ── Event ID resolution ──

    def _resolve_event_ids(self, raw_ids: List[str]) -> List[str]:
        """Resolve event IDs, supporting prefix match for truncated IDs from agents."""
        resolved: List[str] = []
        for rid in raw_ids:
            if not rid:
                continue
            if rid in self.events:
                resolved.append(rid)
                continue
            # Prefix match: agent may have sent truncated ID (e.g. "sha256:e5b089af6")
            matches = [k for k in self.events if k.startswith(rid)]
            if len(matches) == 1:
                resolved.append(matches[0])
        return resolved

    # ── Maintenance ──

    def decay_genes(self, factor: float, floor: float) -> int:
        """Multiply confidence of all non-DEPRECATED genes by ``factor``, clamped to ``>= floor``.

        Returns the count of genes that were actually decayed (DEPRECATED genes are skipped
        and not counted).
        """
        count = 0
        for gene in self.genes.values():
            if gene.status == AssetStatus.DEPRECATED:
                continue
            gene.confidence = max(floor, gene.confidence * factor)
            count += 1
        return count

    # ── Serialization ──

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "genes": {gid: g.to_dict() for gid, g in self.genes.items()},
            "events": {cid: c.to_dict() for cid, c in self.events.items()},
            "feedback": self.feedback.to_dict(),
            "dispute_threads": {
                tid: {
                    "asset_id": t.asset_id, "status": t.status,
                    "op_agent_id": t.op_agent_id,
                    "created_turn": t.created_turn,
                    "resolved_turn": t.resolved_turn,
                    "resolution": t.resolution,
                }
                for tid, t in self.dispute_threads.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> "AssetStore":
        store = cls(config=config)
        for gid, gd in data.get("genes", {}).items():
            store.genes[gid] = Gene.from_dict(gd)
        for cid, cd in data.get("events", {}).items():
            store.events[cid] = Event.from_dict(cd)
        feedback_data = data.get("feedback", {})
        if feedback_data:
            store.feedback.load_from_dict(feedback_data)
        return store
