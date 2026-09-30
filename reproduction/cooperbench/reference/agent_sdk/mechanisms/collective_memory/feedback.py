"""
FeedbackLedger — tracks votes and computes asset/agent credibility.

Single-track model: Gene confidence is always driven by credibility-weighted votes.
Events update contributor credibility (indirectly weighting votes) but never
directly change gene confidence or trigger deprecation.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Set
from .types import (
    FeedbackRecord, AssetFeedbackStats, AgentCredibility,
    FeedbackType,
)


class FeedbackLedger:

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.records: List[FeedbackRecord] = []
        self._asset_stats: Dict[str, AssetFeedbackStats] = {}
        self._agent_credibility: Dict[str, AgentCredibility] = {}
        # Configurable deltas (defaults, tunable later)
        self._task_cred_like_delta = float(self.config.get("task_cred_like_delta", 0.02))
        self._task_cred_dislike_delta = float(self.config.get("task_cred_dislike_delta", 0.03))
        self._task_cred_success_delta = float(self.config.get("task_cred_success_delta", 0.05))
        self._task_cred_failure_delta = float(self.config.get("task_cred_failure_delta", 0.04))
        self._social_cred_action_delta = float(self.config.get("social_cred_action_delta", 0.01))
        self._vote_gdp_base = float(self.config.get("vote_gdp_base", 3.0))

    # ── Voting ──

    def record_vote(
        self,
        asset_id: str,
        vote_type: str,
        voter_id: str,
        turn: int,
        reason: str = "",
    ) -> float:
        """Record a vote and return reward value."""
        record = FeedbackRecord(
            voter_id=voter_id,
            asset_id=asset_id,
            feedback_type=vote_type,
            turn=turn,
            reason=reason,
        )
        self.records.append(record)

        stats = self._asset_stats.setdefault(asset_id, AssetFeedbackStats())
        stats.unique_voters.add(voter_id)
        if vote_type == FeedbackType.LIKE:
            stats.like_count += 1
        elif vote_type == FeedbackType.DISLIKE:
            stats.dislike_count += 1

        # Voter social credibility bump
        voter_cred = self._agent_credibility.setdefault(voter_id, AgentCredibility())
        voter_cred.social_credibility = min(1.0, voter_cred.social_credibility + self._social_cred_action_delta)

        return self._vote_gdp_base

    def compute_vote_confidence(self, asset_id: str) -> float:
        """Laplace-smoothed confidence from votes."""
        stats = self._asset_stats.get(asset_id)
        if stats is None:
            return 0.3
        likes = stats.like_count
        dislikes = stats.dislike_count
        return (likes + 1) / (likes + dislikes + 2)

    def count_consecutive_dislikes(self, asset_id: str) -> int:
        """Count consecutive unique dislikes from the most recent vote backwards.

        Walks vote records for this asset in reverse chronological order.
        Stops at the first non-dislike. Used for vote-based deprecation.
        """
        records = sorted(
            [r for r in self.records if r.asset_id == asset_id],
            key=lambda r: r.turn, reverse=True,
        )
        count = 0
        for r in records:
            if r.feedback_type == FeedbackType.DISLIKE:
                count += 1
            else:
                break
        return count

    def compute_weighted_vote_confidence(
        self, asset_id: str, initial_confidence: float = 0.3,
    ) -> float:
        """Credibility-weighted Laplace confidence from votes.

        Each voter's like/dislike is weighted by their task_credibility.
        The Laplace prior is anchored at initial_confidence so that a
        dislike always decreases confidence and a like always increases it.
        """
        records = [r for r in self.records if r.asset_id == asset_id]
        if not records:
            return initial_confidence

        weighted_likes = sum(
            self.get_agent_credibility(r.voter_id).task_credibility
            for r in records if r.feedback_type == FeedbackType.LIKE
        )
        weighted_dislikes = sum(
            self.get_agent_credibility(r.voter_id).task_credibility
            for r in records if r.feedback_type == FeedbackType.DISLIKE
        )
        alpha = 2.0
        return (weighted_likes + alpha * initial_confidence) / (weighted_likes + weighted_dislikes + alpha)

    # ── Agent credibility updates ──

    def update_contributor_credibility(
        self, contributor_id: str, event: str,
    ) -> None:
        """Update contributor's task_credibility based on events.
        event: 'event_success' | 'event_failure' | 'liked' | 'disliked'
        """
        cred = self._agent_credibility.setdefault(contributor_id, AgentCredibility())
        if event == "event_success":
            cred.task_credibility = min(1.0, cred.task_credibility + self._task_cred_success_delta)
        elif event == "event_failure":
            cred.task_credibility = max(0.0, cred.task_credibility - self._task_cred_failure_delta)
        elif event == "liked":
            cred.task_credibility = min(1.0, cred.task_credibility + self._task_cred_like_delta)
        elif event == "disliked":
            cred.task_credibility = max(0.0, cred.task_credibility - self._task_cred_dislike_delta)

    def update_uploader_credibility(self, agent_id: str) -> None:
        """Bump social_credibility for uploading Gene/Event."""
        cred = self._agent_credibility.setdefault(agent_id, AgentCredibility())
        cred.social_credibility = min(1.0, cred.social_credibility + self._social_cred_action_delta)

    # ── Queries ──

    def get_asset_stats(self, asset_id: str) -> AssetFeedbackStats:
        return self._asset_stats.get(asset_id, AssetFeedbackStats())

    def get_agent_credibility(self, agent_id: str) -> AgentCredibility:
        return self._agent_credibility.get(agent_id, AgentCredibility())

    def get_all_agent_credibilities(self) -> Dict[str, float]:
        """Return {agent_id: task_credibility} for retrieval ranking."""
        return {aid: c.task_credibility for aid, c in self._agent_credibility.items()}

    def agent_already_voted(self, asset_id: str, voter_id: str) -> bool:
        return any(
            r.asset_id == asset_id and r.voter_id == voter_id
            for r in self.records
        )

    # ── Dispute helpers ──

    def get_engagement_count(self, asset_id: str) -> int:
        stats = self.get_asset_stats(asset_id)
        return stats.like_count + stats.dislike_count

    def get_negative_ratio(self, asset_id: str) -> float:
        stats = self.get_asset_stats(asset_id)
        total = stats.like_count + stats.dislike_count
        if total == 0:
            return 0.0
        return stats.dislike_count / total

    def get_voters_by_side(self, asset_id: str) -> Dict[str, List[str]]:
        like_voters = []
        dislike_voters = []
        seen_like = set()
        seen_dislike = set()
        for r in self.records:
            if r.asset_id != asset_id:
                continue
            if r.feedback_type == FeedbackType.LIKE and r.voter_id not in seen_like:
                like_voters.append(r.voter_id)
                seen_like.add(r.voter_id)
            elif r.feedback_type == FeedbackType.DISLIKE and r.voter_id not in seen_dislike:
                dislike_voters.append(r.voter_id)
                seen_dislike.add(r.voter_id)
        return {"like": like_voters, "dislike": dislike_voters}

    # ── Serialization ──

    def to_dict(self) -> Dict[str, Any]:
        return {
            "records": [
                {"voter_id": r.voter_id, "asset_id": r.asset_id,
                 "feedback_type": r.feedback_type, "turn": r.turn, "reason": r.reason}
                for r in self.records
            ],
            "agent_credibility": {
                aid: {"task": c.task_credibility, "social": c.social_credibility}
                for aid, c in self._agent_credibility.items()
            },
        }

    def load_from_dict(self, data: Dict[str, Any]) -> None:
        self.records = []
        self._asset_stats = {}
        self._agent_credibility = {}
        for rd in data.get("records", []):
            rec = FeedbackRecord(**rd)
            self.records.append(rec)
            stats = self._asset_stats.setdefault(rec.asset_id, AssetFeedbackStats())
            stats.unique_voters.add(rec.voter_id)
            if rec.feedback_type == FeedbackType.LIKE:
                stats.like_count += 1
            elif rec.feedback_type == FeedbackType.DISLIKE:
                stats.dislike_count += 1
        for aid, cd in data.get("agent_credibility", {}).items():
            self._agent_credibility[aid] = AgentCredibility(
                task_credibility=cd.get("task", 0.5),
                social_credibility=cd.get("social", 0.5),
            )
