"""EpisodeSummarizer (Phase 6) — readable summary of a CLOSED episode (LLM or template)."""
from __future__ import annotations

from typing import Any, Dict, Optional

from environments.org_env.llm.client import LLMError, OrgLLMClient
from environments.org_env.llm.prompts import SUMMARY_SYSTEM, summary_user
from environments.org_env.llm.schemas import EPISODE_SUMMARY_SCHEMA


def _product_changes(episode) -> str:
    arts = [o for o in episode.linked_object_ids if o.startswith(("art_", "issue_"))]
    return ("product artifacts touched: " + ", ".join(arts)) if arts else "no product artifact changes"


class EpisodeSummarizer:
    def summarize(self, episode: Any, world: Any, client: Optional[OrgLLMClient] = None) -> Dict[str, Any]:
        if client is not None:
            try:
                from environments.org_env.llm.prompt_assets import agent_identity_for, system_for
                agent = world.agents.get(getattr(episode, "primary_agent_id", None))
                system = system_for(agent, world, "episode_summary", SUMMARY_SYSTEM)
                # Prefix Cache Rule: per-agent identity travels in the USER message so the
                # system prompt stays byte-identical across agents (see prompt_assets).
                _identity = agent_identity_for(agent, world)
                _identity = (_identity + "\n\n") if _identity else ""
                ctx = {"episode_type": episode.episode_type, "title": episode.title,
                       "problem": episode.problem_statement, "conflict": episode.conflict_summary,
                       "decision": episode.decision_summary, "participants": list(episode.participants),
                       "timeline": [t for t in episode.timeline][:20],
                       "produced": {"artifacts": episode.produced_artifacts,
                                    "protocols": episode.produced_protocols,
                                    "tasks": episode.produced_tasks},
                       "wishes": list(episode.linked_wish_ids)}
                data = client.generate_json(system, _identity + summary_user(ctx), EPISODE_SUMMARY_SCHEMA)
                data["produced_wishes"] = list(episode.linked_wish_ids)
                data["produced_proposals"] = list(episode.linked_proposal_ids)
                data.setdefault("product_change_summary", _product_changes(episode))
                return data
            except (LLMError, Exception):
                pass
        return {"title": episode.title, "trigger_summary": episode.problem_statement,
                "participant_summary": ", ".join(episode.participants),
                "conflict_summary": episode.conflict_summary or "",
                "decision_summary": episode.decision_summary or "",
                "outcome_summary": episode.outcome_summary or "",
                "product_change_summary": _product_changes(episode),
                "produced_wishes": list(episode.linked_wish_ids),
                "produced_proposals": list(episode.linked_proposal_ids), "open_questions": []}


__all__ = ["EpisodeSummarizer"]
