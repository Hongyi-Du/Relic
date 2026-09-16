"""WishInterpreter (Phase 5) — extract structured Wishes from a reflection via LLM.

Mirrors the reflection manager's template extraction but lets an LLM re-interpret
the reflection text. Always traceable: each wish keeps source_reflection_id +
raw_reflection_excerpt.
"""
from __future__ import annotations

from typing import Any, List, Optional

from environments.org_env.llm.client import LLMError, OrgLLMClient
from environments.org_env.llm.prompts import WISH_SYSTEM, wish_user
from environments.org_env.llm.schemas import WISH_EXTRACTION_SCHEMA
from environments.org_env.reflection.objects import Wish


class WishInterpreter:
    def extract(self, reflection: Any, world: Any, rm: Any,
                client: Optional[OrgLLMClient]) -> Optional[List[Wish]]:
        """Return Wish objects, or None to fall back to template extraction."""
        if client is None:
            return None
        try:
            from environments.org_env.llm.prompt_assets import (
                agent_identity_for,
                system_for,
            )
            agent = world.agents.get(reflection.agent_id) if world is not None else None
            system = system_for(agent, world, "wish", WISH_SYSTEM)
            # Prefix Cache Rule: per-agent identity travels in the USER message so the
            # system prompt stays byte-identical across agents (see prompt_assets).
            _identity = agent_identity_for(agent, world)
            _identity = (_identity + "\n\n") if _identity else ""
            data = client.generate_json(system,
                                        _identity + wish_user(reflection.to_dict()),
                                        WISH_EXTRACTION_SCHEMA)
        except (LLMError, Exception):
            return None
        raw = data.get("wishes") or []
        if not raw:
            return None
        out: List[Wish] = []
        tick = reflection.tick
        ep_id = reflection.source_episode_ids[0] if reflection.source_episode_ids else None
        for w in raw:
            rm._wseq += 1
            out.append(Wish(
                wish_id=f"wish_{rm._wseq}", agent_id=reflection.agent_id,
                source_reflection_id=reflection.reflection_id, source_episode_id=ep_id,
                source_event_ids=list(reflection.source_event_ids),
                raw_reflection_excerpt=w.get("raw_reflection_excerpt", "") or reflection.self_assessment,
                wish_type=w.get("wish_type", "tool_need"),
                interpreted_need=w.get("interpreted_need", ""),
                target_problem=w.get("target_problem", reflection.team_assessment),
                self_related=bool(w.get("self_related", False)),
                team_related=bool(w.get("team_related", True)),
                suggested_improvement=w.get("suggested_improvement", w.get("interpreted_need", "")),
                missing_support_type=w.get("missing_support_type", "tool"),
                urgency=float(w.get("urgency", 0.5) or 0.5),
                expected_benefit=w.get("expected_benefit", ""),
                risk_if_unaddressed=w.get("risk_if_unaddressed", ""),
                related_object_ids=list(reflection.source_object_ids),
                related_agent_ids=[reflection.agent_id],
                related_episode_ids=list(reflection.source_episode_ids),
                status="open", created_at_tick=tick, updated_at_tick=tick))
        return out


__all__ = ["WishInterpreter"]
