"""SurfaceRealizer (Phase 3) — verbalize an ALREADY-DECIDED speech act (LLM or template).

Hard constraint: no new facts, no new promises, no target change (validated).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from environments.org_env.llm.client import LLMError, OrgLLMClient
from environments.org_env.llm.prompts import SURFACE_SYSTEM, surface_user
from environments.org_env.llm.schemas import SURFACE_TEXT_SCHEMA


class SurfaceRealizer:
    def realize(self, *, actor: str, speech_act: str, target: Optional[str] = None,
                reason: str = "", tone: Optional[Dict[str, float]] = None,
                channel_id: Optional[str] = None, episode_context: str = "",
                client: Optional[OrgLLMClient] = None) -> Dict[str, Any]:
        ctx = {"actor": actor, "speech_act": speech_act, "target": target, "reason": reason,
               "tone_constraints": tone or {}, "channel_id": channel_id,
               "episode_context": episode_context}
        if client is not None:
            try:
                out = client.generate_json(SURFACE_SYSTEM, surface_user(ctx), SURFACE_TEXT_SCHEMA)
                if out.get("surface_text"):
                    out["contains_new_facts"] = bool(out.get("contains_new_facts", False))
                    return out
            except (LLMError, Exception):
                pass
        # template fallback
        base = speech_act.replace("_", " ")
        txt = f"[{base}] {reason}".strip() if reason else f"[{base}]"
        if target:
            txt += f" (re: {target})"
        return {"surface_text": txt, "style_tags": [base], "contains_new_facts": False}


__all__ = ["SurfaceRealizer"]
