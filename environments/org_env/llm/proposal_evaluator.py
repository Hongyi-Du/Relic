"""ProposalEvaluator (Phase 7/§9) — score a proposal. Scores ONLY; cannot approve."""
from __future__ import annotations

from typing import Any, Dict, Optional

from environments.org_env.llm.client import LLMError, OrgLLMClient
from environments.org_env.llm.prompts import EVAL_SYSTEM, eval_user
from environments.org_env.llm.schemas import PROPOSAL_EVAL_SCHEMA


class ProposalEvaluator:
    def evaluate(self, proposal: Any, world: Any, client: Optional[OrgLLMClient] = None) -> Dict[str, Any]:
        if client is not None:
            try:
                from environments.org_env.llm.prompt_assets import system_for
                system = system_for(None, world, "proposal_eval", EVAL_SYSTEM)
                return client.generate_json(system, eval_user(proposal.to_dict()), PROPOSAL_EVAL_SCHEMA)
            except LLMError:
                pass
            except Exception:
                pass
        return self._template(proposal, world)

    def _template(self, proposal, world) -> Dict[str, Any]:
        # heuristic: feasible if required actions exist + few risks; useful if benefits;
        # adoption scaled by how many approvers exist.
        try:
            from environments.org_env.backend.actions import registered_action_types
            known = set(registered_action_types())
        except Exception:
            known = set()
        req = proposal.required_actions or []
        have = sum(1 for a in req if a in known)
        feas = 0.5 + 0.4 * (have / len(req) if req else 1.0)
        useful = min(0.9, 0.4 + 0.15 * len(proposal.expected_benefits or []))
        risk = min(0.8, 0.2 + 0.15 * len(proposal.risks or []))
        adopt = round(max(0.2, min(0.9, useful - 0.25 * risk + 0.18)), 3)
        blocking = []
        if proposal.approval_required_from:
            blocking.append("needs approval from " + ", ".join(proposal.approval_required_from[:2]))
        return {"feasibility_score": round(min(0.95, feas), 3), "usefulness_score": round(useful, 3),
                "risk_score": round(risk, 3), "adoption_score": adopt,
                "blocking_issues": blocking, "suggested_revision": "scope the first version narrowly"}


__all__ = ["ProposalEvaluator"]
