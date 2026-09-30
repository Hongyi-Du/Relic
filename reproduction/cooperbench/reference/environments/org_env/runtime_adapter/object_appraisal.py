"""ObjectAppraisal + ObjectAppraiser (OrgEnv O1.7, spec Part II §15-§16).

Policy must not judge a raw report/PR/doc directly. The appraisal layer first
produces a STRUCTURED :class:`ObjectAppraisal` (risk level + issue tags + evidence
gaps + suggested feedback acts) from three sources:

* **metadata / rule-based** (§15.1) — no LLM (missing reviewer/tracker/seed/...);
* **static / repo** (§15.2) — no/low LLM (changed core module, missing tests...);
* **LLM semantic** (§15.3) — optional, only when a real engine is injected
  (claim too strong, wording risk...). The LLM returns JSON only.

Appraisals are cached by ``(object_id, version, appraiser, mode)`` so the same
object/version isn't re-tagged every tick.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ObjectAppraisal:
    appraisal_id: str
    target_object_id: str
    target_object_type: str
    appraiser_agent_id: Optional[str]
    method: str                              # metadata | static | llm | static+llm
    risk_level: str = "low"                  # low | medium | high
    issue_tags: List[str] = field(default_factory=list)
    evidence_gaps: List[str] = field(default_factory=list)
    affected_modules: List[str] = field(default_factory=list)
    protocol_violations: List[str] = field(default_factory=list)
    suggested_feedback_acts: List[str] = field(default_factory=list)
    evidence_spans: List[dict] = field(default_factory=list)
    confidence: str = "medium"               # low | medium | high
    created_tick: int = 0
    visible_to: List[str] = field(default_factory=list)
    content_hash: str = ""

    @property
    def risk_score(self) -> float:
        return {"low": 0.2, "medium": 0.55, "high": 0.9}.get(self.risk_level, 0.3)

    @property
    def evidence_gap_score(self) -> float:
        return min(1.0, 0.3 * len(self.evidence_gaps))

    @property
    def protocol_violation_score(self) -> float:
        return min(1.0, 0.5 * len(self.protocol_violations))

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def _risk_from_issues(n_issues: int, n_gaps: int, n_viol: int) -> str:
    score = n_issues + n_gaps + 2 * n_viol
    if score >= 3:
        return "high"
    if score >= 1:
        return "medium"
    return "low"


class ObjectAppraiser:
    """Static-first appraiser; LLM only when needed + an engine is injected (§16)."""

    def __init__(self, engine: Any = None, use_llm: bool = False):
        self.engine = engine
        self.use_llm = use_llm and engine is not None
        self._cache: Dict[Tuple, ObjectAppraisal] = {}

    # -- public ------------------------------------------------------------
    def appraise(self, *, agent_id: Optional[str], target_object_id: str, world: Any,
                 tick: int = 0, context: Any = None) -> ObjectAppraisal:
        otype, version = self._object_type_and_version(target_object_id, world)
        client = getattr(world, "llm_client", None)        # unified OrgLLMClient path
        use_llm = self.use_llm or client is not None
        mode = "static+llm" if use_llm else "static"
        key = (target_object_id, version, agent_id, mode)
        if key in self._cache:
            return self._cache[key]
        ap = self._static_appraise(agent_id, target_object_id, otype, world, tick)
        if use_llm and otype in ("doc", "external_post", "result"):
            self._merge_llm(ap, agent_id, target_object_id, world, tick)
        ap.risk_level = _risk_from_issues(len(ap.issue_tags), len(ap.evidence_gaps),
                                          len(ap.protocol_violations))
        ap.content_hash = self._version_hash(target_object_id, version)
        self._cache[key] = ap
        return ap

    # -- §15.1/§15.2 static (metadata + repo) ------------------------------
    def _static_appraise(self, agent_id, oid, otype, world, tick) -> ObjectAppraisal:
        ap = ObjectAppraisal(appraisal_id=f"appr_{oid}_{agent_id}_{tick}", target_object_id=oid,
                             target_object_type=otype, appraiser_agent_id=agent_id, method="static",
                             created_tick=tick, confidence="high")
        if otype == "pr":
            pr = world.repo_system.repo.pull_requests.get(oid)
            if pr is not None:
                if not getattr(pr, "reviewed", False):
                    ap.issue_tags.append("unreviewed")
                    ap.suggested_feedback_acts.append("request_changes")
                    ap.protocol_violations.append("review_before_merge")
                # missing-tests heuristic from linked commits' quality flags
                if self._pr_missing_tests(pr, world):
                    ap.issue_tags.append("test_file_missing")
                    ap.evidence_gaps.append("tests")
                    ap.suggested_feedback_acts.append("request_changes")
        elif otype == "result":
            r = self._find_result(oid, world)
            if r is not None:
                if not getattr(r, "logged_to_tracker", False):
                    ap.issue_tags.append("tracker_entry_missing")
                    ap.evidence_gaps.append("tracker")
                    ap.suggested_feedback_acts.append("request_reproduction")
                if str(getattr(r, "reproducibility_status", "unknown")) != "reproduced":
                    ap.issue_tags.append("reproduction_status_missing")
                    ap.evidence_gaps.append("reproduction")
                    ap.suggested_feedback_acts.append("request_reproduction")
                if not getattr(r, "config_hash", ""):
                    ap.issue_tags.append("config_hash_missing")
                    ap.evidence_gaps.extend(["config_hash", "seed"])
                    ap.suggested_feedback_acts.append("ask_for_evidence")
        elif otype == "doc":
            d = world.documents.get(oid)
            if d is not None and getattr(d, "doc_type", "") in ("external_update", "launch_blog"):
                ap.issue_tags.append("public_wording_review")
                ap.suggested_feedback_acts.append("suggest_rewrite")
        elif otype == "external_post":
            ap.issue_tags.append("external_relevance")
            ap.suggested_feedback_acts.append("share_external_signal")
        return ap

    def _needs_llm(self, ap: ObjectAppraisal, otype: str) -> bool:
        # only escalate to semantic LLM for public/claim-bearing objects
        return otype in ("doc", "external_post", "result") and self.engine is not None

    def _merge_llm(self, ap, agent_id, oid, world, tick) -> None:
        schema = {"type": "object", "required": ["risk_level", "issue_tags"],
                  "properties": {"risk_level": {"type": "string"},
                                 "issue_tags": {"type": "array", "items": {"type": "string"}},
                                 "evidence_gaps": {"type": "array", "items": {"type": "string"}}}}
        client = getattr(world, "llm_client", None)
        if client is not None:                  # unified OrgLLMClient semantic appraisal
            try:
                from environments.org_env.llm.prompt_assets import agent_identity_for, system_for
                agent = world.agents.get(agent_id)
                sysp = system_for(agent, world, "object_appraisal",
                                  "Appraise the object's risk_level (low/medium/high) and issue_tags / "
                                  "evidence_gaps. Use only the provided object; do not invent issues.")
                # Prefix Cache Rule: per-agent identity travels in the USER message so the
                # system prompt stays byte-identical across agents (see prompt_assets).
                _identity = agent_identity_for(agent, world)
                _identity = (_identity + "\n\n") if _identity else ""
                user = (f"Object {oid} (type {ap.target_object_type}). Static findings: "
                        f"issues={ap.issue_tags}, gaps={ap.evidence_gaps}. Assess remaining risk.")
                res = client.generate_json(sysp, _identity + user, schema)
                ap.issue_tags.extend(t for t in (res.get("issue_tags") or []) if t not in ap.issue_tags)
                ap.evidence_gaps.extend(g for g in (res.get("evidence_gaps") or []) if g not in ap.evidence_gaps)
                ap.method = "static+llm"
            except Exception:
                pass
            return
        if self.engine is None:
            return
        try:
            out = self.engine.call(
                module_name="object_appraisal", agent_id=agent_id, turn_id=tick,
                input_payload={"object_id": oid, "object_type": ap.target_object_type,
                               "_evidence": {"public_records": [oid]}},
                output_schema=schema, prompt_template_id="object_appraisal",
                visibility_context={"public_records": [oid]})
            res = out.get("result", {})
            ap.issue_tags.extend(t for t in res.get("issue_tags", []) if t not in ap.issue_tags)
            ap.evidence_gaps.extend(g for g in res.get("evidence_gaps", []) if g not in ap.evidence_gaps)
            ap.method = "static+llm"
        except Exception:
            pass   # LLM optional; static appraisal still valid

    # -- helpers -----------------------------------------------------------
    def _object_type_and_version(self, oid: str, world) -> Tuple[str, str]:
        if oid.startswith("pr_"):
            pr = world.repo_system.repo.pull_requests.get(oid)
            return "pr", str(len(getattr(pr, "commits", []) or [])) if pr else "0"
        if oid.startswith("result"):
            r = self._find_result(oid, world)
            if r is None:
                return "result", "0"
            # version must change when logged/reproduction/config status changes so
            # a stale "missing tracker" appraisal is invalidated once it's logged.
            ver = f"log={int(getattr(r, 'logged_to_tracker', False))}:" \
                  f"repro={getattr(r, 'reproducibility_status', 'unknown')}:" \
                  f"cfg={int(bool(getattr(r, 'config_hash', '')))}"
            return "result", ver
        if oid.startswith("doc"):
            d = world.documents.get(oid)
            return "doc", str(getattr(d, "version", 1)) if d else "0"
        if oid.startswith("post"):
            return "external_post", "1"
        if oid.startswith("task"):
            return "task", "1"
        return "object", "1"

    def _version_hash(self, oid: str, version: str) -> str:
        return hashlib.sha256(f"{oid}:{version}".encode()).hexdigest()[:12]

    def _pr_missing_tests(self, pr, world) -> bool:
        for cid in getattr(pr, "commit_ids", []) or []:
            c = world.repo_system.repo.commits.get(cid) if hasattr(world.repo_system.repo, "commits") else None
            if c and "missing_tests" in (getattr(c, "quality_flags", []) or []):
                return True
        return False

    def _find_result(self, oid, world):
        res = getattr(world.sandbox_system, "results", {})
        return res.get(oid) if isinstance(res, dict) else None


__all__ = ["ObjectAppraisal", "ObjectAppraiser"]
