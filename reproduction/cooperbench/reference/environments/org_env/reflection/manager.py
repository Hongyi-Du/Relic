"""ReflectionManager — episode/event pressure -> agent reflection -> wishes.

Reflection generation is template-based by default (deterministic, LLM-free) and
optionally LLM-driven when ``world.text_engine`` is present (mirrors the text-layer
pattern). Every reflection is written into long-term ``AgentMemory``, an
``AgentLogEntry`` trace, the world event log + event graph, and linked back to its
episode; wishes are EXTRACTED from the reflection's improvement ideas (never
invented) and keep ``source_reflection_id`` + ``raw_reflection_excerpt``.

Generic only — keyed by role / episode-type / need-type / work-state, never by a
specific agent id.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from environments.org_env.reflection.objects import (
    AgentLogEntry,
    AgentMemory,
    AgentReflection,
    Wish,
    canon_wish_type,
    make_wish_fingerprint,
    support_type_for,
)

REFLECT_COOLDOWN = 8          # min ticks between an agent's reflections
PERIODIC_REFLECT_EVERY = 48   # team-level reflection cadence
MAX_REFLECTIONS_PER_TICK = 2  # bound cost / avoid spam

# Wish sparsity / dedup caps (preflight §7.3 / §8.1).
MAX_WISHES_PER_REFLECTION = 1
MAX_OPEN_WISHES_PER_AGENT = 4
MAX_OPEN_WISHES_GLOBAL = 20
WISH_MIN_URGENCY_FOR_NEW = 0.75
SIMILAR_WISH_THRESHOLD = 0.78

# preflight v3 §5.4: ground a wish in concrete product objects from its TEXT, so wishes
# stop coming back with empty related_object_ids. (phrase -> product artifact / issue id)
WISH_OBJECT_GROUNDING = [
    (("readme", "overpromise", "over-promise", "overclaim", "over-claim"), "art_README_md"),
    (("claim tracker", "claim_tracker", "evidence link", "evidence-link"), "art_tools_claim_tracker_py"),
    (("source credibility", "source tracker", "source_tracker"), "art_tools_source_tracker_py"),
    (("report writer", "report_writer", "quality gate", "quality-gate"), "art_tools_report_writer_py"),
    (("eval stub", "eval_stub", "metric", "metrics"), "art_eval_eval_stub_py"),
    (("product design", "product_design", "workflow", "research loop"), "art_docs_product_design_md"),
    (("cheap mode", "cheap-mode"), "issue_7"),
    (("onboarding",), "issue_6"),
]

# episode_type -> reflection ontology (team line + improvement ideas). Each idea:
# need_type / missing_support_type / description (first-person) / urgency / risk /
# self|team. GENERIC (no agent ids).
EPISODE_IMPROVEMENT: Dict[str, Dict[str, Any]] = {
    "claim_dispute_episode": {
        "team": "the team repeatedly loses time when experiment results are challenged for missing evidence",
        "ideas": [
            {"need_type": "protocol_need", "missing_support_type": "protocol",
             "description": "a result-evidence protocol defining what evidence (seed/config/trace) "
                            "is required before a result enters a report",
             "urgency": 0.75, "risk": "weak claims keep entering reports", "team": True, "self": False},
            {"need_type": "tool_need", "missing_support_type": "tool",
             "description": "a lightweight result-logging template that records seed, cost, config and trace",
             "urgency": 0.6, "risk": "results keep being challenged after the fact", "team": False, "self": True},
        ]},
    "experiment_episode": {
        "team": "cheap-mode experiments lack standard traces, so results are hard to verify later",
        "ideas": [
            {"need_type": "tool_need", "missing_support_type": "tool",
             "description": "a cheap-mode benchmark runner that automatically logs cost, seed, config and trace",
             "urgency": 0.8, "risk": "future results will keep being challenged", "team": False, "self": True},
        ]},
    "protocol_formation_episode": {
        "team": "norms are proposed but adoption and enforcement are inconsistent",
        "ideas": [
            {"need_type": "workflow_need", "missing_support_type": "workflow",
             "description": "a clear adoption + enforcement workflow so a proposed rule actually takes hold",
             "urgency": 0.6, "risk": "good rules get proposed and then ignored", "team": True, "self": False},
        ]},
    "launch_crunch_episode": {
        "team": "launch pressure trades quality for speed and overloads a few people",
        "ideas": [
            {"need_type": "workflow_need", "missing_support_type": "workflow",
             "description": "a launch checklist that gates ship-readiness (tests/review/docs) under deadline",
             "urgency": 0.7, "risk": "shipping under pressure keeps breaking quality", "team": True, "self": False},
            {"need_type": "role_clarity_need", "missing_support_type": "role_clarity",
             "description": "clearer ownership during a crunch so work is not duplicated or dropped",
             "urgency": 0.5, "risk": "crunches stay chaotic", "team": True, "self": False},
        ]},
    "feedback_ingestion_episode": {
        "team": "external feedback enters ad-hoc and routing it internally is slow",
        "ideas": [
            {"need_type": "workflow_need", "missing_support_type": "workflow",
             "description": "a customer-feedback intake workflow that routes a signal to an owner and a task",
             "urgency": 0.6, "risk": "customer signals get lost or handled twice", "team": True, "self": False},
        ]},
    "customer_triage_episode": {
        "team": "customer issues are triaged inconsistently without a shared template",
        "ideas": [
            {"need_type": "artifact_need", "missing_support_type": "artifact",
             "description": "a reusable customer-triage template (severity, owner, impact, status)",
             "urgency": 0.55, "risk": "triage quality depends on who happens to do it", "team": True, "self": False},
        ]},
}

# role -> first-person self-assessment flavor (generic, role-keyed).
ROLE_SELF_LINE = {
    "fast_engineer": "I move fast, but my results are not always easy for others to verify",
    "reliability": "I keep having to ask for evidence and reproducibility after the fact",
    "cofounder": "I spend a lot of time reviewing and chasing evidence",
    "founder": "I push hard for momentum and sometimes outrun our process",
    "editorial": "I catch quality and clarity issues late, when they are expensive to fix",
    "community": "I see customer pain arrive faster than we can route it internally",
    "external_voice": "I have to turn rough internal work into external-facing material under time pressure",
    "artifact_design": "I keep rebuilding the same artifacts because we lack reusable templates",
}
_DEFAULT_SELF_LINE = "I notice recurring friction in how we work"


def _rules_and_what_they_cost(world: Any) -> List[Dict[str, Any]]:
    """The rules the organization binds itself with, and what each is costing.

    Reflection used to receive a list of protocol ids and nothing else. The
    prompt asks a member to say when "an existing rule is the problem and should
    be relaxed or repealed", but no count for any rule reached any prompt
    anywhere in the system, so the question could only ever be answered from
    whichever refusals happened to land on the member personally.

    That is not enough to see the shape that matters. One B3 arm kept a gate
    that refused 290 merges over 318 ticks; each member met it a few times,
    filed another rule to satisfy it, and nobody was in a position to notice
    that it was the reason nothing shipped.
    """
    reg = getattr(world, "protocol_registry", None)
    if reg is None:
        return []
    try:
        from environments.org_env.backend.protocol.harm import rules_and_their_refusals
    except Exception:
        return list(getattr(reg, "protocols", {}).keys())
    rows = rules_and_their_refusals(world)
    # An arm that inherited its rules as prose has no registry to read, and
    # would reflect as though the organization had never been told anything.
    # The rules belong here for both; what differs is that only the executable
    # form has a record of turning work away, which is the treatment rather
    # than a difference in what the member knows.
    try:
        from environments.org_env.experiments.capability_transfer import (
            inherited_capability_texts,
        )

        rows.extend({"rule": text} for text in inherited_capability_texts(world))
    except Exception:
        pass
    return rows


def _programbench_phase_context(world: Any) -> Dict[str, Any] | None:
    """Agent-visible live phase context for the explicit adapted profile only."""

    if "programbench_profile_state" not in getattr(world, "__dict__", {}):
        return None
    from environments.org_env.programbench import programbench_agent_phase_context

    return programbench_agent_phase_context(world)


def _programbench_retrieved_surfaces(
    world: Any, agent_id: str
) -> List[Dict[str, Any]] | None:
    """Targeted public retrievals available to this agent's cognition only."""

    if "programbench_profile_state" not in getattr(world, "__dict__", {}):
        return None
    from environments.org_env.programbench import (
        programbench_retrieved_public_surfaces,
    )

    return [
        dict(row)
        for row in programbench_retrieved_public_surfaces(world, agent_id)
    ]


class ReflectionManager:
    def __init__(self) -> None:
        self.reflections: Dict[str, AgentReflection] = {}
        self.wishes: Dict[str, Wish] = {}
        self._rseq = 0
        self._wseq = 0
        self._last_periodic = 0

    # ====================== triggers ===================================== #
    def maybe_trigger_reflection(self, agent_id: str, world: Any, *, reason: str = "",
                                 episode: Any = None) -> bool:
        mem = self._memory(world, agent_id)
        tick = int(getattr(world, "world_tick", 0))
        if mem.last_reflection_tick is not None and tick - mem.last_reflection_tick < REFLECT_COOLDOWN:
            return False
        return True

    def on_episode_closed(self, episode: Any, world: Any) -> List[AgentReflection]:
        """Episode close is the primary reflection trigger (spec §3)."""
        out: List[AgentReflection] = []
        agents = [a for a in [getattr(episode, "primary_agent_id", None)] if a]
        # plus one other key participant (most stressed) so team reflection emerges
        others = [a for a in getattr(episode, "participants", []) if a not in agents]
        others.sort(key=lambda a: -self._stress(world, a))
        for aid in agents + others[:1]:
            if len(out) >= MAX_REFLECTIONS_PER_TICK:
                break
            if self.maybe_trigger_reflection(aid, world, reason="episode_close", episode=episode):
                out.append(self.reflect(aid, world, episode=episode, reason="episode_close"))
        return out

    def periodic(self, world: Any) -> List[AgentReflection]:
        """Every PERIODIC_REFLECT_EVERY ticks the most-stressed agent reflects on team state."""
        tick = int(getattr(world, "world_tick", 0))
        if tick == 0 or tick - self._last_periodic < PERIODIC_REFLECT_EVERY:
            return []
        self._last_periodic = tick
        agents = sorted(getattr(world, "agents", {}).keys(), key=lambda a: -self._stress(world, a))
        out = []
        for aid in agents[:1]:
            if self.maybe_trigger_reflection(aid, world, reason="periodic"):
                out.append(self.reflect(aid, world, episode=None, reason="periodic"))
        return out

    # ====================== orchestration ================================ #
    def reflect(self, agent_id: str, world: Any, *, episode: Any = None,
                reason: str = "", batch_budget: Optional[List[int]] = None) -> AgentReflection:
        ctx = self.build_reflection_context(agent_id, world, episode)
        refl = self.generate_reflection(agent_id, ctx, world, episode, reason)
        self.reflections[refl.reflection_id] = refl
        self._write_memory(world, refl)
        self._write_log(world, refl, episode)
        self._write_event(world, refl, episode)
        # Everything above is the agent thinking about its own work: the
        # reflection lands in its memory, its log and the event record, and it
        # informs what that agent does next. Every condition gets it.
        #
        # Everything below is where a private thought starts becoming a shared
        # rule, and that is the mechanism the top rung is there to test. Without
        # institutionalization the ideas stay in memory, which is what an
        # organization that never writes anything down looks like.
        # preflight §7/§8/§9: ideas -> map + dedup/merge + sparse create (NOT 1:1).
        if not getattr(world, "institutionalization_enabled", True):
            return refl
        wishes = self.integrate_wishes(refl, world, episode, batch_budget=batch_budget)
        for wsh in wishes:
            refl.created_wish_ids.append(wsh.wish_id)
            self._write_wish_side_effects(world, wsh, episode)
        if episode is not None:
            self._link_episode(episode, refl, wishes)
        return refl

    # ====================== context ====================================== #
    def build_reflection_context(self, agent_id: str, world: Any, episode: Any = None) -> Dict[str, Any]:
        agent = world.agents.get(agent_id)
        role = getattr(agent, "role", "") if agent else ""
        ws = agent.work_state.snapshot() if agent and hasattr(agent, "work_state") else {}
        acts = [a for a in getattr(world, "action_log", []) if a.get("agent_id") == agent_id][-8:]
        fails = [a for a in acts if not a.get("success", True)]
        mem = self._memory(world, agent_id)
        open_eps = []
        closed_eps = []
        mgr = getattr(world, "episode_manager", None)
        if mgr is not None:
            for ep in mgr.episodes.values():
                (open_eps if ep.status == "open" else closed_eps).append(ep.episode_type)
        protocols = _rules_and_what_they_cost(world)
        from environments.org_env.cooperbench.reflection_grounding import (
            cooperbench_reflection_current_state,
        )
        authoritative_state = cooperbench_reflection_current_state(world, agent_id)
        programbench_context = _programbench_phase_context(world)
        programbench_retrieved = _programbench_retrieved_surfaces(
            world, agent_id
        )
        from environments.org_env.llm.prompt_assets import agent_identity_for, render_product_context
        from environments.org_env.cooperbench.visibility import is_strict_coop, visible_brief_context
        failures: Dict[str, Any] = {}
        try:
            from environments.org_env.experiments.ablations import (
                INSTITUTIONALIZATION,
                mechanism_disabled,
            )
            if (getattr(world, "institutionalization_enabled", False)
                    and not mechanism_disabled(world, INSTITUTIONALIZATION)):
                from environments.org_env.reflection.failure_digest import (
                    recent_failure_digest,
                )
                failures = recent_failure_digest(world, agent_id)
        except Exception:  # noqa: BLE001  a missing digest must not stop reflection
            failures = {}
        return {
            "agent_id": agent_id, "role": role,
            **(
                {"programbench_workflow_state": programbench_context}
                if programbench_context is not None
                else {}
            ),
            **(
                {
                    "programbench_targeted_public_retrievals": (
                        programbench_retrieved
                    )
                }
                if programbench_retrieved
                else {}
            ),
            # First, ahead of the 6876-character product context: what the gates
            # said is the thing to reason from, and it should not sit behind the
            # bulk. Only this arm has the key, so nothing below B3 is reordered.
            **({"what_has_been_failing": failures} if failures else {}),
            **(
                {"authoritative_current_state": authoritative_state}
                if authoritative_state is not None
                else {}
            ),
            "product_context": render_product_context(world, actor_id=agent_id),
            **({"cooperbench_visible_briefs": visible_brief_context(world, agent_id)}
               if is_strict_coop(world) else {}),
            "profile": dict(getattr(agent, "profile", {}) or {}),
            "work_state": {k: ws.get(k) for k in ("stress", "fatigue", "burnout_risk", "morale")},
            # action_log already records what each act was aimed at; keeping only
            # the verb left reflection unable to tell "edited listutils" from
            # "edited something", so no memory of the work could form.
            "recent_actions": [{"action": a.get("action_type"), "target": a.get("target"),
                                "success": a.get("success", True)} for a in acts],
            # The verb and the target: that a CI run failed. What it said is in
            # `what_has_been_failing` above.
            "recent_failures": [{"action": a.get("action_type"), "target": a.get("target")}
                                for a in fails],
            "episode_type": getattr(episode, "episode_type", None),
            "episode_problem": getattr(episode, "problem_statement", "") if episode else "",
            "episode_conflict": getattr(episode, "conflict_summary", "") if episode else "",
            "open_episode_types": open_eps, "closed_episode_types": closed_eps,
            "existing_protocols": protocols,
            "existing_unresolved_needs": list(mem.unresolved_needs),
            # Reflection drops message OBJECTS as pollution, which is right —
            # a bag of unordered message records is not material to reason over.
            # But dropping them left B3 unable to reflect on anything that was
            # said, since the profile policy never reads a prompt and this is
            # its only route in. What is admitted here is the transcript, in
            # order, which is a different thing from the inbox.
            "recent_conversation": list(mem.conversation[-20:]),
        }

    # ====================== generation (template default, LLM optional) === #
    def generate_reflection(self, agent_id, context, world, episode, reason) -> AgentReflection:
        self._rseq += 1
        rid = f"refl_{self._rseq}"
        tick = int(getattr(world, "world_tick", 0))
        data = None
        model = None
        client = getattr(world, "llm_client", None)
        if client is not None:                       # Phase 4: unified OrgLLMClient path
            data, model = self._llm_client_reflect(client, agent_id, context, world)
        if data is None:
            engine = getattr(world, "text_engine", None)
            if engine is not None:
                data, model = self._llm_reflect(engine, agent_id, context)
        if data is None:
            data = self._template_reflect(context, world)
        ideas = data.get("improvement_ideas", [])

        def _needs(flag: str) -> list:
            """Descriptions of the ideas marked for one audience.

            The flag was read with .get and the description by subscript, so an
            idea that carried `team` without a `description` raised KeyError out
            of reflect, up through world.step, and killed the run: one arm died
            at t35 having spent a hundred minutes. The model is free to omit a
            key the schema does not require, so an idea missing its text is
            dropped rather than fatal. Non-dict entries go the same way.
            """
            out = []
            for idea in ideas:
                if not isinstance(idea, dict) or not idea.get(flag):
                    continue
                text = str(idea.get("description") or "").strip()
                if text:
                    out.append(text)
            return out

        refl = AgentReflection(
            reflection_id=rid, agent_id=agent_id, tick=tick,
            source_episode_ids=[episode.episode_id] if episode is not None else [],
            source_event_ids=list(getattr(episode, "linked_event_ids", [])[:6]) if episode else [],
            source_object_ids=list(getattr(episode, "linked_object_ids", [])[:8]) if episode else [],
            self_assessment=data.get("self_assessment", ""),
            team_assessment=data.get("team_assessment", ""),
            perceived_blockers=data.get("perceived_blockers", []),
            perceived_repeated_failures=data.get("perceived_repeated_failures", []),
            perceived_team_needs=_needs("team"),
            perceived_self_needs=_needs("self"),
            improvement_ideas=ideas, raw_text=data.get("raw_text", ""),
            llm_model=model, trigger_reason=reason)
        return refl

    def _template_reflect(self, ctx: Dict[str, Any], world: Any) -> Dict[str, Any]:
        role = ctx.get("role", "")
        st = ctx.get("work_state", {})
        etype = ctx.get("episode_type")
        self_line = ROLE_SELF_LINE.get(role, _DEFAULT_SELF_LINE)
        stress = float(st.get("stress") or 0.0)
        fatigue = float(st.get("fatigue") or 0.0)
        burnout = float(st.get("burnout_risk") or 0.0)
        if stress > 0.6 or fatigue > 0.6:
            self_line += f" — and I'm under heavy load (stress {stress:.2f}, fatigue {fatigue:.2f})"
        spec = EPISODE_IMPROVEMENT.get(etype) if etype else None
        team_line = spec["team"] if spec else "we keep hitting the same kind of friction across work"
        ideas = [dict(i) for i in (spec["ideas"] if spec else [])]
        # state-driven self need (overload) — generic
        if burnout > 0.6 or stress > 0.7:
            ideas.append({"need_type": "resource_need", "missing_support_type": "resource",
                          "description": "more capacity / load-balancing so the same people don't burn out",
                          "urgency": round(min(0.9, 0.5 + burnout), 2),
                          "risk": "key people may disengage or leave", "team": True, "self": True})
        if not ideas:
            ideas.append({"need_type": "coordination_need", "missing_support_type": "coordination",
                          "description": "a clearer way to coordinate so the same problem stops recurring",
                          "urgency": 0.4, "risk": "recurring friction", "team": True, "self": False})
        blockers = []
        if ctx.get("episode_conflict"):
            blockers.append(ctx["episode_conflict"])
        if ctx.get("recent_failures"):
            failed = {
                f"{f['action']} on {f['target']}" if isinstance(f, dict) and f.get("target")
                else (f["action"] if isinstance(f, dict) else str(f))
                for f in ctx["recent_failures"]
            }
            blockers.append("recent failed actions: " + ", ".join(sorted(failed)))
        repeated = []
        if etype and ctx.get("closed_episode_types", []).count(etype) >= 2:
            repeated.append(f"repeated {etype.replace('_episode','')} episodes")
        raw = (f"{self_line}. I think {team_line}. "
               + " ".join(f"I need {i['description']}." for i in ideas))
        return {"self_assessment": self_line[0].upper() + self_line[1:] + ".",
                "team_assessment": team_line[0].upper() + team_line[1:] + ".",
                "perceived_blockers": blockers, "perceived_repeated_failures": repeated,
                "improvement_ideas": ideas, "raw_text": raw}

    def _llm_client_reflect(self, client, agent_id, context, world=None):
        """Phase 4: generate the reflection via the unified OrgLLMClient (template fallback)."""
        try:
            from environments.org_env.llm.prompt_assets import agent_identity_for, system_for
            from environments.org_env.llm.prompts import REFLECTION_SYSTEM, reflection_user
            from environments.org_env.llm.schemas import REFLECTION_SCHEMA
            agent = world.agents.get(agent_id) if world is not None else None
            system = system_for(agent, world, "reflection", REFLECTION_SYSTEM)
            # Prefix Cache Rule: per-agent identity travels in the USER message so the
            # system prompt stays byte-identical across agents (see prompt_assets).
            _identity = agent_identity_for(agent, world)
            _identity = (_identity + "\n\n") if _identity else ""
            res = client.generate_json(system, _identity + reflection_user(context), REFLECTION_SCHEMA)
            # robustness: a real model may return strings as lists / mixed shapes
            def _s(v):
                return " ".join(str(x) for x in v) if isinstance(v, list) else str(v or "")
            res["self_assessment"] = _s(res.get("self_assessment"))
            res["team_assessment"] = _s(res.get("team_assessment"))
            for f in ("perceived_blockers", "perceived_self_needs", "perceived_team_needs",
                      "perceived_repeated_failures"):
                res[f] = [str(x) for x in (res.get(f) or [])]
            # An empty idea list is a valid assessment, not a model failure.
            # Keep the model's reflection instead of inventing a template need;
            # the reader filters individual ideas that have no description.
            ideas = [i for i in (res.get("improvement_ideas") or []) if isinstance(i, dict)]
            res["improvement_ideas"] = ideas
            for i in ideas:
                i.setdefault("missing_support_type", str(i.get("need_type", "tool_need")).replace("_need", ""))
                i.setdefault("team", True)
                i.setdefault("self", False)
                i.setdefault("urgency", 0.6)
                i.setdefault("risk", i.get("risk_if_unaddressed", ""))
                # Every other key a consumer reads is defaulted here; this one was
                # not, and the consumers read it by subscript. An idea that came
                # back with `team` but no `description` raised KeyError out of
                # reflect, through world.step, and ended the run -- one arm died
                # at t35 with a hundred minutes spent. Left empty rather than
                # invented, so a caller can tell it apart from a real answer.
                i.setdefault("description", "")
            return res, getattr(client, "provider", "llm")
        except Exception:
            return None, None

    def _llm_reflect(self, engine, agent_id, context):
        """Optional LLM reflection (deep_reflection role). Returns (data, model) or (None, None)."""
        schema = {"type": "object", "properties": {
            "self_assessment": {"type": "string"}, "team_assessment": {"type": "string"},
            "perceived_blockers": {"type": "array"}, "improvement_ideas": {"type": "array"},
            "raw_text": {"type": "string"}}}
        try:
            out = engine.call(module_name="reflection", agent_id=agent_id, turn_id=0,
                              input_payload={"context": context}, output_schema=schema,
                              prompt_template_id="reflection", model_role="deep_reflection",
                              visibility_context={"episodes": [], "memories": []})
            res = out.get("result") or {}
            if not res.get("improvement_ideas"):
                return None, None
            # normalize idea shape
            for i in res["improvement_ideas"]:
                i.setdefault("missing_support_type", i.get("need_type", "tool_need").replace("_need", ""))
                i.setdefault("team", True)
                i.setdefault("self", False)
                i.setdefault("urgency", 0.6)
                i.setdefault("risk", "")
            return res, (out.get("metadata", {}) or {}).get("model", "llm")
        except Exception:
            return None, None

    # ====================== wish integration (map + sparse + dedup) ====== #
    def integrate_wishes(self, reflection: AgentReflection, world: Any,
                         episode: Any = None, *, batch_budget: Optional[List[int]] = None) -> List[Wish]:
        """Turn a reflection's improvement ideas into AT MOST a few stable wishes
        (preflight §7-§9): map need_type, dedup/merge into existing open wishes, and
        only create a new wish when it clears the urgency/issue/founder bar + caps.
        Ideas that don't qualify stay in memory only (already written there).
        ``batch_budget`` is a 1-element [remaining] list the batch manager uses to
        enforce MAX_NEW_WISHES_PER_BATCH across agents."""
        tick = reflection.tick
        aid = reflection.agent_id
        ideas = sorted(reflection.improvement_ideas or [],
                       key=lambda i: -self._coerce_urgency(i.get("urgency")))
        created: List[Wish] = []
        open_wishes = [w for w in self.wishes.values() if w.status in ("open", "interpreted")]
        open_agent = sum(1 for w in open_wishes if w.agent_id == aid)
        open_global = len(open_wishes)
        for idea in ideas:
            desc = str(idea.get("description", "")).strip()
            if not desc:
                continue
            wtype = canon_wish_type(idea.get("need_type") or idea.get("missing_support_type"))
            from environments.org_env.cooperbench.reflection_grounding import (
                stale_cooperbench_governance_need,
            )
            stale_governance, adopted_protocol_ids = (
                stale_cooperbench_governance_need(
                    world,
                    wtype,
                    desc,
                    reflection_text=" ".join(
                        [
                            reflection.self_assessment,
                            reflection.team_assessment,
                            *reflection.perceived_blockers,
                            *reflection.perceived_repeated_failures,
                            *reflection.perceived_team_needs,
                            *reflection.perceived_self_needs,
                        ]
                    ),
                )
            )
            if stale_governance:
                getattr(world, "events", []).append(
                    {
                        "type": "reflection_grounding_event",
                        "subtype": "discarded_already_satisfied_governance_need",
                        "reflection_id": reflection.reflection_id,
                        "agent_id": reflection.agent_id,
                        "tick": int(tick),
                        "wish_type": wtype,
                        "adopted_protocol_ids": adopted_protocol_ids,
                        "description": desc[:400],
                    }
                )
                continue
            # a wish is about a PRODUCT gap — keep product artifacts/issues/results/docs as
            # primary related objects; drop meeting/mnote/protocol/message pollution (review #4).
            raw_objs = [o for o in (idea.get("related_object_ids")
                                    or reflection.source_object_ids) if o]
            related_objs = [o for o in raw_objs
                            if str(o).startswith(("art_", "issue_", "result_", "doc_"))
                            and not str(o).startswith(("doc_experiment_tracker",))]
            # §5.4: ground from the wish text so related_object_ids is never empty
            self._ground_wish_objects(idea, reflection, world, related_objs)
            related_issues = [o for o in related_objs if str(o).startswith("issue_")]
            target = self._best_target_problem(reflection, idea, world)
            urgency = self._coerce_urgency(idea.get("urgency"))
            risk = idea.get("risk_if_unaddressed") or idea.get("risk") or "recurring friction"
            fp = make_wish_fingerprint(wtype, target, related_objs)
            # dedup/merge into a similar OPEN wish (§8)
            match = self._find_similar_wish(wtype, target, related_objs, fp)
            if match is not None:
                self._merge_into_wish(match, reflection, idea, urgency, risk,
                                      related_objs, related_issues, world)
                continue
            # sparse-create gating (§7.2/§7.3)
            if len(created) >= MAX_WISHES_PER_REFLECTION:
                break
            if open_agent >= MAX_OPEN_WISHES_PER_AGENT or open_global >= MAX_OPEN_WISHES_GLOBAL:
                break
            if batch_budget is not None and batch_budget[0] <= 0:
                break
            if not self._allow_new_wish(idea, urgency, related_issues, world, aid):
                continue
            self._wseq += 1
            w = Wish(
                wish_id=f"wish_{self._wseq}", agent_id=aid,
                source_reflection_id=reflection.reflection_id,
                source_reflection_ids=[reflection.reflection_id],
                supporting_agent_ids=[aid], support_count=1,
                source_episode_id=(episode.episode_id if episode is not None else None),
                source_event_ids=list(reflection.source_event_ids),
                raw_reflection_excerpt=f"I need {desc}.", wish_type=wtype, fingerprint=fp,
                interpreted_need=self._compress(desc), target_problem=target,
                self_related=bool(idea.get("self")), team_related=bool(idea.get("team", True)),
                suggested_improvement=desc, missing_support_type=support_type_for(wtype),
                urgency=urgency, expected_benefit=self._infer_benefit(idea, risk),
                risk_if_unaddressed=risk,
                related_object_ids=list(related_objs), related_issue_ids=list(related_issues),
                related_agent_ids=[aid], related_episode_ids=list(reflection.source_episode_ids),
                related_channel_ids=list(getattr(episode, "linked_channels", []) or []) if episode else [],
                status="open", created_at_tick=tick, updated_at_tick=tick)
            self.wishes[w.wish_id] = w
            created.append(w)
            open_agent += 1
            open_global += 1
            if batch_budget is not None:
                batch_budget[0] -= 1
        return created

    def _allow_new_wish(self, idea: Dict[str, Any], urgency: float,
                        related_issues: List[str], world: Any, agent_id: str = "") -> bool:
        """A new wish needs a real anchor + one of the §7.2 conditions:
        high urgency / high-priority issue / founder-or-cofounder push."""
        if not idea.get("description") or not (idea.get("risk_if_unaddressed") or idea.get("risk")):
            return False
        if urgency >= WISH_MIN_URGENCY_FOR_NEW:
            return True
        # high-priority product issue makes it worth a stable wish
        arts = getattr(world, "product_artifacts", {}) or {}
        for iid in related_issues:
            a = arts.get(iid)
            if a is not None and str(getattr(a, "priority", "")) == "high":
                return True
        # founder / cofounder explicitly pushes a moderate need (§7.2 #5)
        a = getattr(world, "agents", {}).get(agent_id)
        if a is not None and getattr(a, "role", "") in ("founder", "cofounder") and urgency >= 0.6:
            return True
        return False

    def _find_similar_wish(self, wtype: str, target: str, related_objs: List[str],
                           fp: str) -> Optional[Wish]:
        """Merge aggressively: an LLM rephrases the same need differently each time,
        so a shared (wish_type + related object) counts as the same wish (§8)."""
        objs = set(related_objs)
        for w in self.wishes.values():
            if w.status not in ("open", "interpreted") or w.wish_type != wtype:
                continue
            if w.fingerprint and w.fingerprint == fp:
                return w
            wobjs = set(w.related_object_ids)
            # same type + ANY shared related object -> same underlying need
            if objs and wobjs and (objs & wobjs):
                return w
            # object-free needs: same type + similar target problem
            if not objs and not wobjs and target and w.target_problem \
                    and self._compress(target)[:40] == self._compress(w.target_problem)[:40]:
                return w
        return None

    def _merge_into_wish(self, w: Wish, reflection: AgentReflection, idea: Dict[str, Any],
                         urgency: float, risk: str, related_objs: List[str],
                         related_issues: List[str], world: Any) -> None:
        new_reflection = reflection.reflection_id not in w.source_reflection_ids
        if new_reflection:
            w.source_reflection_ids.append(reflection.reflection_id)
        new_supporter = reflection.agent_id not in w.supporting_agent_ids
        if new_supporter:
            w.supporting_agent_ids.append(reflection.agent_id)
        w.support_count = len(w.supporting_agent_ids)
        w.urgency = max(w.urgency, urgency)
        if risk and risk not in w.risk_if_unaddressed:
            w.risk_if_unaddressed = (w.risk_if_unaddressed + "; " + risk).strip("; ")[:240]
        w.related_object_ids = sorted(set(w.related_object_ids) | set(related_objs))
        w.related_issue_ids = sorted(set(w.related_issue_ids) | set(related_issues))
        w.updated_at_tick = int(getattr(world, "world_tick", reflection.tick))
        # v8 #4: only record a reinforcement when this reflection adds NEW signal
        # (a new supporter or a new source reflection) — not on every re-evaluation,
        # which previously wrote several identical wish_reinforced events per tick.
        if new_reflection or new_supporter:
            getattr(world, "events", []).append(
                {"type": "wish_event", "subtype": "reinforced", "agent_id": reflection.agent_id,
                 "tick": w.updated_at_tick, "object_id": w.wish_id, "support_count": w.support_count})

    def _ground_wish_objects(self, idea: Dict[str, Any], reflection: AgentReflection,
                             world: Any, related_objs: List[str]) -> None:
        """Preflight v3 §5.4: scan the wish text for product nouns and attach the matching
        artifact/issue id (only ids that actually exist), so wishes are product-grounded."""
        arts = getattr(world, "product_artifacts", {}) or {}
        text = " ".join([str(idea.get("description", "")),
                         str(idea.get("suggested_improvement", "")),
                         str(idea.get("interpreted_need", "")),
                         str(getattr(reflection, "team_assessment", "") or ""),
                         str(getattr(reflection, "raw_reflection_excerpt", "") or "")]).lower()
        for phrases, oid in WISH_OBJECT_GROUNDING:
            if oid in related_objs or oid not in arts:
                continue
            if any(ph in text for ph in phrases):
                related_objs.append(oid)

    def _best_target_problem(self, reflection: AgentReflection, idea: Dict[str, Any],
                             world: Any) -> str:
        """Concrete problem statement (§9: NOT the team_assessment text dump)."""
        arts = getattr(world, "product_artifacts", {}) or {}
        for o in (idea.get("related_object_ids") or []):
            a = arts.get(o)
            if a is not None and getattr(a, "problem", ""):
                return self._compress(a.problem)
        if reflection.perceived_blockers:
            return self._compress(reflection.perceived_blockers[0])
        return self._compress(reflection.team_assessment.split(".")[0])

    @staticmethod
    def _coerce_urgency(v: Any) -> float:
        """LLMs may return urgency as a number or a word ('high'/'medium'/'low')."""
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return {"high": 0.8, "medium": 0.5, "moderate": 0.5, "low": 0.3,
                    "very high": 0.9, "critical": 0.95}.get(str(v or "").strip().lower(), 0.5)

    @staticmethod
    def _compress(text: str, limit: int = 160) -> str:
        s = " ".join(str(text or "").split())
        return s[:limit]

    @staticmethod
    def _infer_benefit(idea: Dict[str, Any], risk: str) -> str:
        return f"reduce risk: {risk}" if risk else "less recurring friction"

    # ====================== side effects (memory / log / event / graph) == #
    def _write_memory(self, world, refl: AgentReflection) -> None:
        mem = self._memory(world, refl.agent_id)
        AgentMemory._push(mem.reflections, refl.self_assessment)
        AgentMemory._push(mem.reflections, refl.team_assessment)
        for b in refl.perceived_blockers:
            AgentMemory._push(mem.repeated_blockers, b)
        for f in refl.perceived_repeated_failures:
            AgentMemory._push(mem.repeated_blockers, f)
        for idea in refl.improvement_ideas:
            AgentMemory._push(mem.unresolved_needs, idea.get("description", ""))
            # a clear lesson if the idea names a concrete risk
            if idea.get("risk"):
                AgentMemory._push(mem.lessons_learned,
                                  f"{idea.get('description', '')} — else "
                                  f"{idea.get('risk', '')}")
        mem.last_reflection_tick = refl.tick

    def _write_log(self, world, refl: AgentReflection, episode) -> None:
        log = getattr(world, "agent_log", None)
        if log is None:
            return
        name = self._name(world, refl.agent_id)
        ideas = ", ".join(i.get("need_type", "") for i in refl.improvement_ideas)
        summary = (f"{name} reflected ({refl.trigger_reason}): {refl.team_assessment} "
                   f"Needs: {ideas}.")
        log.append(AgentLogEntry(
            log_id=f"log_{len(log)}", agent_id=refl.agent_id, tick=refl.tick,
            entry_type="reflection", summary=summary,
            related_event_ids=list(refl.source_event_ids),
            related_episode_ids=list(refl.source_episode_ids),
            related_object_ids=list(refl.source_object_ids),
            raw_payload=refl.to_dict()))

    def _write_event(self, world, refl: AgentReflection, episode) -> None:
        ev = {"type": "reflection_event", "subtype": "reflected", "agent_id": refl.agent_id,
              "tick": refl.tick, "object_id": refl.reflection_id,
              "episode_ids": list(refl.source_episode_ids),
              "summary": refl.team_assessment}
        getattr(world, "events", []).append(ev)
        eg = getattr(world, "event_graph", None)
        from environments.org_env.experiments.ablations import EVENT_GRAPH, mechanism_disabled
        if eg is not None and not mechanism_disabled(world, EVENT_GRAPH):
            eg.add_node(refl.reflection_id, "reflection")
            eg.add_edge(refl.agent_id, "reflected", refl.reflection_id)
            for eid in refl.source_episode_ids:
                eg.add_edge(eid, "reflected", refl.reflection_id)

    def _write_wish_side_effects(self, world, wsh: Wish, episode) -> None:
        log = getattr(world, "agent_log", None)
        if log is not None:
            log.append(AgentLogEntry(
                log_id=f"log_{len(log)}", agent_id=wsh.agent_id, tick=wsh.created_at_tick,
                entry_type="wish_created",
                summary=f"{self._name(world, wsh.agent_id)} expressed a {wsh.wish_type}: {wsh.interpreted_need}",
                related_episode_ids=list(wsh.related_episode_ids),
                related_object_ids=list(wsh.related_object_ids),
                raw_payload=wsh.to_dict()))
        getattr(world, "events", []).append(
            {"type": "wish_event", "subtype": "created_from_reflection", "agent_id": wsh.agent_id,
             "tick": wsh.created_at_tick, "object_id": wsh.wish_id,
             "source_reflection_id": wsh.source_reflection_id,
             "episode_ids": list(wsh.related_episode_ids)})
        eg = getattr(world, "event_graph", None)
        from environments.org_env.experiments.ablations import EVENT_GRAPH, mechanism_disabled
        if eg is not None and not mechanism_disabled(world, EVENT_GRAPH):
            eg.add_node(wsh.wish_id, "wish")
            eg.add_edge(wsh.source_reflection_id, "produced_wish", wsh.wish_id)

    def _link_episode(self, episode, refl: AgentReflection, wishes: List[Wish]) -> None:
        if refl.reflection_id not in episode.linked_reflection_ids:
            episode.linked_reflection_ids.append(refl.reflection_id)
        for w in wishes:
            if w.wish_id not in episode.linked_wish_ids:
                episode.linked_wish_ids.append(w.wish_id)

    # ====================== decision-context (memory feeds future acts) == #
    def context_for_decision(self, agent_id: str, world: Any) -> Dict[str, Any]:
        mem = self._memory(world, agent_id)
        programbench_context = _programbench_phase_context(world)
        programbench_retrieved = _programbench_retrieved_surfaces(
            world, agent_id
        )
        ow = [w.to_dict() for w in self.wishes.values()
              if w.agent_id == agent_id and w.status in ("open", "interpreted")]
        return {
            **(
                {"programbench_workflow_state": programbench_context}
                if programbench_context is not None
                else {}
            ),
            **(
                {
                    "programbench_targeted_public_retrievals": (
                        programbench_retrieved
                    )
                }
                if programbench_retrieved
                else {}
            ),
            "recent_reflections": list(mem.reflections[-3:]),
            "lessons_learned": list(mem.lessons_learned[-3:]),
            "unresolved_needs": list(mem.unresolved_needs[-5:]),
            # The conversation, as opposed to the conclusions this agent drew on
            # its own. Everything else here is self-generated, so without it a
            # decision could not be influenced by anything anyone said. The full
            # transcript is kept in memory; this is the window a prompt can hold.
            "recent_conversation": list(mem.conversation[-20:]),
            "open_wishes": [w["wish_id"] for w in ow],
            "open_wish_needs": [w["interpreted_need"] for w in ow],
        }

    # ====================== snapshot ===================================== #
    def snapshot(self, world: Any) -> Dict[str, Any]:
        items = [r.to_dict() for r in self.reflections.values()]
        items.sort(key=lambda x: x["tick"])
        by_agent: Dict[str, int] = {}
        for r in self.reflections.values():
            by_agent[r.agent_id] = by_agent.get(r.agent_id, 0) + 1
        return {"items": items, "recent": items[-12:], "by_agent": by_agent,
                "total": len(self.reflections)}

    def wishes_snapshot(self) -> Dict[str, Any]:
        items = [w.to_dict() for w in self.wishes.values()]
        items.sort(key=lambda x: x["created_at_tick"])
        by_type: Dict[str, int] = {}
        for w in self.wishes.values():
            by_type[w.wish_type] = by_type.get(w.wish_type, 0) + 1
        return {"items": items, "by_type": by_type, "total": len(self.wishes)}

    def memories_snapshot(self, world: Any) -> Dict[str, Any]:
        out = {}
        for aid in getattr(world, "agents", {}):
            mem = self._memory(world, aid)
            d = mem.to_dict()
            d["open_wishes"] = [w.wish_id for w in self.wishes.values()
                                if w.agent_id == aid and w.status in ("open", "interpreted")]
            out[aid] = d
        return out

    # ====================== helpers ====================================== #
    def _memory(self, world, agent_id: str) -> AgentMemory:
        store = world.agent_memories
        if agent_id not in store:
            store[agent_id] = AgentMemory(agent_id=agent_id)
        return store[agent_id]

    def _stress(self, world, agent_id: str) -> float:
        a = world.agents.get(agent_id)
        if a is None or not hasattr(a, "work_state"):
            return 0.0
        ws = a.work_state.snapshot()
        return float(ws.get("stress") or 0.0) + float(ws.get("burnout_risk") or 0.0)

    def _name(self, world, agent_id) -> str:
        a = getattr(world, "agents", {}).get(agent_id)
        return getattr(a, "name", agent_id) if a else (agent_id or "someone")


__all__ = ["ReflectionManager", "REFLECT_COOLDOWN", "PERIODIC_REFLECT_EVERY"]
