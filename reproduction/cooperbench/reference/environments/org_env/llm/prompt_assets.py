"""Initial prompt assets — company brief, role mandates, agent identity, grounding
rules, and product-substrate context. Every LLM call composes its system prompt as
``build_agent_system_prompt(agent, world, module) + module_instruction`` so the model
acts as a SPECIFIC agent in a SPECIFIC messy company, not a generic assistant.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any, Dict, List, Optional

from environments.org_env.product.seed import DEFAULT_COMPANY_CONFIG

COMPANY_BRIEF_TEMPLATE = """You are simulating one agent inside {company_name}, {company_framing}.

Company context:
- Company: {company_name}
- Project / product codename: {product_name}
- Current product stage: {product_stage}.
- Product goal: {product_purpose}.
- {discovery_narrative}
- {work_narrative}

Organizational setting:
- A small team with differentiated roles.
- Recurring tension between speed, quality, evidence, usability, and external credibility.
- {decision_narrative}

Important: You are not a generic assistant. You are one specific agent. Your decisions should reflect your role, persona, memory, current episodes, active product problems, and available actions."""


# Only these single-line strings from ``world.company_config`` may enter the
# shared prompt through this module.  The defaults deliberately reproduce the
# original LanternScout prompt byte for byte.  Task-family profiles can opt in
# to a different narrative without handing an arbitrary config mapping to the
# formatter (or changing the native prompt).
COMPANY_PROMPT_TEXT_LIMITS: Dict[str, int] = {
    "company_name": 160,
    "product_name": 160,
    "product_purpose": 640,
    "company_framing": 320,
    "product_stage": 240,
    "discovery_narrative": 640,
    "work_narrative": 640,
    "decision_narrative": 640,
    "product_context_files_heading": 160,
    "product_context_gaps_heading": 160,
    "product_context_issues_heading": 160,
}

_COMPANY_BRIEF_DEFAULTS: Dict[str, str] = {
    "company_name": DEFAULT_COMPANY_CONFIG["company_name"],
    "product_name": DEFAULT_COMPANY_CONFIG["product_name"],
    "product_purpose": DEFAULT_COMPANY_CONFIG["product_purpose"],
    "company_framing": "an early-stage AI company building a new research agent",
    "product_stage": "messy pre-launch prototype",
    "discovery_narrative": (
        "The company is still discovering the right product shape, workflow, "
        "evaluation standard, and value proposition."
    ),
    "work_narrative": (
        "The team must turn scattered starter code, rough docs, experiments, "
        "reports, and external feedback into a coherent research-agent product."
    ),
    "decision_narrative": (
        "The team must decide what the research agent should do, how to validate "
        "outputs, what tools/workflows it needs, and what protocols are required "
        "before claims or reports can be trusted."
    ),
}

_PRODUCT_CONTEXT_HEADING_DEFAULTS: Dict[str, str] = {
    "product_context_files_heading": "Existing files (with gaps):",
    "product_context_gaps_heading": "Known product gaps:",
    "product_context_issues_heading": "Open issues:",
}

_COMPANY_NARRATIVE_ACTIVATORS = (
    "company_framing",
    "discovery_narrative",
    "work_narrative",
    "decision_narrative",
)

GLOBAL_GROUNDING_RULES = """You must obey these rules:
1. Use only the provided world context.
2. Do not invent objects, people, protocols, tasks, experiments, messages, or results.
3. If you reference an object, use its provided id.
4. If you choose an action, it must be from available_actions.
5. If you choose a target object, it must be from valid_targets.
6. If uncertain, say so in rationale or lower confidence.
7. You may propose / reason / summarize / verbalize, but you cannot directly mutate the world.
8. Do not declare an action succeeded unless the context says it did.
9. Do not adopt protocols, create tools, close episodes, update graphs, or change state directly.
10. Output must match the requested JSON schema exactly."""

ROLE_MANDATES: Dict[str, str] = {
    "founder": "You are responsible for direction, urgency, external legitimacy, and keeping the company moving. "
               "You push for visible progress, public narrative, demos, and alignment; you may underweight "
               "operational detail or process friction under pressure.",
    "cofounder": "You are responsible for architecture, evidence standards, institutional memory, and long-term "
                 "technical quality. You connect work to reusable systems/protocols/evaluation; you may centralize "
                 "decisions or raise the bar too high when weak evidence appears.",
    "reliability": "You are responsible for reproducibility, tracking, verification, and operational correctness. "
                   "You care whether results can be repeated, audited, and trusted; you may sound rigid or blunt "
                   "when process is bypassed.",
    "community": "You interpret external signals, customer pain, community feedback, and public perception, and "
                 "translate outside pressure into internal priorities; you may amplify customer concerns.",
    "editorial": "You own wording, evidence alignment, claim quality, and report clarity. You challenge vague "
                 "claims, suggest rewrites, and prevent unsupported public statements.",
    "artifact_design": "You own templates, workflow artifacts, checklists, trackers, and reusable work structures. "
                       "You turn repeated friction into practical artifacts others can use.",
    "external_voice": "You own public-facing documentation, launch material, response drafts, and user-readable "
                      "explanations; you translate internal work into external communication.",
    "external_docs": "You own public-facing documentation, launch material, and user-readable explanations.",
    "fast_engineer": "You own fast implementation, quick demos, debugging, cheap pilots, and visible product motion. "
                     "You may prioritize speed over traceability unless constrained by protocol or review.",
    "default": "You are a contributing member of the team; act according to your skills and the current needs.",
}

# trait -> (low hint, high hint) for prose rendering
_TRAIT_HINTS = {
    "dominance": ("defers / avoids pushing decisions", "tends to push decisions and visible direction"),
    "conformity": ("may resist established procedures", "follows established procedures"),
    "urgency_bias": ("favors deliberate pacing", "favors fast visible movement"),
    "social_tact": ("may create friction when communicating under pressure", "communicates smoothly"),
    "risk_aversion": ("tolerates risk / moves fast", "is cautious and flags risk early"),
    "quality_bar": ("accepts rough output", "insists on high quality / evidence"),
    "long_termism": ("optimizes for the short term", "optimizes for durable systems"),
    "speed_bias": ("prefers careful work", "prefers speed and quick iteration"),
    "process_resistance": ("comfortable with process", "resists process and gates"),
}


def _validated_prompt_values(
    world: Any,
    defaults: Mapping[str, str],
    *,
    ignored_overrides: frozenset[str] = frozenset(),
) -> Dict[str, str]:
    """Read an allowlisted set of prompt strings from ``company_config``.

    Values are deliberately restricted to short, printable, single-line strings.
    A malformed value fails closed instead of being coerced with ``str(...)`` or
    silently copied into a system prompt.  Keys outside ``defaults`` are ignored,
    so substrate metadata and evaluator plumbing can never be rendered by this
    helper accidentally.
    """

    raw = getattr(world, "company_config", {}) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("invalid_prompt_config:company_config_must_be_mapping")

    values: Dict[str, str] = {}
    for field, default in defaults.items():
        value = raw[field] if field in raw and field not in ignored_overrides else default
        if type(value) is not str:
            raise ValueError(
                f"invalid_prompt_config:{field}:expected_nonempty_string"
            )
        if not value or value != value.strip():
            raise ValueError(
                f"invalid_prompt_config:{field}:expected_nonempty_string"
            )
        if not value.isprintable():
            raise ValueError(f"invalid_prompt_config:{field}:control_character")
        limit = COMPANY_PROMPT_TEXT_LIMITS[field]
        if len(value) > limit:
            raise ValueError(f"invalid_prompt_config:{field}:too_long")
        values[field] = value
    return values


def render_company_brief(world: Any) -> str:
    raw = getattr(world, "company_config", {}) or {}
    # ``product_stage`` predates configurable prompt framing: OSS substrates
    # already populate it, while the old renderer hard-coded the native stage.
    # Treat it as a prompt override only when a new narrative field activates
    # the overlay, preserving profile-off/native OSS bytes.
    stage_is_legacy_metadata = (
        isinstance(raw, Mapping)
        and not any(field in raw for field in _COMPANY_NARRATIVE_ACTIVATORS)
    )
    return COMPANY_BRIEF_TEMPLATE.format(
        **_validated_prompt_values(
            world,
            _COMPANY_BRIEF_DEFAULTS,
            ignored_overrides=(
                frozenset({"product_stage"})
                if stage_is_legacy_metadata
                else frozenset()
            ),
        )
    )


def render_top_traits(profile: Dict[str, float], k: int = 6) -> str:
    if not profile:
        return "- (no traits)"
    items = sorted(profile.items(), key=lambda kv: -abs(float(kv[1]) - 0.5))[:k]
    lines = []
    for name, v in items:
        v = float(v)
        hi = v > 0.6
        lo = v < 0.4
        lvl = "high" if hi else ("low" if lo else "mid")
        hint = _TRAIT_HINTS.get(name)
        tail = (f": {hint[1] if hi else hint[0]}") if hint and lvl != "mid" else ""
        lines.append(f"- {lvl} {name} ({v:.2f}){tail}")
    return "\n".join(lines)


def render_top_skills(skills: Dict[str, float], k: int = 6) -> str:
    if not skills:
        return "- (no skills)"
    items = sorted(skills.items(), key=lambda kv: -float(kv[1]))[:k]
    return "\n".join(f"- {n} ({float(v):.2f})" for n, v in items)


def render_failure_modes(fms: List[str]) -> str:
    return "\n".join(f"- {f}" for f in (fms or [])) or "- (none recorded)"


def render_communication_style(style: Dict[str, Any]) -> str:
    if not style:
        return "- (default)"
    return "\n".join(f"- {k}: {v}" for k, v in style.items())


def render_agent_memory(world: Any, agent_id: str) -> str:
    rm = getattr(world, "reflection_manager", None)
    if rm is None:
        return "- (no memory yet)"
    ctx = rm.context_for_decision(agent_id, world)
    parts = []
    if ctx.get("programbench_workflow_state"):
        parts.append(
            "Current ProgramBench workflow state (public organizational evidence only):\n"
            + json.dumps(
                ctx["programbench_workflow_state"],
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    if ctx.get("programbench_targeted_public_retrievals"):
        parts.append(
            "ProgramBench public contract surfaces this agent explicitly retrieved:\n"
            + json.dumps(
                ctx["programbench_targeted_public_retrievals"],
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    if ctx.get("recent_reflections"):
        parts.append("Recent reflections:\n" + "\n".join(f"  - {r}" for r in ctx["recent_reflections"]))
    if ctx.get("lessons_learned"):
        parts.append("Lessons learned:\n" + "\n".join(f"  - {r}" for r in ctx["lessons_learned"]))
    if ctx.get("unresolved_needs"):
        parts.append("Unresolved needs:\n" + "\n".join(f"  - {r}" for r in ctx["unresolved_needs"]))
    if ctx.get("open_wishes"):
        parts.append("Open wishes: " + ", ".join(ctx["open_wishes"]))
    from environments.org_env.cooperbench.feedback import actor_feedback_context
    feedback = actor_feedback_context(world, agent_id)
    if feedback:
        parts.append("Read communication and current test feedback:\n" + json.dumps(feedback, ensure_ascii=False))
    return "\n".join(parts) or "- (no memory yet)"


def render_product_context(world: Any, max_files: int = 12,
                           max_issues: Optional[int] = None, *,
                           actor_id: Optional[str] = None) -> str:
    """Render public product state plus only this actor's granted Cooper briefs.

    A strict-Cooper caller that omits actor_id receives issue metadata only;
    native OrgEnv retains its team-visible public issue behavior.
    """
    from environments.org_env.cooperbench.visibility import (
        is_strict_coop, visible_feature_brief,
    )
    strict_coop = is_strict_coop(world)
    ps = getattr(world, "product", None)
    arts = getattr(world, "product_artifacts", {}) or {}
    from environments.org_env.cooperbench.source_reads import actor_source_read_view, visible_source_artifacts
    source_view = actor_source_read_view(world, actor_id)
    if ps is None:
        return "(no product substrate)"
    headings = _validated_prompt_values(world, _PRODUCT_CONTEXT_HEADING_DEFAULTS)
    retrieval_context = None
    if (source_view is not None
            and getattr(world, "_cooperbench_source_policy", None) == "feature_independent"
            and max_files > 0):
        from environments.org_env.cooperbench.source_retrieval import (
            assert_actor_source_context_current, retrieve_actor_source_context,
        )
        # The directory needs ranked IDs, not source excerpts. Request at most
        # one byte per file and discard it: large files must not consume the
        # excerpt budget and silently reduce the number of metadata rows.
        file_limit = min(max_files, 384)
        retrieval_context = retrieve_actor_source_context(
            world, actor_id, query="",
            limits={"max_files": file_limit, "max_bytes": file_limit, "max_file_bytes": 1},
        )
        source_view.assert_current(world)
        files = [arts[row["artifact_id"]] for row in retrieval_context["files"]
                 if row["artifact_id"] in arts
                 and row["artifact_id"] in source_view.catalog
                 and arts[row["artifact_id"]].artifact_type != "issue"]
    else:
        files = [a for a in visible_source_artifacts(world, source_view)
                 if a.artifact_type != "issue"][:max(0, max_files)]
    issues = [a for a in arts.values() if a.artifact_type == "issue" and a.status == "open"]
    if max_issues is not None:
        issues = issues[:max_issues]
    lines = [f"Project: {ps.name}", f"Stage: {ps.stage}", "",
             headings["product_context_files_heading"]]
    for a in files:
        if source_view is not None:
            lines.append(f"- {source_view.catalog[a.artifact_id]} [{a.artifact_id}] (visible in your desk)")
        else:
            gap = ("; ".join(a.known_gaps)) if a.known_gaps else "ok"
            lines.append(f"- {a.linked_file_path or a.title} [{a.artifact_id}] ({a.status}): {gap}")
    lines.append("")
    lines.append(headings["product_context_gaps_heading"])
    for g in ([] if source_view is not None else ps.known_systemic_issues[:10]):
        lines.append(f"- {g}")
    lines.append("")
    lines.append(headings["product_context_issues_heading"])
    for a in issues:
        lines.append(f"- {a.artifact_id}: {a.title} ({a.priority})")
        # The reported text — what breaks, how to reproduce, what "fixed" means.
        # A title alone ("barrel list sort") names the bug without describing it,
        # which is how a founder can work an issue for a whole run without ever
        # learning what it asks for. This is the public issue as filed; the
        # held-out issues and hidden tests live in evaluator-only assets.
        if strict_coop:
            problem = (visible_feature_brief(world, actor_id, a.artifact_id) or "") if actor_id else ""
        else:
            problem = str(getattr(a, "problem", "") or "").strip()
        for row in problem.splitlines():
            lines.append(f"    {row}")
    if retrieval_context is not None:
        assert_actor_source_context_current(world, actor_id, retrieval_context)
    return "\n".join(lines)


def render_persona_for_llm(
    agent: Any,
    *,
    profile_conditioning_enabled: bool = True,
) -> str:
    """v4 §11: the LLM sees an ABSTRACT persona (traits / skills / failure modes /
    communication style) — never the policy graph's exact act:* / speech:* tendency
    nodes or feature weights, so it doesn't overfit to role-specific action labels. The
    policy layer still uses the full graph internally."""
    skills = "Skills:\n" + render_top_skills(getattr(agent, "skills", {}) or {})
    if not profile_conditioning_enabled:
        return (
            "Profile representation: flat role and professional skills "
            "(structured persona graph disabled for this condition).\n\n"
            + skills
        )
    return (
        "Persona traits:\n" + render_top_traits(getattr(agent, "profile", {}) or {}) + "\n\n"
        + skills + "\n\n"
        + "Failure modes:\n" + render_failure_modes(getattr(agent, "failure_modes", []) or []) + "\n\n"
        + "Communication style:\n" + render_communication_style(getattr(agent, "communication_style", {}) or {}))


def build_agent_system_prompt(agent: Any, world: Any, module_name: str) -> str:
    """DEPRECATED shape: agent identity inside the system prompt.

    Retained only so existing callers/tests keep working. New code must use
    :func:`system_for` (shared prefix) plus :func:`agent_identity_for`
    (per-agent block, prepended to the USER message). See the Prefix Cache Rule
    note in :func:`system_for`.
    """
    return (
        system_for(None, world, module_name, "").rstrip("\n")
        + "\n\n"
        + agent_identity_for(agent, world)
    )


def agent_identity_for(agent: Any, world: Any) -> str:
    """Per-agent, per-tick block: identity, role mandate, persona, memory.

    This is everything that DIFFERS between agents in the same step, so it must
    travel in the user message. Keeping it here (rather than in the system
    prompt) is what allows the provider to reuse the cached shared prefix.
    """
    if agent is None:
        return ""
    role = getattr(agent, "role", "")
    return (
        "You are this agent:\n"
        + f"Name: {getattr(agent, 'name', agent.id)}\n"
        + f"Codename: {getattr(agent, 'codename', '')}\n"
        + f"Role: {role}\n\n"
        + "Role mandate:\n" + (
            ROLE_MANDATES.get(role, ROLE_MANDATES["default"])
            if getattr(world, "role_mandates_enabled", True)
            # P0 homogeneity: role labels stay (governance gates read them) but
            # the per-role behavioural prior does not, or the "no stable role
            # differences" arm would still hand eight agents eight different
            # mandates.
            else ROLE_MANDATES["default"]
        ) + "\n\n"
        + render_persona_for_llm(
            agent,
            profile_conditioning_enabled=bool(
                getattr(world, "profile_conditioning_enabled", True)
            ),
        ) + "\n\n"
        + "Current memory:\n" + render_agent_memory(world, agent.id)
    )


def system_for(agent: Any, world: Any, module_name: str, module_instruction: str) -> str:
    """Shared system prompt: company brief + global rules + module instruction.

    Prefix Cache Rule (CLAUDE.md, CRITICAL): the system message MUST be
    byte-identical across all agents in a step. Providers reuse a cached KV
    prefix only when the leading tokens match, so a single differing token
    (an agent name, a per-tick memory line) costs a full prefill on every call.

    This used to embed agent name/codename/role/persona AND per-tick memory, so
    no two calls in a run shared a prefix: a real 336-tick run reported
    cached_prompt_tokens=0 across all 301 calls while paying for 521,834 prompt
    tokens. The per-agent block now lives in :func:`agent_identity_for` and is
    prepended to the USER message by each call site.

    The ``agent`` parameter is accepted (and ignored) so call sites can pass it
    without having to know this rule; passing one never changes the result.
    """
    shared = (
        render_company_brief(world) + "\n\n"
        + "Global rules:\n" + GLOBAL_GROUNDING_RULES + "\n\n"
        + f"Current module: {module_name}"
    )
    if module_instruction:
        return shared + "\n\n" + module_instruction
    return shared


__all__ = [
    "COMPANY_BRIEF_TEMPLATE", "COMPANY_PROMPT_TEXT_LIMITS",
    "GLOBAL_GROUNDING_RULES", "ROLE_MANDATES",
    "render_company_brief", "render_product_context", "render_agent_memory",
    "render_top_traits", "render_top_skills", "render_failure_modes", "render_communication_style",
    "render_persona_for_llm",
    "build_agent_system_prompt", "agent_identity_for", "system_for",
]
