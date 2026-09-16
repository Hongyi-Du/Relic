"""Prompt assets — organization brief, role mandates, identity, and grounding
rules, and product-substrate context. Every LLM call composes its system prompt as
``build_agent_system_prompt(agent, world, module) + module_instruction`` so the model
acts as a specific member of the current organization.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from environments.org_env.product.seed import DEFAULT_COMPANY_CONFIG

COMPANY_BRIEF_TEMPLATE = """You are simulating one member inside {company_name}, a small software organization maintaining a frozen open-source workload.

Company context:
- Company: {company_name}
- Current project: {product_name}
- Current stage: {product_stage}
- Product goal: {product_purpose}
- The team can inspect the starter tree, public issues, public contracts, and declared verification commands supplied by the workload.
- Evaluator-private assets are unavailable to the team and must never be guessed or requested.

Organizational setting:
- A small team with differentiated roles.
- Recurring tension between speed, correctness, maintainability, usability, and external credibility.
- The team must decide how to repair the current OSS project, verify changes through its declared public surface, review them, and ship a defensible release.

Important: You are not a generic assistant. You are one specific agent. Your decisions should reflect your role, persona, memory, current episodes, active product problems, and available actions."""

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


def render_company_brief(world: Any) -> str:
    cfg = dict(DEFAULT_COMPANY_CONFIG)
    cfg.update(getattr(world, "company_config", {}) or {})
    return COMPANY_BRIEF_TEMPLATE.format(**cfg)


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
    if ctx.get("recent_reflections"):
        parts.append("Recent reflections:\n" + "\n".join(f"  - {r}" for r in ctx["recent_reflections"]))
    if ctx.get("lessons_learned"):
        parts.append("Lessons learned:\n" + "\n".join(f"  - {r}" for r in ctx["lessons_learned"]))
    if ctx.get("unresolved_needs"):
        parts.append("Unresolved needs:\n" + "\n".join(f"  - {r}" for r in ctx["unresolved_needs"]))
    if ctx.get("open_wishes"):
        parts.append("Open wishes: " + ", ".join(ctx["open_wishes"]))
    return "\n".join(parts) or "- (no memory yet)"


def render_product_context(world: Any, max_files: int = 12,
                           max_issues: Optional[int] = None) -> str:
    ps = getattr(world, "product", None)
    arts = getattr(world, "product_artifacts", {}) or {}
    if ps is None:
        return "(no product substrate)"
    files = [a for a in arts.values() if a.artifact_type != "issue"][:max_files]
    issues = [a for a in arts.values() if a.artifact_type == "issue" and a.status == "open"]
    if max_issues is not None:
        issues = issues[:max_issues]
    lines = [f"Project: {ps.name}", f"Stage: {ps.stage}", "",
             "Existing files (with gaps):"]
    for a in files:
        gap = ("; ".join(a.known_gaps)) if a.known_gaps else "ok"
        lines.append(f"- {a.linked_file_path or a.title} [{a.artifact_id}] ({a.status}): {gap}")
    lines.append("")
    lines.append("Known product gaps:")
    for g in ps.known_systemic_issues[:10]:
        lines.append(f"- {g}")
    lines.append("")
    lines.append("Open issues:")
    for a in issues:
        lines.append(f"- {a.artifact_id}: {a.title} ({a.priority})")
        # The reported text — what breaks, how to reproduce, what "fixed" means.
        # A title alone ("barrel list sort") names the bug without describing it,
        # which is how a founder can work an issue for a whole run without ever
        # learning what it asks for. This is the public issue as filed; the
        # held-out issues and hidden tests live in evaluator-only assets.
        problem = str(getattr(a, "problem", "") or "").strip()
        for row in problem.splitlines():
            lines.append(f"    {row}")
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
    "COMPANY_BRIEF_TEMPLATE", "GLOBAL_GROUNDING_RULES", "ROLE_MANDATES",
    "render_company_brief", "render_product_context", "render_agent_memory",
    "render_top_traits", "render_top_skills", "render_failure_modes", "render_communication_style",
    "render_persona_for_llm",
    "build_agent_system_prompt", "agent_identity_for", "system_for",
]
