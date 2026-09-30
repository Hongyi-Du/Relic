"""Demand-first task selection for realistic product experience.

The curated DEFAULT_TASK_DEMANDS / DOMAIN_TASK_DEMANDS tables were hand-tuned
for one frontend-tooling product. When the experienced artifact declares its
own task space (``task_fit_distribution``), user demand is derived over that
declared space instead — otherwise every sampled job-to-be-done misses the
artifact's declared tasks and ``_task_fit`` collapses to its mean fallback,
flattening the fit signal across substrates. The curated tables remain the
fallback for the repo-digest substrate and for artifacts that declare no
task space (e.g. the org's own public release artifact).
"""

from __future__ import annotations

from .hashing import stable_hash
from .randomness import SeededRandom
from .schemas import AgentProfile, AgentState, Artifact, clamp01


DEFAULT_TASK_DEMANDS: dict[str, float] = {
    "dev_server_start": 0.18,
    "production_build": 0.16,
    "legacy_browser_support": 0.12,
    "framework_integration": 0.16,
    "plugin_extension": 0.10,
    "dependency_upgrade": 0.12,
    "team_workflow": 0.10,
    "documentation_lookup": 0.06,
}

DOMAIN_TASK_DEMANDS: dict[str, dict[str, float]] = {
    "frontend_product": {
        "dev_server_start": 0.22,
        "production_build": 0.19,
        "legacy_browser_support": 0.14,
        "framework_integration": 0.19,
        "plugin_extension": 0.08,
        "dependency_upgrade": 0.10,
        "team_workflow": 0.06,
        "documentation_lookup": 0.02,
    },
    "library_author": {
        "plugin_extension": 0.22,
        "framework_integration": 0.20,
        "dependency_upgrade": 0.15,
        "production_build": 0.13,
        "legacy_browser_support": 0.12,
        "team_workflow": 0.08,
        "dev_server_start": 0.07,
        "documentation_lookup": 0.03,
    },
    "ops_maintainer": {
        "production_build": 0.24,
        "dependency_upgrade": 0.20,
        "team_workflow": 0.16,
        "legacy_browser_support": 0.12,
        "framework_integration": 0.10,
        "dev_server_start": 0.10,
        "plugin_extension": 0.05,
        "documentation_lookup": 0.03,
    },
    "learner_builder": {
        "documentation_lookup": 0.20,
        "dev_server_start": 0.20,
        "framework_integration": 0.16,
        "dependency_upgrade": 0.12,
        "production_build": 0.12,
        "team_workflow": 0.10,
        "legacy_browser_support": 0.06,
        "plugin_extension": 0.04,
    },
}


def demand_weights_for_profile(profile: AgentProfile) -> dict[str, float]:
    weights = dict(DEFAULT_TASK_DEMANDS)
    weights.update(DOMAIN_TASK_DEMANDS.get(profile.primary_domain, {}))
    for task_id, weight in profile.task_demand_weights.items():
        weights[task_id] = weights.get(task_id, 0.0) + 5.0 * weight
    total = sum(max(0.0, value) for value in weights.values())
    if total <= 0:
        return dict(DEFAULT_TASK_DEMANDS)
    return {task: clamp01(max(0.0, weight) / total) for task, weight in weights.items()}


def project_task_demands(
    *, agent: AgentState, task_space
) -> dict[str, float]:
    """Project the profile's demand onto an arbitrary task space.

    Demand the profile already expresses for tasks in the space is kept
    verbatim; the mass its profile ties up in tasks outside the space is
    redistributed (largest shares first) over the tasks it has no opinion on,
    in a per-agent stable-hash order. This preserves both an individual
    agent's expressed needs and population-level demand heterogeneity,
    without consuming any rng draws. Returns {} for an empty task space.
    """

    task_space = tuple(dict.fromkeys(task_space))
    if not task_space:
        return {}
    profile_weights = {
        task: max(0.0, weight)
        for task, weight in (
            agent.profile.task_demand_weights or DEFAULT_TASK_DEMANDS
        ).items()
    }
    weights = {
        task: profile_weights[task] for task in task_space if task in profile_weights
    }
    unexpressed = sorted(
        (
            weight
            for task, weight in profile_weights.items()
            if task not in task_space
        ),
        reverse=True,
    )
    missing = sorted(
        (task for task in task_space if task not in profile_weights),
        key=lambda task: (stable_hash({"agent": agent.id, "task": task}), task),
    )
    for index, task in enumerate(missing):
        weights[task] = unexpressed[index % len(unexpressed)] if unexpressed else 0.0
    total = sum(weights.values())
    if total <= 0:
        template = sorted(DEFAULT_TASK_DEMANDS.values(), reverse=True)
        order = sorted(
            task_space,
            key=lambda task: (stable_hash({"agent": agent.id, "task": task}), task),
        )
        weights = {
            task: template[index % len(template)] for index, task in enumerate(order)
        }
        total = sum(weights.values())
    return {task: weight / total for task, weight in sorted(weights.items())}


def substrate_task_demands(
    *, agent: AgentState, artifact: Artifact
) -> dict[str, float] | None:
    """Demand weights over the artifact's declared task space.

    Returns None when the artifact is the curated repo-digest substrate or
    declares no task space, meaning the caller must keep the curated tables.
    Otherwise projects the profile's demand onto the declared task space via
    ``project_task_demands``.
    """

    from .product_experience import _looks_like_repo_digest_artifact

    declared = artifact.task_fit_distribution
    if not declared or _looks_like_repo_digest_artifact(artifact):
        return None
    return project_task_demands(agent=agent, task_space=declared)


def sample_user_task(
    *,
    agent: AgentState,
    artifact: Artifact,
    rng: SeededRandom,
) -> str:
    """Sample the user's job-to-be-done before artifact fit is evaluated."""

    weights = substrate_task_demands(agent=agent, artifact=artifact)
    if weights is None:
        weights = demand_weights_for_profile(agent.profile)
    task_ids = sorted(set(weights).union(artifact.task_fit_distribution))
    if not task_ids:
        return "general_task"
    weighted: list[float] = []
    for task_id in task_ids:
        demand = weights.get(task_id, 0.0)
        artifact_visibility = 0.04 * artifact.task_fit_distribution.get(task_id, 0.0)
        urgency = 0.08 * agent.profile.domain_need + 0.04 * agent.profile.deadline_pressure
        weighted.append(max(0.001, demand + artifact_visibility + urgency * demand))
    index = rng.choice_index("event", weighted, kind=f"demand_task:{artifact.id}")
    return task_ids[index]
