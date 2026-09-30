"""Forum/social-network calibration docking metrics."""

from __future__ import annotations

from dataclasses import dataclass

from .hashing import stable_hash
from .schemas import ContentItem, SocietyState


@dataclass(frozen=True)
class CalibrationTarget:
    metric_name: str
    lower: float
    upper: float
    source_label: str
    source_url: str | None = None


@dataclass(frozen=True)
class CalibrationMetricResult:
    metric_name: str
    observed: float
    lower: float
    upper: float
    within_interval: bool
    normalized_distance: float
    source_label: str
    source_url: str | None = None


@dataclass(frozen=True)
class CalibrationDockingReport:
    report_id: str
    tick_count: int
    population_size: int
    content_count: int
    metric_results: dict[str, CalibrationMetricResult]
    passed: bool
    caveats: tuple[str, ...]

    def hash(self) -> str:
        return stable_hash(self)


def _thread_depth(content_id: str, content: dict[str, ContentItem]) -> int:
    depth = 0
    current = content.get(content_id)
    seen: set[str] = set()
    while current and current.parent_id and current.parent_id not in seen:
        seen.add(current.id)
        depth += 1
        current = content.get(current.parent_id)
    return depth


def measure_forum_metrics(state: SocietyState) -> dict[str, float]:
    content_items = list(state.content.values())
    public_items = [item for item in content_items if item.visibility_scope == "public"]
    action_events = [event for event in state.event_log if event.kind.value == "action"]
    public_action_events = [
        event for event in action_events
        if event.public_visibility == "public"
    ]
    active_public_actors = {
        event.actor_id for event in public_action_events if event.actor_id
    }
    all_action_actors = {
        event.actor_id for event in action_events if event.actor_id
    }
    reply_items = [item for item in public_items if item.kind.value == "reply"]
    repost_items = [item for item in public_items if item.kind.value == "repost"]
    depths = [_thread_depth(item.id, state.content) for item in public_items]
    roots = {item.id for item in public_items if item.parent_id is None}
    child_parent_ids = {item.parent_id for item in public_items if item.parent_id}
    cascades = len(roots.intersection(child_parent_ids))
    controversy_items = [
        item for item in public_items
        if item.controversy_proxy > 0 or item.kind.value in {"callout", "failure_report", "rumor"}
    ]
    population = max(1, len(state.agents))
    tick_count = max(1, state.tick)
    public_count = len(public_items)
    return {
        "post_rate_per_agent_tick": public_count / (population * tick_count),
        "reply_share": len(reply_items) / max(1, public_count),
        "repost_share": len(repost_items) / max(1, public_count),
        "silent_user_fraction": 1.0 - (len(active_public_actors) / population),
        "active_user_fraction": len(all_action_actors) / population,
        "mean_thread_depth": sum(depths) / len(depths) if depths else 0.0,
        "cascade_root_fraction": cascades / max(1, len(roots)),
        "controversy_share": len(controversy_items) / max(1, public_count),
    }


def build_calibration_docking_report(
    *,
    state: SocietyState,
    targets: tuple[CalibrationTarget, ...],
    report_id: str = "society_core_calibration_docking_v15",
) -> CalibrationDockingReport:
    measurements = measure_forum_metrics(state)
    results: dict[str, CalibrationMetricResult] = {}
    caveats: list[str] = []
    for target in targets:
        observed = measurements.get(target.metric_name)
        if observed is None:
            caveats.append(f"missing_metric:{target.metric_name}")
            continue
        width = max(target.upper - target.lower, 1e-9)
        if target.lower <= observed <= target.upper:
            distance = 0.0
        elif observed < target.lower:
            distance = (target.lower - observed) / width
        else:
            distance = (observed - target.upper) / width
        results[target.metric_name] = CalibrationMetricResult(
            metric_name=target.metric_name,
            observed=observed,
            lower=target.lower,
            upper=target.upper,
            within_interval=target.lower <= observed <= target.upper,
            normalized_distance=distance,
            source_label=target.source_label,
            source_url=target.source_url,
        )
    if not targets:
        caveats.append("no_calibration_targets_supplied")
    if not state.content:
        caveats.append("no_content_observed")
    passed = bool(results) and all(result.within_interval for result in results.values()) and not caveats
    return CalibrationDockingReport(
        report_id=report_id,
        tick_count=state.tick,
        population_size=len(state.agents),
        content_count=len(state.content),
        metric_results=results,
        passed=passed,
        caveats=tuple(caveats),
    )


def default_forum_calibration_targets() -> tuple[CalibrationTarget, ...]:
    source_url = "https://github.com/HackerNews/API;https://www.reddit.com/dev/api/;https://docs.github.com/rest/issues"
    return (
        CalibrationTarget(
            metric_name="reply_share",
            lower=0.15,
            upper=0.85,
            source_label="protocol_locked_broad_prior:hn_reddit_github_reply_share",
            source_url=source_url,
        ),
        CalibrationTarget(
            metric_name="silent_user_fraction",
            lower=0.2,
            upper=0.95,
            source_label="protocol_locked_broad_prior:hn_reddit_github_participation",
            source_url=source_url,
        ),
        CalibrationTarget(
            metric_name="mean_thread_depth",
            lower=0.05,
            upper=8.0,
            source_label="protocol_locked_broad_prior:hn_reddit_github_thread_depth",
            source_url=source_url,
        ),
    )
