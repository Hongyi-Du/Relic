"""Bridge sanitized Society-Core public traces into OrgEnv public records."""

from __future__ import annotations

from typing import Any

from environments.org_env.backend.community.objects import ExternalProfile, Post
from society_core.hashing import stable_hash
from society_core.org_adapter import CompanyVisibleTrace


def _bounded_probability(value: float | None, *, default: float = 0.5) -> float:
    if value is None:
        return default
    return max(0.0, min(1.0, float(value)))


def _mirror_trace_to_community(world: Any, trace: CompanyVisibleTrace) -> None:
    """Expose one sanitized trace through the normal read/search feed."""
    community = getattr(world, "community", None)
    if community is None:
        return
    author_id = trace.author_id or "society_public_participant"
    if author_id not in community.profiles:
        community.profiles[author_id] = ExternalProfile(
            external_agent_id=author_id,
            name=author_id,
            role=trace.upgrade_contribution_mode or "external_user",
            credibility=_bounded_probability(trace.upgrade_confidence),
            reputation=_bounded_probability(trace.upgrade_technical_depth),
            topic_interests=[
                topic
                for topic in (trace.upgrade_theme, trace.artifact_id)
                if topic
            ],
            activity_level=0.5,
        )
    post_id = f"society_post_{trace.event_id}"
    if post_id in community.posts:
        return
    topic = (
        trace.upgrade_theme
        or trace.artifact_id
        or trace.action_type
        or "external_society"
    )
    summary = trace.public_summary
    if trace.requested_change and trace.requested_change not in summary:
        summary = f"{summary} Requested change: {trace.requested_change}".strip()
    community.posts[post_id] = Post(
        post_id=post_id,
        author_id=author_id,
        topic=topic,
        content_summary=summary[:1000],
        credibility=_bounded_probability(trace.upgrade_confidence),
        linked_external_docs=list(trace.objective_observation_refs),
        created_tick=int(trace.tick),
        visibility="public",
    )


def ingest_society_public_traces(
    world: Any,
    traces: tuple[CompanyVisibleTrace, ...],
    *,
    source_run_id: str = "society_core",
) -> tuple[dict, ...]:
    """Append deduplicated public traces to ``world.public_records``.

    This is intentionally one-way and sanitized. It never imports private society
    body state, beliefs, local observations, DMs, or detector outputs into OrgEnv.
    """
    if not hasattr(world, "public_records"):
        world.public_records = []
    existing = {
        record.get("source_event_id")
        for record in world.public_records
        if record.get("type") == "external_society_public_trace"
    }
    ingested: list[dict] = []
    for trace in traces:
        _mirror_trace_to_community(world, trace)
        if trace.event_id in existing:
            continue
        record = {
            "record_id": f"society_trace_{trace.event_id}",
            "type": "external_society_public_trace",
            "source": source_run_id,
            "source_event_id": trace.event_id,
            "tick": trace.tick,
            "action_type": trace.action_type,
            "artifact_id": trace.artifact_id,
            "content_id": trace.content_id,
            "channel_id": trace.channel_id,
            "summary": trace.public_summary,
            "support_refs": list(trace.support_refs),
            "author_id": trace.author_id,
            "upgrade_suggestion_id": trace.upgrade_suggestion_id,
            "upgrade_theme": trace.upgrade_theme,
            "upgrade_priority": trace.upgrade_priority,
            "upgrade_confidence": trace.upgrade_confidence,
            "upgrade_contribution_mode": trace.upgrade_contribution_mode,
            "upgrade_technical_depth": trace.upgrade_technical_depth,
            "upgrade_novelty_score": trace.upgrade_novelty_score,
            "source_experience_ref": trace.source_experience_ref,
            "objective_observation_refs": list(
                trace.objective_observation_refs
            ),
            "objective_observation_set_hash": (
                trace.objective_observation_set_hash
            ),
            "objective_observation_manifest_hash": (
                trace.objective_observation_manifest_hash
            ),
            "generator_source": trace.generator_source,
            "requested_change": trace.requested_change,
            "semantic_cluster_id": trace.semantic_cluster_id,
            "payload_hash": stable_hash(trace),
            "visibility": "company_public_record",
        }
        world.public_records.append(record)
        ingested.append(record)
        existing.add(trace.event_id)
    if ingested:
        if not hasattr(world, "events"):
            world.events = []
        world.events.append(
            {
                "type": "external_society_traces_ingested",
                "source": source_run_id,
                "count": len(ingested),
                "source_event_ids": [record["source_event_id"] for record in ingested],
            }
        )
    return tuple(ingested)
