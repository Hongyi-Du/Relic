"""Pre-institutional emergence from event evidence chains."""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import EmergentInstitution, ExternalEvent, SocietyState


REGULARITY_ACTIONS = {"ask_question", "challenge_claim", "verify", "compare_sources"}
EXPECTATION_ACTIONS = {"ask_for_explanation", "challenge_claim"}
THIRD_PARTY_ACTIONS = {"support_callout", "oppose_callout", "mediate_conflict", "warn_peer"}
SOCIAL_COST_ACTIONS = {"reduce_trust", "avoid_future_interaction", "exclude_from_attention"}
ROLE_ACTIONS = {"write_tutorial", "mediate_conflict", "help_peer"}
PROCEDURE_ACTIONS = {"repost", "reply"}


@dataclass(frozen=True)
class InstitutionWindowEvidence:
    window_index: int
    start_tick: int
    end_tick: int
    regularity_actor_count: int
    expectation_event_count: int
    third_party_event_count: int
    social_cost_event_count: int
    role_actor_ids: tuple[str, ...]
    procedure_content_refs: tuple[str, ...]
    support_event_refs: tuple[str, ...]
    chain_complete: bool
    component_completeness: float


@dataclass(frozen=True)
class InstitutionLongRunReport:
    report_id: str
    window_size: int
    min_stable_windows: int
    total_window_count: int
    stable_window_count: int
    chain_completeness: float
    role_persistence_score: float
    procedure_persistence_score: float
    support_event_count: int
    first_stable_tick: int | None
    claim_ready: bool
    claim_status: str
    support_events: tuple[str, ...]
    windows: tuple[InstitutionWindowEvidence, ...]
    caveats: tuple[str, ...] = ()


def _events_by_action(events: list[ExternalEvent], action_types: set[str]) -> list[ExternalEvent]:
    return [event for event in events if event.action_type in action_types]


def infer_institution_candidate(
    *,
    event_log: list[ExternalEvent],
    tick: int,
    pattern: str = "evidence_request",
) -> EmergentInstitution | None:
    regularity = _events_by_action(event_log, REGULARITY_ACTIONS)
    expectation = _events_by_action(event_log, EXPECTATION_ACTIONS)
    third_party = _events_by_action(event_log, THIRD_PARTY_ACTIONS)
    social_cost = _events_by_action(event_log, SOCIAL_COST_ACTIONS)
    role = _events_by_action(event_log, ROLE_ACTIONS)
    procedure = [
        event for event in _events_by_action(event_log, PROCEDURE_ACTIONS)
        if event.content_id or event.observation_refs
    ]
    regularity_actors = {event.actor_id for event in regularity if event.actor_id}
    role_actors = tuple(sorted({event.actor_id for event in role if event.actor_id}))
    if (
        len(regularity_actors) < 3
        or len(expectation) < 2
        or len(third_party) < 2
        or len(social_cost) < 1
        or len(role_actors) < 1
        or len(procedure) < 2
    ):
        return None
    support_events = tuple(
        event.event_id
        for event in (regularity + expectation + third_party + social_cost + role + procedure)[:30]
    )
    procedure_refs = tuple(
        event.content_id
        for event in procedure
        if event.content_id
    )
    stability_components = (
        min(1.0, len(regularity_actors) / 8.0),
        min(1.0, len(expectation) / 5.0),
        min(1.0, len(third_party) / 4.0),
        min(1.0, len(social_cost) / 3.0),
        min(1.0, len(role_actors) / 3.0),
        min(1.0, len(procedure) / 4.0),
    )
    stability = 1.0
    for component in stability_components:
        stability *= max(0.001, component)
    stability = stability ** (1 / len(stability_components))
    behavioral_feedback = {
        "verify_expectation": min(1.0, 0.18 + 0.42 * stability),
        "explanation_expectation": min(1.0, 0.12 + 0.35 * stability),
        "help_role_salience": min(1.0, 0.10 + 0.28 * stability),
        "callout_social_cost": min(1.0, 0.08 + 0.24 * stability),
        "trust_sanction_salience": min(1.0, 0.08 + 0.30 * stability),
    }
    return EmergentInstitution(
        id=f"institution_{pattern}",
        pattern=pattern,
        stage="institution_like_structure",
        created_at=tick,
        support_event_refs=support_events,
        role_actor_ids=role_actors,
        procedure_content_refs=procedure_refs,
        stability_score=stability,
        behavioral_feedback=behavioral_feedback,
    )


def update_emergent_institutions(state: SocietyState) -> tuple[str, ...]:
    candidate = infer_institution_candidate(
        event_log=list(state.event_log),
        tick=state.tick,
    )
    if candidate is None:
        return ()
    previous = state.emergent_institutions.get(candidate.id)
    if previous and previous.support_event_refs == candidate.support_event_refs:
        return ()
    state.emergent_institutions[candidate.id] = candidate
    return (candidate.id,)


def build_institution_long_run_report(
    *,
    event_log: list[ExternalEvent] | tuple[ExternalEvent, ...],
    window_size: int = 50,
    min_stable_windows: int = 3,
) -> InstitutionLongRunReport:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if min_stable_windows <= 0:
        raise ValueError("min_stable_windows must be positive")
    windows = tuple(
        _institution_window_evidence(
            window_index=window_index,
            events=events,
            window_size=window_size,
        )
        for window_index, events in _events_by_window(tuple(event_log), window_size).items()
    )
    stable_windows = tuple(window for window in windows if window.chain_complete)
    chain_completeness = (
        sum(window.component_completeness for window in windows) / len(windows)
        if windows
        else 0.0
    )
    role_persistence_score = _actor_persistence_score(
        tuple(window.role_actor_ids for window in stable_windows)
    )
    procedure_persistence_score = min(
        1.0,
        sum(1 for window in stable_windows if window.procedure_content_refs)
        / max(1, min_stable_windows),
    )
    claim_ready = (
        len(stable_windows) >= min_stable_windows
        and chain_completeness >= 0.80
        and role_persistence_score > 0
        and procedure_persistence_score >= 1.0
    )
    support_events = tuple(
        dict.fromkeys(
            event_id
            for window in stable_windows
            for event_id in window.support_event_refs
        )
    )
    caveats: list[str] = []
    if len(stable_windows) < min_stable_windows:
        caveats.append("fewer_stable_windows_than_required")
    if role_persistence_score <= 0:
        caveats.append("role_persistence_not_established")
    if procedure_persistence_score < 1.0:
        caveats.append("procedure_persistence_not_established")
    return InstitutionLongRunReport(
        report_id="institution_long_run_" + _long_run_hash_components(
            windows=windows,
            window_size=window_size,
            min_stable_windows=min_stable_windows,
        ),
        window_size=window_size,
        min_stable_windows=min_stable_windows,
        total_window_count=len(windows),
        stable_window_count=len(stable_windows),
        chain_completeness=round(chain_completeness, 6),
        role_persistence_score=round(role_persistence_score, 6),
        procedure_persistence_score=round(procedure_persistence_score, 6),
        support_event_count=len(support_events),
        first_stable_tick=stable_windows[0].start_tick if stable_windows else None,
        claim_ready=claim_ready,
        claim_status="supported" if claim_ready else "insufficient_evidence",
        support_events=support_events,
        windows=windows,
        caveats=tuple(caveats),
    )


def _events_by_window(
    event_log: tuple[ExternalEvent, ...],
    window_size: int,
) -> dict[int, tuple[ExternalEvent, ...]]:
    windows: dict[int, list[ExternalEvent]] = {}
    for event in event_log:
        windows.setdefault(max(0, event.tick // window_size), []).append(event)
    return {
        window_index: tuple(events)
        for window_index, events in sorted(windows.items())
    }


def _institution_window_evidence(
    *,
    window_index: int,
    events: tuple[ExternalEvent, ...],
    window_size: int,
) -> InstitutionWindowEvidence:
    regularity = tuple(event for event in events if event.action_type in REGULARITY_ACTIONS)
    expectation = tuple(event for event in events if event.action_type in EXPECTATION_ACTIONS)
    third_party = tuple(event for event in events if event.action_type in THIRD_PARTY_ACTIONS)
    social_cost = tuple(event for event in events if event.action_type in SOCIAL_COST_ACTIONS)
    role = tuple(event for event in events if event.action_type in ROLE_ACTIONS)
    procedure = tuple(
        event
        for event in events
        if event.action_type in PROCEDURE_ACTIONS
        and (event.content_id or event.observation_refs)
    )
    regularity_actors = {event.actor_id for event in regularity if event.actor_id}
    role_actor_ids = tuple(sorted({event.actor_id for event in role if event.actor_id}))
    procedure_refs = tuple(
        event.content_id
        for event in procedure
        if event.content_id
    )
    component_flags = (
        len(regularity_actors) >= 3,
        len(expectation) >= 2,
        len(third_party) >= 2,
        len(social_cost) >= 1,
        len(role_actor_ids) >= 1,
        len(procedure) >= 2,
    )
    support_events = tuple(
        event.event_id
        for event in (
            regularity
            + expectation
            + third_party
            + social_cost
            + role
            + procedure
        )[:40]
    )
    return InstitutionWindowEvidence(
        window_index=window_index,
        start_tick=window_index * window_size,
        end_tick=(window_index + 1) * window_size - 1,
        regularity_actor_count=len(regularity_actors),
        expectation_event_count=len(expectation),
        third_party_event_count=len(third_party),
        social_cost_event_count=len(social_cost),
        role_actor_ids=role_actor_ids,
        procedure_content_refs=procedure_refs,
        support_event_refs=support_events,
        chain_complete=all(component_flags),
        component_completeness=sum(float(flag) for flag in component_flags) / len(component_flags),
    )


def _actor_persistence_score(role_actor_windows: tuple[tuple[str, ...], ...]) -> float:
    if not role_actor_windows:
        return 0.0
    actor_window_counts: dict[str, int] = {}
    for actors in role_actor_windows:
        for actor_id in set(actors):
            actor_window_counts[actor_id] = actor_window_counts.get(actor_id, 0) + 1
    persistent = sum(1 for count in actor_window_counts.values() if count >= 2)
    return min(1.0, persistent / max(1, len(actor_window_counts)))


def _long_run_hash_components(
    *,
    windows: tuple[InstitutionWindowEvidence, ...],
    window_size: int,
    min_stable_windows: int,
) -> str:
    from .hashing import stable_hash

    return stable_hash(
        {
            "window_size": window_size,
            "min_stable_windows": min_stable_windows,
            "windows": windows,
        }
    )[:24]
