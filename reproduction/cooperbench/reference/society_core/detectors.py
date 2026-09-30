"""Read-only detector scaffolding for Society-Core."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

from .hashing import stable_hash
from .schemas import ExternalEvent, SocietyState


@dataclass(frozen=True)
class DetectorReport:
    name: str
    continuous_score: float
    uncertainty: float
    stage_label: str | None = None
    support_events: tuple[str, ...] = ()
    counterevidence_events: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()


class DetectorSuite:
    """Read-only post-hoc detectors.

    The suite intentionally accepts only event logs or immutable snapshots. Runtime
    code must never pass detector output back into state transition.
    """

    def detect_attention_cluster(self, event_log: list[ExternalEvent]) -> DetectorReport:
        public_events = [
            event for event in event_log
            if event.public_visibility == "public" or event.action_type in {"post", "repost", "reply"}
        ]
        actors = {event.actor_id for event in public_events if event.actor_id}
        score = min(1.0, len(actors) / 10.0)
        return DetectorReport(
            name="attention_cluster",
            continuous_score=score,
            uncertainty=1.0 / max(1, len(event_log)),
            stage_label="attention_cluster" if score >= 0.5 else None,
            support_events=tuple(event.event_id for event in public_events[:20]),
        )

    def detect_artifact_diffusion(
        self,
        event_log: list[ExternalEvent],
        artifact_id: str,
        *,
        exposure_count: int | None = None,
    ) -> DetectorReport:
        trial_events = [
            event for event in event_log
            if event.artifact_id == artifact_id
            and event.action_type in {"try_artifact_on_task", "reuse_artifact", "inspect_artifact"}
        ]
        trial_actors = {
            event.actor_id for event in trial_events
            if event.actor_id and event.action_type in {"try_artifact_on_task", "reuse_artifact"}
        }
        repeat_actors = {
            actor for actor in trial_actors
            if sum(1 for event in trial_events if event.actor_id == actor) > 1
        }
        denominator = max(1, exposure_count if exposure_count is not None else max(10, len(trial_actors)))
        score = min(1.0, (len(trial_actors) + 0.5 * len(repeat_actors)) / denominator)
        stage = None
        if score > 0:
            stage = "trial_interest"
        if score >= 0.25 and len(trial_actors) >= 5:
            stage = "broad_trial_diffusion"
        return DetectorReport(
            name="artifact_diffusion",
            continuous_score=score,
            uncertainty=1.0 / max(1, len(trial_events)),
            stage_label=stage,
            support_events=tuple(event.event_id for event in trial_events[:20]),
            caveats=("post_hoc_read_only",),
        )

    def detect_artifact_payment(
        self,
        event_log: list[ExternalEvent],
        artifact_id: str,
        *,
        exposure_count: int | None = None,
    ) -> DetectorReport:
        payment_events = [
            event for event in event_log
            if event.artifact_id == artifact_id
            and event.action_type == "pay_for_artifact"
        ]
        payment_actors = {event.actor_id for event in payment_events if event.actor_id}
        denominator = max(1, exposure_count if exposure_count is not None else max(10, len(payment_actors)))
        score = min(1.0, len(payment_actors) / denominator)
        return DetectorReport(
            name="artifact_payment",
            continuous_score=score,
            uncertainty=1.0 / max(1, len(payment_events)),
            stage_label="paid_usage" if len(payment_actors) >= 3 else None,
            support_events=tuple(event.event_id for event in payment_events[:20]),
            caveats=("payment_is_endogenous_not_success_label",),
        )

    def detect_trust_shift(self, event_log: list[ExternalEvent]) -> DetectorReport:
        trust_events = [
            event for event in event_log
            if event.action_type in {"reduce_trust", "warn_peer", "avoid_future_interaction"}
        ]
        score = -min(1.0, len(trust_events) / 10.0)
        return DetectorReport(
            name="trust_shift",
            continuous_score=score,
            uncertainty=1.0 / max(1, len(trust_events)),
            stage_label="negative_trust_shift" if trust_events else None,
            support_events=tuple(event.event_id for event in trust_events[:20]),
        )

    def detect_norm_emergence(self, event_log: list[ExternalEvent], pattern: str = "evidence_request") -> DetectorReport:
        regularity_events = [
            event for event in event_log
            if event.action_type in {"ask_question", "challenge_claim", "verify", "compare_sources"}
        ]
        expectation_events = [
            event for event in event_log
            if event.action_type in {"ask_for_explanation", "challenge_claim"}
        ]
        third_party_events = [
            event for event in event_log
            if event.action_type in {"support_callout", "oppose_callout", "mediate_conflict", "warn_peer"}
        ]
        social_cost_events = [
            event for event in event_log
            if event.action_type in {"reduce_trust", "avoid_future_interaction", "exclude_from_attention"}
        ]
        scale = max(8.0, sqrt(max(1, len(event_log))))
        regularity_actors = {event.actor_id for event in regularity_events if event.actor_id}
        components = [
            len(regularity_actors) / (len(regularity_actors) + scale),
            len(expectation_events) / (len(expectation_events) + 0.75 * scale),
            len(third_party_events) / (len(third_party_events) + 0.5 * scale),
            len(social_cost_events) / (len(social_cost_events) + 0.4 * scale),
        ]
        score = 1.0
        for component in components:
            score *= max(0.001, component)
        score = score ** (1 / len(components))
        stage = None
        if components[0] > 0 and components[1] == 0:
            stage = "behavioral_regularity"
        if components[0] > 0 and components[1] > 0:
            stage = "norm_candidate"
        if components[2] > 0 and components[3] > 0:
            stage = "enforced_norm"
        return DetectorReport(
            name=f"norm_{pattern}",
            continuous_score=score,
            uncertainty=1.0 / max(1, len(regularity_events) + len(expectation_events)),
            stage_label=stage,
            support_events=tuple(
                event.event_id
                for event in (regularity_events + expectation_events + third_party_events + social_cost_events)[:20]
            ),
            caveats=("synthetic_construct_tests_required_for_main_claims", "event_volume_scaled_score"),
        )

    def detect_institution_emergence(self, event_log: list[ExternalEvent]) -> DetectorReport:
        norm = self.detect_norm_emergence(event_log)
        role_events = [
            event for event in event_log
            if event.action_type in {"write_tutorial", "mediate_conflict", "help_peer"}
        ]
        referral_events = [
            event for event in event_log
            if event.action_type in {"repost", "reply"} and event.content_id
        ]
        scale = max(8.0, sqrt(max(1, len(event_log))))
        role_actor_count = len({event.actor_id for event in role_events if event.actor_id})
        procedure_score = len(referral_events) / (len(referral_events) + 0.6 * scale)
        role_score = role_actor_count / (role_actor_count + 0.45 * scale)
        score = (max(0.001, norm.continuous_score) * max(0.001, role_score) * max(0.001, procedure_score)) ** (1 / 3)
        stage = None
        if role_score > 0:
            stage = "proto_role_structure"
        if role_score > 0 and procedure_score > 0:
            stage = "proto_procedure"
        if norm.stage_label == "enforced_norm" and role_score > 0 and procedure_score > 0:
            stage = "institution_like_structure"
        return DetectorReport(
            name="institution_emergence",
            continuous_score=score,
            uncertainty=max(norm.uncertainty, 1.0 / max(1, len(role_events) + len(referral_events))),
            stage_label=stage,
            support_events=tuple(event.event_id for event in (role_events + referral_events)[:20]),
            caveats=("collapse_resistance_not_tested", "event_volume_scaled_score") if stage else ("insufficient_gates",),
        )

    def detect_belief_shift(self, event_log: list[ExternalEvent], claim_id: str | None = None) -> DetectorReport:
        belief_events = [
            event for event in event_log
            if "belief_delta" in event.typed_payload
            and (claim_id is None or event.typed_payload.get("claim_id") == claim_id)
        ]
        total = sum(abs(float(event.typed_payload.get("belief_delta", 0.0))) for event in belief_events)
        score = min(1.0, total / 5.0)
        return DetectorReport(
            name="belief_shift",
            continuous_score=score,
            uncertainty=1.0 / max(1, len(belief_events)),
            stage_label="belief_shift" if score > 0 else None,
            support_events=tuple(event.event_id for event in belief_events[:20]),
            caveats=("influence_claim_requires_homophily_controls",) if score > 0 else (),
        )

    def detect_organizational_capability_signal(
        self,
        state: SocietyState,
        *,
        minimum_theme_events: int = 2,
    ) -> dict[str, DetectorReport]:
        """Read out public capability-formation signals without changing state.

        This is a matrix-level proxy for whether the outside society generated
        reusable, organization-facing capability pressure. Historical workspace
        ledgers remain the stronger evidence for company-side L2-L4 claims.
        """

        public_events = [
            event for event in state.event_log
            if event.public_visibility == "public" or event.company_visible_flag
        ]
        events_by_theme: dict[str, list[ExternalEvent]] = {}
        for event in public_events:
            for theme in _capability_event_themes(event, state):
                events_by_theme.setdefault(theme, []).append(event)
        theme_levels: dict[str, int] = {}
        support_by_level: dict[int, list[str]] = {level: [] for level in range(1, 5)}
        caveats = [
            "external_society_proxy",
            "workspace_ledger_required_for_company_side_capability_claim",
        ]
        for theme, events in sorted(events_by_theme.items()):
            if len(events) < minimum_theme_events:
                continue
            level = _capability_level_from_public_events(events)
            if level <= 0:
                continue
            theme_levels[theme] = level
            for threshold in range(1, min(level, 4) + 1):
                support_by_level[threshold].extend(event.event_id for event in events[:8])
        denominator = max(1, len(events_by_theme))
        reports: dict[str, DetectorReport] = {}
        for level, name in (
            (1, "organizational_capability_l1_plus"),
            (2, "organizational_capability_l2_plus"),
            (3, "organizational_capability_l3_plus"),
            (4, "organizational_capability_transfer_l4"),
        ):
            count = sum(1 for value in theme_levels.values() if value >= level)
            score = min(1.0, count / denominator)
            reports[name] = DetectorReport(
                name=name,
                continuous_score=score,
                uncertainty=1.0 / max(1, len(support_by_level[level])),
                stage_label=f"L{level}_plus_public_capability_signal" if count else None,
                support_events=tuple(dict.fromkeys(support_by_level[level]))[:20],
                caveats=tuple(caveats),
            )
        return reports

    def run_all(self, state: SocietyState) -> dict[str, DetectorReport]:
        reports = {
            "attention_cluster": self.detect_attention_cluster(list(state.event_log)),
            "trust_shift": self.detect_trust_shift(list(state.event_log)),
            "norm_emergence": self.detect_norm_emergence(list(state.event_log)),
            "institution_emergence": self.detect_institution_emergence(list(state.event_log)),
            "belief_shift": self.detect_belief_shift(list(state.event_log)),
        }
        reports.update(self.detect_organizational_capability_signal(state))
        for artifact_id in state.artifacts:
            exposure_count = sum(
                1
                for agent in state.agents.values()
                if artifact_id in agent.artifact_access
            )
            reports[f"artifact_diffusion:{artifact_id}"] = self.detect_artifact_diffusion(
                list(state.event_log), artifact_id, exposure_count=exposure_count
            )
            reports[f"artifact_payment:{artifact_id}"] = self.detect_artifact_payment(
                list(state.event_log), artifact_id, exposure_count=exposure_count
            )
        return reports


def detector_output_hash(reports: dict[str, DetectorReport]) -> str:
    return stable_hash(reports)


def _capability_event_themes(event: ExternalEvent, state: SocietyState) -> tuple[str, ...]:
    themes: list[str] = []
    payload = event.typed_payload if isinstance(event.typed_payload, dict) else {}
    suggestion = payload.get("upgrade_suggestion")
    if isinstance(suggestion, dict):
        theme = suggestion.get("theme")
        if isinstance(theme, str) and theme:
            themes.append(_slug(theme))
    action_payload = payload.get("action_payload")
    if isinstance(action_payload, dict):
        product_experience = action_payload.get("product_experience")
        if isinstance(product_experience, dict):
            for key in ("blocked_stage", "failure_event", "task_type"):
                value = product_experience.get(key)
                if isinstance(value, str) and value:
                    themes.append(_slug(value))
    if event.content_id and event.content_id in state.content:
        content = state.content[event.content_id]
        themes.extend(_slug(key) for key, value in content.topic_vector.items() if value >= 0.5)
    if event.artifact_id and event.action_type in {
        "share_failure",
        "share_success",
        "write_tutorial",
        "suggest_upgrade",
        "challenge_claim",
        "verify",
        "compare_sources",
        "warn_peer",
        "reduce_trust",
        "mediate_conflict",
    }:
        themes.append(_slug(event.artifact_id))
    return tuple(theme for theme in dict.fromkeys(themes) if theme)


def _capability_level_from_public_events(events: list[ExternalEvent]) -> int:
    action_types = {event.action_type for event in events}
    actor_ids = {event.actor_id for event in events if event.actor_id}
    ticks = {event.tick for event in events}
    artifact_ids = {event.artifact_id for event in events if event.artifact_id}
    has_proposal = "suggest_upgrade" in action_types or "ask_question" in action_types
    has_adoption = bool(
        action_types.intersection(
            {
                "try_artifact_on_task",
                "reuse_artifact",
                "pay_for_artifact",
                "reply",
                "repost",
                "write_tutorial",
                "share_success",
            }
        )
    )
    has_repeated_use = len(actor_ids) >= 2 and (
        len(events) >= 3
        or bool(action_types.intersection({"reuse_artifact", "write_tutorial", "share_success"}))
    )
    has_persistence = len(ticks) >= 2 or len(events) >= 4
    has_enforcement_or_repair = bool(
        action_types.intersection(
            {
                "share_failure",
                "warn_peer",
                "challenge_claim",
                "verify",
                "compare_sources",
                "reduce_trust",
                "mediate_conflict",
            }
        )
    )
    has_transfer = len(artifact_ids) >= 2 and has_persistence
    if has_proposal and has_adoption and has_repeated_use and has_persistence and has_transfer:
        return 4
    if has_proposal and has_adoption and has_repeated_use and has_persistence and has_enforcement_or_repair:
        return 3
    if has_proposal and has_adoption and has_repeated_use and has_persistence:
        return 2
    if has_proposal and has_adoption:
        return 1
    return 0


def _slug(value: str) -> str:
    return "_".join("".join(ch.lower() if ch.isalnum() else " " for ch in str(value)).split())
