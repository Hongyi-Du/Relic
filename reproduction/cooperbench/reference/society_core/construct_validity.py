"""Synthetic construct-validity fixtures for read-only detectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .detectors import DetectorReport, DetectorSuite
from .hashing import stable_hash
from .schemas import EventKind, ExternalEvent


@dataclass(frozen=True)
class ConstructValidityCheck:
    fixture_id: str
    detector_name: str
    expectation: str
    observed_score: float
    observed_stage: str | None
    passed: bool
    required: bool = True
    caveats: tuple[str, ...] = ()
    support_events: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConstructValidityReport:
    report_id: str
    suite_version: str
    fixture_count: int
    required_passed: bool
    checks: tuple[ConstructValidityCheck, ...]

    def hash(self) -> str:
        return stable_hash(self)


def _event(event_id: str, action_type: str, actor: str = "agent_a", **kwargs) -> ExternalEvent:
    return ExternalEvent(
        event_id=event_id,
        tick=int(kwargs.pop("tick", 0)),
        kind=EventKind.ACTION,
        actor_id=actor,
        action_type=action_type,
        public_visibility=kwargs.pop("public_visibility", "public"),
        **kwargs,
    )


def _check(
    *,
    fixture_id: str,
    detector_name: str,
    expectation: str,
    report: DetectorReport,
    predicate: Callable[[DetectorReport], bool],
    required: bool = True,
) -> ConstructValidityCheck:
    return ConstructValidityCheck(
        fixture_id=fixture_id,
        detector_name=detector_name,
        expectation=expectation,
        observed_score=report.continuous_score,
        observed_stage=report.stage_label,
        passed=predicate(report),
        required=required,
        caveats=report.caveats,
        support_events=report.support_events,
    )


def run_construct_validity_suite() -> ConstructValidityReport:
    detector = DetectorSuite()
    checks: list[ConstructValidityCheck] = []

    artifact_mentions_only = [
        _event("artifact_mention_1", "post", "provider", artifact_id="tool_x"),
        _event("artifact_mention_2", "post", "provider", artifact_id="tool_x"),
    ]
    checks.append(
        _check(
            fixture_id="artifact_mentions_only_negative",
            detector_name="artifact_diffusion",
            expectation="mentions by provider alone do not count as trial diffusion",
            report=detector.detect_artifact_diffusion(artifact_mentions_only, "tool_x", exposure_count=50),
            predicate=lambda report: report.continuous_score == 0 and report.stage_label is None,
        )
    )

    artifact_trials = [
        _event("artifact_trial_1", "try_artifact_on_task", "agent_a", artifact_id="tool_x"),
        _event("artifact_trial_2", "reuse_artifact", "agent_a", artifact_id="tool_x"),
        _event("artifact_trial_3", "try_artifact_on_task", "agent_b", artifact_id="tool_x"),
    ]
    checks.append(
        _check(
            fixture_id="artifact_trials_positive",
            detector_name="artifact_diffusion",
            expectation="independent trials create positive diffusion evidence",
            report=detector.detect_artifact_diffusion(artifact_trials, "tool_x", exposure_count=20),
            predicate=lambda report: report.continuous_score > 0 and report.stage_label == "trial_interest",
        )
    )

    broad_trials = [
        _event(f"broad_trial_{index}", "try_artifact_on_task", f"agent_{index}", artifact_id="tool_x")
        for index in range(12)
    ]
    checks.append(
        _check(
            fixture_id="artifact_broad_trials_positive",
            detector_name="artifact_diffusion",
            expectation="broad exposed-agent trial share reaches broad-trial stage",
            report=detector.detect_artifact_diffusion(broad_trials, "tool_x", exposure_count=24),
            predicate=lambda report: report.stage_label == "broad_trial_diffusion",
        )
    )

    norm_regular = [
        _event("norm_regular_1", "verify", "agent_a"),
        _event("norm_regular_2", "compare_sources", "agent_b"),
        _event("norm_regular_3", "verify", "agent_c"),
    ]
    checks.append(
        _check(
            fixture_id="norm_regular_without_expectation",
            detector_name="norm_emergence",
            expectation="repeated behavior without expectation is not an enforced norm",
            report=detector.detect_norm_emergence(norm_regular),
            predicate=lambda report: report.stage_label == "behavioral_regularity",
        )
    )

    norm_enforced = [
        _event("norm_enforced_1", "verify", "agent_a"),
        _event("norm_enforced_2", "challenge_claim", "agent_b"),
        _event("norm_enforced_3", "support_callout", "agent_c"),
        _event("norm_enforced_4", "reduce_trust", "agent_d"),
    ]
    checks.append(
        _check(
            fixture_id="norm_enforced_positive",
            detector_name="norm_emergence",
            expectation="expectation, third-party response, and social cost produce enforced norm candidate",
            report=detector.detect_norm_emergence(norm_enforced),
            predicate=lambda report: report.stage_label == "enforced_norm" and report.continuous_score > 0,
        )
    )

    institution_positive = [
        _event("inst_1", "verify", "agent_a"),
        _event("inst_2", "challenge_claim", "agent_b"),
        _event("inst_3", "support_callout", "agent_c"),
        _event("inst_4", "reduce_trust", "agent_d"),
        _event("inst_5", "write_tutorial", "helper"),
        _event("inst_6", "mediate_conflict", "helper"),
        _event("inst_7", "help_peer", "helper"),
        _event("inst_8", "reply", "newcomer", content_id="guide"),
        _event("inst_9", "repost", "newcomer_2", content_id="guide"),
        _event("inst_10", "reply", "newcomer_3", content_id="guide"),
    ]
    checks.append(
        _check(
            fixture_id="institution_like_positive",
            detector_name="institution_emergence",
            expectation="norm, role, and procedure evidence produce institution-like stage",
            report=detector.detect_institution_emergence(institution_positive),
            predicate=lambda report: report.stage_label == "institution_like_structure",
        )
    )

    belief_positive = [
        _event("belief_1", "read", "agent_a", typed_payload={"claim_id": "claim_x", "belief_delta": 0.4}),
        _event("belief_2", "read", "agent_b", typed_payload={"claim_id": "claim_x", "belief_delta": -0.2}),
    ]
    checks.append(
        _check(
            fixture_id="belief_shift_positive_with_caveat",
            detector_name="belief_shift",
            expectation="belief deltas produce shift but retain homophily caveat",
            report=detector.detect_belief_shift(belief_positive, claim_id="claim_x"),
            predicate=lambda report: (
                report.stage_label == "belief_shift"
                and "influence_claim_requires_homophily_controls" in report.caveats
            ),
        )
    )

    trust_negative = [
        _event("trust_1", "reduce_trust", "agent_a"),
        _event("trust_2", "warn_peer", "agent_b"),
    ]
    checks.append(
        _check(
            fixture_id="trust_shift_negative_direction",
            detector_name="trust_shift",
            expectation="negative social reactions create negative trust-shift score",
            report=detector.detect_trust_shift(trust_negative),
            predicate=lambda report: report.continuous_score < 0 and report.stage_label == "negative_trust_shift",
        )
    )

    required_passed = all(check.passed for check in checks if check.required)
    return ConstructValidityReport(
        report_id="society_core_detector_construct_validity_v12",
        suite_version="v12.1",
        fixture_count=len(checks),
        required_passed=required_passed,
        checks=tuple(checks),
    )


def construct_validity_suite_hash() -> str:
    return run_construct_validity_suite().hash()
