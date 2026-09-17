"""Checks for the neutral source capability-registry compatibility module."""

from __future__ import annotations

from environments.org_env.experiments.capability_carriers import (
    CARRIER_KIND_PROTOCOL,
    canonical_capability,
)
from relic.governance.capabilities import (
    CAPABILITY_KIND_ORGANIZATIONAL,
    CAPABILITY_KIND_TECHNICAL_FIX,
    ORGANIZATIONAL_CAPABILITY_LABELS,
    capability_label,
    classify_capability_kind,
)


def test_source_registry_compatibility_preserves_punctuation_slugging() -> None:
    """The HCI source's three registry imports share this neutral closure."""

    assert classify_capability_kind("Review/Gate") == CAPABILITY_KIND_ORGANIZATIONAL
    assert capability_label("customer:triage") == "customer triage workflow"
    assert classify_capability_kind("new-repository-bug") == CAPABILITY_KIND_TECHNICAL_FIX
    assert canonical_capability("review_gate", CARRIER_KIND_PROTOCOL) == "review_gate"
    assert canonical_capability("not_a_registered_capability", CARRIER_KIND_PROTOCOL) == ""
    assert set(ORGANIZATIONAL_CAPABILITY_LABELS) == {
        "budget_rule",
        "claim_evidence_binding",
        "customer_triage",
        "evidence_workflow",
        "experiment_tracker",
        "external_signal_loop",
        "ownership_map",
        "privacy_boundary",
        "release_governance",
        "review_gate",
        "shared_memory",
        "source_credibility_workflow",
        "workflow_integration",
    }
