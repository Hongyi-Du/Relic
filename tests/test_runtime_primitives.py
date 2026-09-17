from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from relic.core.domain import DomainAction, DomainState, VitalState
from relic.decision.contracts import ActionCandidate, CandidateSource
from relic.evaluation.time_machine import oracle_blocking_reasons
from relic.research.hashing import stable_hash
from relic.research.openai_runtime import (
    DEFAULT_HEADERS_ENV,
    DISABLE_RESPONSE_STORAGE_ENV,
    configured_openai_default_headers,
    openai_response_storage_disabled,
    prepare_openai_response_request,
)


def test_environment_contracts_have_safe_defaults() -> None:
    state = DomainState()
    action = DomainAction(action_type="inspect_task")

    assert state.run_id == "run"
    assert state.agents == {}
    assert action.parameters == {}
    assert VitalState().guard_violated() is False


def test_action_candidate_serializes_only_executable_fields() -> None:
    candidate = ActionCandidate(
        action_type="inspect_task",
        parameters={"task_id": "task-1"},
        source=CandidateSource.INSTITUTION,
        rationale="required by an adopted protocol",
    )

    assert candidate.to_action_dict() == {
        "type": "inspect_task",
        "parameters": {"task_id": "task-1"},
    }


def test_stable_hash_is_order_independent_for_mappings() -> None:
    assert stable_hash({"a": 1, "b": [2, 3]}) == stable_hash(
        {"b": [2, 3], "a": 1}
    )


@pytest.mark.parametrize(
    ("baseline_status", "reference_status", "expected"),
    [
        ("failed", "passed", []),
        ("infra_error", "passed", []),
        (
            "passed",
            "passed",
            ["baseline_not_failed:oracle-1:passed"],
        ),
        (
            "failed",
            "failed",
            ["reference_not_passed:oracle-1:failed"],
        ),
    ],
)
def test_oracle_qualification_rule_is_centralized(
    baseline_status: str,
    reference_status: str,
    expected: list[str],
) -> None:
    assert oracle_blocking_reasons(
        "oracle-1",
        baseline_status=baseline_status,
        reference_status=reference_status,
    ) == expected


def test_openai_runtime_uses_relic_environment_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(DEFAULT_HEADERS_ENV, json.dumps({"X-Provider": "test"}))
    monkeypatch.setenv(DISABLE_RESPONSE_STORAGE_ENV, "true")

    assert DEFAULT_HEADERS_ENV.startswith("RELIC_")
    assert configured_openai_default_headers() == {"X-Provider": "test"}
    assert openai_response_storage_disabled() is True
    assert prepare_openai_response_request({"model": "example"})["store"] is False


def test_default_runtime_does_not_eagerly_load_legacy_optional_modules() -> None:
    """Compatibility surfaces may exist, but the default runtime must stay local.

    The release still contains lazy adapters for historical integrations and
    product-specific paths.  A source-text ban therefore rejects valid
    compatibility code.  Import the public runtime in a clean interpreter and
    assert that the optional SocietyCore module is not pulled in before an
    operator explicitly selects one of those paths.
    """

    root = Path(__file__).resolve().parents[1]
    probe = subprocess.run(
        (
            sys.executable,
            "-c",
            (
                "import sys; "
                "import relic.cell_worker; "
                "from environments.org_env.backend.simulation.world import OrgWorld; "
                "assert not any(name == 'society_core' or "
                "name.startswith('society_core.') for name in sys.modules)"
            ),
        ),
        cwd=root,
        env={"PATH": str(Path(sys.executable).parent)},
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
