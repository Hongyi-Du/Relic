"""Fail-closed runtime identity checks for CooperBench public execution."""
from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "cooperbench_evaluator_runtime_v1"


def evaluator_runtime_receipt(executor: Any) -> dict[str, Any]:
    """Return the execution fields that determine a Cooper sandbox runtime."""

    policy = getattr(executor, "policy", None)
    if policy is None:
        raise RuntimeError("cooperbench_evaluator_runtime_policy_missing")
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": str(getattr(policy, "backend", "") or ""),
        "container_image": str(
            getattr(policy, "container_image", "") or ""
        ),
        "container_platform": str(
            getattr(policy, "container_platform", "") or ""
        ),
        "clear_container_entrypoint": bool(
            getattr(policy, "clear_container_entrypoint", False)
        ),
        "network_enabled": bool(getattr(policy, "network_enabled", True)),
    }


def require_evaluator_runtime(
    executor: Any,
    *,
    expected_backend: str,
    expected_image: str,
    expected_platform: str,
) -> dict[str, Any]:
    """Prove the executor uses the exact task image requested by CooperBench."""

    receipt = evaluator_runtime_receipt(executor)
    expected = {
        "backend": str(expected_backend),
        "container_image": str(expected_image),
        "container_platform": str(expected_platform),
        "clear_container_entrypoint": True,
        "network_enabled": False,
    }
    mismatches = [
        key for key, value in expected.items() if receipt.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            "cooperbench_evaluator_runtime_mismatch:"
            + ",".join(sorted(mismatches))
        )
    return receipt


__all__ = [
    "SCHEMA_VERSION",
    "evaluator_runtime_receipt",
    "require_evaluator_runtime",
]
