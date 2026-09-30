"""Mechanism ablations for the controlled OrgEnv experiment.

The default is an empty set, which preserves the existing runtime exactly.
Scenario parameters are preferred for reproducibility; ``ORG_MECHANISM_ABLATIONS``
is a convenience surface for single runs and accepts comma-separated names.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, ClassVar, Iterable, Mapping, Optional

PROFILE_POLICY = "profile_policy"
INSTITUTIONALIZATION = "institutionalization"
EVENT_GRAPH = "event_graph"
EXTERNAL_BRIDGE = "external_bridge"
PROTOCOL_ENFORCEMENT = "protocol_enforcement"
CAPABILITY_MEMORY = "capability_memory"
PRODUCT_WORKFLOW = "product_workflow"
# The lived-body layer: attention/fatigue/stress/burnout/morale plus the
# circadian availability gates that decide whether an agent is awake and online.
# Disabling it freezes those vitals at their neutral defaults and makes every
# agent continuously available, which is what an organization study wants when
# it is not also making a claim about human work rhythm. Payroll-driven trust
# and retention are organizational, not physiological, and are left running.
WORK_RHYTHM = "work_rhythm"

MECHANISMS = frozenset(
    {
        PROFILE_POLICY,
        INSTITUTIONALIZATION,
        EVENT_GRAPH,
        EXTERNAL_BRIDGE,
        PROTOCOL_ENFORCEMENT,
        CAPABILITY_MEMORY,
        PRODUCT_WORKFLOW,
        WORK_RHYTHM,
    }
)

_ALIASES = {
    "profile": PROFILE_POLICY,
    "profile_graph": PROFILE_POLICY,
    "institution": INSTITUTIONALIZATION,
    "event_history": EVENT_GRAPH,
    "external": EXTERNAL_BRIDGE,
    "bridge": EXTERNAL_BRIDGE,
    "enforcement": PROTOCOL_ENFORCEMENT,
    "memory": CAPABILITY_MEMORY,
    "capability_ledger": CAPABILITY_MEMORY,
    "executable_product_workflow": PRODUCT_WORKFLOW,
    "circadian": WORK_RHYTHM,
    "fatigue": WORK_RHYTHM,
    "lived_body": WORK_RHYTHM,
    "vitals": WORK_RHYTHM,
    # paper_evidence.py ablation_id -> runtime switch binding
    "no_capability_memory": CAPABILITY_MEMORY,
    "no_executable_product_workflow": PRODUCT_WORKFLOW,
    "no_work_rhythm": WORK_RHYTHM,
}


def _canonical_name(value: Any) -> str:
    name = str(value).strip().lower().replace("-", "_")
    return _ALIASES.get(name, name)


@dataclass(frozen=True)
class MechanismAblations:
    SCHEMA_VERSION: ClassVar[str] = "orgenv_mechanism_ablations_v1"

    disabled: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        unknown = set(self.disabled) - set(MECHANISMS)
        if unknown:
            raise ValueError(f"unknown mechanism ablations: {sorted(unknown)}")

    def is_disabled(self, mechanism: str) -> bool:
        return _canonical_name(mechanism) in self.disabled

    @property
    def arm_id(self) -> str:
        return "full" if not self.disabled else "without_" + "_and_".join(sorted(self.disabled))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "arm_id": self.arm_id,
            "disabled": sorted(self.disabled),
            "enabled": sorted(MECHANISMS - self.disabled),
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_values(cls, values: Iterable[Any]) -> "MechanismAblations":
        disabled = frozenset(_canonical_name(value) for value in values if str(value).strip())
        return cls(disabled=disabled)


def resolve_mechanism_ablations(
    config: Optional[Any] = None,
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> MechanismAblations:
    """Resolve list/dict/string config; explicit scenario config beats the env."""

    raw = config
    if raw is None:
        env = environ if environ is not None else os.environ
        raw = env.get("ORG_MECHANISM_ABLATIONS", "")
    if isinstance(raw, MechanismAblations):
        return raw
    if isinstance(raw, str):
        return MechanismAblations.from_values(raw.split(","))
    if isinstance(raw, Mapping):
        if "disabled" in raw:
            values = raw.get("disabled") or ()
            if isinstance(values, str):
                values = values.split(",")
            return MechanismAblations.from_values(values)
        return MechanismAblations.from_values(
            name for name, disabled in raw.items() if bool(disabled)
        )
    if raw is None:
        return MechanismAblations()
    return MechanismAblations.from_values(raw)


def mechanism_disabled(world: Any, mechanism: str) -> bool:
    config = getattr(world, "mechanism_ablations", None)
    return bool(config and config.is_disabled(mechanism))


__all__ = [
    "CAPABILITY_MEMORY",
    "EVENT_GRAPH",
    "EXTERNAL_BRIDGE",
    "INSTITUTIONALIZATION",
    "MECHANISMS",
    "MechanismAblations",
    "PRODUCT_WORKFLOW",
    "PROFILE_POLICY",
    "PROTOCOL_ENFORCEMENT",
    "WORK_RHYTHM",
    "mechanism_disabled",
    "resolve_mechanism_ablations",
]
