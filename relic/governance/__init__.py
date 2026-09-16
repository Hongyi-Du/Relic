"""Governance primitives shared by Relic experiment runtimes."""

from .capabilities import (
    CAPABILITY_KIND_ORGANIZATIONAL,
    CAPABILITY_KIND_TECHNICAL_FIX,
    EXPLORATORY_CAPABILITIES,
    ORGANIZATIONAL_CAPABILITY_LABELS,
    PREREGISTERED_CAPABILITIES,
    capability_label,
    classify_capability_kind,
)

__all__ = [
    "CAPABILITY_KIND_ORGANIZATIONAL",
    "CAPABILITY_KIND_TECHNICAL_FIX",
    "EXPLORATORY_CAPABILITIES",
    "ORGANIZATIONAL_CAPABILITY_LABELS",
    "PREREGISTERED_CAPABILITIES",
    "capability_label",
    "classify_capability_kind",
]
