"""Product substrates for OrgEnv (OSS Time-Machine brief §3).

Two substrates seed the company's product:

* ``synthetic_lanternscout`` — the hand-written messy research-agent (default / debug / dev).
* ``oss_time_machine``       — a real OSS project's early runnable release (frozen locally), with
  future code / release notes / hidden behavior tests withheld from the agents.

``product/seed.py::seed_product`` routes to the substrate named in
``company_config["product_substrate"]["type"]``. Downstream product-artifact / repo-workflow /
materialization / release / market / event-graph systems are unchanged.
"""
from __future__ import annotations

from environments.org_env.product.substrates.base import (
    OSS_TIME_MACHINE,
    SUBSTRATE_TYPES,
    SYNTHETIC_LANTERNSCOUT,
    HiddenTestSpec,
    HistoricalIssue,
    OSSSubstrateSpec,
)

__all__ = [
    "SYNTHETIC_LANTERNSCOUT", "OSS_TIME_MACHINE", "SUBSTRATE_TYPES",
    "HistoricalIssue", "HiddenTestSpec", "OSSSubstrateSpec",
]
