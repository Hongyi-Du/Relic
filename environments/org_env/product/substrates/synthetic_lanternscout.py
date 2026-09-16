"""Synthetic LanternScout substrate (brief §6.1).

Thin wrapper around the existing hand-written LanternScout seeding so it can be referenced uniformly
as a substrate. The implementation still lives in ``product/seed.py`` (unchanged) — this only
re-exports it so both substrates have a parallel ``seed_substrate`` entry point.
"""
from __future__ import annotations

from typing import Any, Dict

from environments.org_env.product.seed import seed_synthetic_lanternscout_product


def seed_substrate(world: Any, substrate_config: Dict[str, Any] = None) -> Any:
    # the synthetic substrate ignores substrate_config; it uses company_config defaults
    return seed_synthetic_lanternscout_product(world, None)


__all__ = ["seed_substrate", "seed_synthetic_lanternscout_product"]
