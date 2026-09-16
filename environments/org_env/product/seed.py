"""Seed a frozen Relic OSS workload into an organization world."""

from __future__ import annotations

from typing import Any

from environments.org_env.product.objects import ProductState


DEFAULT_COMPANY_CONFIG = {
    "company_name": "Relic organization",
    "product_name": "frozen OSS workload",
    "product_stage": "early runnable OSS release",
    "product_purpose": (
        "Improve a frozen open-source starter from agent-visible issues and "
        "public contracts while preserving release correctness."
    ),
}


def seed_product(world: Any, config: dict[str, Any] | None = None) -> ProductState:
    """Seed the only public paper substrate: a frozen OSS time machine."""

    supplied = dict(config or {})
    substrate = supplied.get("product_substrate")
    if not isinstance(substrate, dict):
        raise ValueError("company_config.product_substrate is required")
    if substrate.get("type") != "oss_time_machine":
        raise ValueError("unsupported_product_substrate")

    from environments.org_env.product.substrates.oss_time_machine import (
        seed_oss_time_machine_product,
    )

    return seed_oss_time_machine_product(world, substrate)


__all__ = ["DEFAULT_COMPANY_CONFIG", "seed_product"]
