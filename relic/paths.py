"""Central path resolution for the Relic release."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return the checkout root unless explicitly overridden for packaging."""
    configured = os.environ.get("RELIC_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def config_root() -> Path:
    return project_root() / "configs"


def benchmark_root() -> Path:
    configured = os.environ.get("RELIC_BENCHMARK_ROOT")
    return Path(configured).expanduser().resolve() if configured else project_root() / "benchmarks"


def default_output_root() -> Path:
    configured = os.environ.get("RELIC_OUTPUT_ROOT")
    return Path(configured).expanduser().resolve() if configured else project_root() / "outputs"

