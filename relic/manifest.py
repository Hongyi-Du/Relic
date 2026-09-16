"""Canonical main-study manifest construction.

This module plans cells only. It never contacts a model provider or starts an
experiment, which makes it safe to use for release checks and dry runs.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from relic.paths import config_root, default_output_root


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"configuration must be a mapping: {path}")
    return payload


def visible_memory_gib() -> float:
    """Return memory visible to the current Linux/WSL process."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return float(pages * page_size) / (1024**3)
    except (ValueError, OSError, AttributeError):
        return 0.0


def recommended_parallelism(memory_gib: float) -> int:
    """Apply the conservative handoff policy of about 16 GiB per cell."""
    if memory_gib < 32:
        return 1
    if memory_gib < 64:
        return 2
    if memory_gib < 100:
        return 4
    return 8


def _load_named_configs(directory: str, names: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        path = config_root() / directory / f"{name.lower()}.yaml"
        payload = load_yaml(path)
        declared = str(payload.get(directory[:-1]) or payload.get("id") or "").lower()
        if declared != name.lower():
            raise ValueError(f"{path} declares {declared!r}, expected {name.lower()!r}")
        result[name] = payload
    return result


def build_main_manifest(
    *,
    model: str,
    output_root: Path | None = None,
    max_parallel: int = 1,
) -> dict[str, Any]:
    """Build the paper's 120-cell, single-model main-study plan."""
    study_path = config_root() / "main-study.yaml"
    study = load_yaml(study_path)
    models = [str(value) for value in study["models"]]
    if model not in models:
        raise ValueError(f"unknown main-study model {model!r}; choose one of {models}")
    if max_parallel < 1:
        raise ValueError("max_parallel must be at least 1")

    arms = [str(value).upper() for value in study["arms"]]
    workloads = [str(value).upper() for value in study["workloads"]]
    seeds = [int(value) for value in study["seeds"]]
    arm_configs = _load_named_configs("arms", arms)
    workload_configs = _load_named_configs("workloads", workloads)
    model_configs = _load_named_configs("models", [model])

    destination = (output_root or default_output_root() / "main-study").resolve()
    cells: list[dict[str, Any]] = []
    for workload in workloads:
        for seed in seeds:
            for arm in arms:
                cell_id = f"{model}__{workload}__{arm}__seed{seed}"
                cells.append(
                    {
                        "cell_id": cell_id,
                        "model": model,
                        "workload": workload,
                        "arm": arm,
                        "seed": seed,
                        "ticks": int(study["ticks"]),
                        "checkpoint_every": int(study["checkpoint_every"]),
                        "sprint_ticks": int(study["sprint_ticks"]),
                        "output_path": str(destination / cell_id),
                        "status": "pending",
                    }
                )

    expected = len(workloads) * len(seeds) * len(arms)
    if expected != 120 or len(cells) != 120:
        raise ValueError(f"canonical single-model main study must contain 120 cells, found {len(cells)}")

    memory_gib = visible_memory_gib()
    recommended = recommended_parallelism(memory_gib)
    return {
        "schema_version": "relic-run-manifest-v1",
        "study": str(study["study"]),
        "paper_design": "10 workloads x 3 seeds x 4 arms",
        "model": model_configs[model],
        "arms": arm_configs,
        "workloads": workload_configs,
        "seeds": seeds,
        "total_cells": len(cells),
        "max_parallel": max_parallel,
        "resource_advice": {
            "visible_memory_gib": round(memory_gib, 2),
            "recommended_max_parallel": recommended,
            "budget_gib_per_active_cell": 16,
            "warning": max_parallel > recommended,
        },
        "output_root": str(destination),
        "generated_at": datetime.now(UTC).isoformat(),
        "cells": cells,
    }


def write_manifest(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
