"""Experiment data objects — Dataset / BenchmarkScenario / Run / ResultRecord
(DESIGN env_org §36-§40/§42).

The Experiment object itself is reused from backend/entities (§33.1); here we add
the data/run/result objects the sandbox produces. All carry config_hash / seed /
environment_hash for reproducibility (acceptance ⑫).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Dataset:
    dataset_id: str
    name: str = ""
    domain: str = ""
    size: int = 0
    quality: float = 0.5
    license: str = "unknown"
    metadata_complete: bool = False
    known_issues: List[str] = field(default_factory=list)
    used_by_experiments: List[str] = field(default_factory=list)
    version: str = "v1"
    content_hash: str = ""


@dataclass
class BenchmarkScenario:
    scenario_id: str
    name: str = ""
    task_type: str = ""
    difficulty: float = 0.5
    expected_behavior: str = ""
    evaluation_metric: str = "accuracy"
    source_dataset: Optional[str] = None
    coverage_tags: List[str] = field(default_factory=list)
    reliability_score: float = 0.5


@dataclass
class Run:
    run_id: str
    experiment_id: str
    sandbox_id: str
    agent_id: str
    cost: float = 0.0
    start_tick: int = 0
    end_tick: int = 0
    status: str = "completed"   # completed|failed|running
    trace_id: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    error_log: str = ""
    seed: int = 0
    config_hash: str = ""
    environment_hash: str = ""


@dataclass
class ResultRecord:
    result_id: str
    experiment_id: str
    run_id: str
    metrics: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.5
    seed: int = 0
    config_hash: str = ""
    environment_hash: str = ""
    reproducibility_status: str = "unknown"   # unknown|reproduced|not_reproduced
    linked_claims: List[str] = field(default_factory=list)
    file_id: Optional[str] = None
    trusted_by: List[str] = field(default_factory=list)
    disputed_by: List[str] = field(default_factory=list)
    logged_to_tracker: bool = False   # local until explicitly exported (§19/§40)


__all__ = ["Dataset", "BenchmarkScenario", "Run", "ResultRecord"]
