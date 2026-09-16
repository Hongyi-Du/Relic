"""Sandbox objects — ExecutionSandbox / SandboxJob (DESIGN env_org §32-§35/§42)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SandboxJob:
    job_id: str
    sandbox_id: str
    agent_id: str
    job_type: str = "run_cheap_pilot"   # run_script|run_experiment|run_paper_baseline|run_cheap_pilot|debug_failure|install_package|load_dataset|evaluate_model|generate_report
    command_summary: str = ""
    config_hash: str = ""
    seed: int = 0
    dataset_id: Optional[str] = None
    benchmark_id: Optional[str] = None
    linked_experiment_id: Optional[str] = None
    start_tick: int = 0
    end_tick: int = 0
    status: str = "completed"   # completed|failed|running
    cost: float = 0.0
    result_id: Optional[str] = None
    failure_reason: str = ""
    reproducibility_score: float = 0.0


@dataclass
class ExecutionSandbox:
    sandbox_id: str
    owner_id: str
    installed_packages: List[str] = field(default_factory=list)
    available_compute: float = 100.0
    local_data: List[str] = field(default_factory=list)
    local_repo_branch: Optional[str] = None
    running_jobs: List[str] = field(default_factory=list)
    completed_jobs: List[str] = field(default_factory=list)
    failed_jobs: List[str] = field(default_factory=list)
    cached_result_ids: List[str] = field(default_factory=list)   # PRIVATE until exported
    paper_reproduction_attempts: int = 0
    environment_hash: str = ""
    cost_spent: float = 0.0
    last_clean_tick: int = 0
    sandbox_status: str = "clean"   # clean|dirty|running_job|broken|stale|out_of_budget


__all__ = ["SandboxJob", "ExecutionSandbox"]
