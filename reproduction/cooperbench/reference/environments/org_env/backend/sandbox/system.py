"""SandboxSystem — symbolic experiment execution (DESIGN env_org §32-§35/§42).

v0 does NOT run real code. A job's success/cost/reproducibility is a
**deterministic** function (seeded by config_hash+seed) of: agent skill · repo
state · dataset quality · environment_hash · compute budget · task difficulty ·
prior notes · whether config is reproducible. Results stay PRIVATE in the
owner's sandbox until explicitly exported (§19/§40).
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

from environments.org_env.backend.experiments.objects import ResultRecord, Run
from environments.org_env.backend.sandbox.objects import ExecutionSandbox, SandboxJob
from environments.org_env.backend.workspace.provenance import content_hash


class SandboxSystem:
    def __init__(self):
        self.sandboxes: Dict[str, ExecutionSandbox] = {}
        self.jobs: Dict[str, SandboxJob] = {}
        self.runs: Dict[str, Run] = {}
        self.results: Dict[str, ResultRecord] = {}
        self._seq = 0

    def ensure_sandbox(self, agent_id: str) -> ExecutionSandbox:
        sid = f"sandbox_{agent_id}"
        if sid not in self.sandboxes:
            self.sandboxes[sid] = ExecutionSandbox(
                sandbox_id=sid, owner_id=agent_id,
                environment_hash=content_hash({"sandbox": sid}))
        return self.sandboxes[sid]

    def _id(self, p: str) -> str:
        self._seq += 1
        return f"{p}_{self._seq}"

    def run_job(self, *, agent_id: str, job_type: str = "run_cheap_pilot",
                experiment_id: str = "", config: Optional[dict] = None,
                seed: int = 0, skill: float = 0.5, difficulty: float = 0.5,
                reproducible_config: bool = True, dataset_quality: float = 0.6,
                cost: float = 10.0, tick: int = 0) -> SandboxJob:
        """Symbolically run a job. Deterministic success from (seed, config)."""
        sb = self.ensure_sandbox(agent_id)
        cfg_hash = content_hash(config or {"job": job_type, "exp": experiment_id})
        rng = random.Random(f"{seed}:{cfg_hash}")
        # success probability (bounded) from the §34 factors
        p = 0.25 + 0.4 * skill + 0.2 * dataset_quality - 0.35 * difficulty \
            + (0.15 if reproducible_config else -0.1) \
            - (0.2 if sb.cost_spent > sb.available_compute else 0.0)
        p = max(0.05, min(0.95, p))
        success = rng.random() < p
        jid = self._id("job")
        rid_seed = rng.random()

        sb.cost_spent += cost
        if sb.cost_spent > sb.available_compute:
            sb.sandbox_status = "out_of_budget"

        job = SandboxJob(job_id=jid, sandbox_id=sb.sandbox_id, agent_id=agent_id,
                         job_type=job_type, config_hash=cfg_hash, seed=seed,
                         linked_experiment_id=experiment_id, start_tick=tick,
                         end_tick=tick + 1, cost=cost,
                         status="completed" if success else "failed",
                         reproducibility_score=(0.8 if reproducible_config else 0.3))
        self.jobs[jid] = job

        if success:
            run_id = self._id("run")
            run = Run(run_id=run_id, experiment_id=experiment_id, sandbox_id=sb.sandbox_id,
                      agent_id=agent_id, cost=cost, start_tick=tick, end_tick=tick + 1,
                      status="completed", metrics={"score": round(0.5 + 0.4 * rid_seed, 3)},
                      seed=seed, config_hash=cfg_hash, environment_hash=sb.environment_hash)
            self.runs[run_id] = run
            res_id = self._id("result")
            res = ResultRecord(result_id=res_id, experiment_id=experiment_id, run_id=run_id,
                               metrics=dict(run.metrics), confidence=round(p, 2), seed=seed,
                               config_hash=cfg_hash, environment_hash=sb.environment_hash,
                               reproducibility_status="reproduced" if reproducible_config else "unknown")
            self.results[res_id] = res
            job.result_id = res_id
            sb.completed_jobs.append(jid)
            sb.cached_result_ids.append(res_id)   # PRIVATE — not in shared tracker yet
        else:
            job.failure_reason = "config_not_reproducible" if not reproducible_config else "low_skill_or_budget"
            sb.failed_jobs.append(jid)
        return job

    def export_result_to_tracker(self, result_id: str) -> Optional[ResultRecord]:
        """Mark a private result as logged to the shared experiment tracker
        (must be an explicit action — §19/§40)."""
        res = self.results.get(result_id)
        if res:
            res.logged_to_tracker = True
        return res

    def visible_results_for(self, agent_id: str) -> List[ResultRecord]:
        """Own sandbox results + any result already exported to the shared tracker."""
        sb = self.sandboxes.get(f"sandbox_{agent_id}")
        own = set(sb.cached_result_ids) if sb else set()
        return [r for rid, r in self.results.items() if rid in own or r.logged_to_tracker]


__all__ = ["SandboxSystem"]
