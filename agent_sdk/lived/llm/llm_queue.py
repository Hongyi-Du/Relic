"""LLM job queue + decision snapshot validation (async infra §9-§15).

Async means an LLM call submitted at tick T may only return at tick T+k, by
which point the agent's state may have changed (moved, lost the resource, got
busy, action became illegal). So every action-affecting LLM call is bound to a
:class:`DecisionSnapshot` taken at submit time; on return the snapshot is
re-validated against the live world (§11/§12):

  * valid        → use the result;
  * revalidated  → minor change but still feasible → use, flag revalidated;
  * stale        → discard, log, fall back (§12.3).

Plus a simple FIFO+priority+timeout :class:`LLMQueue` (§14) and an
:class:`LLMJob` record (§10) written to ``LLMJobLog`` (§15). Blocking mode (§13):
a timed-out job uses a safe fallback action. Replay safety comes from the
engine's ``read_only`` cache policy (a miss raises rather than calling a model).
"""
from __future__ import annotations

import hashlib
import itertools
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


def _hash(obj: Any) -> str:
    import json
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# §11 DecisionSnapshot
# --------------------------------------------------------------------------- #
@dataclass
class DecisionSnapshot:
    decision_snapshot_id: str
    run_id: str
    agent_id: str
    created_tick: int
    world_tick: int
    agent_location: Optional[List[float]] = None
    agent_energy: float = 0.0
    agent_hp: float = 0.0
    inventory_hash: str = ""
    self_state_hash: str = ""
    perception_packet_id: Optional[str] = None
    memory_context_id: Optional[str] = None
    active_plan_ids: List[str] = field(default_factory=list)
    active_episode_ids: List[str] = field(default_factory=list)
    visible_object_ids: List[str] = field(default_factory=list)
    visible_agent_ids: List[str] = field(default_factory=list)
    candidate_pool_hash: str = ""
    mechanism_context_hash: str = ""
    world_state_version: int = 0
    agent_state_version: int = 0
    target_resource_ids: List[str] = field(default_factory=list)
    target_agent_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # §12 validate on LLM return
    def validate(self, agent: Any, scene: Any, *, move_tol: float = 0.5,
                 energy_tol: float = 20.0) -> Tuple[str, str]:
        """Return (status, reason): 'valid' / 'revalidated' / 'stale'."""
        # agent gone / dead / no longer available
        if agent is None:
            return "stale", "agent no longer available"
        if getattr(agent, "hp", 1) <= 0:
            return "stale", "agent dead"
        # moved significantly
        loc = [getattr(agent, "x", 0.0), getattr(agent, "y", 0.0)]
        if self.agent_location is not None:
            d = ((loc[0] - self.agent_location[0]) ** 2 + (loc[1] - self.agent_location[1]) ** 2) ** 0.5
            if d > move_tol:
                return "stale", "agent moved"
        # inventory changed
        if _hash(getattr(agent, "inventory", {})) != self.inventory_hash:
            return "stale", "agent inventory changed"
        # target resource gone
        for rid in self.target_resource_ids:
            patch = next((r for r in getattr(scene, "resources", []) if r.get("id") == rid), None)
            if patch is None or float(patch.get("amount", 0)) <= 0:
                return "stale", f"target resource {rid} gone"
        # target agent no longer nearby
        for tid in self.target_agent_ids:
            tgt = next((g for g in getattr(scene, "agents", []) if g.id == tid), None)
            if tgt is None:
                return "stale", f"target agent {tid} gone"
            d = ((loc[0] - tgt.x) ** 2 + (loc[1] - tgt.y) ** 2) ** 0.5
            if d > getattr(agent, "comm_radius", 12):
                return "stale", f"target agent {tid} out of range"
        # energy drifted a lot but action may still be feasible → revalidated
        if abs(getattr(agent, "energy", 0.0) - self.agent_energy) > energy_tol:
            return "revalidated", "energy drifted; re-feasibility-checked"
        return "valid", ""


def capture_snapshot(agent: Any, scene: Any, *, run_id: str, world_tick: int,
                     packet: Any = None, memory_context_id: Optional[str] = None,
                     candidate_pool: Optional[List[Any]] = None,
                     target_resource_ids: Optional[List[str]] = None,
                     target_agent_ids: Optional[List[str]] = None) -> DecisionSnapshot:
    p = (packet.to_dict() if hasattr(packet, "to_dict") else (packet or {}))
    return DecisionSnapshot(
        decision_snapshot_id=f"ds:{run_id}:{world_tick}:{agent.id}",
        run_id=run_id, agent_id=agent.id, created_tick=world_tick, world_tick=world_tick,
        agent_location=[getattr(agent, "x", 0.0), getattr(agent, "y", 0.0)],
        agent_energy=float(getattr(agent, "energy", 0.0)), agent_hp=float(getattr(agent, "hp", 0.0)),
        inventory_hash=_hash(getattr(agent, "inventory", {})),
        self_state_hash=_hash({"e": getattr(agent, "energy", 0), "hp": getattr(agent, "hp", 0)}),
        perception_packet_id=p.get("perception_packet_id"),
        memory_context_id=memory_context_id,
        active_plan_ids=list(getattr(agent, "active_plan_ids", [])),
        active_episode_ids=list(getattr(agent, "active_episode_ids", [])),
        visible_object_ids=[o.get("id") for o in p.get("visible_objects", []) if o.get("id")],
        visible_agent_ids=list(p.get("visible_agents", [])),
        candidate_pool_hash=_hash([getattr(c, "action_type", c) for c in (candidate_pool or [])]),
        target_resource_ids=list(target_resource_ids or []),
        target_agent_ids=list(target_agent_ids or []),
    )


# --------------------------------------------------------------------------- #
# §10 LLMJob
# --------------------------------------------------------------------------- #
@dataclass
class LLMJob:
    llm_job_id: str
    run_id: str
    agent_id: str
    created_tick: int
    module_name: str
    model_role: str = "default"
    input_payload_hash: str = ""
    decision_snapshot_id: str = ""
    visibility_context_hash: str = ""
    expected_schema_id: str = ""
    status: str = "queued"           # queued/running/completed/failed/cancelled/stale/replayed_from_cache
    priority: int = 5                # lower = higher priority
    deadline_tick: int = 10 ** 9
    cancel_if_stale: bool = True
    cache_policy: str = "read_write"
    ready_tick: int = 0              # earliest tick the result is available (queue latency)
    result_llm_call_id: Optional[str] = None
    result_ready_tick: int = -1
    validation_status: str = ""
    stale_reason: str = ""
    fallback_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# §14 LLM queue (FIFO + priority + timeout)
# --------------------------------------------------------------------------- #
@dataclass
class _Entry:
    job: LLMJob
    snapshot: DecisionSnapshot
    call_kwargs: Dict[str, Any]
    fallback: Optional[Callable[[], Any]]
    seq: int


class LLMQueue:
    def __init__(self, engine: Any, journal: Any = None, *, max_concurrent: int = 4,
                 timeout_ticks: int = 3, queue_latency: int = 0):
        self.engine = engine
        self.journal = journal
        self.max_concurrent = max_concurrent
        self.timeout_ticks = timeout_ticks
        self.queue_latency = queue_latency
        self._pending: List[_Entry] = []
        self._counter = itertools.count()

    def submit(self, *, module_name: str, agent_id: str, world_tick: int,
               snapshot: DecisionSnapshot, call_kwargs: Dict[str, Any],
               priority: int = 5, model_role: str = "default", cache_policy: str = "read_write",
               cancel_if_stale: bool = True, deadline_tick: Optional[int] = None,
               fallback: Optional[Callable[[], Any]] = None) -> LLMJob:
        seq = next(self._counter)
        job = LLMJob(
            llm_job_id=f"job:{agent_id}:{world_tick}:{seq}", run_id=getattr(self.engine, "run_id", "run"),
            agent_id=agent_id, created_tick=world_tick, module_name=module_name, model_role=model_role,
            input_payload_hash=_hash(call_kwargs.get("input_payload", {})),
            decision_snapshot_id=snapshot.decision_snapshot_id,
            visibility_context_hash=_hash(call_kwargs.get("visibility_context", {})),
            expected_schema_id=str(call_kwargs.get("prompt_template_id", "")),
            priority=priority, cache_policy=cache_policy, cancel_if_stale=cancel_if_stale,
            deadline_tick=deadline_tick if deadline_tick is not None else world_tick + self.timeout_ticks,
            ready_tick=world_tick + self.queue_latency,
        )
        self._pending.append(_Entry(job=job, snapshot=snapshot, call_kwargs=dict(call_kwargs),
                                    fallback=fallback, seq=seq))
        return job

    def process_due(self, world_tick: int, scene: Any) -> List[Tuple[LLMJob, Dict[str, Any]]]:
        """Process up to ``max_concurrent`` ready jobs (FIFO within priority).
        Returns [(job, outcome)] where outcome is the engine result or a fallback
        ``{"fallback": ...}``."""
        agents = {g.id: g for g in getattr(scene, "agents", [])}
        ready = [e for e in self._pending if e.job.ready_tick <= world_tick]
        ready.sort(key=lambda e: (e.job.priority, e.seq))
        out: List[Tuple[LLMJob, Dict[str, Any]]] = []
        for e in ready[: self.max_concurrent]:
            self._pending.remove(e)
            job, snap = e.job, e.snapshot
            # §13 timeout → fallback
            if world_tick > job.deadline_tick:
                job.status = "cancelled"; job.stale_reason = "timeout"; job.fallback_used = True
                job.result_ready_tick = world_tick
                self._log(job, world_tick, snapshot_validation="timeout")
                out.append((job, {"fallback": e.fallback() if e.fallback else None, "reason": "timeout"}))
                continue
            agent = agents.get(job.agent_id)
            vstatus, reason = snap.validate(agent, scene)
            job.validation_status = vstatus
            if vstatus == "stale" and job.cancel_if_stale:
                job.status = "stale"; job.stale_reason = reason; job.fallback_used = True
                job.result_ready_tick = world_tick
                self._log(job, world_tick, snapshot_validation="stale")
                out.append((job, {"fallback": e.fallback() if e.fallback else None, "stale_reason": reason}))
                continue
            # snapshot valid (or revalidated) → run the engine call
            result = self.engine.call(**e.call_kwargs)
            job.result_llm_call_id = result.get("metadata", {}).get("llm_call_id")
            job.status = ("replayed_from_cache" if result.get("metadata", {}).get("cache_hit")
                          else "completed")
            job.result_ready_tick = world_tick
            self._log(job, world_tick, snapshot_validation=vstatus,
                      stale_on_return=(vstatus == "revalidated"))
            out.append((job, result))
        return out

    def pending_count(self) -> int:
        return len(self._pending)

    def _log(self, job: LLMJob, world_tick: int, *, snapshot_validation: str,
             stale_on_return: bool = False) -> None:
        if self.journal is None:
            return
        self.journal.record(
            "LLMJobLog", source_module="llm_queue", turn_id=world_tick, agent_id=job.agent_id,
            llm_job_id=job.llm_job_id, created_tick=job.created_tick,
            started_tick=job.ready_tick, finished_tick=world_tick,
            queue_wait_ticks=max(0, world_tick - job.created_tick),
            runtime_ticks=max(0, world_tick - job.ready_tick), deadline_tick=job.deadline_tick,
            status=job.status, stale_on_return=stale_on_return,
            snapshot_validation_result=snapshot_validation, fallback_used=job.fallback_used,
            cancel_reason=job.stale_reason, result_llm_call_id=job.result_llm_call_id)
