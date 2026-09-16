"""Minimal live OrgWorld session used by the CooperBench B3-2 adapter.

It keeps the generic B3 session and explicit two-person roster seams needed by
the adapter, while deliberately excluding HCI UI state and alternate
execution-profile support from the public Relic release.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Sequence

from relic.core.domain import DomainScenarioConfig

from environments.org_env.backend.simulation.world import OrgWorld
from environments.org_env.runtime_adapter.snapshot import org_lived_full_snapshot


class OrgLiveFrameBuffer:
    """Bounded deep-frame buffer for a native OrgWorld session."""

    def __init__(self, max_frames: int = 2000) -> None:
        self.frames: List[Dict[str, Any]] = []
        self.max_frames = max_frames

    def add_frame(self, frame: Dict[str, Any]) -> None:
        self.frames.append(frame)
        if len(self.frames) > self.max_frames:
            self.frames = self.frames[-self.max_frames:]

    def frames_since(self, tick: int) -> List[Dict[str, Any]]:
        return [frame for frame in self.frames if frame.get("tick", -1) > tick]

    @property
    def latest_tick(self) -> int:
        return self.frames[-1]["tick"] if self.frames else -1


class OrgInspectorSession:
    """Own a native OrgWorld plus a frame buffer and simulation controls."""

    def __init__(
        self,
        *,
        seed: int = 42,
        n_agents: int = 8,
        policy_mode: str = "mock",
        max_frames: int = 600,
        load_llm: bool = False,
        approval_mode: str = "auto",
        experiment_condition: Optional[str] = None,
        baseline_sprint_ticks: Optional[int] = None,
        member_ids: Optional[Sequence[str]] = None,
        noncanonical_roster_variant: Optional[str] = None,
        defer_initial_readiness_check: bool = False,
    ) -> None:
        self.seed = seed
        self.n_agents = n_agents
        self.policy_mode = policy_mode
        self.load_llm = load_llm
        self.approval_mode = approval_mode
        self.experiment_condition = experiment_condition
        self.baseline_sprint_ticks = baseline_sprint_ticks
        self.member_ids = (
            tuple(str(item).strip() for item in member_ids)
            if member_ids is not None
            else None
        )
        self.noncanonical_roster_variant = str(
            noncanonical_roster_variant or ""
        ).strip()
        self.buffer = OrgLiveFrameBuffer(max_frames=max_frames)
        self.world: Optional[OrgWorld] = None
        self.is_running = False
        self.created = time.time()
        self._defer_initial_readiness_check = bool(defer_initial_readiness_check)
        try:
            self.reset(seed=seed)
        finally:
            self._defer_initial_readiness_check = False

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        n_agents: Optional[int] = None,
        policy_mode: Optional[str] = None,
        experiment_condition: Optional[str] = None,
        baseline_sprint_ticks: Optional[int] = None,
    ) -> Dict[str, Any]:
        if seed is not None:
            self.seed = seed
        if n_agents is not None:
            self.n_agents = n_agents
        if policy_mode is not None:
            self.policy_mode = policy_mode
        if experiment_condition is not None:
            self.experiment_condition = experiment_condition
        if baseline_sprint_ticks is not None:
            self.baseline_sprint_ticks = baseline_sprint_ticks

        params: dict[str, Any] = {
            "num_internal_agents": self.n_agents,
            "policy_mode": self.policy_mode,
        }
        if self.member_ids is not None:
            params["internal_agent_ids"] = list(self.member_ids)
        for key, value in {
            "experiment_phase": os.environ.get("ORG_EXPERIMENT_PHASE"),
            "arm_id": os.environ.get("ORG_EXPERIMENT_ARM_ID"),
            "evaluation_perturbation": os.environ.get(
                "ORG_EXPERIMENT_EVALUATION_PERTURBATION"
            ),
        }.items():
            if value not in (None, ""):
                params[key] = value

        from environments.org_env.config.baseline_conditions import resolve_condition

        raw_condition = self.experiment_condition or os.environ.get(
            "ORG_EXPERIMENT_CONDITION"
        )
        condition = resolve_condition(raw_condition)
        params["action_selection_mode"] = condition.action_selection_mode
        if raw_condition:
            params["experiment_condition"] = condition.condition_id
            params["experiment_condition_explicit"] = True
            params["num_internal_agents"] = condition.roster_size
            self.n_agents = condition.roster_size
        if self.noncanonical_roster_variant:
            if self.member_ids is None:
                raise ValueError(
                    "a noncanonical roster variant requires explicit member_ids"
                )
            params["noncanonical_roster_variant"] = self.noncanonical_roster_variant
            params["num_internal_agents"] = len(self.member_ids)
            self.n_agents = len(self.member_ids)

        sprint_ticks = self.baseline_sprint_ticks
        if sprint_ticks is None:
            raw_sprint_ticks = os.environ.get("ORG_BASELINE_SPRINT_TICKS")
            sprint_ticks = int(raw_sprint_ticks) if raw_sprint_ticks else None
        if sprint_ticks is not None:
            params["baseline_sprint_ticks"] = max(1, int(sprint_ticks))

        execution_profile = (
            os.environ.get("ORG_EXECUTION_PROFILE", "native") or "native"
        ).strip()
        if execution_profile != "native":
            raise RuntimeError(
                "only native execution is available in the public Relic Cooper runtime"
            )

        substrate = (os.environ.get("ORG_PRODUCT_SUBSTRATE", "") or "").strip()
        mode = (os.environ.get("ORG_OSS_MODE", "dev") or "dev").strip()
        if mode == "formal" and substrate != "oss_time_machine":
            raise RuntimeError(
                "ORG_OSS_MODE=formal requires ORG_PRODUCT_SUBSTRATE=oss_time_machine"
            )
        corpus = "v0"
        if substrate == "oss_time_machine":
            dataset = os.environ.get("ORG_OSS_DATASET", "") or ""
            params["experiment_mode"] = mode
            params["company_config"] = {
                "product_substrate": {
                    "type": "oss_time_machine",
                    "dataset_id": dataset,
                    "repository_id": os.environ.get("ORG_OSS_REPOSITORY_ID", "") or "",
                    "anonymize": os.environ.get("ORG_OSS_ANONYMIZE", "0")
                    in ("1", "true", "True"),
                    "control": os.environ.get("ORG_OSS_CONTROL", "none") or "none",
                    "mode": mode,
                }
            }
            corpus = "oss-v0"

        scenario = DomainScenarioConfig(
            name="org_default", seed=self.seed, corpus_version=corpus, params=params
        )
        self.world = OrgWorld(scenario).build()
        self.world.set_approval_mode(self.approval_mode)
        if self.load_llm:
            try:
                from environments.org_env.experiments.resources import (
                    attach_metered_llm_client,
                )
                from environments.org_env.llm.config import load_org_llm_client

                client, decides = load_org_llm_client()
                if client is None:
                    self.world.llm_client_load_error = "load_org_llm_client_returned_none"
                else:
                    self.world.llm_client = attach_metered_llm_client(self.world, client)
                    self.world.llm_decides_actions = decides
            except Exception as exc:
                self.world.llm_client_load_error = repr(exc)
        if not self._defer_initial_readiness_check:
            self.world.ensure_action_selection_ready()
        self.buffer = OrgLiveFrameBuffer(max_frames=self.buffer.max_frames)
        self.is_running = False
        self._capture()
        return {
            "reset": True,
            "seed": self.seed,
            "tick": self.world.world_tick,
            "experiment_condition": self.world.experiment_condition,
            "action_selection_mode": self.world.action_selection_mode,
        }

    def _capture(self) -> Dict[str, Any]:
        if self.world is None:
            raise RuntimeError("org_world_missing")
        frame = org_lived_full_snapshot(self.world, mode="live")
        self.buffer.add_frame(frame)
        return frame

    def step(self, n: int = 1) -> Dict[str, Any]:
        if self.world is None:
            raise RuntimeError("org_world_missing")
        self.is_running = True
        try:
            for _ in range(max(1, int(n))):
                self.world.step()
                self._capture()
        finally:
            self.is_running = False
        return {
            "tick": self.world.world_tick,
            "captured": True,
            "latest_tick": self.buffer.latest_tick,
        }

    def run_ticks(self, n: int) -> Dict[str, Any]:
        return self.step(n)

    def pause(self) -> Dict[str, Any]:
        self.is_running = False
        return {"running": False, "tick": getattr(self.world, "world_tick", 0)}

    def full(self) -> Dict[str, Any]:
        return self.buffer.frames[-1] if self.buffer.frames else {}

    def frames_since(self, since: int = -1) -> Dict[str, Any]:
        return {"frames": self.buffer.frames_since(since), "latest_tick": self.buffer.latest_tick}

    def state(self) -> Dict[str, Any]:
        frame = self.full()
        return {
            "tick": frame.get("tick", 0),
            "running": self.is_running,
            "agents": list((frame.get("agents") or {}).keys()),
        }


__all__ = ["OrgInspectorSession", "OrgLiveFrameBuffer"]
