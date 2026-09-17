"""OrgEnv live inspector session — frame buffer + sim controls (frontend spec §7/§12).

`OrgLiveFrameBuffer` keeps a rolling window of deep frames; `OrgInspectorSession`
owns an `OrgWorld`, steps it on demand (step / run_ticks / reset), captures a frame
after every tick, and serves them to the frontend (`full`, `frames_since`). The
same frames feed live-tailing AND replay export — replay = a saved frame list.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Mapping, Optional

from relic.core.domain import DomainScenarioConfig
from environments.org_env.backend.simulation.world import OrgWorld
from environments.org_env.runtime_adapter.snapshot import org_lived_full_snapshot


class OrgLiveFrameBuffer:
    def __init__(self, max_frames: int = 2000) -> None:
        self.frames: List[Dict[str, Any]] = []
        self.max_frames = max_frames

    def add_frame(self, frame: Dict[str, Any]) -> None:
        self.frames.append(frame)
        if len(self.frames) > self.max_frames:
            self.frames = self.frames[-self.max_frames:]

    def frames_since(self, tick: int) -> List[Dict[str, Any]]:
        return [f for f in self.frames if f.get("tick", -1) > tick]

    @property
    def latest_tick(self) -> int:
        return self.frames[-1]["tick"] if self.frames else -1


class OrgInspectorSession:
    """Holds a running OrgWorld + a frame buffer + sim controls."""

    def __init__(self, *, seed: int = 42, n_agents: int = 8, policy_mode: str = "mock",
                 max_frames: int = 600, load_llm: bool = False, approval_mode: str = "auto",
                 experiment_condition: Optional[str] = None,
                 baseline_sprint_ticks: Optional[int] = None,
                 defer_initial_readiness_check: bool = False) -> None:
        self.seed = seed
        self.n_agents = n_agents
        self.policy_mode = policy_mode
        self.load_llm = load_llm        # opt-in: wire the configured LLM client (off in tests)
        self.approval_mode = approval_mode   # auto (debug) | semi_auto | agent (realistic)
        self.experiment_condition = experiment_condition
        self.baseline_sprint_ticks = baseline_sprint_ticks
        self.last_error: Optional[str] = None
        self.buffer = OrgLiveFrameBuffer(max_frames=max_frames)
        self.world: Optional[OrgWorld] = None
        self.is_running = False
        self.created = time.time()
        self._defer_initial_readiness_check = bool(
            defer_initial_readiness_check
        )
        try:
            self.reset(seed=seed)
        finally:
            # This exception exists only for a checkpoint candidate that will
            # immediately replace the provisional world.  Any later reset or
            # step must enforce the ordinary readiness gate.
            self._defer_initial_readiness_check = False

    # -- lifecycle ---------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, n_agents: Optional[int] = None,
              policy_mode: Optional[str] = None,
              experiment_condition: Optional[str] = None,
              baseline_sprint_ticks: Optional[int] = None) -> Dict[str, Any]:
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
        params = {"num_internal_agents": self.n_agents, "policy_mode": self.policy_mode}
        experiment_metadata = {
            "experiment_phase": os.environ.get("ORG_EXPERIMENT_PHASE"),
            "arm_id": os.environ.get("ORG_EXPERIMENT_ARM_ID"),
            "evaluation_perturbation": os.environ.get(
                "ORG_EXPERIMENT_EVALUATION_PERTURBATION"
            ),
        }
        params.update(
            {
                key: value
                for key, value in experiment_metadata.items()
                if value not in (None, "")
            }
        )
        raw_condition = self.experiment_condition or os.environ.get(
            "ORG_EXPERIMENT_CONDITION"
        )
        from environments.org_env.config.baseline_conditions import resolve_condition

        condition = resolve_condition(raw_condition)
        # The condition is authoritative. In particular, neither
        # ORG_LLM_ACTIONS nor ORG_ACTION_SELECTION_MODE can turn B3 into an
        # LLM-direct condition.
        params["action_selection_mode"] = condition.action_selection_mode
        if raw_condition:
            params["experiment_condition"] = condition.condition_id
            params["experiment_condition_explicit"] = True
            params["num_internal_agents"] = condition.roster_size
            self.n_agents = condition.roster_size
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
                "unsupported_execution_profile_in_relic_release"
            )
        corpus = "v0"
        _substrate = (os.environ.get("ORG_PRODUCT_SUBSTRATE", "") or "").strip()
        _mode = (os.environ.get("ORG_OSS_MODE", "dev") or "dev").strip()
        # Formal paper cells must use the frozen OSS substrate.  The public
        # release intentionally exposes only this native execution route.
        if _mode == "formal" and _substrate != "oss_time_machine":
            raise RuntimeError(
                "ORG_OSS_MODE=formal requires ORG_PRODUCT_SUBSTRATE=oss_time_machine"
            )
        # Relic deliberately excludes the source's synthetic/Nature-adjacent
        # default company.  Even mock replays therefore bind one frozen public
        # benchmark pack rather than constructing a different hidden world.
        if not _substrate:
            _substrate = "oss_time_machine"
        if _substrate == "oss_time_machine":
            _ds = os.environ.get("ORG_OSS_DATASET", "mini_blobstore_v1")
            params["experiment_mode"] = _mode
            params["company_config"] = {"product_substrate": {
                "type": "oss_time_machine",
                "dataset_id": _ds,
                "repository_id": os.environ.get("ORG_OSS_REPOSITORY_ID", "") or "",
                "anonymize": (os.environ.get("ORG_OSS_ANONYMIZE", "0") in ("1", "true", "True")),
                "control": os.environ.get("ORG_OSS_CONTROL", "none") or "none",
                "mode": _mode}}
            corpus = "oss-v0"
        sc = DomainScenarioConfig(name="org_default", seed=self.seed, corpus_version=corpus,
                                  params=params)
        self.world = OrgWorld(sc).build()
        self.world.set_approval_mode(self.approval_mode)   # governance friction mode
        self._apply_capability_transfer()
        if self.load_llm:                     # opt-in LLM cognitive layer from config/llm(.local).yaml
            try:
                from environments.org_env.llm.config import load_org_llm_client
                from environments.org_env.experiments.resources import attach_metered_llm_client
                client, decides = load_org_llm_client()
                if client is not None:
                    self.world.llm_client = attach_metered_llm_client(self.world, client)
                    # Compatibility assignment only. OrgWorld derives the effective
                    # WHAT selector from the condition and client availability.
                    self.world.llm_decides_actions = decides
                else:
                    # load_org_llm_client is fail-soft by design (returns (None,
                    # False) on any config problem). A session that ASKED for the
                    # LLM layer but got no client is a treatment-delivery failure,
                    # not a preference: record it loudly so downstream gates
                    # (llm_client_not_attached) and humans can see why.
                    self.world.llm_client_load_error = "load_org_llm_client_returned_none"
                    print(
                        "[org_live] WARNING: load_llm requested but no LLM client "
                        "attached (config missing/disabled/invalid); running "
                        "rule/template-only",
                        flush=True,
                    )
            except Exception as exc:
                self.world.llm_client_load_error = repr(exc)
                print(
                    f"[org_live] WARNING: LLM client load failed ({exc!r}); "
                    "running rule/template-only",
                    flush=True,
                )
        if not self._defer_initial_readiness_check:
            self.world.ensure_action_selection_ready()
        self.buffer = OrgLiveFrameBuffer(max_frames=self.buffer.max_frames)
        self.is_running = False
        self._capture()                       # tick-0 frame so the UI has initial state
        return {
            "reset": True,
            "seed": self.seed,
            "tick": self.world.world_tick,
            "experiment_condition": self.world.experiment_condition,
            "action_selection_mode": self.world.action_selection_mode,
        }

    def _capture(self) -> Dict[str, Any]:
        frame = org_lived_full_snapshot(self.world, mode="live")
        self.buffer.add_frame(frame)
        return frame

    def _apply_capability_transfer(self) -> None:
        """Inherit a source run's organizational state, for a transfer arm.

        Applied at build time, before tick one, because the arms differ only in
        the state the target world starts with. A missing or unreadable bundle
        raises: an arm that quietly ran with nothing inherited would be a fifth
        copy of the reference wearing another arm's label.
        """
        arm = os.environ.get("ORG_TRANSFER_ARM", "") or ""
        if not arm:
            return
        from environments.org_env.experiments.capability_transfer import (
            inject_capability_bundle,
        )

        bundle_path = os.environ.get("ORG_TRANSFER_SOURCE_BUNDLE", "") or ""
        if not bundle_path or not os.path.isfile(bundle_path):
            raise RuntimeError(
                f"transfer_source_bundle_missing:{arm}:{bundle_path or '(unset)'}"
            )
        with open(bundle_path, encoding="utf-8") as handle:
            bundle = json.load(handle)
        receipt = inject_capability_bundle(
            self.world,
            bundle,
            roster_origin=os.environ.get("ORG_TRANSFER_ROSTER_ORIGIN", ""),
            capability_form=os.environ.get("ORG_TRANSFER_CAPABILITY_FORM", ""),
            frozen_episodes=int(
                os.environ.get("ORG_TRANSFER_FROZEN_EPISODES", "0") or 0
            ),
            fixed_protocol_landscape=(
                os.environ.get("ORG_TRANSFER_FIXED_PROTOCOL_LANDSCAPE", "0")
                == "1"
            ),
        )
        receipt["arm_id"] = arm
        print(
            f"[org_live] transfer arm {arm}: roster={receipt['roster_origin']} "
            f"capabilities={receipt['capability_form']} "
            f"protocols={len(receipt['protocols_injected'])} "
            f"documents={len(receipt['documents_injected'])} "
            f"text={len(receipt['prose_entries_injected'])} "
            f"members={len(receipt['roster_applied'])} "
            f"frozen_episodes={receipt['frozen_episodes']} "
            f"fixed_protocol_landscape={receipt['fixed_protocol_landscape']}"
        )

    # -- controls ----------------------------------------------------------
    def step(self, n: int = 1) -> Dict[str, Any]:
        steps = max(1, int(n))
        self.is_running = True
        try:
            for _ in range(steps):
                self.world.step()
                self._capture()
        except Exception as exc:
            self.last_error = repr(exc)
            raise
        finally:
            self.is_running = False
        return {"tick": self.world.world_tick, "captured": True,
                "latest_tick": self.buffer.latest_tick}

    def run_ticks(self, n: int) -> Dict[str, Any]:
        return self.step(n)

    def pause(self) -> Dict[str, Any]:
        self.is_running = False
        return {"running": False, "tick": self.world.world_tick}

    # -- checkpoint / resume (faithful pickle of the whole world) ----------
    def save_checkpoint(self, path: str, *, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Save the full world (incl. external market) at the current tick — resumable later."""
        from environments.org_env.runtime_adapter.checkpoint import save_world_checkpoint
        return save_world_checkpoint(self.world, path, meta={
            "seed": self.seed, "approval_mode": self.approval_mode,
            "load_llm": self.load_llm,
            "experiment_condition": getattr(
                self.world, "experiment_condition", None
            ),
            "action_selection_mode": getattr(
                self.world, "action_selection_mode", None
            ),
            **(meta or {})})

    def load_checkpoint(
        self,
        path: str,
        *,
        reattach_llm: Optional[bool] = None,
        expected_target_tick: Optional[int] = None,
        remaining_ticks: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Replace the live world with a checkpointed one, re-attach the LLM client, and
        re-capture an initial frame so the inspector shows the resumed state immediately."""
        from environments.org_env.runtime_adapter.checkpoint import load_world_checkpoint
        want_llm = self.load_llm if reattach_llm is None else reattach_llm
        world, info = load_world_checkpoint(
            path,
            load_llm=want_llm,
            expected_target_tick=expected_target_tick,
        )
        from environments.org_env.product.substrates.eval_assets import (
            validate_formal_oss_world,
        )

        validate_formal_oss_world(
            world,
            require_formal=(
                (os.environ.get("ORG_OSS_MODE", "") or "").strip() == "formal"
            ),
        )
        self.world = world
        try:
            self.world.set_approval_mode(self.approval_mode)
        except Exception:
            pass
        self.world.ensure_action_selection_ready()
        self.seed = info.get("seed") or self.seed
        self.buffer = OrgLiveFrameBuffer(max_frames=self.buffer.max_frames)
        self.is_running = False
        self._capture()
        return {
            "loaded": True,
            "tick": self.world.world_tick,
            "from": os.path.abspath(path),
            "checkpoint_tick": info.get("tick"),
            "checkpoint_created": info.get("created"),
            "case_plan_fingerprint": info.get("case_plan_fingerprint"),
            "target_tick": info.get("target_tick"),
            "source_provenance_fingerprint": info.get(
                "source_provenance_fingerprint"
            ),
            "model_binding_fingerprint": info.get(
                "model_binding_fingerprint"
            ),
            "resource_budget_fingerprint": info.get(
                "resource_budget_fingerprint"
            ),
        }

    # -- views -------------------------------------------------------------
    def full(self) -> Dict[str, Any]:
        if self.buffer.frames:
            return self.buffer.frames[-1]
        return self._capture()

    def frames_since(self, since: int = -1) -> Dict[str, Any]:
        frames = self.buffer.frames_since(since)
        return {"frames": frames, "latest_tick": self.buffer.latest_tick,
                "running": self.is_running, "mode": "live"}

    def state(self) -> Dict[str, Any]:
        f = self.full()
        return {"tick": f["tick"], "day": f["day"], "hour": f["hour"], "phase": f["phase"],
                "running": self.is_running, "company": f["company"], "agents": list(f["agents"].keys()),
                "last_error": self.last_error, **self.runtime_metadata()}

    def runtime_metadata(self) -> Dict[str, Any]:
        """Non-sensitive identity facts for a live UI.

        The browser should be able to distinguish a real ``OrgWorld`` event
        stream from static presentation copy, and model-backed autonomy from
        the deterministic/rule-template fallback.  This deliberately reports
        no provider configuration, prompts, keys, or private evaluator data.
        """
        world = self.world
        llm_attached = bool(getattr(world, "llm_client", None))
        return {
            "engine": "org_env.OrgWorld",
            "event_source": "live_org_world_seat_filtered_projection",
            "world_tick": int(getattr(world, "world_tick", 0)),
            "llm": {
                "requested": bool(self.load_llm),
                "attached": llm_attached,
                "mode": "model_backed" if llm_attached else "rule_template_only",
                "load_error": bool(getattr(world, "llm_client_load_error", None)),
            },
            "policy_mode": str(getattr(world, "policy_mode", self.policy_mode)),
            "action_selection_mode": str(
                getattr(world, "action_selection_mode", "unknown")
            ),
        }

    # -- replay ------------------------------------------------------------
    def to_replay(self, name: str = "org_run", extra_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        from environments.org_env.experiments.resources import experiment_resource_snapshot
        agents = sorted({a for f in self.buffer.frames for a in (f.get("agents") or {})})
        meta = {"name": name, "scenario": "org_default", "seed": self.seed,
                "created": self.created, "ticks": len(self.buffer.frames),
                "agents": agents,
                "experiment_condition": getattr(
                    self.world, "experiment_condition", None
                ),
                "action_selection_mode": getattr(
                    self.world, "action_selection_mode", None
                ),
                "mechanism_ablations": getattr(
                    self.world, "mechanism_ablations", None
                ).to_dict() if getattr(
                    self.world, "mechanism_ablations", None
                ) is not None else None,
                "experiment_resources": experiment_resource_snapshot(self.world)}

        # Add extra metadata (user_id, hci_logs, etc.)
        if extra_meta:
            meta.update(extra_meta)

        return {"meta": meta, "frames": self.buffer.frames}

    def to_replay_delta(self, name: str = "org_run", extra_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Compact replay: one base frame + per-tick structural deltas (frontend spec —
        replay size). Reconstructs byte-identically via replay_delta.expand_delta_replay."""
        from environments.org_env.runtime_adapter.replay_delta import build_delta_replay
        full = self.to_replay(name, extra_meta=extra_meta)
        return build_delta_replay(full["meta"], full["frames"])

    def save_replay(self, path: str, name: str = "org_run", *, delta: bool = True, extra_meta: Optional[Dict[str, Any]] = None) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = self.to_replay_delta(name, extra_meta=extra_meta) if delta else self.to_replay(name, extra_meta=extra_meta)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        return path


__all__ = ["OrgLiveFrameBuffer", "OrgInspectorSession"]
