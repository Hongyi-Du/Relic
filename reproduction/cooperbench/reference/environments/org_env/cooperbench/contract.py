"""Fail-closed data contract for the two-member CooperBench treatment."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


CONTRACT_SCHEMA_VERSION = "orgenv_cooperbench_b3_two_agent_v3"
TREATMENT_ID = "relic_cooperbench_b3_two_agent"
DELIVERY_MODE = "identical_joint_mainline"
ADAPTER_NAME = "orgenv_b3_two_agent"


class CooperContractError(ValueError):
    """The caller did not supply the preregistered two-member treatment."""


def _required_text(value: Any, name: str) -> str:
    rendered = str(value or "").strip()
    if not rendered:
        raise CooperContractError(f"{name}_required")
    return rendered


def _bounded_int(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise CooperContractError(f"{name}_invalid") from error
    if parsed < minimum or parsed > maximum:
        raise CooperContractError(f"{name}_out_of_range:{minimum}:{maximum}")
    return parsed


@dataclass(frozen=True)
class FeatureCall:
    """One of CooperBench's two concurrent calls for a feature pair."""

    run_id: str
    agent_id: str
    agents: tuple[str, ...]
    task: str
    image: str
    model_name: str
    backend: str
    log_dir: str
    ticks: int = 168
    seed: int = 6101
    provider: str = "openai"
    reasoning_effort: str = "high"
    worker_timeout_seconds: int = 28_800
    max_surface_files: int = 16_384
    max_surface_bytes: int = 1024 * 1024 * 1024

    @classmethod
    def from_adapter_call(
        cls,
        *,
        task: str,
        image: str,
        agent_id: str,
        agents: list[str] | tuple[str, ...] | None,
        model_name: str,
        config: Mapping[str, Any] | None,
        log_dir: str | None,
    ) -> "FeatureCall":
        cfg = dict(config or {})
        raw_agents = tuple(str(item).strip() for item in (agents or ()))
        return cls(
            run_id=_required_text(cfg.get("run_id"), "run_id"),
            agent_id=_required_text(agent_id, "agent_id"),
            agents=raw_agents,
            task=_required_text(task, "task"),
            image=_required_text(image, "image"),
            model_name=_required_text(model_name, "model_name"),
            backend=_required_text(cfg.get("backend", "docker"), "backend").lower(),
            log_dir=_required_text(log_dir, "log_dir"),
            ticks=_bounded_int(
                cfg.get("orgenv_ticks", 168), "orgenv_ticks", minimum=24, maximum=1000
            ),
            seed=_bounded_int(
                cfg.get("orgenv_seed", 6101), "orgenv_seed", minimum=0, maximum=2**31 - 1
            ),
            provider=_required_text(
                cfg.get("orgenv_provider", "openai"), "orgenv_provider"
            ).lower(),
            reasoning_effort=_required_text(
                cfg.get("orgenv_reasoning_effort", "high"),
                "orgenv_reasoning_effort",
            ).lower(),
            worker_timeout_seconds=_bounded_int(
                cfg.get("orgenv_worker_timeout_seconds", 28_800),
                "orgenv_worker_timeout_seconds",
                minimum=60,
                maximum=86_400,
            ),
            max_surface_files=_bounded_int(
                cfg.get("orgenv_max_surface_files", 16_384),
                "orgenv_max_surface_files",
                minimum=24,
                maximum=32_768,
            ),
            max_surface_bytes=_bounded_int(
                cfg.get("orgenv_max_surface_bytes", 1024 * 1024 * 1024),
                "orgenv_max_surface_bytes",
                minimum=256 * 1024,
                maximum=1024 * 1024 * 1024,
            ),
        ).validated()

    def validated(self) -> "FeatureCall":
        if self.backend != "docker":
            raise CooperContractError(
                f"docker_backend_required_for_image_extraction:{self.backend}"
            )
        if len(self.agents) != 2 or len(set(self.agents)) != 2:
            raise CooperContractError("exactly_two_unique_agents_required")
        if self.agent_id not in self.agents:
            raise CooperContractError("agent_id_not_in_pair")
        if self.provider not in {"openai", "http", "generic"}:
            raise CooperContractError(f"unsupported_orgenv_provider:{self.provider}")
        if self.reasoning_effort not in {
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
        }:
            raise CooperContractError(
                f"unsupported_orgenv_reasoning_effort:{self.reasoning_effort}"
            )
        path = Path(self.log_dir).expanduser()
        if not path.is_absolute():
            raise CooperContractError("absolute_log_dir_required")
        return self

    def pair_signature(self) -> tuple[Any, ...]:
        """Fields which must be byte-for-byte common to both feature calls."""

        return (
            self.run_id,
            self.agents,
            self.image,
            self.model_name,
            self.backend,
            str(Path(self.log_dir).resolve()),
            self.ticks,
            self.seed,
            self.provider,
            self.reasoning_effort,
            self.worker_timeout_seconds,
            self.max_surface_files,
            self.max_surface_bytes,
        )


@dataclass(frozen=True)
class PairRequest:
    schema_version: str
    treatment_id: str
    delivery_mode: str
    run_id: str
    agents: tuple[str, str]
    tasks: Mapping[str, str]
    image: str
    model_name: str
    backend: str
    log_dir: str
    ticks: int
    seed: int
    provider: str
    reasoning_effort: str
    max_surface_files: int
    max_surface_bytes: int

    @classmethod
    def from_calls(cls, calls: Mapping[str, FeatureCall]) -> "PairRequest":
        if len(calls) != 2:
            raise CooperContractError("pair_request_requires_two_calls")
        first = next(iter(calls.values())).validated()
        if set(calls) != set(first.agents):
            raise CooperContractError("pair_call_agents_mismatch")
        for call in calls.values():
            if call.validated().pair_signature() != first.pair_signature():
                raise CooperContractError("pair_execution_config_mismatch")
        ordered_agents = (first.agents[0], first.agents[1])
        return cls(
            schema_version=CONTRACT_SCHEMA_VERSION,
            treatment_id=TREATMENT_ID,
            delivery_mode=DELIVERY_MODE,
            run_id=first.run_id,
            agents=ordered_agents,
            tasks={agent: calls[agent].task for agent in ordered_agents},
            image=first.image,
            model_name=first.model_name,
            backend=first.backend,
            log_dir=str(Path(first.log_dir).resolve()),
            ticks=first.ticks,
            seed=first.seed,
            provider=first.provider,
            reasoning_effort=first.reasoning_effort,
            max_surface_files=first.max_surface_files,
            max_surface_bytes=first.max_surface_bytes,
        ).validated()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PairRequest":
        agents = tuple(str(item) for item in (payload.get("agents") or ()))
        if len(agents) != 2:
            raise CooperContractError("pair_request_requires_two_agents")
        request = cls(
            schema_version=str(payload.get("schema_version") or ""),
            treatment_id=str(payload.get("treatment_id") or ""),
            delivery_mode=str(payload.get("delivery_mode") or ""),
            run_id=str(payload.get("run_id") or ""),
            agents=(agents[0], agents[1]),
            tasks=dict(payload.get("tasks") or {}),
            image=str(payload.get("image") or ""),
            model_name=str(payload.get("model_name") or ""),
            backend=str(payload.get("backend") or ""),
            log_dir=str(payload.get("log_dir") or ""),
            ticks=int(payload.get("ticks") or 0),
            seed=int(payload.get("seed") or 0),
            provider=str(payload.get("provider") or ""),
            reasoning_effort=str(payload.get("reasoning_effort") or ""),
            max_surface_files=int(payload.get("max_surface_files") or 0),
            max_surface_bytes=int(payload.get("max_surface_bytes") or 0),
        )
        return request.validated()

    def validated(self) -> "PairRequest":
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise CooperContractError("pair_request_schema_mismatch")
        if self.treatment_id != TREATMENT_ID:
            raise CooperContractError("pair_request_treatment_mismatch")
        if self.delivery_mode != DELIVERY_MODE:
            raise CooperContractError("pair_request_delivery_mode_mismatch")
        if len(set(self.agents)) != 2 or set(self.tasks) != set(self.agents):
            raise CooperContractError("pair_request_agent_task_bijection_required")
        if any(not str(self.tasks[agent]).strip() for agent in self.agents):
            raise CooperContractError("pair_request_task_required")
        if self.backend != "docker":
            raise CooperContractError("pair_request_docker_backend_required")
        for name, value in (
            ("run_id", self.run_id),
            ("image", self.image),
            ("model_name", self.model_name),
            ("log_dir", self.log_dir),
        ):
            _required_text(value, name)
        if not Path(self.log_dir).is_absolute():
            raise CooperContractError("pair_request_absolute_log_dir_required")
        if not 24 <= int(self.ticks) <= 1000:
            raise CooperContractError("pair_request_ticks_out_of_range")
        if not 24 <= int(self.max_surface_files) <= 32_768:
            raise CooperContractError("pair_request_surface_file_limit_invalid")
        if not 256 * 1024 <= int(self.max_surface_bytes) <= 1024 * 1024 * 1024:
            raise CooperContractError("pair_request_surface_byte_limit_invalid")
        return self

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["agents"] = list(self.agents)
        payload["tasks"] = dict(self.tasks)
        return payload


@dataclass(frozen=True)
class PairOutcome:
    schema_version: str
    treatment_id: str
    delivery_mode: str
    run_id: str
    agents: tuple[str, str]
    status: str
    joint_patch: str
    usage_totals: Mapping[str, int] = field(default_factory=dict)
    communications_by_agent: Mapping[str, list[dict[str, Any]]] = field(
        default_factory=dict
    )
    receipt: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PairOutcome":
        agents = tuple(str(item) for item in (payload.get("agents") or ()))
        if len(agents) != 2:
            raise CooperContractError("pair_outcome_requires_two_agents")
        outcome = cls(
            schema_version=str(payload.get("schema_version") or ""),
            treatment_id=str(payload.get("treatment_id") or ""),
            delivery_mode=str(payload.get("delivery_mode") or ""),
            run_id=str(payload.get("run_id") or ""),
            agents=(agents[0], agents[1]),
            status=str(payload.get("status") or ""),
            joint_patch=str(payload.get("joint_patch") or ""),
            usage_totals=dict(payload.get("usage_totals") or {}),
            communications_by_agent=dict(
                payload.get("communications_by_agent") or {}
            ),
            receipt=dict(payload.get("receipt") or {}),
            error=(str(payload.get("error")) if payload.get("error") else None),
        )
        return outcome.validated()

    def validated(self) -> "PairOutcome":
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise CooperContractError("pair_outcome_schema_mismatch")
        if self.treatment_id != TREATMENT_ID:
            raise CooperContractError("pair_outcome_treatment_mismatch")
        if self.delivery_mode != DELIVERY_MODE:
            raise CooperContractError("pair_outcome_delivery_mode_mismatch")
        if len(set(self.agents)) != 2:
            raise CooperContractError("pair_outcome_unique_agents_required")
        if self.status not in {"Submitted", "Error"}:
            raise CooperContractError(f"pair_outcome_status_invalid:{self.status}")
        if self.status == "Submitted" and not self.joint_patch.strip():
            raise CooperContractError("submitted_pair_outcome_patch_empty")
        if self.status == "Error" and not self.error:
            raise CooperContractError("error_pair_outcome_reason_required")
        return self

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["agents"] = list(self.agents)
        payload["usage_totals"] = dict(self.usage_totals)
        payload["communications_by_agent"] = dict(self.communications_by_agent)
        payload["receipt"] = dict(self.receipt)
        return payload


__all__ = [
    "ADAPTER_NAME",
    "CONTRACT_SCHEMA_VERSION",
    "DELIVERY_MODE",
    "TREATMENT_ID",
    "CooperContractError",
    "FeatureCall",
    "PairOutcome",
    "PairRequest",
]
