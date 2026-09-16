"""Compile one canonical Relic main-study cell from frozen configuration.

The compiled payload is deliberately independent of timestamps, host memory,
output paths, and credentials.  Its digest is therefore suitable for checkpoint
identity and resume validation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import environments

from environments.org_env.config.baseline_conditions import resolve_condition
from environments.org_env.experiments.resources import FrozenResourceBudget
from relic.benchmark import benchmark_directory, load_benchmark_manifest, tree_digest
from relic.manifest import load_yaml
from relic.paths import config_root, default_output_root, project_root

CELL_SPEC_SCHEMA_VERSION = "relic-cell-spec-v1"
SOURCE_BRANCH = "hci-human-seat"
SOURCE_COMMIT = "dda36fb563375060ae8d8850300db01eb4695d29"
SOURCE_GIT_TREE = "d7276c13312b13d4a030d82c3accc253349d708d"
SOURCE_REPOSITORY = "https://github.com/Hongyi-Du/SocioGenesis"
MECHANISM_ABLATIONS = ("work_rhythm",)

_STUDY_KEYS = frozenset(
    {
        "schema_version",
        "study",
        "models",
        "workloads",
        "seeds",
        "arms",
        "ticks",
        "checkpoint_every",
        "sprint_ticks",
        "work_rhythm_enabled",
        "approval_mode",
        "resource_policy",
        "condition_invariants",
        "formal_evaluator_policy",
        "aggregation",
    }
)
_ARM_KEYS = frozenset(
    {
        "schema_version",
        "arm",
        "paper_label",
        "roster_size",
        "action_selection",
        "profile_conditioning",
        "capability_learning",
        "institutionalization",
        "runtime_protocol_binding",
        "protocol_masked_preselection",
        "temporary_team",
        "profile_assignment",
        "private_memory_and_appraisal",
        "shared_workspace_channels_meetings",
        "organization_reflection_to_institution_path",
    }
)
_MODEL_KEYS = frozenset(
    {
        "schema_version",
        "model",
        "paper_label",
        "provider",
        "runtime_model_default",
        "runtime_model_env",
        "wire_api",
        "json_transport",
        "reasoning_effort",
        "request_timeout_seconds",
        "max_retries",
        "retry_backoff_seconds",
    }
)
_WORKLOAD_KEYS = frozenset(
    {
        "schema_version",
        "workload",
        "paper_label",
        "category",
        "paper_family",
        "paper_status",
        "frozen_pack_id",
        "benchmark_path",
        "manifest_path",
        "scoring_units",
    }
)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def stable_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


@lru_cache(maxsize=1)
def release_runtime_tree() -> dict[str, Any]:
    """Fingerprint the exact release runtime/config bytes used by a cell.

    Upstream provenance says where the implementation was distilled from; this
    digest separately binds what Python and canonical YAML bytes actually run.
    """

    roots = (
        ("relic", Path(__file__).resolve().parent, {".py"}),
        ("environments", Path(environments.__file__).resolve().parent, {".py"}),
        ("configs", config_root(), {".yaml", ".yml", ".json"}),
    )
    records: list[tuple[str, str]] = []
    for label, root, suffixes in roots:
        if not root.is_dir():
            raise ValueError(f"release_runtime_root_missing:{label}")
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
            if path.suffix not in suffixes or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(root).as_posix()
            records.append((f"{label}/{relative}", hashlib.sha256(path.read_bytes()).hexdigest()))
    for name in ("pyproject.toml", "uv.lock"):
        path = project_root() / name
        if path.is_file():
            records.append((name, hashlib.sha256(path.read_bytes()).hexdigest()))
    benchmark_manifest = benchmark_directory() / "manifest.yaml"
    records.append(
        (
            "benchmarks/relic-main-v1/manifest.yaml",
            hashlib.sha256(benchmark_manifest.read_bytes()).hexdigest(),
        )
    )
    digest = hashlib.sha256()
    for relative, file_hash in sorted(records):
        digest.update(relative.encode("utf-8") + b"\0" + file_hash.encode("ascii") + b"\n")
    return {
        "algorithm": "sha256 over sorted relative-path, NUL, file-sha256, newline",
        "file_count": len(records),
        "tree_sha256": digest.hexdigest(),
    }


def _json_copy(payload: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(canonical_json_bytes(dict(payload)))


def _strict_keys(
    payload: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    required: frozenset[str],
    label: str,
) -> None:
    observed = set(payload)
    missing = sorted(required - observed)
    unknown = sorted(observed - allowed)
    if missing:
        raise ValueError(f"{label}_missing_fields:{','.join(missing)}")
    if unknown:
        raise ValueError(f"{label}_unknown_fields:{','.join(unknown)}")


def _normalized_id(value: str, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if not normalized:
        raise ValueError(f"{label}_empty")
    return normalized


def _validate_arm_config(arm: str, payload: Mapping[str, Any]) -> str:
    required = _ARM_KEYS - {"protocol_masked_preselection"}
    _strict_keys(payload, allowed=_ARM_KEYS, required=required, label=f"arm_{arm}")
    if payload.get("schema_version") != "relic-arm-v1":
        raise ValueError(f"arm_{arm}_schema_version_mismatch")
    if _normalized_id(str(payload.get("arm") or ""), label="arm") != arm:
        raise ValueError(f"arm_{arm}_identity_mismatch")

    condition = resolve_condition(arm)
    comparisons = {
        "roster_size": condition.roster_size,
        "action_selection": condition.action_selection_mode,
        "profile_conditioning": condition.profile_conditioning_enabled,
        "capability_learning": condition.capability_learning_enabled,
        "institutionalization": condition.institutionalization_enabled,
        "temporary_team": condition.temporary_team,
        "profile_assignment": condition.profile_assignment,
        "runtime_protocol_binding": condition.institutionalization_enabled,
        "organization_reflection_to_institution_path": (
            condition.institutionalization_enabled
        ),
        "private_memory_and_appraisal": True,
        "shared_workspace_channels_meetings": condition.roster_size > 1,
        "protocol_masked_preselection": False,
    }
    for field, expected in comparisons.items():
        observed = payload.get(field, False)
        if observed != expected:
            raise ValueError(
                f"arm_{arm}_{field}_mismatch:expected={expected!r}:observed={observed!r}"
            )
    return condition.condition_id


def _validate_study(payload: Mapping[str, Any]) -> None:
    _strict_keys(payload, allowed=_STUDY_KEYS, required=_STUDY_KEYS, label="main_study")
    if payload.get("schema_version") != "relic-main-study-v1":
        raise ValueError("main_study_schema_version_mismatch")
    if payload.get("study") != "relic-main-v1":
        raise ValueError("main_study_identity_mismatch")
    if bool(payload.get("work_rhythm_enabled")):
        raise ValueError("main_study_work_rhythm_must_be_disabled")
    if payload.get("approval_mode") != "semi_auto":
        raise ValueError("main_study_approval_mode_mismatch")
    if int(payload.get("ticks", 0)) <= 0:
        raise ValueError("main_study_ticks_invalid")
    checkpoint_every = int(payload.get("checkpoint_every", 0))
    if checkpoint_every <= 0 or int(payload["ticks"]) % checkpoint_every:
        raise ValueError("main_study_checkpoint_cadence_invalid")
    invariants = payload.get("condition_invariants")
    expected_invariants = {
        "candidate_feature_execution_implementations": "shared",
        "action_registry_and_tool_surface": "same",
        "model_decoding_and_retries": "same",
        "task_visible_information": "same",
        "private_memory_and_appraisal": "enabled",
        "shared_workspace_channels_meetings": {
            "b0": "not_applicable",
            "b1": "enabled",
            "b2": "enabled",
            "b3": "enabled",
        },
        "persistent_roster_for_whole_run": {
            "b0": "not_applicable",
            "b1": "enabled",
            "b2": "enabled",
            "b3": "enabled",
        },
    }
    if invariants != expected_invariants:
        raise ValueError("main_study_condition_invariants_mismatch")
    evaluator = payload.get("formal_evaluator_policy")
    if not isinstance(evaluator, Mapping):
        raise ValueError("main_study_evaluator_policy_invalid")
    evaluator_expectations = {
        "allowed_backends": ["docker", "apptainer"],
        "trust_level": "untrusted",
        "network_enabled": False,
        "platform": "linux/amd64",
        "digest_pinned_image_required": True,
        "image_environment_variable": "RELIC_EVALUATOR_CONTAINER_IMAGE",
        "paper_runtime_environment_variable": "ORG_EVALUATOR_CONTAINER_IMAGE",
    }
    for field, expected in evaluator_expectations.items():
        if evaluator.get(field) != expected:
            raise ValueError(f"main_study_evaluator_{field}_mismatch")


def _load_named_config(directory: str, name: str) -> dict[str, Any]:
    path = config_root() / directory / f"{name}.yaml"
    return load_yaml(path)


def _selected_benchmark_row(workload: str, pack: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_benchmark_manifest()
    if manifest.get("schema_version") != "relic-benchmark-manifest-v1":
        raise ValueError("benchmark_manifest_schema_version_mismatch")
    source = {
        "repository": manifest.get("source_repository"),
        "branch": manifest.get("source_branch"),
        "commit": manifest.get("source_commit"),
    }
    expected_source = {
        "repository": SOURCE_REPOSITORY,
        "branch": SOURCE_BRANCH,
        "commit": SOURCE_COMMIT,
        "git_tree": SOURCE_GIT_TREE,
    }
    if source != {key: expected_source[key] for key in ("repository", "branch", "commit")}:
        raise ValueError("benchmark_source_provenance_mismatch")
    rows = [
        row
        for row in manifest.get("workloads", [])
        if isinstance(row, Mapping) and str(row.get("id") or "").lower() == workload
    ]
    if len(rows) != 1:
        raise ValueError(f"benchmark_workload_binding_invalid:{workload}")
    row = dict(rows[0])
    if row.get("pack") != pack:
        raise ValueError(f"benchmark_pack_binding_mismatch:{workload}")
    return _json_copy(row), expected_source


@dataclass(frozen=True)
class CellSpec:
    output_root: Path
    study: str
    model: str
    workload: str
    arm: str
    seed: int
    dataset_id: str
    condition_id: str
    ticks: int
    checkpoint_every: int
    sprint_ticks: int
    approval_mode: str
    mechanism_ablations: tuple[str, ...]
    study_config: dict[str, Any]
    model_config: dict[str, Any]
    workload_config: dict[str, Any]
    arm_config: dict[str, Any]
    benchmark_entry: dict[str, Any]
    source: dict[str, Any]
    runtime_source: dict[str, Any]

    @property
    def cell_id(self) -> str:
        return f"{self.model}__{self.workload.upper()}__{self.arm.upper()}__seed{self.seed}"

    @property
    def run_id(self) -> str:
        return f"{self.study}__{self.cell_id}"

    @property
    def cell_dir(self) -> Path:
        return (
            self.output_root
            / self.study
            / self.model
            / self.workload
            / self.arm
            / f"seed-{self.seed}"
        )

    @property
    def resource_budget(self) -> FrozenResourceBudget:
        return FrozenResourceBudget(max_ticks=self.ticks)

    @property
    def model_binding_fingerprint(self) -> str:
        return stable_sha256(self.model_config)

    @property
    def source_provenance_fingerprint(self) -> str:
        return stable_sha256(
            {"upstream_source": self.source, "release_runtime": self.runtime_source}
        )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": CELL_SPEC_SCHEMA_VERSION,
            "study": self.study,
            "cell_id": self.cell_id,
            "run_id": self.run_id,
            "model": self.model,
            "workload": self.workload.upper(),
            "arm": self.arm.upper(),
            "seed": self.seed,
            "dataset_id": self.dataset_id,
            "condition_id": self.condition_id,
            "ticks": self.ticks,
            "checkpoint_every": self.checkpoint_every,
            "sprint_ticks": self.sprint_ticks,
            "approval_mode": self.approval_mode,
            "mechanism_ablations": list(self.mechanism_ablations),
            "resource_budget": self.resource_budget.to_dict(),
            "source": _json_copy(self.source),
            "runtime_source": _json_copy(self.runtime_source),
            "benchmark_entry": _json_copy(self.benchmark_entry),
            "configs": {
                "study": _json_copy(self.study_config),
                "model": _json_copy(self.model_config),
                "workload": _json_copy(self.workload_config),
                "arm": _json_copy(self.arm_config),
            },
        }

    @property
    def fingerprint(self) -> str:
        return stable_sha256(self.canonical_payload())

    def document(self) -> dict[str, Any]:
        return {**self.canonical_payload(), "cell_spec_sha256": self.fingerprint}


def compile_cell_spec(
    *,
    model: str,
    workload: str,
    arm: str,
    seed: int,
    output_root: Path | None = None,
    verify_pack: bool = True,
) -> CellSpec:
    """Compile and validate one of the paper's 120 canonical cells."""

    model_id = _normalized_id(model, label="model")
    workload_id = _normalized_id(workload, label="workload")
    arm_id = _normalized_id(arm, label="arm")
    study = load_yaml(config_root() / "main-study.yaml")
    _validate_study(study)

    allowed_models = tuple(str(value).lower() for value in study["models"])
    allowed_workloads = tuple(str(value).lower() for value in study["workloads"])
    allowed_arms = tuple(str(value).lower() for value in study["arms"])
    allowed_seeds = tuple(int(value) for value in study["seeds"])
    if model_id not in allowed_models:
        raise ValueError(f"model_not_in_main_study:{model_id}")
    if workload_id not in allowed_workloads:
        raise ValueError(f"workload_not_in_main_study:{workload_id}")
    if arm_id not in allowed_arms:
        raise ValueError(f"arm_not_in_main_study:{arm_id}")
    if int(seed) not in allowed_seeds:
        raise ValueError(f"seed_not_in_main_study:{seed}")

    model_config = _load_named_config("models", model_id)
    _strict_keys(
        model_config,
        allowed=_MODEL_KEYS,
        required=_MODEL_KEYS - {"runtime_model_default", "runtime_model_env"},
        label=f"model_{model_id}",
    )
    if model_config.get("schema_version") != "relic-model-v1":
        raise ValueError(f"model_{model_id}_schema_version_mismatch")
    if str(model_config.get("model") or "").lower() != model_id:
        raise ValueError(f"model_{model_id}_identity_mismatch")
    provider = str(model_config.get("provider") or "").strip().lower()
    if provider not in {"openai", "http", "mock"}:
        raise ValueError(f"model_{model_id}_provider_unsupported")
    wire_api = str(model_config.get("wire_api") or "").strip().lower().replace("-", "_")
    if wire_api not in {"chat_completions", "responses"}:
        raise ValueError(f"model_{model_id}_wire_api_invalid")
    json_transport = (
        str(model_config.get("json_transport") or "")
        .strip()
        .lower()
        .replace("-", "_")
    )
    if json_transport not in {"native", "prompt_only"}:
        raise ValueError(f"model_{model_id}_json_transport_invalid")
    runtime_fields = [
        field
        for field in ("runtime_model_default", "runtime_model_env")
        if str(model_config.get(field) or "").strip()
    ]
    if len(runtime_fields) != 1:
        raise ValueError(f"model_{model_id}_runtime_binding_invalid")

    workload_config = _load_named_config("workloads", workload_id)
    _strict_keys(
        workload_config,
        allowed=_WORKLOAD_KEYS,
        required=_WORKLOAD_KEYS,
        label=f"workload_{workload_id}",
    )
    if workload_config.get("schema_version") != "relic-workload-v1":
        raise ValueError(f"workload_{workload_id}_schema_version_mismatch")
    if str(workload_config.get("workload") or "").lower() != workload_id:
        raise ValueError(f"workload_{workload_id}_identity_mismatch")
    dataset_id = str(workload_config.get("frozen_pack_id") or "").strip()
    expected_benchmark_path = f"benchmarks/relic-main-v1/packs/{dataset_id}"
    if workload_config.get("benchmark_path") != expected_benchmark_path:
        raise ValueError(f"workload_{workload_id}_benchmark_path_mismatch")
    if workload_config.get("manifest_path") != f"{expected_benchmark_path}/manifest.yaml":
        raise ValueError(f"workload_{workload_id}_manifest_path_mismatch")

    arm_config = _load_named_config("arms", arm_id)
    condition_id = _validate_arm_config(arm_id, arm_config)
    if int(study["sprint_ticks"]) != resolve_condition(arm_id).default_sprint_ticks:
        raise ValueError(f"arm_{arm_id}_sprint_ticks_mismatch")

    benchmark_entry, source = _selected_benchmark_row(workload_id, dataset_id)
    if verify_pack:
        pack_path = benchmark_directory() / "packs" / dataset_id
        count, digest = tree_digest(pack_path)
        if count != int(benchmark_entry["file_count"]):
            raise ValueError(f"benchmark_pack_file_count_mismatch:{dataset_id}")
        if digest != benchmark_entry["tree_sha256"]:
            raise ValueError(f"benchmark_pack_tree_digest_mismatch:{dataset_id}")

    return CellSpec(
        output_root=(output_root or default_output_root()).expanduser().resolve(),
        study=str(study["study"]),
        model=model_id,
        workload=workload_id,
        arm=arm_id,
        seed=int(seed),
        dataset_id=dataset_id,
        condition_id=condition_id,
        ticks=int(study["ticks"]),
        checkpoint_every=int(study["checkpoint_every"]),
        sprint_ticks=int(study["sprint_ticks"]),
        approval_mode=str(study["approval_mode"]),
        mechanism_ablations=MECHANISM_ABLATIONS,
        study_config=_json_copy(study),
        model_config=_json_copy(model_config),
        workload_config=_json_copy(workload_config),
        arm_config=_json_copy(arm_config),
        benchmark_entry=benchmark_entry,
        source=source,
        runtime_source=release_runtime_tree(),
    )


def load_frozen_cell_spec(cell_dir: Path) -> CellSpec:
    """Load a frozen spec and prove it still matches canonical repository config."""

    path = Path(cell_dir).expanduser().resolve() / "cell-spec.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != CELL_SPEC_SCHEMA_VERSION:
        raise ValueError("cell_spec_document_invalid")
    declared_hash = str(payload.get("cell_spec_sha256") or "")
    canonical = {key: value for key, value in payload.items() if key != "cell_spec_sha256"}
    if stable_sha256(canonical) != declared_hash:
        raise ValueError("cell_spec_document_hash_mismatch")
    compiled = compile_cell_spec(
        model=str(payload.get("model") or ""),
        workload=str(payload.get("workload") or ""),
        arm=str(payload.get("arm") or ""),
        seed=int(payload.get("seed")),
        output_root=Path(cell_dir).expanduser().resolve(),
    )
    if compiled.fingerprint != declared_hash:
        raise ValueError("cell_spec_no_longer_matches_canonical_config")
    return compiled


__all__ = [
    "CELL_SPEC_SCHEMA_VERSION",
    "CellSpec",
    "MECHANISM_ABLATIONS",
    "SOURCE_BRANCH",
    "SOURCE_COMMIT",
    "SOURCE_GIT_TREE",
    "canonical_json_bytes",
    "compile_cell_spec",
    "load_frozen_cell_spec",
    "release_runtime_tree",
    "stable_sha256",
]
