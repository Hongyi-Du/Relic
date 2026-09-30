"""Post-rollout OSS evaluation and experiment-record persistence.

The candidate workspace is materialized only after the rollout has stopped.  Hidden
results stay in evaluator-owned files and are never attached to the simulation world.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from environments.org_env.product.substrates.eval_assets import (
    oss_eval_assets,
    oss_evaluator_config,
    resolve_oss_evaluator_spec,
    validate_formal_oss_world,
)
from society_core.hashing import canonicalize, stable_hash
from society_core.code_landing.event_store import redact_sensitive_payload
from society_core.safe_files import read_regular_file_text

FINAL_EVIDENCE_SCHEMA_VERSION = "orgenv_oss_final_evaluation_v1"
PROGRAMBENCH_DUAL_EVALUATION_SCHEMA_VERSION = (
    "programbench_sealed_dual_evaluation_v3"
)
PROGRAMBENCH_PRIMARY_SELECTION_SCHEMA_VERSION = (
    "programbench_public_primary_selection_v1"
)
_MANUAL_STATUSES = frozenset({"passed", "failed", "infra_error"})
_PLAN_HASH_FIELDS = (
    "dataset_id",
    "product_name",
    "starter_ref",
    "reference_ref",
    "starter_repo_digest",
    "reference_repo_digest",
    "hidden_suite_hash",
    "evaluator_environment_hash",
    "required_release_coverage",
    "covered_release_versions",
    "oracles",
    "public_contracts",
    "operational_ready",
    "formal_environment_ready",
    "blocking_reasons",
)
_EVIDENCE_HASH_FIELDS = (
    "task_id",
    "command",
    "owner",
    "kind",
    "base_status",
    "candidate_status",
    "required",
    "baseline_repo_digest",
    "candidate_repo_digest",
    "execution_policy_hash",
    "exit_code",
    "stdout_hash",
    "stderr_hash",
)

ManualCheckRunner = Callable[[Path, str, Mapping[str, Any]], Mapping[str, Any]]
_MAX_FINAL_EVIDENCE_BYTES = 20_000_000
_PINNED_IMAGE = re.compile(r"^.+@sha256:[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class FinalEvaluationArtifact:
    """Validated evaluator evidence, newly created or content-addressedly reused."""

    path: Path
    payload: dict[str, Any]
    reused: bool

    @property
    def result(self) -> Mapping[str, Any]:
        value = self.payload.get("result")
        return value if isinstance(value, Mapping) else {}


def manifest_manual_release_checks(
    manifest: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], ...]:
    """Return validated checks declared by ``evaluation.manual_release_checks``."""

    evaluation = (manifest or {}).get("evaluation") or {}
    raw = evaluation.get("manual_release_checks") or ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise RuntimeError("evaluation.manual_release_checks must be a list")
    checks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise RuntimeError("manual release check declaration must be an object")
        check_id = str(item.get("check_id") or "").strip()
        if not check_id:
            raise RuntimeError("manual release check declaration has no check_id")
        if check_id in seen:
            raise RuntimeError(f"duplicate manual release check: {check_id}")
        seen.add(check_id)
        checks.append(dict(item))
    return tuple(checks)


def manifest_requires_manual_release_checks(
    manifest: Mapping[str, Any] | None,
) -> bool:
    """Whether this pack declares any evaluator-owned manual release checks."""

    return bool(manifest_manual_release_checks(manifest))


def run_docker_launch_check(
    candidate_root: Path,
    candidate_digest: str,
    declaration: Mapping[str, Any],
) -> dict[str, Any]:
    """Build, restricted-run, and probe ``/health`` for a declared Docker check."""

    check_id = str(declaration.get("check_id") or "docker_launch")
    container_name = f"sociogenesis-oss-{candidate_digest[:12]}-{os.getpid()}"
    image_id = ""
    environment = _docker_environment()
    try:
        dockerfile, base_images = _validate_dockerfile_build_policy(
            candidate_root,
            declaration,
        )
        for base_image in base_images:
            inspect_base = subprocess.run(
                ["docker", "image", "inspect", base_image],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                env=environment,
            )
            if inspect_base.returncode != 0:
                return {
                    "check_id": check_id,
                    "status": "infra_error",
                    "stage": "build_policy",
                    "output_tail": _bounded_redacted_output(
                        f"pinned base image not preloaded: {base_image}"
                    ),
                }
        build = subprocess.run(
            [
                "docker",
                "build",
                "--quiet",
                "--network=none",
                "--pull=false",
                "--file",
                str(dockerfile),
                str(candidate_root),
            ],
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("ORG_OSS_DOCKER_BUILD_TIMEOUT", "900")),
            check=False,
            env=environment,
        )
        if build.returncode != 0:
            return {
                "check_id": check_id,
                "status": "failed",
                "stage": "build",
                "output_tail": _bounded_redacted_output(
                    (build.stdout or "") + (build.stderr or "")
                ),
            }
        lines = (build.stdout or "").strip().splitlines()
        if not lines:
            return {
                "check_id": check_id,
                "status": "infra_error",
                "stage": "build",
                "output_tail": "docker build returned no image id",
            }
        image_id = lines[-1]
        image_user = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                "--format={{.Config.User}}",
                image_id,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=environment,
        )
        declared_user = (image_user.stdout or "").strip()
        if image_user.returncode != 0 or declared_user in {"", "0", "root"}:
            return {
                "check_id": check_id,
                "status": "failed",
                "stage": "image_policy",
                "image_id": image_id,
                "output_tail": "final image must declare a non-root user",
            }
        launch = subprocess.run(
            [
                "docker",
                "run",
                "--detach",
                "--network",
                "none",
                "--read-only",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,size=64m",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--memory",
                "512m",
                "--cpus",
                "1",
                "--pids-limit",
                "128",
                "--name",
                container_name,
                image_id,
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env=environment,
        )
        if launch.returncode != 0:
            return {
                "check_id": check_id,
                "status": "failed",
                "stage": "launch",
                "image_id": image_id,
                "output_tail": _bounded_redacted_output(
                    (launch.stdout or "") + (launch.stderr or "")
                ),
            }

        health: subprocess.CompletedProcess[str] | None = None
        for _ in range(15):
            inspect = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "--format={{.State.Running}}",
                    container_name,
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                env=environment,
            )
            if (
                inspect.returncode != 0
                or (inspect.stdout or "").strip().lower() != "true"
            ):
                break
            health = subprocess.run(
                [
                    "docker",
                    "exec",
                    container_name,
                    "python",
                    "-c",
                    (
                        "import urllib.request;"
                        "response=urllib.request.urlopen("
                        "'http://127.0.0.1:8000/health',timeout=2);"
                        "assert response.status==200;"
                        "assert b'healthy' in response.read()"
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                env=environment,
            )
            if health.returncode == 0:
                break
            time.sleep(1)
        logs = subprocess.run(
            ["docker", "logs", container_name],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=environment,
        )
        ready = health is not None and health.returncode == 0
        return {
            "check_id": check_id,
            "status": "passed" if ready else "failed",
            "stage": "health",
            "image_id": image_id,
            "output_tail": _bounded_redacted_output(
                (
                    (health.stdout or "") + (health.stderr or "")
                    if health is not None
                    else ""
                )
                + (logs.stdout or "")
                + (logs.stderr or "")
            ),
        }
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        return {
            "check_id": check_id,
            "status": "infra_error",
            "stage": "controller",
            "image_id": image_id,
            "output_tail": _bounded_redacted_output(str(exc)),
        }
    finally:
        try:
            subprocess.run(
                ["docker", "rm", "--force", container_name],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass


def _docker_environment() -> dict[str, str]:
    allowed = (
        "PATH",
        "HOME",
        "TMPDIR",
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_CONFIG",
        "DOCKER_TLS_VERIFY",
        "DOCKER_CERT_PATH",
    )
    return {
        name: value
        for name in allowed
        if (value := os.environ.get(name)) not in (None, "")
    }


def _validate_dockerfile_build_policy(
    candidate_root: Path,
    declaration: Mapping[str, Any],
) -> tuple[Path, tuple[str, ...]]:
    relative = Path(str(declaration.get("dockerfile") or "Dockerfile"))
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError("dockerfile_path_invalid")
    dockerfile = (candidate_root / relative).resolve()
    try:
        dockerfile.relative_to(candidate_root.resolve())
    except ValueError as exc:
        raise ValueError("dockerfile_path_outside_candidate") from exc
    content = read_regular_file_text(
        candidate_root,
        relative,
        max_bytes=1_000_000,
    )
    logical = content.replace("\\\n", " ")
    stages: set[str] = set()
    base_images: list[str] = []
    from_count = 0
    for raw_line in logical.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = shlex.split(line, comments=True)
        if not tokens or tokens[0].casefold() != "from":
            continue
        from_count += 1
        index = 1
        while index < len(tokens) and tokens[index].startswith("--"):
            index += 1
        if index >= len(tokens):
            raise ValueError("dockerfile_from_image_missing")
        image = tokens[index]
        if "$" in image:
            raise ValueError("dockerfile_dynamic_base_forbidden")
        if image.casefold() not in stages and image != "scratch":
            if not _PINNED_IMAGE.fullmatch(image):
                raise ValueError(f"dockerfile_base_not_digest_pinned:{image}")
            base_images.append(image)
        if index + 2 < len(tokens) and tokens[index + 1].casefold() == "as":
            stages.add(tokens[index + 2].casefold())
    if from_count == 0:
        raise ValueError("dockerfile_has_no_valid_from")
    return dockerfile, tuple(dict.fromkeys(base_images))


def _bounded_redacted_output(value: str) -> str:
    redacted = str(redact_sensitive_payload(value))
    return redacted[-2000:]


DEFAULT_MANUAL_CHECK_RUNNERS: Mapping[str, ManualCheckRunner] = {
    "docker_launch": run_docker_launch_check,
}


def _run_manual_checks(
    *,
    declarations: Sequence[Mapping[str, Any]],
    enabled: bool,
    candidate_root: Path,
    candidate_digest: str,
    runners: Mapping[str, ManualCheckRunner],
) -> list[dict[str, Any]]:
    if not enabled:
        return []
    results: list[dict[str, Any]] = []
    for declaration in declarations:
        check_id = str(declaration["check_id"])
        runner = runners.get(check_id)
        if runner is None:
            results.append(
                {
                    "check_id": check_id,
                    "status": "infra_error",
                    "stage": "controller",
                    "output_tail": f"no evaluator runner registered for {check_id}",
                }
            )
            continue
        try:
            raw = runner(candidate_root, candidate_digest, declaration)
            result = dict(raw)
        except Exception as exc:  # pragma: no cover - runner containment
            result = {
                "status": "infra_error",
                "stage": "controller",
                "output_tail": str(exc)[-2000:],
            }
        result["check_id"] = check_id
        if result.get("status") not in _MANUAL_STATUSES:
            result["status"] = "infra_error"
            result.setdefault("stage", "controller")
            result.setdefault("output_tail", "manual check returned an invalid status")
        results.append(result)
    return results


def _manual_checks_pass(
    declarations: Sequence[Mapping[str, Any]],
    *,
    enabled: bool,
    results: Sequence[Mapping[str, Any]],
) -> bool:
    if not declarations:
        return True
    if not enabled or len(results) != len(declarations):
        return False
    for declaration, result in zip(declarations, results):
        if result.get("check_id") != declaration.get("check_id"):
            return False
        if result.get("status") != "passed":
            return False
        if (
            declaration.get("check_id") == "docker_launch"
            and result.get("stage") != "health"
        ):
            return False
    return True


def _verify_plan_payload(plan: Mapping[str, Any]) -> None:
    if not all(field in plan for field in (*_PLAN_HASH_FIELDS, "plan_hash")):
        raise ValueError("final evidence has an incomplete qualified plan")
    expected = stable_hash({field: plan[field] for field in _PLAN_HASH_FIELDS})
    if plan["plan_hash"] != expected:
        raise ValueError(
            "final evidence plan_hash mismatch: "
            f"supplied={plan['plan_hash']} computed={expected}"
        )
    if bool(plan.get("formal_ready")) != (not bool(plan.get("blocking_reasons"))):
        raise ValueError("final evidence plan formal_ready mismatch")


def _verify_evidence_records(result: Mapping[str, Any]) -> None:
    records = result.get("evidence_records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("final evidence result has no evidence_records")
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("final evidence record must be an object")
        expected = stable_hash(
            {field: record.get(field) for field in _EVIDENCE_HASH_FIELDS}
        )
        if record.get("evidence_hash") != expected:
            raise ValueError("final evaluator evidence_hash mismatch")
        if record.get("evidence_id") != f"verification_evidence_{expected[:24]}":
            raise ValueError("final evaluator evidence_id mismatch")


def _artifact_hash(payload: Mapping[str, Any]) -> str:
    return stable_hash(
        {key: value for key, value in payload.items() if key != "artifact_hash"}
    )


def validate_final_evaluation_evidence(
    payload: Mapping[str, Any],
    *,
    dataset_id: str,
    plan_hash: str,
    candidate_digest: str,
    manifest: Mapping[str, Any],
    manual_checks_enabled: bool,
) -> dict[str, Any]:
    """Validate complete plan/result/manual lineage before evidence is reused."""

    if payload.get("schema_version") != FINAL_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("final evidence schema mismatch")
    if payload.get("artifact_hash") != _artifact_hash(payload):
        raise ValueError("final evidence artifact_hash mismatch")
    plan = payload.get("qualified_plan")
    result = payload.get("result")
    if not isinstance(plan, Mapping) or not isinstance(result, Mapping):
        raise ValueError("final evidence must contain qualified_plan and result")
    _verify_plan_payload(plan)
    _verify_evidence_records(result)
    if (
        payload.get("dataset_id") != dataset_id
        or plan.get("dataset_id") != dataset_id
        or result.get("dataset_id") != dataset_id
    ):
        raise ValueError("final evidence dataset mismatch")
    if (
        payload.get("plan_hash") != plan_hash
        or plan.get("plan_hash") != plan_hash
        or result.get("plan_hash") != plan_hash
    ):
        raise ValueError("final evidence plan mismatch")
    if (
        payload.get("candidate_repo_digest") != candidate_digest
        or result.get("candidate_repo_digest") != candidate_digest
    ):
        raise ValueError("final evidence candidate mismatch")
    if payload.get("evaluation") != result:
        raise ValueError("final evidence evaluation alias mismatch")

    from environments.org_env.experiments.records import normalize_final_evaluation

    normalize_final_evaluation(
        payload,
        expected_dataset_id=dataset_id,
        expected_plan_hash=plan_hash,
        expected_candidate_repo_digest=candidate_digest,
        dataset_manifest=manifest,
    )
    declarations = manifest_manual_release_checks(manifest)
    recorded_declarations = payload.get("manual_check_declarations")
    if recorded_declarations != canonicalize(declarations):
        raise ValueError("final evidence manual check declarations mismatch")
    if payload.get("manual_checks_enabled") is not manual_checks_enabled:
        raise ValueError("final evidence manual check mode mismatch")
    manual_results = payload.get("manual_checks")
    if not isinstance(manual_results, list):
        raise ValueError("final evidence manual_checks must be a list")
    expected_ids = [str(item["check_id"]) for item in declarations]
    observed_ids = [str(item.get("check_id") or "") for item in manual_results]
    if manual_checks_enabled:
        if observed_ids != expected_ids:
            raise ValueError("final evidence manual check result mismatch")
    elif manual_results:
        raise ValueError("disabled manual checks produced results")
    expected_release_ready = bool(
        result.get("formal_claim_ready")
    ) and _manual_checks_pass(
        declarations,
        enabled=manual_checks_enabled,
        results=manual_results,
    )
    if payload.get("formal_release_ready") is not expected_release_ready:
        raise ValueError("final evidence formal_release_ready mismatch")
    return dict(payload)


def _formal_mode(world: Any) -> bool:
    params = getattr(getattr(world, "scenario", None), "params", {}) or {}
    return str(params.get("experiment_mode") or "") == "formal"


def _sandboxed_evaluator_mode(world: Any) -> bool:
    params = getattr(getattr(world, "scenario", None), "params", {}) or {}
    return str(params.get("experiment_mode") or "") in {"formal", "pilot"}


def _frozen_evaluator_executor():
    from society_core.execution import ExecutionPolicy, build_command_executor

    backend = str(os.environ.get("ORG_EVALUATOR_BACKEND") or "")
    if backend == "bubblewrap":
        virtualenv_path = os.environ.get("ORG_EVALUATOR_VENV")
        if not virtualenv_path:
            raise RuntimeError(
                "pilot_evaluator_runtime_binding_missing:virtualenv_path"
            )
        return build_command_executor(
            ExecutionPolicy(
                trust_level="untrusted",
                backend="bubblewrap",
                virtualenv_path=virtualenv_path,
                network_enabled=False,
            )
        )
    values = {
        "backend": backend,
        "container_image": os.environ.get("ORG_EVALUATOR_CONTAINER_IMAGE"),
        "container_platform": os.environ.get("ORG_EVALUATOR_CONTAINER_PLATFORM"),
    }
    missing = sorted(key for key, value in values.items() if not value)
    if missing:
        raise RuntimeError(
            "formal_evaluator_runtime_binding_missing:" + ",".join(missing)
        )
    return build_command_executor(
        ExecutionPolicy(
            trust_level="untrusted",
            backend=str(values["backend"]),
            container_image=str(values["container_image"]),
            container_platform=str(values["container_platform"]),
            network_enabled=False,
        )
    )


def run_final_evaluation(
    world: Any,
    *,
    output_dir: str | os.PathLike[str],
    run_tag: str = "",
    timeout_seconds: int | None = None,
    manual_check_runners: Mapping[str, ManualCheckRunner] | None = None,
    plan_builder: Callable[..., Any] | None = None,
    candidate_evaluator: Callable[..., Any] | None = None,
    candidate_exporter: Callable[..., Any] | None = None,
) -> FinalEvaluationArtifact | None:
    """Evaluate one content-addressed final candidate after rollout completion.

    Dependency-injection parameters keep unit tests fast; production callers use the
    frozen evaluator, materializer, and manifest-declared manual check registry.
    """

    programbench_profile_id = _active_programbench_final_profile(world)
    programbench_primary_selection: Mapping[str, Any] | None = None
    if programbench_profile_id is not None:
        # There is deliberately no early terminal profile state.  Reject the
        # final boundary before a plan is built or any tree is exported or
        # shown to an evaluator.
        _validate_programbench_final_horizon(world)
        programbench_primary_selection = _programbench_public_primary_selection(
            world
        )
    assets = oss_eval_assets(world) or {}
    config = oss_evaluator_config(world)
    if _formal_mode(world):
        # Loaded formal checkpoints are re-qualified here as well as on session load.
        validate_formal_oss_world(
            world,
            require_formal=True,
            persist_qualification=False,
        )
    if not assets or not config.get("run_oss_final_evaluation"):
        return None

    dataset_id = str(assets.get("dataset_id") or "")
    manifest = assets.get("manifest") or {}
    if not dataset_id or not isinstance(manifest, Mapping):
        raise RuntimeError("final evaluator could not resolve dataset manifest")
    declarations = manifest_manual_release_checks(manifest)
    manual_enabled = bool(config.get("run_oss_manual_checks", bool(declarations)))
    timeout = int(
        timeout_seconds
        if timeout_seconds is not None
        else os.environ.get("ORG_OSS_QUALIFICATION_TIMEOUT", "180")
    )

    default_plan_builder = plan_builder is None
    default_candidate_evaluator = candidate_evaluator is None
    evaluator_executor = (
        _frozen_evaluator_executor()
        if _sandboxed_evaluator_mode(world)
        and (default_plan_builder or default_candidate_evaluator)
        else None
    )
    if default_plan_builder:
        from society_core.time_machine_evaluation import (
            build_time_machine_evaluation_plan,
        )

        if programbench_profile_id is not None:
            from society_core.programbench_evaluation import is_programbench_spec

            resolved_profile_spec = resolve_oss_evaluator_spec(assets)
            if not is_programbench_spec(resolved_profile_spec):
                raise RuntimeError(
                    "programbench_final_profile_requires_programbench_spec"
                )
        plan_builder = build_time_machine_evaluation_plan
    if default_candidate_evaluator:
        from society_core.time_machine_evaluation import (
            evaluate_time_machine_candidate,
        )

        candidate_evaluator = evaluate_time_machine_candidate
    default_candidate_exporter = candidate_exporter is None
    if default_candidate_exporter:
        from environments.org_env.product.materialize import export_product_repo

        candidate_exporter = export_product_repo
    from society_core.time_machine_evaluation import _repo_digest

    plan_arguments: dict[str, Any] = {
        "dataset_id": dataset_id,
        "timeout_seconds": timeout,
    }
    if default_plan_builder:
        # External frozen packs are located by ``dataset_dir`` but identified in
        # evidence by their manifest project id.  Supplying the already verified
        # vault spec preserves both properties and avoids a second lookup under
        # the repository-local dataset root.
        plan_arguments["spec"] = resolve_oss_evaluator_spec(assets)
        plan_arguments["executor"] = evaluator_executor
    plan = plan_builder(**plan_arguments)
    expected_environment_hash = os.environ.get(
        "ORG_EVALUATOR_EXPECTED_ENVIRONMENT_HASH"
    )
    expected_qualification_hash = os.environ.get(
        "ORG_EVALUATOR_EXPECTED_QUALIFICATION_HASH"
    )
    if (
        expected_environment_hash
        and plan.evaluator_environment_hash != expected_environment_hash
    ):
        raise RuntimeError("formal_evaluator_environment_hash_mismatch")
    if expected_qualification_hash and plan.plan_hash != expected_qualification_hash:
        raise RuntimeError("formal_evaluator_qualification_hash_mismatch")
    plan_payload = canonicalize(plan)
    _verify_plan_payload(plan_payload)
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if programbench_profile_id is not None:
        return _run_programbench_dual_final_evaluation(
            world=world,
            assets=assets,
            manifest=manifest,
            declarations=declarations,
            manual_enabled=manual_enabled,
            manual_check_runners=manual_check_runners,
            plan=plan,
            plan_payload=plan_payload,
            output_root=output_root,
            run_tag=run_tag,
            timeout=timeout,
            candidate_evaluator=candidate_evaluator,
            candidate_exporter=candidate_exporter,
            default_candidate_exporter=default_candidate_exporter,
            default_candidate_evaluator=default_candidate_evaluator,
            evaluator_executor=evaluator_executor,
            profile_id=programbench_profile_id,
            primary_selection=programbench_primary_selection,
        )

    with tempfile.TemporaryDirectory(prefix="oss_final_candidate_") as temporary_dir:
        candidate_root = Path(temporary_dir) / "candidate"
        candidate_exporter(
            world,
            str(candidate_root),
            prefer_mainline=True,
        )
        candidate_digest = _repo_digest(candidate_root)
        output_path = _final_evaluation_output_path(
            output_root,
            dataset_id=dataset_id,
            plan_hash=plan.plan_hash,
            candidate_digest=candidate_digest,
        )
        if _long_path_compatible(output_path).exists():
            payload = _read_final_evidence(output_root, output_path)
            validated = validate_final_evaluation_evidence(
                payload,
                dataset_id=dataset_id,
                plan_hash=plan.plan_hash,
                candidate_digest=candidate_digest,
                manifest=manifest,
                manual_checks_enabled=manual_enabled,
            )
            return FinalEvaluationArtifact(output_path, validated, True)

        evaluator_arguments: dict[str, Any] = {
            "timeout_seconds": timeout,
        }
        if default_candidate_evaluator:
            evaluator_arguments["executor"] = evaluator_executor
        result = candidate_evaluator(
            plan,
            candidate_root,
            **evaluator_arguments,
        )
        result_payload = canonicalize(result)
        if result_payload.get("candidate_repo_digest") != candidate_digest:
            raise RuntimeError("final evaluator returned a different candidate digest")
        manual_results = _run_manual_checks(
            declarations=declarations,
            enabled=manual_enabled,
            candidate_root=candidate_root,
            candidate_digest=candidate_digest,
            runners=manual_check_runners or DEFAULT_MANUAL_CHECK_RUNNERS,
        )
        # Route-agnostic functional score, beside (never instead of) the hidden
        # oracles: of the user-reachable capabilities the reference release
        # added, how many the candidate also exposes. Three of eight measured
        # oracle failures were route disagreements on work that functionally
        # landed; this is the number that credits them, and it is the number
        # the 90%-functional-overlap goal is stated in. Evaluator-only data
        # (it reads the reference repo), so it must never reach agents.
        functional_overlap_payload: dict[str, Any] | None = None
        try:
            from society_core.functional_overlap import functional_overlap

            spec = resolve_oss_evaluator_spec(assets)
            functional_overlap_payload = functional_overlap(
                starter_dir=spec.starter_repo_dir,
                reference_dir=spec.reference_repo_dir,
                candidate_dir=candidate_root,
            ).to_dict()
        except Exception as exc:
            # A missing score must say so, not sink the evaluation: the hidden
            # suite already ran and its evidence stands on its own.
            functional_overlap_payload = {
                "schema_version": "functional_overlap_v2",
                "error": f"{type(exc).__name__}: {exc}"[:300],
            }
        payload: dict[str, Any] = {
            "schema_version": FINAL_EVIDENCE_SCHEMA_VERSION,
            "run_tag": run_tag,
            "functional_overlap": functional_overlap_payload,
            "world_tick": int(getattr(world, "world_tick", 0)),
            "dataset_id": dataset_id,
            "plan_hash": plan.plan_hash,
            "candidate_repo_digest": candidate_digest,
            "starter_repo_digest": plan.starter_repo_digest,
            "reference_repo_digest": plan.reference_repo_digest,
            "hidden_suite_hash": plan.hidden_suite_hash,
            "evaluator_environment_hash": plan.evaluator_environment_hash,
            "manual_check_declarations": canonicalize(declarations),
            "manual_checks_enabled": manual_enabled,
            "manual_checks": manual_results,
            "formal_release_ready": bool(result.formal_claim_ready)
            and _manual_checks_pass(
                declarations,
                enabled=manual_enabled,
                results=manual_results,
            ),
            "qualified_plan": plan_payload,
            "result": result_payload,
            # Compatibility for the first Gitingest evaluator payload.
            "evaluation": result_payload,
        }
        payload["artifact_hash"] = _artifact_hash(payload)
        validated = validate_final_evaluation_evidence(
            payload,
            dataset_id=dataset_id,
            plan_hash=plan.plan_hash,
            candidate_digest=candidate_digest,
            manifest=manifest,
            manual_checks_enabled=manual_enabled,
        )
        published = _publish_json_exclusive(output_path, validated)
        if not published:
            concurrent = _read_final_evidence(output_root, output_path)
            validated = validate_final_evaluation_evidence(
                concurrent,
                dataset_id=dataset_id,
                plan_hash=plan.plan_hash,
                candidate_digest=candidate_digest,
                manifest=manifest,
                manual_checks_enabled=manual_enabled,
            )
            return FinalEvaluationArtifact(output_path, validated, True)
        return FinalEvaluationArtifact(output_path, validated, False)


def _active_programbench_final_profile(world: Any) -> str | None:
    """Resolve the opt-in profile without accepting a half-attached world.

    The scenario flag is controller-owned while the profile state is attached
    only after the public pack has passed task-family detection.  Requiring
    both prevents an unrelated OSS run from entering a ProgramBench evaluator
    merely because a stale dictionary survived in memory.
    """

    from environments.org_env.programbench.leaderboard_profile import (
        PROFILE_ID,
        programbench_profile_active,
        validate_programbench_profile_state_for_step,
    )

    params = getattr(getattr(world, "scenario", None), "params", {}) or {}
    configured = str(params.get("execution_profile") or "native")
    attached = programbench_profile_active(world)
    claimed = "programbench_profile_state" in getattr(world, "__dict__", {})
    materialized = str(world.__dict__.get("execution_profile") or "native")
    if claimed:
        try:
            validate_programbench_profile_state_for_step(world)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "programbench_final_profile_state_invalid"
            ) from error
    if configured == "native":
        if claimed and not attached:
            raise RuntimeError("programbench_final_profile_state_invalid")
        if attached or materialized != "native":
            raise RuntimeError("programbench_final_profile_attachment_mismatch")
        return None
    if configured != PROFILE_ID:
        raise RuntimeError("programbench_final_profile_unknown")
    if not attached or materialized != PROFILE_ID:
        raise RuntimeError("programbench_final_profile_attachment_mismatch")
    return PROFILE_ID


PROGRAMBENCH_DIAGNOSTIC_ENV = "ORG_PROGRAMBENCH_DIAGNOSTIC_EVALUATION"


def programbench_diagnostic_evaluation_enabled() -> bool:
    """Whether this process may score a ProgramBench world before its horizon.

    The horizon gate exists to stop a running organization seeing a hidden
    aggregate while it can still act on what it saw. That is worth protecting,
    and it is protected precisely elsewhere: the dual evaluation records
    hidden_feedback_returned_to_world, and nothing writes an evaluator result
    back into the world.

    Refusing every tick but 336 protects the same thing a second time, bluntly,
    and the cost is paid entirely by analysis that cannot possibly leak: a
    detached copy of a finished run has nobody to tell. It also makes a
    trajectory unobtainable, so a run can be scored but never explained -- no
    curve, no answer to when a capability appeared, no way to see where progress
    stopped. The 197/285 mid-run figure quoted for the previous arm was taken by
    presenting a copy at its horizon anyway, which is this, without the label.

    Opt-in, per process, and never set by a run: an artifact produced this way
    is a diagnostic and says so.
    """

    return str(os.environ.get(PROGRAMBENCH_DIAGNOSTIC_ENV, "")).strip().lower() in {
        "1", "true", "yes", "on"
    }


def _validate_programbench_final_horizon(world: Any) -> None:
    """Require the fresh v5 ProgramBench run to reach exactly t336."""

    from environments.org_env.programbench import (
        PROFILE_STATE_SCHEMA_VERSION,
        get_programbench_profile_state,
    )

    state = get_programbench_profile_state(world)
    if not isinstance(state, Mapping) or state.get("schema_version") != (
        PROFILE_STATE_SCHEMA_VERSION
    ):
        raise RuntimeError("programbench_final_profile_state_invalid")
    horizon = state.get("run_horizon_ticks")
    world_tick = getattr(world, "world_tick", None)
    if isinstance(horizon, bool) or horizon != 336:
        raise RuntimeError("programbench_final_requires_tick_336")
    if isinstance(world_tick, bool) or world_tick != horizon:
        # A world short of its horizon is scorable only as a diagnostic, and
        # only because this process was told to. The seal still means t336.
        if not programbench_diagnostic_evaluation_enabled():
            raise RuntimeError("programbench_final_requires_tick_336")


def _programbench_dual_evaluation_output_path(
    root: Path,
    *,
    dataset_id: str,
    plan_hash: str,
    profile_id: str,
    candidate_view: str,
    workspace_digest: str,
    mainline_digest: str,
    primary_selection: Mapping[str, Any],
) -> Path:
    evidence_key = stable_hash(
        {
            "schema_version": PROGRAMBENCH_DUAL_EVALUATION_SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "plan_hash": plan_hash,
            "profile_id": profile_id,
            "candidate_view": candidate_view,
            "workspace_candidate_repo_digest": workspace_digest,
            "mainline_candidate_repo_digest": mainline_digest,
            "primary_selection": canonicalize(primary_selection),
        }
    )
    return root / f"final_evaluation_programbench_dual_{evidence_key}.json"


def _programbench_candidate_freeze(
    *,
    dataset_id: str,
    plan_hash: str,
    evaluator_environment_hash: str,
    profile_id: str,
    candidate_view: str,
    workspace_digest: str,
    mainline_digest: str,
    primary_selection: Mapping[str, Any],
) -> dict[str, str]:
    payload = {
        "dataset_id": dataset_id,
        "plan_hash": plan_hash,
        "evaluator_environment_hash": evaluator_environment_hash,
        "profile_id": profile_id,
        "candidate_view": candidate_view,
        "workspace_candidate_repo_digest": workspace_digest,
        "mainline_candidate_repo_digest": mainline_digest,
        "workspace_public_content_digest": str(
            primary_selection.get("selected_repo_digest") or ""
        ),
        "mainline_public_content_digest": str(
            primary_selection.get("mainline_repo_digest") or ""
        ),
        "primary_selection": canonicalize(primary_selection),
    }
    return {**payload, "freeze_hash": stable_hash(payload)}


def _programbench_public_primary_selection(world: Any) -> dict[str, Any]:
    """Select the final tree using only current public rollout evidence."""

    from environments.org_env.backend.repo.workflow import (
        programbench_integration_branches,
    )
    from environments.org_env.product.materialize import (
        PROGRAMBENCH_CANDIDATE_VIEW_SCHEMA_VERSION,
        programbench_integration_candidate_digest,
        programbench_mainline_digest,
        programbench_probe_corpus_digest,
    )
    from environments.org_env.programbench import (
        PROFILE_STATE_SCHEMA_VERSION,
        get_programbench_profile_state,
        programbench_current_public_candidate_attestation,
        validate_programbench_profile_state_for_step,
    )

    try:
        validate_programbench_profile_state_for_step(world)
    except (TypeError, ValueError) as error:
        raise RuntimeError("programbench_final_profile_state_invalid") from error
    state = get_programbench_profile_state(world)
    if not isinstance(state, Mapping) or state.get("schema_version") != (
        PROFILE_STATE_SCHEMA_VERSION
    ):
        raise RuntimeError("programbench_final_profile_state_invalid")
    _validate_programbench_final_horizon(world)
    mainline_digest = programbench_mainline_digest(world)
    signals = state.get("signals")
    if not isinstance(signals, Mapping):
        raise RuntimeError("programbench_final_profile_signals_invalid")
    candidate_digest = ""
    corpus_digest = ""
    active_branches: Sequence[Any] = ()
    candidate_surface_available = True
    try:
        candidate_digest = programbench_integration_candidate_digest(world)
        corpus_digest = programbench_probe_corpus_digest(world)
        active_branches = programbench_integration_branches(world)
    except Exception:  # noqa: BLE001 - malformed live work falls back to mainline
        candidate_surface_available = False
    verification_candidate = str(
        state.get("public_verification_candidate_repo_digest") or ""
    )
    verification_corpus = str(
        state.get("public_verification_probe_corpus_digest") or ""
    )
    verification_view = str(
        state.get("public_verification_candidate_view_schema") or ""
    )
    try:
        attestation_reason, live_attestation = (
            programbench_current_public_candidate_attestation(world)
        )
    except Exception:  # noqa: BLE001 - final candidate selection falls back
        attestation_reason = "programbench_live_attestation_exception"
        live_attestation = None
    live_attestation_current = bool(
        attestation_reason is None and isinstance(live_attestation, Mapping)
    )
    candidate_attested = bool(
        candidate_surface_available
        and candidate_digest
        and corpus_digest
        and len(active_branches) == 1
        and str(state.get("phase") or "") == "develop"
        and signals.get("public_verification_complete") is True
        and signals.get("unresolved_public_mismatch_count") == 0
        and state.get("latest_reference_probe_required") is False
        and verification_corpus == corpus_digest
        and verification_candidate == candidate_digest
        and verification_view == PROGRAMBENCH_CANDIDATE_VIEW_SCHEMA_VERSION
        and str(state.get("public_candidate_repo_digest") or "")
        == candidate_digest
        and live_attestation_current
        and live_attestation.get("candidate_repo_digest") == candidate_digest
        and live_attestation.get("probe_corpus_digest") == corpus_digest
    )
    strict_mainline_digest = str(
        state.get("strict_verified_mainline_digest") or ""
    )
    strict_mainline_attested = strict_mainline_digest == mainline_digest
    strict_mainline_attestation_kind = str(
        state.get("strict_mainline_attestation_kind") or ""
    )
    if not strict_mainline_attested or strict_mainline_attestation_kind not in {
        "trusted_starter_baseline",
        "public_verified_merge",
    }:
        raise RuntimeError("programbench_strict_mainline_attestation_stale")
    if candidate_attested:
        selected_view = "integration_candidate"
        reason = "current_public_differential_zero_mismatch"
        selected_digest = candidate_digest
    else:
        selected_view = "strict_mainline_safety_fallback"
        reason = "integration_candidate_not_currently_attested"
        selected_digest = mainline_digest
        if not candidate_surface_available:
            reason = "integration_candidate_surface_invalid_mainline_fallback"
    return canonicalize(
        {
            "schema_version": PROGRAMBENCH_PRIMARY_SELECTION_SCHEMA_VERSION,
            "candidate_surface_schema_version": (
                PROGRAMBENCH_CANDIDATE_VIEW_SCHEMA_VERSION
            ),
            "selected_view": selected_view,
            "selection_reason": reason,
            "selected_repo_digest": selected_digest,
            "integration_candidate_repo_digest": candidate_digest,
            "mainline_repo_digest": mainline_digest,
            "current_probe_corpus_digest": corpus_digest,
            "verification_probe_corpus_digest": verification_corpus or None,
            "verification_candidate_repo_digest": verification_candidate or None,
            "verification_candidate_view_schema": verification_view or None,
            "active_integration_branch_count": len(active_branches),
            "candidate_public_attestation_current": candidate_attested,
            "public_attestation_schema_version": (
                live_attestation.get("schema_version")
                if isinstance(live_attestation, Mapping)
                else None
            ),
            "public_attestation_reason": attestation_reason,
            "public_attestation": canonicalize(live_attestation),
            "live_delivery_attestation_current": live_attestation_current,
            "strict_mainline_attestation_current": strict_mainline_attested,
            "strict_mainline_attestation_kind": (
                strict_mainline_attestation_kind
            ),
        }
    )


def _programbench_result_hash(result: Mapping[str, Any]) -> str:
    evidence_hashes = result.get("evidence_hashes")
    if evidence_hashes is None:
        records = result.get("evidence_records")
        if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
            evidence_hashes = tuple(
                item.get("evidence_hash")
                for item in records
                if isinstance(item, Mapping)
            )
    return stable_hash(
        {
            "dataset_id": result.get("dataset_id"),
            "plan_hash": result.get("plan_hash"),
            "candidate_repo_digest": result.get("candidate_repo_digest"),
            "status": result.get("status"),
            "outcomes": result.get("outcomes"),
            "evidence_hashes": evidence_hashes,
            "formal_claim_ready": result.get("formal_claim_ready"),
        }
    )


def _validate_programbench_candidate_result(
    result: Mapping[str, Any],
    *,
    label: str,
    dataset_id: str,
    plan_hash: str,
    candidate_digest: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise RuntimeError(f"programbench_dual_result_not_mapping:{label}")
    if (
        result.get("dataset_id") != dataset_id
        or result.get("plan_hash") != plan_hash
        or result.get("candidate_repo_digest") != candidate_digest
    ):
        raise RuntimeError(f"programbench_dual_result_identity_mismatch:{label}")
    if result.get("result_hash") != _programbench_result_hash(result):
        raise RuntimeError(f"programbench_dual_result_hash_mismatch:{label}")
    try:
        _verify_evidence_records(result)
        from environments.org_env.experiments.records import (
            normalize_final_evaluation,
        )

        normalize_final_evaluation(
            result,
            expected_dataset_id=dataset_id,
            expected_plan_hash=plan_hash,
            expected_candidate_repo_digest=candidate_digest,
            dataset_manifest=manifest,
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"programbench_dual_result_invalid:{label}"
        ) from error
    infrastructure_errors = result.get("infrastructure_error_count")
    if (
        isinstance(infrastructure_errors, bool)
        or not isinstance(infrastructure_errors, int)
        or infrastructure_errors != 0
        or result.get("status") in {"infra_error", "blocked_invalid_plan"}
    ):
        raise RuntimeError(f"programbench_dual_infrastructure_failure:{label}")
    return canonicalize(result)


def _programbench_statuses(result: Mapping[str, Any]) -> dict[str, str]:
    outcomes = result.get("outcomes")
    if not isinstance(outcomes, Sequence) or isinstance(outcomes, (str, bytes)):
        raise RuntimeError("programbench_dual_outcomes_missing")
    statuses: dict[str, str] = {}
    for row in outcomes:
        if not isinstance(row, Mapping):
            raise RuntimeError("programbench_dual_outcome_invalid")
        test_id = str(row.get("test_id") or "")
        status = str(row.get("candidate_status") or "")
        if not test_id or test_id in statuses:
            raise RuntimeError("programbench_dual_outcome_identity_invalid")
        statuses[test_id] = status
    return statuses


def _programbench_delivery_gap(
    workspace_result: Mapping[str, Any],
    mainline_result: Mapping[str, Any],
) -> dict[str, Any]:
    workspace_statuses = _programbench_statuses(workspace_result)
    mainline_statuses = _programbench_statuses(mainline_result)
    if set(workspace_statuses) != set(mainline_statuses):
        raise RuntimeError("programbench_dual_outcome_set_mismatch")
    workspace_passed = {
        test_id for test_id, status in workspace_statuses.items() if status == "passed"
    }
    mainline_passed = {
        test_id for test_id, status in mainline_statuses.items() if status == "passed"
    }
    workspace_rate = workspace_result.get("candidate_pass_rate")
    mainline_rate = mainline_result.get("candidate_pass_rate")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (workspace_rate, mainline_rate)
    ):
        raise RuntimeError("programbench_dual_pass_rate_invalid")
    workspace_count = len(workspace_passed)
    mainline_count = len(mainline_passed)
    return {
        "schema_version": "programbench_delivery_gap_v1",
        "workspace_candidate_passed_count": workspace_count,
        "mainline_realization_passed_count": mainline_count,
        "passed_count_delta": workspace_count - mainline_count,
        "unrealized_workspace_passed_count": len(
            workspace_passed - mainline_passed
        ),
        "workspace_only_passed_test_ids": sorted(workspace_passed - mainline_passed),
        "mainline_only_passed_test_ids": sorted(mainline_passed - workspace_passed),
        "workspace_candidate_pass_rate": float(workspace_rate),
        "mainline_realization_pass_rate": float(mainline_rate),
        "pass_rate_delta": round(float(workspace_rate) - float(mainline_rate), 12),
        "delivery_realization_ratio": (
            round(mainline_count / workspace_count, 12)
            if workspace_count
            else None
        ),
    }


def _programbench_candidate_summary(
    result: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    statuses = _programbench_statuses(result)
    return {
        "role": role,
        "candidate_repo_digest": result["candidate_repo_digest"],
        "result_hash": result["result_hash"],
        "status": result["status"],
        "candidate_pass_rate": result["candidate_pass_rate"],
        "passed_count": sum(status == "passed" for status in statuses.values()),
        "total_count": len(statuses),
        "formal_claim_ready": result["formal_claim_ready"],
    }


def _build_programbench_dual_extension(
    *,
    profile_id: str,
    candidate_view: str,
    plan_hash: str,
    evaluator_environment_hash: str,
    candidate_freeze: Mapping[str, Any],
    workspace_result: Mapping[str, Any],
    mainline_result: Mapping[str, Any],
    mainline_manual_checks: Sequence[Mapping[str, Any]],
    mainline_formal_release_ready: bool,
    primary_selection: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": PROGRAMBENCH_DUAL_EVALUATION_SCHEMA_VERSION,
        "profile_id": profile_id,
        "candidate_view": candidate_view,
        # A score taken before the horizon is a diagnostic and has to say so in
        # the artifact, not only in whatever the caller happened to name the
        # output directory. The sealed boundary is the one at t336.
        "evaluation_boundary": (
            "pre_horizon_diagnostic"
            if programbench_diagnostic_evaluation_enabled()
            else "post_rollout_sealed"
        ),
        "hidden_feedback_returned_to_world": False,
        "plan_hash": plan_hash,
        "evaluator_environment_hash": evaluator_environment_hash,
        "candidate_freeze": canonicalize(candidate_freeze),
        "primary_selection": canonicalize(primary_selection),
        "workspace_candidate": _programbench_candidate_summary(
            workspace_result,
            role="primary_programbench_candidate",
        ),
        "mainline_realization": {
            **_programbench_candidate_summary(
                mainline_result,
                role="mainline_delivery_realization",
            ),
            "result": canonicalize(mainline_result),
            "manual_checks": canonicalize(mainline_manual_checks),
            "formal_release_ready": bool(mainline_formal_release_ready),
        },
        "delivery_gap": _programbench_delivery_gap(
            workspace_result,
            mainline_result,
        ),
    }
    payload["evidence_hash"] = stable_hash(payload)
    return canonicalize(payload)


def _validate_programbench_dual_extension(
    payload: Mapping[str, Any],
    *,
    profile_id: str,
    candidate_view: str,
    plan: Any,
    workspace_digest: str,
    mainline_digest: str,
    manifest: Mapping[str, Any],
    declarations: Sequence[Mapping[str, Any]],
    manual_enabled: bool,
    primary_selection: Mapping[str, Any],
) -> dict[str, Any]:
    extension = payload.get("programbench_dual_evaluation")
    if not isinstance(extension, Mapping):
        raise RuntimeError("programbench_dual_evidence_missing")
    if canonicalize(payload.get("programbench_primary_selection")) != (
        canonicalize(primary_selection)
    ):
        raise RuntimeError("programbench_primary_selection_evidence_mismatch")
    mainline = extension.get("mainline_realization")
    if not isinstance(mainline, Mapping) or not isinstance(
        mainline.get("result"), Mapping
    ):
        raise RuntimeError("programbench_dual_mainline_evidence_missing")
    workspace_result = payload.get("result")
    if not isinstance(workspace_result, Mapping):
        raise RuntimeError("programbench_dual_workspace_evidence_missing")
    validated_workspace = _validate_programbench_candidate_result(
        workspace_result,
        label="workspace",
        dataset_id=plan.dataset_id,
        plan_hash=plan.plan_hash,
        candidate_digest=workspace_digest,
        manifest=manifest,
    )
    validated_mainline = _validate_programbench_candidate_result(
        mainline["result"],
        label="mainline",
        dataset_id=plan.dataset_id,
        plan_hash=plan.plan_hash,
        candidate_digest=mainline_digest,
        manifest=manifest,
    )
    recorded_checks = mainline.get("manual_checks")
    if not isinstance(recorded_checks, list):
        raise RuntimeError("programbench_dual_mainline_manual_checks_invalid")
    expected_ids = [str(item["check_id"]) for item in declarations]
    observed_ids = [str(item.get("check_id") or "") for item in recorded_checks]
    if (manual_enabled and observed_ids != expected_ids) or (
        not manual_enabled and recorded_checks
    ):
        raise RuntimeError("programbench_dual_mainline_manual_checks_mismatch")
    expected_mainline_release = bool(validated_mainline.get("formal_claim_ready")) and (
        _manual_checks_pass(
            declarations,
            enabled=manual_enabled,
            results=recorded_checks,
        )
    )
    candidate_freeze = _programbench_candidate_freeze(
        dataset_id=plan.dataset_id,
        plan_hash=plan.plan_hash,
        evaluator_environment_hash=plan.evaluator_environment_hash,
        profile_id=profile_id,
        candidate_view=candidate_view,
        workspace_digest=workspace_digest,
        mainline_digest=mainline_digest,
        primary_selection=primary_selection,
    )
    expected = _build_programbench_dual_extension(
        profile_id=profile_id,
        candidate_view=candidate_view,
        plan_hash=plan.plan_hash,
        evaluator_environment_hash=plan.evaluator_environment_hash,
        candidate_freeze=candidate_freeze,
        workspace_result=validated_workspace,
        mainline_result=validated_mainline,
        mainline_manual_checks=recorded_checks,
        mainline_formal_release_ready=expected_mainline_release,
        primary_selection=primary_selection,
    )
    if canonicalize(extension) != expected:
        raise RuntimeError("programbench_dual_evidence_mismatch")
    return dict(payload)


def _run_programbench_dual_final_evaluation(
    *,
    world: Any,
    assets: Mapping[str, Any],
    manifest: Mapping[str, Any],
    declarations: Sequence[Mapping[str, Any]],
    manual_enabled: bool,
    manual_check_runners: Mapping[str, ManualCheckRunner] | None,
    plan: Any,
    plan_payload: Mapping[str, Any],
    output_root: Path,
    run_tag: str,
    timeout: int,
    candidate_evaluator: Callable[..., Any],
    candidate_exporter: Callable[..., Any],
    default_candidate_exporter: bool,
    default_candidate_evaluator: bool,
    evaluator_executor: Any,
    profile_id: str,
    primary_selection: Mapping[str, Any] | None,
) -> FinalEvaluationArtifact:
    from society_core.programbench_evaluation import is_programbench_spec
    from society_core.time_machine_evaluation import _repo_digest

    if not is_programbench_spec(getattr(plan, "_spec", None)):
        raise RuntimeError("programbench_final_profile_requires_programbench_spec")
    if not bool(getattr(plan, "formal_ready", False)):
        raise RuntimeError("programbench_dual_plan_not_formal_ready")
    if not isinstance(primary_selection, Mapping):
        raise RuntimeError("programbench_primary_selection_missing")
    primary_selection = canonicalize(primary_selection)
    select_mainline = primary_selection.get("selected_view") == (
        "strict_mainline_safety_fallback"
    )

    with tempfile.TemporaryDirectory(
        prefix="programbench_final_candidates_"
    ) as temporary_dir:
        temporary_root = Path(temporary_dir)
        workspace_root = temporary_root / "integration_candidate"
        mainline_root = temporary_root / "mainline_candidate"
        candidate_view = (
            None
            if default_candidate_exporter
            else "injected_workspace_exporter"
        )
        if default_candidate_exporter:
            from environments.org_env.product.materialize import (
                PROGRAMBENCH_CANDIDATE_VIEW_SCHEMA_VERSION,
                export_programbench_integration_candidate,
                export_programbench_mainline,
                programbench_exported_repo_digest,
            )

            candidate_view = PROGRAMBENCH_CANDIDATE_VIEW_SCHEMA_VERSION
            if select_mainline:
                export_programbench_mainline(world, str(workspace_root))
            else:
                export_programbench_integration_candidate(world, str(workspace_root))
            export_programbench_mainline(world, str(mainline_root))
        else:
            from environments.org_env.product.materialize import (
                programbench_exported_repo_digest,
            )

            # Preserve the long-standing dependency-injection contract used by
            # tests and external evaluators.  The v2 evidence explicitly labels
            # this as injected rather than pretending it has production view
            # semantics.
            candidate_exporter(
                world,
                str(workspace_root),
                prefer_mainline=select_mainline,
            )
            candidate_exporter(
                world,
                str(mainline_root),
                prefer_mainline=True,
            )
        if not isinstance(candidate_view, str) or not candidate_view:
            raise RuntimeError("programbench_candidate_view_schema_missing")
        # Both immutable identities are fixed before either candidate is shown
        # to the hidden evaluator.  A later result therefore cannot influence
        # which second candidate is selected.
        workspace_public_digest = programbench_exported_repo_digest(
            workspace_root
        )
        mainline_public_digest = programbench_exported_repo_digest(
            mainline_root
        )
        if workspace_public_digest != str(
            primary_selection.get("selected_repo_digest") or ""
        ):
            raise RuntimeError("programbench_primary_selection_export_mismatch")
        if mainline_public_digest != str(
            primary_selection.get("mainline_repo_digest") or ""
        ):
            raise RuntimeError("programbench_mainline_selection_export_mismatch")
        # Hidden results and the evaluation cache remain bound to the richer
        # workspace-execution identity, not the public content SHA-1 above.
        workspace_digest = _repo_digest(workspace_root)
        mainline_digest = _repo_digest(mainline_root)
        candidate_freeze = _programbench_candidate_freeze(
            dataset_id=plan.dataset_id,
            plan_hash=plan.plan_hash,
            evaluator_environment_hash=plan.evaluator_environment_hash,
            profile_id=profile_id,
            candidate_view=candidate_view,
            workspace_digest=workspace_digest,
            mainline_digest=mainline_digest,
            primary_selection=primary_selection,
        )
        output_path = _programbench_dual_evaluation_output_path(
            output_root,
            dataset_id=plan.dataset_id,
            plan_hash=plan.plan_hash,
            profile_id=profile_id,
            candidate_view=candidate_view,
            workspace_digest=workspace_digest,
            mainline_digest=mainline_digest,
            primary_selection=primary_selection,
        )
        if _long_path_compatible(output_path).exists():
            payload = _read_final_evidence(output_root, output_path)
            validated = validate_final_evaluation_evidence(
                payload,
                dataset_id=plan.dataset_id,
                plan_hash=plan.plan_hash,
                candidate_digest=workspace_digest,
                manifest=manifest,
                manual_checks_enabled=manual_enabled,
            )
            validated = _validate_programbench_dual_extension(
                validated,
                profile_id=profile_id,
                candidate_view=candidate_view,
                plan=plan,
                workspace_digest=workspace_digest,
                mainline_digest=mainline_digest,
                manifest=manifest,
                declarations=declarations,
                manual_enabled=manual_enabled,
                primary_selection=primary_selection,
            )
            return FinalEvaluationArtifact(output_path, validated, True)

        evaluator_arguments: dict[str, Any] = {"timeout_seconds": timeout}
        if default_candidate_evaluator:
            evaluator_arguments["executor"] = evaluator_executor
        workspace_result = _validate_programbench_candidate_result(
            canonicalize(
                candidate_evaluator(
                    plan,
                    workspace_root,
                    **evaluator_arguments,
                )
            ),
            label="workspace",
            dataset_id=plan.dataset_id,
            plan_hash=plan.plan_hash,
            candidate_digest=workspace_digest,
            manifest=manifest,
        )
        if workspace_digest == mainline_digest:
            mainline_result = canonicalize(workspace_result)
        else:
            mainline_result = _validate_programbench_candidate_result(
                canonicalize(
                    candidate_evaluator(
                        plan,
                        mainline_root,
                        **evaluator_arguments,
                    )
                ),
                label="mainline",
                dataset_id=plan.dataset_id,
                plan_hash=plan.plan_hash,
                candidate_digest=mainline_digest,
                manifest=manifest,
            )
        runners = manual_check_runners or DEFAULT_MANUAL_CHECK_RUNNERS
        workspace_manual_results = _run_manual_checks(
            declarations=declarations,
            enabled=manual_enabled,
            candidate_root=workspace_root,
            candidate_digest=workspace_digest,
            runners=runners,
        )
        mainline_manual_results = _run_manual_checks(
            declarations=declarations,
            enabled=manual_enabled,
            candidate_root=mainline_root,
            candidate_digest=mainline_digest,
            runners=runners,
        )
        if workspace_digest == mainline_digest:
            mainline_manual_results = canonicalize(workspace_manual_results)
        functional_overlap_payload: dict[str, Any] | None = None
        try:
            from society_core.functional_overlap import functional_overlap

            spec = resolve_oss_evaluator_spec(dict(assets))
            functional_overlap_payload = functional_overlap(
                starter_dir=spec.starter_repo_dir,
                reference_dir=spec.reference_repo_dir,
                candidate_dir=workspace_root,
            ).to_dict()
        except Exception as exc:
            functional_overlap_payload = {
                "schema_version": "functional_overlap_v2",
                "error": f"{type(exc).__name__}: {exc}"[:300],
            }
        mainline_release_ready = bool(mainline_result["formal_claim_ready"]) and (
            _manual_checks_pass(
                declarations,
                enabled=manual_enabled,
                results=mainline_manual_results,
            )
        )
        extension = _build_programbench_dual_extension(
            profile_id=profile_id,
            candidate_view=candidate_view,
            plan_hash=plan.plan_hash,
            evaluator_environment_hash=plan.evaluator_environment_hash,
            candidate_freeze=candidate_freeze,
            workspace_result=workspace_result,
            mainline_result=mainline_result,
            mainline_manual_checks=mainline_manual_results,
            mainline_formal_release_ready=mainline_release_ready,
            primary_selection=primary_selection,
        )
        payload: dict[str, Any] = {
            "schema_version": FINAL_EVIDENCE_SCHEMA_VERSION,
            "run_tag": run_tag,
            "functional_overlap": functional_overlap_payload,
            "world_tick": int(getattr(world, "world_tick", 0)),
            "dataset_id": plan.dataset_id,
            "plan_hash": plan.plan_hash,
            "candidate_repo_digest": workspace_digest,
            "starter_repo_digest": plan.starter_repo_digest,
            "reference_repo_digest": plan.reference_repo_digest,
            "hidden_suite_hash": plan.hidden_suite_hash,
            "evaluator_environment_hash": plan.evaluator_environment_hash,
            "manual_check_declarations": canonicalize(declarations),
            "manual_checks_enabled": manual_enabled,
            "manual_checks": workspace_manual_results,
            "formal_release_ready": bool(workspace_result["formal_claim_ready"])
            and _manual_checks_pass(
                declarations,
                enabled=manual_enabled,
                results=workspace_manual_results,
            ),
            "qualified_plan": canonicalize(plan_payload),
            "result": workspace_result,
            "evaluation": workspace_result,
            "programbench_dual_evaluation": extension,
            "programbench_primary_selection": canonicalize(primary_selection),
        }
        payload["artifact_hash"] = _artifact_hash(payload)
        validated = validate_final_evaluation_evidence(
            payload,
            dataset_id=plan.dataset_id,
            plan_hash=plan.plan_hash,
            candidate_digest=workspace_digest,
            manifest=manifest,
            manual_checks_enabled=manual_enabled,
        )
        validated = _validate_programbench_dual_extension(
            validated,
            profile_id=profile_id,
            candidate_view=candidate_view,
            plan=plan,
            workspace_digest=workspace_digest,
            mainline_digest=mainline_digest,
            manifest=manifest,
            declarations=declarations,
            manual_enabled=manual_enabled,
            primary_selection=primary_selection,
        )
        published = _publish_json_exclusive(output_path, validated)
        if not published:
            concurrent = _read_final_evidence(output_root, output_path)
            validated = validate_final_evaluation_evidence(
                concurrent,
                dataset_id=plan.dataset_id,
                plan_hash=plan.plan_hash,
                candidate_digest=workspace_digest,
                manifest=manifest,
                manual_checks_enabled=manual_enabled,
            )
            validated = _validate_programbench_dual_extension(
                validated,
                profile_id=profile_id,
                candidate_view=candidate_view,
                plan=plan,
                workspace_digest=workspace_digest,
                mainline_digest=mainline_digest,
                manifest=manifest,
                declarations=declarations,
                manual_enabled=manual_enabled,
                primary_selection=primary_selection,
            )
            return FinalEvaluationArtifact(output_path, validated, True)
        return FinalEvaluationArtifact(output_path, validated, False)


def _read_final_evidence(root: Path, path: Path) -> dict[str, Any]:
    return json.loads(
        read_regular_file_text(
            _long_path_compatible(root),
            _long_path_compatible(path),
            max_bytes=_MAX_FINAL_EVIDENCE_BYTES,
        )
    )


def _final_evaluation_key(
    *,
    dataset_id: str,
    plan_hash: str,
    candidate_digest: str,
) -> str:
    """Return the full content-addressed identity for one final evaluation."""
    return stable_hash(
        {
            "dataset_id": dataset_id,
            "plan_hash": plan_hash,
            "candidate_repo_digest": candidate_digest,
        }
    )


def _final_evaluation_output_path(
    root: Path,
    *,
    dataset_id: str,
    plan_hash: str,
    candidate_digest: str,
) -> Path:
    """Choose a bounded evidence name while retaining legacy evidence reuse.

    The former name embedded two 64-character digests and could exceed the
    Windows path limit under ordinary pytest and evaluator roots.  The compact
    name retains a full SHA-256 identity over all three key fields; loading the
    evidence still validates those fields against the caller's expected values.
    """
    legacy = root / (
        f"final_evaluation_{dataset_id}_{plan_hash}_{candidate_digest}.json"
    )
    try:
        if legacy.exists():
            return legacy
    except OSError:
        # Probing an overlong legacy path can itself fail on Windows.  It cannot
        # name an existing usable file in that environment, so use the compact
        # representation below.
        pass
    evidence_key = _final_evaluation_key(
        dataset_id=dataset_id,
        plan_hash=plan_hash,
        candidate_digest=candidate_digest,
    )
    return root / f"final_evaluation_{evidence_key}.json"


def _final_evaluation_temp_key(
    path: Path,
    payload: Mapping[str, Any],
) -> str:
    dataset_id = payload.get("dataset_id")
    plan_hash = payload.get("plan_hash")
    candidate_digest = payload.get("candidate_repo_digest")
    if all(isinstance(value, str) and value for value in (
        dataset_id,
        plan_hash,
        candidate_digest,
    )):
        return _final_evaluation_key(
            dataset_id=dataset_id,
            plan_hash=plan_hash,
            candidate_digest=candidate_digest,
        )
    return stable_hash({"final_evaluation_output_name": path.name})


def _long_path_compatible(path: Path) -> Path:
    """Return a Windows extended path without changing its file identity."""
    if os.name != "nt":
        return path
    absolute = os.path.abspath(os.fspath(path))
    if absolute.startswith("\\\\?\\"):
        return Path(absolute)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)


def _publish_json_exclusive(
    path: Path,
    payload: Mapping[str, Any],
) -> bool:
    body = (
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    if len(body) > _MAX_FINAL_EVIDENCE_BYTES:
        raise ValueError("final_evidence_too_large")
    publish_root = _long_path_compatible(path.parent)
    publish_root.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".fe-{_final_evaluation_temp_key(path, payload)}-",
        suffix=".tmp",
        dir=publish_root,
    )
    temporary = Path(temporary_name)
    try:
        fchmod = getattr(os, "fchmod", None)
        if callable(fchmod):
            fchmod(descriptor, 0o600)
        else:
            os.chmod(temporary, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(
                _long_path_compatible(temporary),
                _long_path_compatible(path),
            )
        except FileExistsError:
            return False
        _fsync_directory(path.parent)
        return True
    finally:
        _long_path_compatible(temporary).unlink(missing_ok=True)
        _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":  # pragma: no cover - Windows compatibility.
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _first_environment(*names: str) -> str | None:
    return next(
        (value for name in names if (value := os.environ.get(name)) not in (None, "")),
        None,
    )


def _environment_value(*names: str) -> Any:
    value = _first_environment(*names)
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _provider_model(world: Any) -> tuple[str, str]:
    client = getattr(world, "llm_client", None)
    inner = getattr(client, "inner", client)
    provider = (
        getattr(inner, "provider", None) or getattr(client, "provider", None) or "none"
    )
    model = getattr(inner, "model", None) or getattr(client, "model", None) or "rules"
    return str(provider), str(model)


def _normalize_time(value: Any) -> Any:
    if value is not None:
        return value
    return dt.datetime.now(dt.timezone.utc)


def _programbench_dual_record_provenance(
    artifact: FinalEvaluationArtifact,
) -> dict[str, Any] | None:
    extension = artifact.payload.get("programbench_dual_evaluation")
    if not isinstance(extension, Mapping):
        return None
    supplied_hash = extension.get("evidence_hash")
    expected_hash = stable_hash(
        {key: value for key, value in extension.items() if key != "evidence_hash"}
    )
    if supplied_hash != expected_hash:
        raise ValueError("programbench dual evidence hash mismatch")
    workspace = extension.get("workspace_candidate")
    mainline = extension.get("mainline_realization")
    delivery_gap = extension.get("delivery_gap")
    if not all(isinstance(value, Mapping) for value in (workspace, mainline, delivery_gap)):
        raise ValueError("programbench dual evidence summary missing")
    return {
        "schema_version": PROGRAMBENCH_DUAL_EVALUATION_SCHEMA_VERSION,
        "profile_id": extension.get("profile_id"),
        "candidate_view": extension.get("candidate_view"),
        "primary_selection": canonicalize(extension.get("primary_selection")),
        "evaluation_boundary": extension.get("evaluation_boundary"),
        "hidden_feedback_returned_to_world": extension.get(
            "hidden_feedback_returned_to_world"
        ),
        "evidence_hash": supplied_hash,
        "workspace_candidate_repo_digest": workspace.get(
            "candidate_repo_digest"
        ),
        "workspace_candidate_result_hash": workspace.get("result_hash"),
        "mainline_candidate_repo_digest": mainline.get("candidate_repo_digest"),
        "mainline_candidate_result_hash": mainline.get("result_hash"),
        "delivery_gap": canonicalize(delivery_gap),
    }


def write_experiment_run_record(
    world: Any,
    output_path: str | os.PathLike[str],
    *,
    final_evaluator: FinalEvaluationArtifact
    | Mapping[str, Any]
    | str
    | os.PathLike[str]
    | None,
    started_at: Any,
    ended_at: Any = None,
    status: str,
    checkpoint: Mapping[str, Any] | str | os.PathLike[str] | None,
    failure_reason: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    replication_id: str | None = None,
    randomization_block: Any = None,
    randomization_order: Any = None,
    provenance: Mapping[str, Any] | None = None,
    model_cutoff_policy: str | None = None,
    contamination_status: str | None = None,
    contamination_probes: Mapping[str, Any] | str | os.PathLike[str] | None = None,
    contamination_clearance: bool | None = None,
) -> dict[str, Any]:
    """Build and atomically persist the canonical ``orgenv_experiment_run_v2``."""

    from environments.org_env.experiments.records import (
        build_experiment_run_record,
    )

    resolved_provider, resolved_model = _provider_model(world)
    final_source: Any = final_evaluator
    if isinstance(final_evaluator, FinalEvaluationArtifact):
        final_source = _long_path_compatible(final_evaluator.path)
    environment_names = (
        "ORG_EXPERIMENT_REPLICATION_ID",
        "ORG_REPLICATION_ID",
        "ORG_EXPERIMENT_RANDOMIZATION_BLOCK",
        "ORG_RANDOMIZATION_BLOCK",
        "ORG_EXPERIMENT_RANDOMIZATION_ORDER",
        "ORG_RANDOMIZATION_ORDER",
    )
    provenance_payload = dict(provenance or {})
    provenance_payload["experiment_environment"] = {
        name: os.environ[name] for name in environment_names if name in os.environ
    }
    if isinstance(final_evaluator, FinalEvaluationArtifact):
        provenance_payload["final_evaluator_reused"] = final_evaluator.reused
        dual_provenance = _programbench_dual_record_provenance(final_evaluator)
        if dual_provenance is not None:
            provenance_payload["execution_profile"] = dual_provenance["profile_id"]
            provenance_payload["programbench_dual_evaluation"] = dual_provenance
    resolved_clearance = contamination_clearance
    if resolved_clearance is None:
        raw_clearance = _first_environment("ORG_CONTAMINATION_CLEARANCE")
        if raw_clearance is not None:
            parsed_clearance = _environment_value("ORG_CONTAMINATION_CLEARANCE")
            if not isinstance(parsed_clearance, bool):
                raise ValueError(
                    "ORG_CONTAMINATION_CLEARANCE must be JSON true or false"
                )
            resolved_clearance = parsed_clearance
    resolved_probes = contamination_probes or _first_environment(
        "ORG_CONTAMINATION_PROBES_PATH"
    )

    record = build_experiment_run_record(
        world,
        provider=provider or resolved_provider,
        model=model or resolved_model,
        replication_id=replication_id
        or _first_environment(
            "ORG_EXPERIMENT_REPLICATION_ID",
            "ORG_REPLICATION_ID",
        ),
        randomization_block=(
            randomization_block
            if randomization_block is not None
            else _environment_value(
                "ORG_EXPERIMENT_RANDOMIZATION_BLOCK",
                "ORG_RANDOMIZATION_BLOCK",
            )
        ),
        randomization_order=(
            randomization_order
            if randomization_order is not None
            else _environment_value(
                "ORG_EXPERIMENT_RANDOMIZATION_ORDER",
                "ORG_RANDOMIZATION_ORDER",
            )
        ),
        started_at=started_at,
        ended_at=_normalize_time(ended_at),
        status=status,
        failure_reason=failure_reason,
        checkpoint=checkpoint,
        provenance=provenance_payload,
        model_cutoff_policy=model_cutoff_policy
        or _first_environment("ORG_MODEL_CUTOFF_POLICY"),
        contamination_status=contamination_status
        or _first_environment("ORG_CONTAMINATION_STATUS"),
        contamination_probes=resolved_probes,
        contamination_clearance=resolved_clearance,
        final_evaluator=final_source,
    )
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    capability_evidence = record.get("organizational_capability_evidence")
    if isinstance(capability_evidence, Mapping):
        evidence_path = path.with_name("organizational_capability_evidence.json")
        evidence_temporary = evidence_path.with_name(
            f".{evidence_path.name}.{os.getpid()}.tmp"
        )
        evidence_temporary.write_text(
            json.dumps(capability_evidence, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(evidence_temporary, evidence_path)
        record["provenance"]["organizational_capability_evidence_file"] = (
            evidence_path.name
        )
    profile_evidence = record.get("profile_causality_evidence")
    if isinstance(profile_evidence, Mapping):
        profile_path = path.with_name("profile_causality_evidence.json")
        profile_temporary = profile_path.with_name(
            f".{profile_path.name}.{os.getpid()}.tmp"
        )
        profile_temporary.write_text(
            json.dumps(profile_evidence, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(profile_temporary, profile_path)
        record["provenance"]["profile_causality_evidence_file"] = profile_path.name
    from environments.org_env.experiments.records import (
        validate_experiment_run_record_schema,
    )

    validate_experiment_run_record_schema(record)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return record


__all__ = [
    "DEFAULT_MANUAL_CHECK_RUNNERS",
    "FINAL_EVIDENCE_SCHEMA_VERSION",
    "PROGRAMBENCH_DUAL_EVALUATION_SCHEMA_VERSION",
    "FinalEvaluationArtifact",
    "manifest_manual_release_checks",
    "manifest_requires_manual_release_checks",
    "run_docker_launch_check",
    "run_final_evaluation",
    "validate_final_evaluation_evidence",
    "write_experiment_run_record",
]
