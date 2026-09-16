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
    validate_formal_oss_world,
)
from relic.research.hashing import canonicalize, stable_hash
from relic.research.redaction import redact_sensitive_payload
from relic.research.safe_files import read_regular_file_text

FINAL_EVIDENCE_SCHEMA_VERSION = "orgenv_oss_final_evaluation_v1"
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
    container_name = f"relic-oss-{candidate_digest[:12]}-{os.getpid()}"
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


def _frozen_evaluator_executor():
    from relic.evaluation.execution import ExecutionPolicy, build_command_executor

    values = {
        "backend": os.environ.get("RELIC_EVALUATOR_BACKEND"),
        "container_image": os.environ.get("RELIC_EVALUATOR_CONTAINER_IMAGE"),
        "container_platform": os.environ.get("RELIC_EVALUATOR_CONTAINER_PLATFORM"),
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
        if _formal_mode(world) and (default_plan_builder or default_candidate_evaluator)
        else None
    )
    if default_plan_builder:
        from relic.evaluation.time_machine import (
            build_time_machine_evaluation_plan,
        )

        plan_builder = build_time_machine_evaluation_plan
    if default_candidate_evaluator:
        from relic.evaluation.time_machine import (
            evaluate_time_machine_candidate,
        )

        candidate_evaluator = evaluate_time_machine_candidate
    if candidate_exporter is None:
        from environments.org_env.product.materialize import export_product_repo

        candidate_exporter = export_product_repo
    from relic.evaluation.time_machine import _repo_digest

    plan_arguments: dict[str, Any] = {
        "dataset_id": dataset_id,
        "timeout_seconds": timeout,
    }
    if default_plan_builder:
        plan_arguments["executor"] = evaluator_executor
    plan = plan_builder(**plan_arguments)
    expected_environment_hash = os.environ.get(
        "RELIC_EVALUATOR_EXPECTED_ENVIRONMENT_HASH"
    )
    expected_qualification_hash = os.environ.get(
        "RELIC_EVALUATOR_EXPECTED_QUALIFICATION_HASH"
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

    with tempfile.TemporaryDirectory(prefix="oss_final_candidate_") as temporary_dir:
        candidate_root = Path(temporary_dir) / "candidate"
        candidate_exporter(
            world,
            str(candidate_root),
            prefer_mainline=True,
        )
        candidate_digest = _repo_digest(candidate_root)
        output_path = output_root / (
            f"final_evaluation_{dataset_id}_{plan.plan_hash}_{candidate_digest}.json"
        )
        if output_path.exists():
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
            from environments.org_env.product.substrates.loader import (
                load_oss_substrate_spec,
            )
            from relic.evaluation.functional_overlap import functional_overlap

            spec = load_oss_substrate_spec(dataset_id)
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


def _read_final_evidence(root: Path, path: Path) -> dict[str, Any]:
    return json.loads(
        read_regular_file_text(
            root,
            path,
            max_bytes=_MAX_FINAL_EVIDENCE_BYTES,
        )
    )


def _publish_json_exclusive(
    path: Path,
    payload: Mapping[str, Any],
) -> bool:
    body = (
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    if len(body) > _MAX_FINAL_EVIDENCE_BYTES:
        raise ValueError("final_evidence_too_large")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        _fsync_directory(path.parent)
        return True
    finally:
        temporary.unlink(missing_ok=True)
        _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
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
        final_source = final_evaluator.path
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
    "FinalEvaluationArtifact",
    "manifest_manual_release_checks",
    "manifest_requires_manual_release_checks",
    "run_docker_launch_check",
    "run_final_evaluation",
    "validate_final_evaluation_evidence",
    "write_experiment_run_record",
]
