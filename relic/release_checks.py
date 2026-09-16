"""Non-destructive release environment gates and bounded smoke checks.

The checks in this module deliberately stop before constructing an LLM client.
They are intended for a release operator (and a thin CLI wrapper), not as a
substitute for a recorded formal experiment.  In particular, an image is only
inspected when it is already available locally; this module never pulls one.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from relic.manifest import recommended_parallelism, visible_memory_gib
from relic.paths import default_output_root, project_root

REPORT_SCHEMA_VERSION = "relic-release-check-report-v1"

# These values are intentionally part of the module's public API.  A CLI can
# return report["exit_code"] unchanged without reimplementing gate semantics.
EXIT_OK = 0
EXIT_CHECK_FAILED = 2
EXIT_FORMAL_GATE_FAILED = 3
EXIT_INVALID_REQUEST = 64

_SCOPES = frozenset({"core", "formal"})
_SMOKE_MODES = frozenset({"mock", "formal"})
_DIGEST_PINNED_IMAGE = re.compile(
    r"(?:^sha256:[0-9a-fA-F]{64}$|@sha256:[0-9a-fA-F]{64}$)"
)
_SUPPORTED_PROVIDERS = frozenset({"openai"})
_CREDENTIAL_ENVIRONMENTS = {
    "openai": ("ORG_LLM_API_KEY", "OPENAI_API_KEY"),
}
_RUNTIME_MODEL_ENVIRONMENT = {
    "openai": "ORG_LLM_API_KEY or OPENAI_API_KEY",
}


def _check(
    name: str,
    status: Literal["pass", "fail", "warn", "skip"],
    code: str,
    **details: Any,
) -> dict[str, Any]:
    """Build a JSON-safe check record with no incidental command output."""

    return {
        "name": name,
        "status": status,
        "code": code,
        "details": details,
    }


def _failed(checks: Iterable[Mapping[str, Any]]) -> bool:
    return any(check.get("status") == "fail" for check in checks)


def _finish(
    *,
    scope: str,
    checks: list[dict[str, Any]],
    recommendations: Mapping[str, Any],
    invalid_request: bool = False,
    smoke_mode: str | None = None,
) -> dict[str, Any]:
    failed = _failed(checks)
    if invalid_request:
        exit_code = EXIT_INVALID_REQUEST
    elif failed and scope == "formal":
        exit_code = EXIT_FORMAL_GATE_FAILED
    elif failed:
        exit_code = EXIT_CHECK_FAILED
    else:
        exit_code = EXIT_OK
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "scope": scope,
        "status": "failed" if failed or invalid_request else "passed",
        "exit_code": exit_code,
        "checks": checks,
        "recommendations": dict(recommendations),
        # This is a behavioural guarantee, not merely a note: no function in
        # this module builds an LLM client or sends a provider request.
        "provider_calls_made": 0,
        "formal_experiment_started": False,
    }
    if smoke_mode is not None:
        report["mode"] = smoke_mode
    return report


def report_exit_code(report: Mapping[str, Any]) -> int:
    """Return the stable process exit value represented by a report."""

    value = report.get("exit_code", EXIT_CHECK_FAILED)
    return int(value) if isinstance(value, int) else EXIT_CHECK_FAILED


def _platform_details() -> tuple[bool, dict[str, Any]]:
    system = platform.system().lower()
    release = platform.release().lower()
    version = platform.version().lower()
    is_wsl = system == "linux" and ("microsoft" in release or "microsoft" in version)
    if is_wsl:
        kind = "wsl"
    elif system == "linux":
        kind = "linux"
    else:
        kind = "non_linux"
    return system == "linux", {
        "system": system or "unknown",
        "architecture": platform.machine().lower() or "unknown",
        "environment": kind,
        "is_wsl": is_wsl,
    }


def _nearest_existing_parent(path: Path) -> Path | None:
    candidate = path.expanduser()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate if candidate.exists() and candidate.is_dir() else None


def _write_location_check(name: str, path: Path) -> dict[str, Any]:
    """Check whether a location can be created without creating it ourselves."""

    parent = _nearest_existing_parent(path)
    if parent is None:
        return _check(name, "fail", "write_location_parent_missing", path=str(path))
    writable = os.access(parent, os.W_OK | os.X_OK)
    return _check(
        name,
        "pass" if writable else "fail",
        "write_location_writable" if writable else "write_location_not_writable",
        path=str(path),
        checked_parent=str(parent),
        write_probe_performed=False,
    )


def _normalise_requested_model(model: str | None) -> tuple[str | None, dict[str, Any] | None]:
    """Resolve a known model without echoing arbitrary caller input into reports."""

    if model is None:
        return None, None
    if not isinstance(model, str) or not model.strip():
        return None, _check("model_selection", "fail", "model_selection_invalid")
    requested = model.strip().lower()
    try:
        from relic.manifest import load_yaml
        from relic.paths import config_root

        study = load_yaml(config_root() / "main-study.yaml")
        configured = {str(value).strip().lower() for value in study.get("models", ())}
    except Exception:
        return None, _check("model_selection", "fail", "canonical_config_unavailable")
    if requested not in configured:
        return None, _check("model_selection", "fail", "model_not_in_main_study")
    return requested, None


def _canonical_checks(model: str | None) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """Compile all canonical cells without executing them or calling a provider."""

    checks: list[dict[str, Any]] = []
    try:
        from relic.benchmark import verify_benchmark
        from relic.cell_spec import compile_cell_spec
        from relic.manifest import load_yaml
        from relic.paths import config_root

        study = load_yaml(config_root() / "main-study.yaml")
        all_models = tuple(str(value).strip().lower() for value in study["models"])
        models = (model,) if model is not None else all_models
        workloads = tuple(str(value).strip().lower() for value in study["workloads"])
        arms = tuple(str(value).strip().lower() for value in study["arms"])
        seeds = tuple(int(value) for value in study["seeds"])
        expected_per_model = len(workloads) * len(arms) * len(seeds)
        if expected_per_model != 120:
            checks.append(
                _check(
                    "canonical_cells",
                    "fail",
                    "canonical_single_model_cell_count_invalid",
                    expected=120,
                    found=expected_per_model,
                )
            )
            return checks, all_models

        compiled_ids: set[str] = set()
        for selected_model in models:
            per_model_ids: set[str] = set()
            for workload in workloads:
                for arm in arms:
                    for seed in seeds:
                        spec = compile_cell_spec(
                            model=selected_model,
                            workload=workload,
                            arm=arm,
                            seed=seed,
                            # A release gate should not write the configured
                            # output location just to validate a CellSpec.
                            output_root=Path(tempfile.gettempdir()),
                            verify_pack=False,
                        )
                        per_model_ids.add(spec.cell_id)
                        compiled_ids.add(spec.cell_id)
            if len(per_model_ids) != 120:
                checks.append(
                    _check(
                        "canonical_cells",
                        "fail",
                        "canonical_cellspec_count_invalid",
                        model=selected_model,
                        expected=120,
                        found=len(per_model_ids),
                    )
                )
                return checks, all_models

        failures = verify_benchmark()
        if failures:
            checks.append(
                _check(
                    "benchmark_integrity",
                    "fail",
                    "benchmark_integrity_failed",
                    failure_count=len(failures),
                )
            )
        else:
            checks.append(
                _check(
                    "benchmark_integrity",
                    "pass",
                    "benchmark_integrity_verified",
                    workload_count=len(workloads),
                )
            )
        checks.append(
            _check(
                "canonical_cells",
                "pass",
                "canonical_cellspecs_compiled",
                models=list(models),
                cells_per_model=120,
                compiled_cell_count=len(compiled_ids),
            )
        )
        return checks, all_models
    except Exception as exc:
        # Configuration errors are deliberately classified, not printed.  A
        # report must remain safe even if a local config contains credentials.
        checks.append(
            _check(
                "canonical_config",
                "fail",
                "canonical_config_or_cellspec_invalid",
                error_type=type(exc).__name__,
            )
        )
        return checks, ()


def _package_import_check() -> dict[str, Any]:
    try:
        importlib.import_module("relic")
        importlib.import_module("environments")
        importlib.import_module("relic.cell_spec")
    except Exception as exc:
        return _check(
            "package_imports",
            "fail",
            "release_package_import_failed",
            error_type=type(exc).__name__,
        )
    return _check("package_imports", "pass", "release_packages_importable")


def _tooling_checks() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for executable, required in (("uv", True), ("git", False)):
        available = shutil.which(executable) is not None
        if available:
            checks.append(
                _check(
                    f"system_dependency_{executable}",
                    "pass",
                    f"{executable}_executable_available",
                )
            )
        else:
            checks.append(
                _check(
                    f"system_dependency_{executable}",
                    "fail" if required else "warn",
                    f"{executable}_executable_not_found",
                )
            )
    return checks


def _optional_release_checks() -> list[dict[str, Any]]:
    credential_names = _CREDENTIAL_ENVIRONMENTS["openai"]
    credential_configured = any(bool(os.environ.get(name)) for name in credential_names)
    inspector_root = project_root() / "inspector"
    inspector_available = inspector_root.is_dir()
    return [
        _check(
            "optional_model_credentials",
            "pass" if credential_configured else "warn",
            (
                "optional_model_credential_available"
                if credential_configured
                else "optional_model_credential_missing"
            ),
            accepted_environment_variables=list(credential_names),
            credential_configured=credential_configured,
        ),
        _check(
            "inspector",
            "pass" if inspector_available else "warn",
            "inspector_available" if inspector_available else "inspector_not_in_current_release",
        ),
    ]


def _provider_checks(models: Iterable[str]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    try:
        from relic.manifest import load_yaml
        from relic.paths import config_root

        for model in models:
            config = load_yaml(config_root() / "models" / f"{model}.yaml")
            provider = str(config.get("provider") or "").strip().lower()
            runtime_default = str(config.get("runtime_model_default") or "").strip()
            runtime_env = str(config.get("runtime_model_env") or "").strip()
            if provider not in _SUPPORTED_PROVIDERS:
                checks.append(
                    _check(
                        "model_provider",
                        "fail",
                        "model_provider_unsupported",
                        model=model,
                        provider=provider or "missing",
                        supported_providers=sorted(_SUPPORTED_PROVIDERS),
                    )
                )
                checks.append(
                    _check(
                        "model_credentials",
                        "skip",
                        "model_credential_not_checked_for_unsupported_provider",
                        model=model,
                    )
                )
                continue
            checks.append(
                _check(
                    "model_provider",
                    "pass",
                    "model_provider_supported",
                    model=model,
                    provider=provider,
                )
            )
            credential_names = _CREDENTIAL_ENVIRONMENTS[provider]
            configured = any(bool(os.environ.get(name)) for name in credential_names)
            checks.append(
                _check(
                    "model_credentials",
                    "pass" if configured else "fail",
                    "model_credential_available" if configured else "model_credential_missing",
                    model=model,
                    accepted_environment_variables=list(credential_names),
                    credential_configured=configured,
                )
            )
            if runtime_default:
                checks.append(
                    _check(
                        "model_runtime_binding",
                        "pass",
                        "model_runtime_default_bound",
                        model=model,
                    )
                )
            elif runtime_env and os.environ.get(runtime_env):
                checks.append(
                    _check(
                        "model_runtime_binding",
                        "pass",
                        "model_runtime_environment_bound",
                        model=model,
                        environment_variable=runtime_env,
                    )
                )
            else:
                checks.append(
                    _check(
                        "model_runtime_binding",
                        "fail",
                        "model_runtime_binding_missing",
                        model=model,
                        environment_variable=runtime_env or _RUNTIME_MODEL_ENVIRONMENT[provider],
                    )
                )
    except Exception as exc:
        checks.append(
            _check(
                "model_provider",
                "fail",
                "model_provider_configuration_invalid",
                error_type=type(exc).__name__,
            )
        )
    return checks


def _run_runtime_command(argv: tuple[str, ...]) -> tuple[bool, str]:
    """Run a short, read-only runtime probe and discard all command output."""

    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, "runtime_command_unavailable"
    return completed.returncode == 0, "runtime_command_succeeded" if completed.returncode == 0 else "runtime_command_failed"


def _docker_image_is_linux_amd64(docker: str, image: str) -> tuple[bool, str]:
    """Inspect an already-local image and verify its declared platform.

    Docker's return code alone says only that an object with this name exists;
    a formal evaluator additionally requires exactly ``linux/amd64``.  The
    inspected value stays internal so a report cannot expose an arbitrary
    image's metadata or registry reference.
    """

    try:
        completed = subprocess.run(
            (
                docker,
                "image",
                "inspect",
                "--format",
                "{{.Os}}/{{.Architecture}}",
                image,
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, "docker_image_unavailable"
    if completed.returncode != 0:
        return False, "docker_image_unavailable"
    if completed.stdout.strip().lower() != "linux/amd64":
        return False, "docker_image_platform_mismatch"
    return True, "docker_image_qualified"


def _apptainer_image_is_cached(apptainer: str, image: str) -> tuple[bool, str]:
    """Check Apptainer's local cache for a pinned registry digest without pulling."""

    match = re.search(r"@sha256:([0-9a-fA-F]{64})$", image)
    if match is None:
        return False, "apptainer_registry_digest_required"
    try:
        completed = subprocess.run(
            (apptainer, "cache", "list", "-v"),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, "apptainer_cache_unavailable"
    if completed.returncode != 0:
        return False, "apptainer_cache_unavailable"
    if match.group(1).lower() not in completed.stdout.lower():
        return False, "apptainer_image_not_cached"
    return True, "apptainer_image_cache_entry_available"


def _formal_evaluator_checks() -> list[dict[str, Any]]:
    """Mirror the cell worker's evaluator binding gate without running it."""

    checks: list[dict[str, Any]] = []
    backend = str(os.environ.get("RELIC_EVALUATOR_BACKEND") or "").strip().lower()
    image = str(os.environ.get("RELIC_EVALUATOR_CONTAINER_IMAGE") or "").strip()
    container_platform = str(
        os.environ.get("RELIC_EVALUATOR_CONTAINER_PLATFORM") or ""
    ).strip().lower()

    if not backend:
        checks.append(
            _check(
                "evaluator_backend",
                "fail",
                "formal_evaluator_runtime_binding_missing",
                missing="backend",
                allowed_backends=["docker", "apptainer"],
            )
        )
    elif backend not in {"docker", "apptainer"}:
        checks.append(
            _check(
                "evaluator_backend",
                "fail",
                "formal_evaluator_backend_forbidden",
                allowed_backends=["docker", "apptainer"],
            )
        )
    else:
        checks.append(
            _check("evaluator_backend", "pass", "formal_evaluator_backend_supported", backend=backend)
        )

    if not image:
        checks.append(
            _check(
                "evaluator_image",
                "fail",
                "formal_evaluator_runtime_binding_missing",
                missing="container_image",
            )
        )
    elif not _DIGEST_PINNED_IMAGE.search(image):
        checks.append(
            _check(
                "evaluator_image",
                "fail",
                "formal_digest_pinned_image_required",
            )
        )
    else:
        checks.append(_check("evaluator_image", "pass", "formal_digest_pinned_image_configured"))

    if not container_platform:
        checks.append(
            _check(
                "evaluator_platform",
                "fail",
                "formal_evaluator_runtime_binding_missing",
                missing="container_platform",
                required="linux/amd64",
            )
        )
    elif container_platform != "linux/amd64":
        checks.append(
            _check(
                "evaluator_platform",
                "fail",
                "formal_evaluator_platform_mismatch",
                required="linux/amd64",
            )
        )
    else:
        checks.append(
            _check("evaluator_platform", "pass", "formal_evaluator_platform_supported", platform="linux/amd64")
        )

    if backend not in {"docker", "apptainer"} or not image or container_platform != "linux/amd64":
        checks.append(
            _check(
                "evaluator_runtime",
                "fail",
                "formal_evaluator_runtime_unqualified",
                reason="runtime_binding_incomplete_or_invalid",
            )
        )
        return checks

    # Keep this in lockstep with cell_worker._preflight_evaluator.  Constructing
    # a policy validates the same pure-Python constraints, with no container or
    # provider activity.
    try:
        from relic.evaluation.execution import ExecutionPolicy

        ExecutionPolicy(
            trust_level="untrusted",
            backend=backend,
            container_image=image,
            container_platform=container_platform,
            network_enabled=False,
        )
    except Exception as exc:
        checks.append(
            _check(
                "evaluator_policy",
                "fail",
                "formal_evaluator_policy_invalid",
                error_type=type(exc).__name__,
            )
        )
        return checks
    checks.append(_check("evaluator_policy", "pass", "formal_evaluator_policy_compatible"))

    if backend == "docker":
        docker = shutil.which("docker")
        if docker is None:
            checks.append(_check("evaluator_runtime_binary", "fail", "docker_executable_not_found"))
            checks.append(_check("evaluator_daemon", "fail", "docker_daemon_unavailable"))
            checks.append(_check("evaluator_image_qualification", "fail", "docker_image_unqualified"))
            return checks
        checks.append(_check("evaluator_runtime_binary", "pass", "docker_executable_available"))
        daemon_ok, _ = _run_runtime_command((docker, "info", "--format", "{{.ServerVersion}}"))
        if not daemon_ok:
            checks.append(_check("evaluator_daemon", "fail", "docker_daemon_unavailable"))
            checks.append(_check("evaluator_image_qualification", "fail", "docker_image_unqualified"))
            return checks
        checks.append(_check("evaluator_daemon", "pass", "docker_daemon_reachable"))
        image_ok, image_code = _docker_image_is_linux_amd64(docker, image)
        checks.append(
            _check(
                "evaluator_image_qualification",
                "pass" if image_ok else "fail",
                image_code,
                image_pull_performed=False,
            )
        )
        return checks

    apptainer = shutil.which("apptainer")
    if apptainer is None:
        checks.append(_check("evaluator_runtime_binary", "fail", "apptainer_executable_not_found"))
        checks.append(_check("evaluator_daemon", "skip", "apptainer_has_no_daemon"))
        checks.append(_check("evaluator_image_qualification", "fail", "apptainer_image_unqualified"))
        return checks
    checks.append(_check("evaluator_runtime_binary", "pass", "apptainer_executable_available"))
    checks.append(_check("evaluator_daemon", "skip", "apptainer_has_no_daemon"))
    slurm_job = str(os.environ.get("SLURM_JOB_ID") or "")
    if not re.fullmatch(r"[0-9]+", slurm_job) or shutil.which("srun") is None:
        checks.append(
            _check(
                "evaluator_apptainer_isolation",
                "fail",
                "apptainer_untrusted_requires_slurm_allocation",
            )
        )
        checks.append(
            _check(
                "evaluator_image_qualification",
                "fail",
                "apptainer_image_unqualified",
                image_pull_performed=False,
            )
        )
        return checks
    checks.append(
        _check(
            "evaluator_apptainer_isolation",
            "pass",
            "apptainer_slurm_allocation_available",
        )
    )
    image_ok, image_code = _apptainer_image_is_cached(apptainer, image)
    checks.append(
        _check(
            "evaluator_image_qualification",
            "pass" if image_ok else "fail",
            image_code,
            image_pull_performed=False,
        )
    )
    return checks


def check_environment(
    *,
    scope: Literal["core", "formal"] = "core",
    model: str | None = None,
    requested_parallelism: int | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe, non-destructive release-readiness report.

    ``scope='formal'`` verifies all the bindings that the formal cell worker
    requires.  It does not call a provider, pull an image, run an evaluator, or
    start a world.  Therefore a green environment gate is necessary but never
    evidence that a formal experiment succeeded.
    """

    checks: list[dict[str, Any]] = []
    if scope not in _SCOPES:
        checks.append(_check("scope", "fail", "release_check_scope_invalid"))
        return _finish(
            scope="invalid",
            checks=checks,
            recommendations={},
            invalid_request=True,
        )

    selected_model, model_error = _normalise_requested_model(model)
    if model_error is not None:
        checks.append(model_error)

    version = sys.version_info
    python_ok = (version.major, version.minor) >= (3, 12)
    checks.append(
        _check(
            "python",
            "pass" if python_ok else "fail",
            "python_version_supported" if python_ok else "python_version_unsupported",
            required="3.12",
            found=f"{version.major}.{version.minor}.{version.micro}",
        )
    )
    checks.append(_package_import_check())
    checks.extend(_tooling_checks())
    checks.extend(_optional_release_checks())

    canonical_checks, configured_models = _canonical_checks(selected_model)
    checks.extend(canonical_checks)

    output_root = default_output_root().expanduser().resolve()
    cache_root = Path(os.environ.get("RELIC_CACHE_ROOT") or output_root / ".relic-cache")
    cache_root = cache_root.expanduser().resolve()
    checks.append(_write_location_check("output_location", output_root))
    checks.append(_write_location_check("cache_location", cache_root))

    linux, system_details = _platform_details()
    checks.append(
        _check(
            "operating_system",
            "pass" if linux else "fail",
            "linux_environment_detected" if linux else "linux_environment_required",
            **system_details,
        )
    )

    memory_gib = round(max(0.0, visible_memory_gib()), 2)
    recommended = recommended_parallelism(memory_gib)
    parallel_details: dict[str, Any] = {
        "visible_memory_gib": memory_gib,
        "recommended_max_parallel": recommended,
        "budget_gib_per_active_cell": 16,
    }
    if requested_parallelism is None:
        checks.append(
            _check("parallelism", "pass", "parallelism_recommendation_available", **parallel_details)
        )
    elif isinstance(requested_parallelism, bool) or not isinstance(requested_parallelism, int) or requested_parallelism < 1:
        checks.append(_check("parallelism", "fail", "requested_parallelism_invalid", **parallel_details))
    elif requested_parallelism > recommended:
        checks.append(
            _check(
                "parallelism",
                "warn",
                "requested_parallelism_above_recommendation",
                requested_parallelism=requested_parallelism,
                **parallel_details,
            )
        )
    else:
        checks.append(
            _check(
                "parallelism",
                "pass",
                "requested_parallelism_within_recommendation",
                requested_parallelism=requested_parallelism,
                **parallel_details,
            )
        )

    if scope == "formal":
        models_for_provider = (selected_model,) if selected_model else configured_models
        if models_for_provider:
            checks.extend(_provider_checks(models_for_provider))
        else:
            checks.append(_check("model_provider", "fail", "model_provider_configuration_unavailable"))
        checks.extend(_formal_evaluator_checks())

    return _finish(
        scope=scope,
        checks=checks,
        recommendations={
            "output_root": str(output_root),
            "cache_root": str(cache_root),
            "recommended_max_parallel": recommended,
            "formal_gate_note": (
                "A passed formal environment gate does not mean a formal experiment was run."
            ),
        },
    )


def _formal_workspace_roundtrip_check() -> dict[str, Any]:
    """Exercise the qualified container boundary without a provider or evaluator.

    This uses the same policy/executor construction as the cell worker.
    ``ensure_command_executor_ready`` writes randomized probe files only in a
    temporary directory and removes them itself. Its network-isolated command
    does not load hidden evaluator assets, start a world, or persist output.
    """

    backend = str(os.environ.get("RELIC_EVALUATOR_BACKEND") or "").strip().lower()
    image = str(os.environ.get("RELIC_EVALUATOR_CONTAINER_IMAGE") or "").strip()
    container_platform = str(
        os.environ.get("RELIC_EVALUATOR_CONTAINER_PLATFORM") or ""
    ).strip().lower()
    try:
        from relic.evaluation.execution import (
            ExecutionPolicy,
            build_command_executor,
            ensure_command_executor_ready,
        )

        policy = ExecutionPolicy(
            trust_level="untrusted",
            backend=backend,
            container_image=image,
            container_platform=container_platform,
            network_enabled=False,
        )
        executor = build_command_executor(policy)
        with tempfile.TemporaryDirectory(prefix="relic_formal_smoke_") as directory:
            ensure_command_executor_ready(
                executor,
                root=Path(directory),
                timeout_seconds=20.0,
            )
    except RuntimeError as exc:
        message = str(exc)
        if message.startswith("execution_backend_preflight_failed:"):
            code = "formal_smoke_executor_preflight_failed"
        elif message.startswith("execution_workspace_roundtrip_failed:"):
            code = "formal_smoke_workspace_roundtrip_failed"
        else:
            code = "formal_smoke_executor_roundtrip_failed"
        return _check(
            "formal_executor_roundtrip",
            "fail",
            code,
            provider_calls_made=0,
            network_enabled=False,
        )
    except Exception as exc:
        return _check(
            "formal_executor_roundtrip",
            "fail",
            "formal_smoke_executor_setup_failed",
            error_type=type(exc).__name__,
            provider_calls_made=0,
            network_enabled=False,
        )
    return _check(
        "formal_executor_roundtrip",
        "pass",
        "formal_smoke_executor_workspace_roundtrip_passed",
        provider_calls_made=0,
        network_enabled=False,
    )


def smoke(
    *,
    mode: Literal["mock", "formal"] = "mock",
    model: str | None = None,
) -> dict[str, Any]:
    """Run a bounded, no-provider smoke check and return the same report shape.

    Mock mode runs an in-process primitive plus temporary output I/O. Formal
    mode first applies the gate and then performs a network-isolated container
    workspace roundtrip through the existing executor API. Neither calls a
    model provider.
    """

    if mode not in _SMOKE_MODES:
        return _finish(
            scope="invalid",
            checks=[_check("mode", "fail", "release_smoke_mode_invalid")],
            recommendations={},
            invalid_request=True,
            smoke_mode="invalid",
        )

    selected_model = model or ("gpt-5.6-terra" if mode == "formal" else None)
    environment = check_environment(
        scope="formal" if mode == "formal" else "core",
        model=selected_model,
    )
    checks = [dict(item) for item in environment["checks"]]
    recommendations = dict(environment["recommendations"])

    if mode == "formal" and environment["status"] != "passed":
        checks.append(
            _check(
                "formal_smoke",
                "fail",
                "formal_smoke_blocked_by_environment_gate",
                evaluator_started=False,
            )
        )
        return _finish(
            scope="formal",
            checks=checks,
            recommendations=recommendations,
            smoke_mode=mode,
        )

    try:
        from relic.cell_spec import compile_cell_spec
        from relic.core.domain import DomainAction, DomainState

        with tempfile.TemporaryDirectory(prefix="relic_release_smoke_") as directory:
            output_root = Path(directory)
            spec = compile_cell_spec(
                model="gpt-5.6-terra",
                workload="w01",
                arm="b0",
                seed=1401,
                output_root=output_root,
                verify_pack=False,
            )
            # Exercise an environment-neutral runtime primitive and a real
            # temporary output write/read path.  This intentionally does not
            # build an OrgWorld, which would seed formal evaluator assets.
            state = DomainState(run_id="release-mock-smoke")
            action = DomainAction(action_type="inspect_task")
            payload = {
                "cell_spec": spec.document(),
                "runtime": {
                    "run_id": state.run_id,
                    "world_tick": state.world_tick,
                    "action": action.action_type,
                },
            }
            output_path = output_root / "mock-smoke.json"
            output_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            restored = json.loads(output_path.read_text(encoding="utf-8"))
            if (
                restored != payload
                or not spec.cell_dir.is_relative_to(output_root)
                or not output_path.is_file()
            ):
                raise RuntimeError("mock_smoke_cellspec_invalid")
    except Exception as exc:
        checks.append(
            _check(
                "mock_runtime",
                "fail",
                "mock_smoke_runtime_validation_failed",
                error_type=type(exc).__name__,
            )
        )
    else:
        checks.append(
            _check(
                "mock_runtime",
                "pass",
                "mock_smoke_runtime_and_output_validated",
                provider_calls_made=0,
                formal_evaluator_started=False,
            )
        )

    if mode == "mock":
        checks.append(
            _check(
                "formal_claim",
                "skip",
                "mock_smoke_is_not_a_formal_experiment",
            )
        )
    else:
        mock_runtime_failed = any(
            check["name"] == "mock_runtime" and check["status"] == "fail"
            for check in checks
        )
        if mock_runtime_failed:
            checks.append(
                _check(
                    "formal_executor_roundtrip",
                    "fail",
                    "formal_smoke_blocked_by_mock_runtime",
                    provider_calls_made=0,
                    network_enabled=False,
                )
            )
        else:
            checks.append(_formal_workspace_roundtrip_check())
        checks.append(
            _check(
                "formal_claim",
                "skip",
                "formal_smoke_does_not_claim_experiment_success",
            )
        )
    return _finish(
        scope="formal" if mode == "formal" else "core",
        checks=checks,
        recommendations=recommendations,
        smoke_mode=mode,
    )


__all__ = [
    "EXIT_CHECK_FAILED",
    "EXIT_FORMAL_GATE_FAILED",
    "EXIT_INVALID_REQUEST",
    "EXIT_OK",
    "REPORT_SCHEMA_VERSION",
    "check_environment",
    "report_exit_code",
    "smoke",
]
