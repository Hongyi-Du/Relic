"""Workspace-backed product experience observations.

This layer converts bounded real command execution into typed product
experience packets. The command output is evidence; the typed kernel still owns
state mutation, payment, and public feedback effects.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from .demand import project_task_demands, sample_user_task
from .execution import CommandExecutor
from .hashing import canonicalize, stable_hash
from .product_experience import artifact_price
from .randomness import SeededRandom
from .schemas import (
    ActionKind,
    AgentState,
    Artifact,
    ProductExperiencePacket,
    ProductUsageStep,
    clamp01,
)
from .workspace_transaction import CandidateWorkspace

MAX_EVIDENCE_CHARS = 360
_SENSITIVE_OUTPUT_PATTERNS = (
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{20,}=*"),
    re.compile(
        r"(?i)\b(?=[A-Za-z0-9_.-]{12,}\b)"
        r"[A-Za-z0-9_.-]*(?:secret|token|password|api[_-]?key)"
        r"[A-Za-z0-9_.-]*\b"
    ),
)


@dataclass(frozen=True)
class RealProductExperienceTask:
    task_type: str
    stage: str
    objective: str
    command: tuple[str, ...]
    relative_cwd: str = "."
    expected_exit_code: int = 0
    expected_stdout_contains: tuple[str, ...] = ()
    expected_stderr_contains: tuple[str, ...] = ()
    timeout_seconds: float = 20.0
    weight: float = 1.0
    failure_theme: str | None = None
    derivation_source_refs: tuple[str, ...] = ()
    expose_output_to_user_model: bool = False
    evaluation_role: str = "natural_journey"


@dataclass(frozen=True)
class RealProductExecutionObservation:
    observation_id: str
    task_type: str
    stage: str
    objective: str
    command: tuple[str, ...]
    exit_code: int | None
    timed_out: bool
    duration_sec: float
    success: bool
    diagnostic_clarity: float
    stdout_excerpt: str
    stderr_excerpt: str
    output_hash: str
    failure_event: str | None
    weight: float
    user_visible_output_excerpt: str = ""


def _observation_from_payload(
    payload: dict,
) -> RealProductExecutionObservation:
    return RealProductExecutionObservation(
        observation_id=str(payload["observation_id"]),
        task_type=str(payload["task_type"]),
        stage=str(payload["stage"]),
        objective=str(payload["objective"]),
        command=tuple(str(item) for item in payload["command"]),
        exit_code=(
            int(payload["exit_code"])
            if payload.get("exit_code") is not None
            else None
        ),
        timed_out=bool(payload["timed_out"]),
        duration_sec=float(payload["duration_sec"]),
        success=bool(payload["success"]),
        diagnostic_clarity=float(payload["diagnostic_clarity"]),
        stdout_excerpt=str(payload["stdout_excerpt"]),
        stderr_excerpt=str(payload["stderr_excerpt"]),
        output_hash=str(payload["output_hash"]),
        failure_event=(
            str(payload["failure_event"])
            if payload.get("failure_event") is not None
            else None
        ),
        weight=float(payload["weight"]),
        user_visible_output_excerpt=str(
            payload.get("user_visible_output_excerpt", "")
        ),
    )


class WorkspaceBackedProductExperienceHarness:
    def __init__(
        self,
        *,
        workspace_root: Path,
        tasks: tuple[RealProductExperienceTask, ...],
        timeout_seconds: float = 20.0,
        executor: CommandExecutor | None = None,
        observation_cache_path: Path | None = None,
        evidence_context_hash: str = "",
        require_natural_journey_tasks: bool = False,
    ) -> None:
        if not workspace_root.exists():
            raise FileNotFoundError(
                f"Product experience workspace does not exist: {workspace_root}"
            )
        if not tasks:
            raise ValueError(
                "Workspace-backed product experience requires at least one task"
            )
        invalid_roles = tuple(
            task.evaluation_role
            for task in tasks
            if task.evaluation_role
            not in {"natural_journey", "specified_contract_probe"}
        )
        if invalid_roles:
            raise ValueError(
                "unsupported_product_experience_evaluation_role:"
                + ",".join(sorted(set(invalid_roles)))
            )
        if require_natural_journey_tasks:
            contract_tasks = tuple(
                task.task_type
                for task in tasks
                if task.evaluation_role != "natural_journey"
            )
            if contract_tasks:
                raise ValueError(
                    "blind_discovery_rejects_specified_contract_tasks:"
                    + ",".join(contract_tasks)
                )
        self.workspace_root = workspace_root
        self.tasks = tasks
        self.timeout_seconds = timeout_seconds
        if executor is None:
            raise ValueError("product_experience_executor_required")
        if observation_cache_path is not None and not evidence_context_hash:
            raise ValueError(
                "observation_cache_requires_evidence_context_hash"
            )
        self.executor = executor
        self.observation_cache_path = observation_cache_path
        self.evidence_context_hash = evidence_context_hash
        self._observation_cache_status = "disabled"
        self._observation_cache_entry_hash = ""
        self._observations: tuple[RealProductExecutionObservation, ...] | None = None

    def observations(self) -> tuple[RealProductExecutionObservation, ...]:
        if self._observations is None:
            cached = self._load_cached_observations()
            if cached is not None:
                self._observations = cached
                return self._observations
            with CandidateWorkspace.create(self.workspace_root) as session:
                self._observations = tuple(
                    execute_real_product_task(
                        task=task,
                        workspace_root=session.root,
                        default_timeout_seconds=self.timeout_seconds,
                        executor=self.executor,
                    )
                    for task in self.tasks
                )
            self._store_cached_observations(self._observations)
        return self._observations

    def packet_for_agent(
        self,
        *,
        agent: AgentState,
        artifact: Artifact,
        rng: SeededRandom,
        tick: int,
        action_kind: ActionKind,
    ) -> ProductExperiencePacket:
        return product_experience_packet_from_real_execution(
            agent=agent,
            artifact=artifact,
            rng=rng,
            tick=tick,
            action_kind=action_kind,
            observations=self.observations(),
        )

    def summary(self) -> dict:
        observations = self.observations()
        return {
            "workspace_root": str(self.workspace_root),
            "observation_cache_status": self._observation_cache_status,
            "observation_cache_entry_hash": (
                self._observation_cache_entry_hash
            ),
            "evidence_context_hash": self.evidence_context_hash,
            "task_count": len(self.tasks),
            "observation_count": len(observations),
            "observation_ids": tuple(
                observation.observation_id for observation in observations
            ),
            "success_count": sum(
                1 for observation in observations if observation.success
            ),
            "failure_count": sum(
                1 for observation in observations if not observation.success
            ),
            "timed_out_count": sum(
                1 for observation in observations if observation.timed_out
            ),
            "observation_hash": stable_hash(observations),
            "objective_execution_scope": "shared_base_release_task_suite",
            "agent_journey_policy": (
                "profile_activity_and_role_conditioned_observation_subset"
            ),
            "independent_agent_workspace_execution": False,
            "tasks": [
                {
                    "task_type": observation.task_type,
                    "stage": observation.stage,
                    "success": observation.success,
                    "exit_code": observation.exit_code,
                    "timed_out": observation.timed_out,
                    "failure_event": observation.failure_event,
                    "output_hash": observation.output_hash,
                    "derivation_source_refs": (
                        self.tasks[index].derivation_source_refs
                    ),
                    "user_visible_output_exposed": bool(
                        observation.user_visible_output_excerpt
                    ),
                    "evaluation_role": self.tasks[index].evaluation_role,
                }
                for index, observation in enumerate(observations)
            ],
        }

    def _load_cached_observations(
        self,
    ) -> tuple[RealProductExecutionObservation, ...] | None:
        path = self.observation_cache_path
        if path is None or not path.exists():
            return None
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"unsafe_observation_cache_entry:{path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != "real_product_observations_v1":
                raise ValueError("observation_cache_schema_mismatch")
            if payload.get("evidence_context_hash") != self.evidence_context_hash:
                raise ValueError("observation_cache_context_mismatch")
            if payload.get("task_manifest_hash") != stable_hash(self.tasks):
                raise ValueError("observation_cache_task_manifest_mismatch")
            observations = tuple(
                _observation_from_payload(item)
                for item in payload["observations"]
            )
            if payload.get("observation_manifest_hash") != stable_hash(
                observations
            ):
                raise ValueError("observation_cache_manifest_mismatch")
            entry_body = {
                key: value
                for key, value in payload.items()
                if key != "entry_hash"
            }
            if payload.get("entry_hash") != stable_hash(entry_body):
                raise ValueError("observation_cache_entry_hash_mismatch")
        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise RuntimeError(
                f"invalid_observation_cache:{type(exc).__name__}"
            ) from exc
        self._observation_cache_status = "cache_hit"
        self._observation_cache_entry_hash = str(payload["entry_hash"])
        return observations

    def _store_cached_observations(
        self,
        observations: tuple[RealProductExecutionObservation, ...],
    ) -> None:
        path = self.observation_cache_path
        if path is None:
            self._observation_cache_status = "disabled"
            return
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.parent.chmod(0o700)
        entry = {
            "schema_version": "real_product_observations_v1",
            "evidence_context_hash": self.evidence_context_hash,
            "task_manifest_hash": stable_hash(self.tasks),
            "observation_manifest_hash": stable_hash(observations),
            "observations": observations,
        }
        entry["entry_hash"] = stable_hash(entry)
        encoded = (
            json.dumps(
                canonicalize(entry),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            path.chmod(0o600)
        finally:
            temporary.unlink(missing_ok=True)
        self._observation_cache_status = "fresh_measurement"
        self._observation_cache_entry_hash = str(entry["entry_hash"])

    def failed_verification_commands(
        self,
        *,
        allowed_observation_ids: tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        """Return exact reproducers, optionally limited to published observations."""

        commands: list[str] = []
        allowed = (
            set(allowed_observation_ids)
            if allowed_observation_ids is not None
            else None
        )
        for task, observation in zip(self.tasks, self.observations(), strict=True):
            if observation.success or (
                allowed is not None and observation.observation_id not in allowed
            ):
                continue
            payload = json.dumps(
                {
                    "argv": observation.command,
                    "relative_cwd": task.relative_cwd,
                    "expected_exit_code": task.expected_exit_code,
                    "expected_stdout_contains": task.expected_stdout_contains,
                    "expected_stderr_contains": task.expected_stderr_contains,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            script = (
                "import json,subprocess,sys;"
                f"spec=json.loads({payload!r});"
                "result=subprocess.run(spec['argv'],cwd=spec['relative_cwd'],"
                "capture_output=True,text=True);"
                "sys.stdout.write(result.stdout);sys.stderr.write(result.stderr);"
                "ok=(result.returncode==spec['expected_exit_code'] and "
                "all(token in result.stdout for token in "
                "spec['expected_stdout_contains']) and "
                "all(token in result.stderr for token in "
                "spec['expected_stderr_contains']));"
                "raise SystemExit(0 if ok else 1)"
            )
            verifier = _resolve_command(
                ("{python}", "-c", script),
                executor=self.executor,
            )
            commands.append(shlex.join(verifier))
        return tuple(dict.fromkeys(commands))


def build_real_product_experience_harness(
    *,
    artifact: Artifact,
    workspace_root: Path,
    timeout_seconds: float = 20.0,
    max_tasks: int = 8,
    tasks: tuple[RealProductExperienceTask, ...] | None = None,
    executor: CommandExecutor | None = None,
    observation_cache_path: Path | None = None,
    evidence_context_hash: str = "",
    require_natural_journey_tasks: bool = False,
) -> WorkspaceBackedProductExperienceHarness:
    task_suite = tasks or infer_real_product_experience_tasks(
        artifact=artifact, workspace_root=workspace_root
    )
    return WorkspaceBackedProductExperienceHarness(
        workspace_root=workspace_root,
        tasks=task_suite[:max_tasks],
        timeout_seconds=timeout_seconds,
        executor=executor,
        observation_cache_path=observation_cache_path,
        evidence_context_hash=evidence_context_hash,
        require_natural_journey_tasks=require_natural_journey_tasks,
    )


def infer_real_product_experience_tasks(
    *,
    artifact: Artifact,
    workspace_root: Path,
) -> tuple[RealProductExperienceTask, ...]:
    if _looks_like_early_vite_artifact(
        artifact=artifact,
        workspace_root=workspace_root,
    ):
        return _early_vite_experience_tasks()
    if _looks_like_repo_digest_artifact(artifact):
        return (
            RealProductExperienceTask(
                task_type="documentation_lookup",
                stage="discover_fit",
                objective="Inspect the product CLI surface.",
                command=_src_python_command(
                    "sys.argv=['gitingest','--help'];"
                    "from gitingest.cli import main;"
                    "main()"
                ),
                timeout_seconds=12.0,
                failure_theme="documentation_and_migration_path",
            ),
            RealProductExperienceTask(
                task_type="local_repo_digest",
                stage="first_run",
                objective="Generate a digest for the current repository workspace.",
                command=_src_python_command(
                    _deterministic_tokenizer_script() + "import gitingest;"
                    "result=gitingest.ingest('.');"
                    "assert result[1] and result[2]"
                ),
                timeout_seconds=25.0,
                failure_theme="task_fit_improvement",
            ),
            RealProductExperienceTask(
                task_type="offline_digest_generation",
                stage="reliability",
                objective="Generate a digest while token metadata is unavailable offline.",
                command=_src_python_command(
                    "import ssl,tempfile,tiktoken;"
                    "from pathlib import Path;"
                    "import gitingest;"
                    "root=Path(tempfile.mkdtemp());"
                    "(root/'sample.py').write_text('print(1)\\n');"
                    "tiktoken.get_encoding=lambda *_a,**_k:"
                    "(_ for _ in ()).throw(ssl.SSLError('offline'));"
                    "assert gitingest.ingest(str(root))"
                ),
                timeout_seconds=15.0,
                failure_theme="token_count_resilience",
                evaluation_role="specified_contract_probe",
            ),
            RealProductExperienceTask(
                task_type="large_repository_digest",
                stage="limits",
                objective="Keep an oversized file out of a bounded digest.",
                command=_src_python_command(
                    _deterministic_tokenizer_script() + "import tempfile;"
                    "from pathlib import Path;"
                    "import gitingest;"
                    "root=Path(tempfile.mkdtemp());"
                    "(root/'small.py').write_text('SMALL_OK\\n');"
                    "(root/'large.py').write_text('B'*5000);"
                    "content=gitingest.ingest(str(root),max_file_size=100)[2];"
                    "assert 'SMALL_OK' in content and 'B'*500 not in content"
                ),
                timeout_seconds=15.0,
                failure_theme="max_file_size_enforcement",
                evaluation_role="specified_contract_probe",
            ),
            RealProductExperienceTask(
                task_type="gitignore_respecting_digest",
                stage="filtering",
                objective="Honor default .gitignore and project-specific .gitingestignore rules.",
                command=_src_python_command(
                    _deterministic_tokenizer_script() + "import tempfile;"
                    "from pathlib import Path;"
                    "import gitingest;"
                    "root=Path(tempfile.mkdtemp());"
                    "(root/'keep.py').write_text('KEEP_ME\\n');"
                    "(root/'git_secret.log').write_text('GIT_SECRET_VALUE\\n');"
                    "(root/'tool_secret.private').write_text('TOOL_SECRET_VALUE\\n');"
                    "(root/'.gitignore').write_text('*.log\\n');"
                    "(root/'.gitingestignore').write_text('*.private\\n');"
                    "content=gitingest.ingest(str(root))[2];"
                    "assert ('KEEP_ME' in content and "
                    "'GIT_SECRET_VALUE' not in content and "
                    "'TOOL_SECRET_VALUE' not in content)"
                ),
                timeout_seconds=15.0,
                failure_theme="ignore_pattern_reliability",
                evaluation_role="specified_contract_probe",
            ),
            RealProductExperienceTask(
                task_type="include_exclude_filtering",
                stage="filtering",
                objective="Apply a git-wildmatch character class consistently.",
                command=_src_python_command(
                    _deterministic_tokenizer_script() + "import tempfile;"
                    "from pathlib import Path;"
                    "import gitingest;"
                    "root=Path(tempfile.mkdtemp());"
                    "(root/'keep.py').write_text('KEEP_ME\\n');"
                    "(root/'secret1.py').write_text('SECRET_VALUE\\n');"
                    "content=gitingest.ingest(str(root),exclude_patterns={'secret[0-9].py'})[2];"
                    "assert 'KEEP_ME' in content and 'SECRET_VALUE' not in content"
                ),
                timeout_seconds=15.0,
                failure_theme="pattern_filtering_consistency",
                evaluation_role="specified_contract_probe",
            ),
            RealProductExperienceTask(
                task_type="submodule_repository_digest",
                stage="repository_features",
                objective="Include a real local Git submodule in the generated digest.",
                command=_src_python_command(
                    _deterministic_tokenizer_script() + _submodule_experience_script()
                ),
                timeout_seconds=25.0,
                failure_theme="include_submodules",
                evaluation_role="specified_contract_probe",
            ),
            RealProductExperienceTask(
                task_type="remote_repo_check",
                stage="portability",
                objective="Check repository reachability without an external curl binary.",
                command=_src_python_command(_http_portability_experience_script()),
                timeout_seconds=15.0,
                failure_theme="http_client_portability",
                evaluation_role="specified_contract_probe",
            ),
        )
    if (workspace_root / "package.json").exists():
        return (
            RealProductExperienceTask(
                task_type="project_smoke_test",
                stage="first_run",
                objective="Run the package test surface exposed by the project.",
                command=("npm", "test"),
                timeout_seconds=45.0,
                failure_theme="production_build_reliability",
            ),
        )
    if (workspace_root / "pyproject.toml").exists():
        return (
            RealProductExperienceTask(
                task_type="python_package_import",
                stage="setup",
                objective="Verify the Python package imports in the workspace.",
                command=(
                    "{python}",
                    "-c",
                    "import pathlib; print(pathlib.Path('.').resolve().name)",
                ),
                timeout_seconds=8.0,
                failure_theme="dependency_resolution",
            ),
        )
    return (
        RealProductExperienceTask(
            task_type="workspace_discovery",
            stage="discover_fit",
            objective="Inspect the workspace enough to form a first-use judgment.",
            command=(
                "{python}",
                "-c",
                "import pathlib; print(len(list(pathlib.Path('.').iterdir())))",
            ),
            timeout_seconds=8.0,
            failure_theme="documentation_and_migration_path",
        ),
    )


def _looks_like_early_vite_artifact(
    *,
    artifact: Artifact,
    workspace_root: Path,
) -> bool:
    package_path = workspace_root / "package.json"
    if artifact.id != "vite_early_dev_server" or not package_path.is_file():
        return False
    try:
        package = json.loads(package_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        package.get("name") == "vite"
        and str(package.get("version", "")).startswith("0.20.")
    )


def _early_vite_experience_tasks() -> tuple[RealProductExperienceTask, ...]:
    """Natural base-release journey derived from the shipped package and README."""

    return (
        RealProductExperienceTask(
            task_type="package_contract_discovery",
            stage="discover_fit",
            objective=(
                "Discover the packaged CLI, JavaScript entrypoint, type surface, "
                "and supported runtime from the released package."
            ),
            command=("node", "-e", _vite_package_contract_script()),
            timeout_seconds=10.0,
            failure_theme="package_surface_reliability",
            derivation_source_refs=("package.json",),
            expose_output_to_user_model=True,
        ),
        RealProductExperienceTask(
            task_type="documented_product_fit",
            stage="discover_constraints",
            objective=(
                "Read the released status, framework-fit, and browser-support "
                "constraints before adopting the tool."
            ),
            command=("node", "-e", _vite_documented_fit_script()),
            timeout_seconds=10.0,
            failure_theme="documentation_and_migration_path",
            derivation_source_refs=(
                "README.md#status",
                "README.md#getting-started",
                "README.md#browser-support",
            ),
            expose_output_to_user_model=True,
        ),
        RealProductExperienceTask(
            task_type="cli_first_run",
            stage="first_run",
            objective=(
                "Open the released command-line help without modifying the project."
            ),
            command=("node", "bin/vite.js", "--help"),
            timeout_seconds=15.0,
            failure_theme="installation_and_dependency_resolution",
            derivation_source_refs=("README.md#getting-started", "bin/vite.js"),
            expose_output_to_user_model=True,
        ),
        RealProductExperienceTask(
            task_type="dev_server_first_page",
            stage="core_workflow",
            objective=(
                "Start the released development server for a minimal module app "
                "and fetch its first page over localhost."
            ),
            command=("node", "-e", _vite_dev_server_first_page_script()),
            timeout_seconds=35.0,
            failure_theme="dev_server_reliability",
            derivation_source_refs=(
                "README.md#getting-started",
                "dist/server/index.d.ts",
            ),
            expose_output_to_user_model=True,
        ),
        RealProductExperienceTask(
            task_type="source_update_cycle",
            stage="feedback_loop",
            objective=(
                "Edit a served JavaScript module and verify the development "
                "server returns the updated source without restarting."
            ),
            command=("node", "-e", _vite_source_update_cycle_script()),
            timeout_seconds=35.0,
            failure_theme="hmr_feedback_loop",
            derivation_source_refs=(
                "README.md#hot-module-replacement",
                "dist/server/index.d.ts",
            ),
            expose_output_to_user_model=True,
        ),
        RealProductExperienceTask(
            task_type="typescript_transform",
            stage="language_workflow",
            objective=(
                "Serve a TypeScript module and verify the documented on-demand "
                "transpilation removes type syntax."
            ),
            command=("node", "-e", _vite_typescript_transform_script()),
            timeout_seconds=35.0,
            failure_theme="typescript_workflow",
            derivation_source_refs=("README.md#typescript",),
            expose_output_to_user_model=True,
        ),
        RealProductExperienceTask(
            task_type="documented_alias_configuration",
            stage="configure_workflow",
            objective=(
                "Apply a documented import alias in a minimal project and verify "
                "the served module resolves through it."
            ),
            command=("node", "-e", _vite_alias_configuration_script()),
            timeout_seconds=35.0,
            failure_theme="configuration_extensibility",
            derivation_source_refs=(
                "README.md#config-file",
                "dist/config.d.ts",
            ),
            expose_output_to_user_model=True,
        ),
        RealProductExperienceTask(
            task_type="production_build_workflow",
            stage="production_build",
            objective=(
                "Build a minimal HTML, JavaScript, and CSS application with the "
                "released production workflow and inspect emitted assets."
            ),
            command=("node", "-e", _vite_production_build_workflow_script()),
            timeout_seconds=60.0,
            failure_theme="production_build_reliability",
            derivation_source_refs=(
                "README.md#production-build",
                "README.md#css--json-importing",
                "dist/build/index.d.ts",
            ),
            expose_output_to_user_model=True,
        ),
    )


def _vite_package_contract_script() -> str:
    return (
        "const fs=require('fs');"
        "const p=JSON.parse(fs.readFileSync('package.json','utf8'));"
        "if(p.name!=='vite'||!p.bin||!p.bin.vite||!p.main||!p.types)"
        "throw new Error('incomplete_package_contract');"
        "if(!p.engines||!p.engines.node)"
        "throw new Error('missing_runtime_contract');"
        "process.stdout.write(JSON.stringify({bin:p.bin.vite,main:p.main,"
        "types:p.types,node:p.engines.node}));"
    )


def _vite_documented_fit_script() -> str:
    return (
        "const fs=require('fs');const s=fs.readFileSync('README.md','utf8');"
        "const facts={experimental:/still experimental/i.test(s),"
        "vue_primary:/primarily designed to work with Vue 3/i.test(s),"
        "other_frameworks:/support other frameworks/i.test(s),"
        "modern_browsers:/targeting modern browsers/i.test(s),"
        "legacy_transform_default:/does not perform any compatibility-oriented "
        "code transforms by default/i.test(s)};"
        "if(Object.values(facts).some(v=>!v))"
        "throw new Error('documented_fit_missing');"
        "process.stdout.write(JSON.stringify(facts));"
    )


def _vite_dev_server_first_page_script() -> str:
    return _vite_server_journey_script(
        files={
            "index.html": (
                '<!doctype html><main id="app">BASE_PAGE</main>'
                '<script type="module" src="/main.js"></script>'
            ),
            "main.js": "document.querySelector('#app').dataset.ready='yes'\n",
        },
        requests=(("/", ("BASE_PAGE", "/main.js")),),
        result_label="first_page_served",
    )


def _vite_source_update_cycle_script() -> str:
    return _vite_server_journey_script(
        files={
            "index.html": '<script type="module" src="/main.js"></script>',
            "main.js": "export const version = 'BEFORE_EDIT'\n",
        },
        requests=(
            ("/main.js?t=1", ("BEFORE_EDIT",)),
            ("/main.js?t=2", ("AFTER_EDIT",)),
        ),
        mutation=("main.js", "export const version = 'AFTER_EDIT'\n"),
        mutate_after_request=0,
        result_label="updated_source_observed",
    )


def _vite_typescript_transform_script() -> str:
    return _vite_server_journey_script(
        files={
            "index.html": '<script type="module" src="/main.ts"></script>',
            "main.ts": (
                "const message: string = 'TS_TRANSFORMED';"
                "console.log(message)\n"
            ),
        },
        requests=(("/main.ts", ("TS_TRANSFORMED",)),),
        rejected_tokens=(": string",),
        result_label="typescript_transformed",
    )


def _vite_alias_configuration_script() -> str:
    return _vite_server_journey_script(
        files={
            "index.html": '<script type="module" src="/main.js"></script>',
            "main.js": (
                "import { message } from '@fixture';console.log(message)\n"
            ),
            "fixture.js": "export const message = 'ALIAS_RESOLVED'\n",
        },
        requests=(("/main.js", ("fixture.js",)),),
        server_options="{alias:{'@fixture':path.join(root,'fixture.js')}}",
        result_label="documented_alias_resolved",
    )


def _vite_server_journey_script(
    *,
    files: dict[str, str],
    requests: tuple[tuple[str, tuple[str, ...]], ...],
    result_label: str,
    rejected_tokens: tuple[str, ...] = (),
    mutation: tuple[str, str] | None = None,
    mutate_after_request: int | None = None,
    server_options: str = "{}",
) -> str:
    files_json = json.dumps(files, sort_keys=True)
    requests_json = json.dumps(requests)
    rejected_json = json.dumps(rejected_tokens)
    mutation_json = json.dumps(mutation)
    return (
        "const fs=require('fs'),path=require('path'),http=require('http');"
        "console.log=()=>{};console.info=()=>{};"
        "const root=fs.mkdtempSync(path.join(process.cwd(),'.xp-vite-'));"
        f"const files={files_json};"
        "for(const [name,body] of Object.entries(files)){"
        "const target=path.join(root,name);"
        "fs.mkdirSync(path.dirname(target),{recursive:true});"
        "fs.writeFileSync(target,body);}"
        f"const configured={server_options};"
        "const options=Object.assign("
        "{root,hostname:'127.0.0.1',silent:true},configured);"
        "const server=require('./dist/server').createServer(options);"
        "const get=(port,url)=>new Promise((resolve,reject)=>{"
        "const req=http.get({hostname:'127.0.0.1',port,path:url},res=>{"
        "let body='';res.setEncoding('utf8');res.on('data',c=>body+=c);"
        "res.on('end',()=>res.statusCode>=200&&res.statusCode<300?"
        "resolve(body):reject(new Error('http_status_'+res.statusCode)));});"
        "req.on('error',reject);});"
        f"const requests={requests_json};const rejected={rejected_json};"
        f"const mutation={mutation_json};"
        f"const mutateAfter={json.dumps(mutate_after_request)};"
        "const timer=setTimeout(()=>{console.error('journey_timeout');"
        "process.exit(3)},30000);"
        "server.listen(0,'127.0.0.1',async()=>{try{"
        "const port=server.address().port;const sizes=[];"
        "for(let i=0;i<requests.length;i++){const [url,tokens]=requests[i];"
        "const body=await get(port,url);"
        "for(const token of tokens){if(!body.includes(token))"
        "throw new Error('missing_response_token_'+token);}"
        "for(const token of rejected){if(body.includes(token))"
        "throw new Error('unexpected_response_token_'+token);}"
        "sizes.push(body.length);"
        "if(mutation&&i===mutateAfter){"
        "fs.writeFileSync(path.join(root,mutation[0]),mutation[1]);"
        "await new Promise(resolve=>setTimeout(resolve,150));}}"
        "clearTimeout(timer);"
        f"process.stdout.write(JSON.stringify({{result:'{result_label}',sizes}}));"
        "server.close(()=>process.exit(0));"
        "}catch(error){clearTimeout(timer);"
        "console.error(error&&error.stack||error);"
        "server.close(()=>process.exit(2));}});"
    )


def _vite_production_build_workflow_script() -> str:
    return (
        "const fs=require('fs'),path=require('path');"
        "const root=fs.mkdtempSync(path.join(process.cwd(),'.xp-build-'));"
        "fs.writeFileSync(path.join(root,'index.html'),"
        "'<main>BUILD_APP</main><script type=\"module\" "
        "src=\"/main.js\"></script>');"
        "fs.writeFileSync(path.join(root,'main.js'),"
        "\"import './style.css';console.log('BUILD_READY')\\n\");"
        "fs.writeFileSync(path.join(root,'style.css'),"
        "'main{color:#123456}\\n');"
        "require('./dist/build').build({root,minify:false,silent:true,"
        "assetsInlineLimit:0}).then(()=>{"
        "const out=path.join(root,'dist');const files=[];"
        "const walk=dir=>{for(const name of fs.readdirSync(dir)){"
        "const target=path.join(dir,name),stat=fs.statSync(target);"
        "stat.isDirectory()?walk(target):files.push(path.relative(out,target));}};"
        "walk(out);if(!files.includes('index.html')||"
        "!files.some(name=>name.endsWith('.js'))||"
        "!files.some(name=>name.endsWith('.css')))"
        "throw new Error('incomplete_production_output_'+files.join(','));"
        "process.stdout.write(JSON.stringify({"
        "result:'production_build_complete',files:files.sort()}));"
        "process.exit(0);"
        "}).catch(error=>{console.error(error&&error.stack||error);"
        "process.exit(2)});"
    )


def _src_python_command(script: str) -> tuple[str, ...]:
    return ("{python}", "-c", "import sys;sys.path.insert(0,'src');" + script)


def _deterministic_tokenizer_script() -> str:
    return (
        "import tiktoken;"
        "tiktoken.get_encoding=lambda *_a,**_k:"
        "type('OfflineEncoding',(),{'encode':lambda self,text,**_k:"
        "list(range(max(1,len(text)//4)))})();"
    )


def _submodule_experience_script() -> str:
    return """
import subprocess
import tempfile
from pathlib import Path

import gitingest


def run_git(cwd, *args):
    subprocess.run(
        ["git", "-c", "protocol.file.allow=always", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


root = Path(tempfile.mkdtemp())
submodule = root / "submodule"
main = root / "main"
for repo, filename, content in (
    (submodule, "subfile.txt", "SUBMODULE_CONTENT\n"),
    (main, "mainfile.txt", "MAIN_CONTENT\n"),
):
    repo.mkdir()
    run_git(repo, "init")
    run_git(repo, "config", "user.name", "Experience")
    run_git(repo, "config", "user.email", "experience@example.com")
    (repo / filename).write_text(content)
    run_git(repo, "add", filename)
    run_git(repo, "commit", "-m", "fixture")
run_git(main, "submodule", "add", submodule.resolve().as_uri(), "dependency")
run_git(main, "commit", "-m", "add submodule")
content = gitingest.ingest(str(main), include_submodules=True)[2]
assert "MAIN_CONTENT" in content and "SUBMODULE_CONTENT" in content
"""


def _http_portability_experience_script() -> str:
    return """
import asyncio
import http.server
import socket
import threading
from unittest.mock import patch

from gitingest.utils.git_utils import check_repo_exists


class Handler(http.server.BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200 if "exists" in self.path else 404)
        self.end_headers()

    def log_message(self, *_args):
        return


sock = socket.socket()
sock.bind(("127.0.0.1", 0))
port = sock.getsockname()[1]
sock.close()
server = http.server.HTTPServer(("127.0.0.1", port), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
real_exec = asyncio.create_subprocess_exec


async def no_curl(program, *args, **kwargs):
    if str(program).endswith("curl"):
        raise FileNotFoundError("curl is unavailable")
    return await real_exec(program, *args, **kwargs)


async def probe():
    exists = await check_repo_exists(f"http://127.0.0.1:{port}/exists")
    missing = await check_repo_exists(f"http://127.0.0.1:{port}/missing")
    return exists, missing


try:
    with patch("asyncio.create_subprocess_exec", side_effect=no_curl):
        exists, missing = asyncio.run(probe())
finally:
    server.shutdown()
    server.server_close()
assert exists is True and missing is False
"""


def execute_real_product_task(
    *,
    task: RealProductExperienceTask,
    workspace_root: Path,
    default_timeout_seconds: float,
    executor: CommandExecutor | None = None,
) -> RealProductExecutionObservation:
    if executor is None:
        raise ValueError("product_experience_executor_required")
    command_executor = executor
    command = _resolve_command(task.command, executor=command_executor)
    cwd = (workspace_root / task.relative_cwd).resolve()
    if not _is_relative_to(cwd, workspace_root.resolve()):
        raise ValueError(f"Task cwd escapes workspace: {task.relative_cwd}")
    timeout_seconds = task.timeout_seconds or default_timeout_seconds
    started = monotonic()
    outcome = command_executor.run(
        root=cwd,
        argv=command,
        timeout_seconds=timeout_seconds,
    )
    exit_code = outcome.exit_code
    timed_out = outcome.status == "timeout"
    stdout = outcome.stdout_tail
    stderr = outcome.stderr_tail
    if outcome.blocked_reason:
        stderr = f"{stderr}\n{outcome.blocked_reason}".strip()
    duration_sec = round(monotonic() - started, 6)
    output_hash = stable_hash(
        {
            "stdout_hash": outcome.stdout_hash,
            "stderr_hash": outcome.stderr_hash,
            "status": outcome.status,
            "exit_code": exit_code,
            "timed_out": timed_out,
        }
    )
    success = (
        not timed_out
        and exit_code == task.expected_exit_code
        and all(token in stdout for token in task.expected_stdout_contains)
        and all(token in stderr for token in task.expected_stderr_contains)
    )
    diagnostic_clarity = _diagnostic_clarity(
        success=success,
        timed_out=timed_out,
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
    )
    failure_event = (
        None
        if success
        else _failure_event(
            task=task, timed_out=timed_out, exit_code=exit_code, stderr=stderr
        )
    )
    observation_id = (
        "real_xp_obs_"
        + stable_hash(
            {
                "task": task,
                "exit_code": exit_code,
                "timed_out": timed_out,
                "output_hash": output_hash,
            }
        )[:24]
    )
    return RealProductExecutionObservation(
        observation_id=observation_id,
        task_type=task.task_type,
        stage=task.stage,
        objective=task.objective,
        command=command,
        exit_code=exit_code,
        timed_out=timed_out,
        duration_sec=duration_sec,
        success=success,
        diagnostic_clarity=diagnostic_clarity,
        stdout_excerpt=_excerpt(stdout),
        stderr_excerpt=_excerpt(stderr),
        output_hash=output_hash,
        failure_event=failure_event,
        weight=clamp01(task.weight),
        user_visible_output_excerpt=(
            _user_visible_output_excerpt(stdout=stdout, stderr=stderr)
            if task.expose_output_to_user_model
            else ""
        ),
    )


def product_experience_packet_from_real_execution(
    *,
    agent: AgentState,
    artifact: Artifact,
    rng: SeededRandom,
    tick: int,
    action_kind: ActionKind,
    observations: tuple[RealProductExecutionObservation, ...],
) -> ProductExperiencePacket:
    if not observations:
        raise ValueError("Real product experience requires at least one observation")
    objective_observation_manifest = observations
    observations = _profile_conditioned_observations(
        agent=agent,
        observations=observations,
    )
    profile = agent.profile
    body = agent.body
    budgets = agent.action_budgets
    task_type = _dominant_task_type(
        agent=agent, artifact=artifact, rng=rng, observations=observations
    )
    weighted_success = _weighted_mean(
        (float(item.success), item.weight) for item in observations
    )
    failure_share = _weighted_mean(
        (float(not item.success), item.weight) for item in observations
    )
    diagnostic_clarity = _weighted_mean(
        (item.diagnostic_clarity, item.weight) for item in observations
    )
    duration_pressure = _mean(
        min(1.0, item.duration_sec / 20.0) for item in observations
    )
    cognitive_capacity = profile.cognitive_capacity
    task_fit = clamp01(
        artifact.task_fit_distribution.get(
            task_type, _mean(artifact.task_fit_distribution.values()) or 0.5
        )
    )
    task_difficulty = clamp01(
        0.22
        + 0.34 * failure_share
        + 0.20 * (1.0 - task_fit)
        + 0.16 * (1.0 - profile.technical_skill)
        + 0.12 * (1.0 - cognitive_capacity)
        + 0.10 * body.stress
        + 0.08 * profile.deadline_pressure
    )
    baseline_success = clamp01(
        0.16
        + 0.42 * profile.technical_skill
        + 0.20 * cognitive_capacity
        + 0.16 * budgets.attention_budget
        + 0.10 * body.physiological_energy
        - 0.30 * task_difficulty
        - 0.14 * body.fatigue
    )
    interpretive_load = clamp01(
        0.26 * failure_share
        + 0.22 * (1.0 - diagnostic_clarity)
        + 0.16 * duration_pressure
        + 0.14 * artifact.learning_curve
        + 0.12 * body.fatigue
        + 0.10 * body.stress
    )
    profile_friction = clamp01(
        interpretive_load
        * (1.22 - 0.34 * profile.technical_skill - 0.18 * cognitive_capacity)
        + 0.08 * profile.budget_sensitivity * artifact_price(artifact)
    )
    workaround_success = clamp01(
        0.12
        + 0.36 * diagnostic_clarity
        + 0.28 * profile.technical_skill
        + 0.16 * cognitive_capacity
        + 0.08 * budgets.attention_budget
        - 0.22 * failure_share
    )
    journey_steps = tuple(
        _usage_step_from_observation(item, profile_friction=profile_friction)
        for item in observations
    )
    blocked_stage = next(
        (step.stage for step in journey_steps if step.error_event), None
    )
    failure_event = next(
        (item.failure_event for item in observations if item.failure_event), None
    )
    objective_success = clamp01(
        0.08
        + 0.62 * weighted_success
        + 0.14 * workaround_success
        + 0.12 * task_fit
        + 0.10 * profile.technical_skill
        - 0.24 * profile_friction
        - 0.08 * float(blocked_stage is not None)
    )
    time_saved = clamp01(
        0.10
        + 0.36 * weighted_success
        + 0.18 * task_fit
        + 0.16 * workaround_success
        - 0.28 * profile_friction
        - 0.18 * duration_pressure
    )
    observed_reliability = clamp01(
        0.12
        + 0.70 * weighted_success
        + 0.18 * diagnostic_clarity
        - 0.14 * failure_share
    )
    success_gain = max(0.0, objective_success - baseline_success)
    first_value_time = clamp01(
        duration_pressure + 0.20 * failure_share + 0.10 * profile_friction
    )
    integration_quality = clamp01(
        0.22
        + 0.42 * weighted_success
        + 0.20 * task_fit
        + 0.16 * workaround_success
        - 0.18 * profile_friction
    )
    repeat_use_value = clamp01(
        0.24 * success_gain
        + 0.22 * time_saved
        + 0.20 * integration_quality
        + 0.18 * observed_reliability
        + 0.16 * workaround_success
        - 0.12 * profile_friction
    )
    packet_id_hash = stable_hash(
        {
            "agent_id": agent.id,
            "artifact_id": artifact.id,
            "tick": tick,
            "observations": tuple(item.observation_id for item in observations),
            "profile": profile,
            "action_kind": action_kind.value,
        }
    )[:20]
    base_experience_id = (
        f"real_xp_{tick:06d}_{agent.id}_{artifact.id}_{packet_id_hash}"
    )
    latest_experience = agent.latest_artifact_experience.get(artifact.id)
    experience_id = base_experience_id
    if latest_experience is not None and latest_experience.tick == tick:
        occurrence = max(
            1,
            agent.artifact_usage_counts.get(artifact.id, 0),
        )
        experience_id = f"{base_experience_id}_u{occurrence:04d}"
    return ProductExperiencePacket(
        experience_id=experience_id,
        artifact_id=artifact.id,
        tick=tick,
        task_type=task_type,
        task_difficulty=task_difficulty,
        baseline_success_without_product=baseline_success,
        objective_success=objective_success,
        success_gain=success_gain,
        time_saved=time_saved,
        friction=profile_friction,
        failure_event=failure_event,
        observed_reliability=observed_reliability,
        attention_cost=clamp01(
            0.08 + 0.34 * profile_friction + 0.08 * duration_pressure
        ),
        budget_cost=artifact_price(artifact),
        comparison_to_current_method=clamp01(
            0.5 + objective_success - baseline_success - 0.22 * profile_friction
        ),
        journey_steps=journey_steps,
        blocked_stage=blocked_stage,
        diagnostic_clarity=diagnostic_clarity,
        workaround_success=workaround_success,
        first_value_time=first_value_time,
        integration_quality=integration_quality,
        repeat_use_value=repeat_use_value,
        objective_observation_refs=tuple(
            item.observation_id for item in observations
        ),
        objective_observation_set_hash=stable_hash(
            tuple(item.observation_id for item in observations)
        ),
        objective_observation_manifest_hash=stable_hash(
            objective_observation_manifest
        ),
        raw_execution_success=weighted_success,
    )


def _usage_step_from_observation(
    observation: RealProductExecutionObservation,
    *,
    profile_friction: float,
) -> ProductUsageStep:
    success = 1.0 if observation.success else 0.0
    stage_friction = clamp01(
        0.10
        + 0.42 * float(not observation.success)
        + 0.24 * min(1.0, observation.duration_sec / 20.0)
        + 0.22 * (1.0 - observation.diagnostic_clarity)
        + 0.18 * profile_friction
    )
    evidence = (
        f"real_exec:{observation.stage}:{observation.task_type}:exit={observation.exit_code}:timeout={observation.timed_out}",
        f"output_hash:{observation.output_hash}",
        f"diagnostic_class:{observation.failure_event or 'none'}",
        f"output_present:{bool(observation.stdout_excerpt or observation.stderr_excerpt)}",
        *(
            (f"public_output:{observation.user_visible_output_excerpt}",)
            if observation.user_visible_output_excerpt
            else ()
        ),
        observation.observation_id,
    )
    return ProductUsageStep(
        stage=observation.stage,
        objective=observation.objective,
        success=success,
        friction=stage_friction,
        time_cost=clamp01(observation.duration_sec / 20.0),
        diagnostic_clarity=observation.diagnostic_clarity,
        workaround_success=clamp01(
            observation.diagnostic_clarity * (0.72 if observation.success else 0.48)
        ),
        error_event=observation.failure_event,
        user_visible_evidence=evidence,
    )


def _resolve_command(
    command: tuple[str, ...],
    *,
    executor: CommandExecutor,
) -> tuple[str, ...]:
    python_executable = (
        "python" if executor.policy.backend != "local" else sys.executable
    )
    return tuple(python_executable if part == "{python}" else part for part in command)


def _diagnostic_clarity(
    *,
    success: bool,
    timed_out: bool,
    stdout: str,
    stderr: str,
    exit_code: int | None,
) -> float:
    if success:
        return 0.86 if stdout.strip() or stderr.strip() else 0.70
    if timed_out:
        return 0.24
    text = f"{stdout}\n{stderr}".lower()
    clarity = 0.24
    if exit_code is not None:
        clarity += 0.10
    if any(
        token in text
        for token in (
            "traceback",
            "error",
            "exception",
            "failed",
            "not found",
            "no module",
        )
    ):
        clarity += 0.28
    if any(
        token in text
        for token in ("usage", "help", "try", "install", "expected", "missing")
    ):
        clarity += 0.18
    if len(text.strip()) > 80:
        clarity += 0.08
    return clamp01(clarity)


def _failure_event(
    *,
    task: RealProductExperienceTask,
    timed_out: bool,
    exit_code: int | None,
    stderr: str,
) -> str:
    if timed_out:
        suffix = "timeout"
    elif "no module" in stderr.lower() or "module not found" in stderr.lower():
        suffix = "missing_module"
    elif "permission" in stderr.lower():
        suffix = "permission"
    elif "not found" in stderr.lower():
        suffix = "missing_command"
    else:
        suffix = f"exit_{exit_code}" if exit_code is not None else "os_error"
    if task.failure_theme:
        return f"{task.failure_theme}:{suffix}"
    return f"real_execution_failure:{task.stage}:{suffix}"


def _excerpt(value: str) -> str:
    sanitized = value
    for pattern in _SENSITIVE_OUTPUT_PATTERNS:
        sanitized = pattern.sub("[REDACTED]", sanitized)
    return " ".join(sanitized.split())[:MAX_EVIDENCE_CHARS]


def _user_visible_output_excerpt(*, stdout: str, stderr: str) -> str:
    combined = "\n".join(
        part for part in (stdout.strip(), stderr.strip()) if part
    )
    return _excerpt(combined)


def _profile_conditioned_observations(
    *,
    agent: AgentState,
    observations: tuple[RealProductExecutionObservation, ...],
) -> tuple[RealProductExecutionObservation, ...]:
    if len(observations) <= 3:
        return observations
    profile = agent.profile
    target_count = {
        "inactive": 3,
        "normal": 5,
        "very_active": len(observations),
    }.get(profile.activity_tier, 5)
    if profile.technical_role == "professional_engineer":
        target_count += 1
    if profile.innovation_role in {"creative_originator", "early_builder"}:
        target_count += 1
    target_count = min(len(observations), max(3, target_count))
    mandatory_indices = tuple(
        index
        for index, observation in enumerate(observations)
        if observation.stage in _MANDATORY_OBSERVATION_STAGES
    )[:2]
    mandatory = set(mandatory_indices)
    demand_weights = project_task_demands(
        agent=agent,
        task_space=(observation.task_type for observation in observations),
    )
    observed_pairs = tuple(
        (observation.task_type, observation.stage) for observation in observations
    )
    role_bonus_pairs = {
        channel: _role_bonus_pairs(channel=channel, observed_pairs=observed_pairs)
        for channel in _ROLE_BONUS_TOKENS
    }
    ranked = sorted(
        (
            index
            for index in range(len(observations))
            if index not in mandatory
        ),
        key=lambda index: _profile_observation_relevance(
            agent=agent,
            observation=observations[index],
            demand_weights=demand_weights,
            role_bonus_pairs=role_bonus_pairs,
        ),
        reverse=True,
    )
    selected = {
        *mandatory,
        *ranked[: max(0, target_count - len(mandatory))],
    }
    return tuple(
        observation
        for index, observation in enumerate(observations)
        if index in selected
    )


_MANDATORY_OBSERVATION_STAGES = frozenset({"discover_fit", "first_run"})

_ROLE_BONUS_TOKENS = {
    "professional_engineer": ("build", "config", "typescript", "update", "plugin"),
    "technical_practitioner": ("server", "build", "typescript", "first_run"),
    "innovation_builder": ("config", "update", "workflow", "feedback"),
}


def _role_bonus_pairs(
    *,
    channel: str,
    observed_pairs: tuple[tuple[str, str], ...],
) -> frozenset[tuple[str, str]]:
    """(task, stage) pairs whose observations earn this role channel's bonus.

    Token-expressed pairs keep the bonus verbatim. The token vocabulary lives
    in the curated frontend task language, so on other substrates a channel
    can end up with no hit outside the mandatory-selected stages — the bonus
    then cannot influence ranking and the role silently stops shaping memory
    selection. In that case the channel falls back to a deterministic
    role-keyed half of the rankable pair space (stable-hash order, rng-free),
    so same-role agents share attention, channels stay pairwise distinct, and
    role heterogeneity survives on an arbitrary substrate.
    """
    tokens = _ROLE_BONUS_TOKENS[channel]
    pairs = tuple(
        dict.fromkeys((task.lower(), stage.lower()) for task, stage in observed_pairs)
    )
    expressed = frozenset(
        pair for pair in pairs if any(token in f"{pair[0]} {pair[1]}" for token in tokens)
    )
    rankable = tuple(
        pair for pair in pairs if pair[1] not in _MANDATORY_OBSERVATION_STAGES
    )
    if expressed.intersection(rankable) or not rankable:
        return expressed
    fallback_order = sorted(
        rankable,
        key=lambda pair: (
            stable_hash(
                {
                    "role_bonus_channel": channel,
                    "task": pair[0],
                    "stage": pair[1],
                }
            ),
            pair,
        ),
    )
    return expressed.union(fallback_order[: (len(rankable) + 1) // 2])


def _profile_observation_relevance(
    *,
    agent: AgentState,
    observation: RealProductExecutionObservation,
    demand_weights: dict[str, float],
    role_bonus_pairs: dict[str, frozenset[tuple[str, str]]],
) -> float:
    profile = agent.profile
    task = observation.task_type.lower()
    stage = observation.stage.lower()
    pair = (task, stage)
    technical_bonus = (
        0.34
        if profile.technical_role == "professional_engineer"
        and pair in role_bonus_pairs["professional_engineer"]
        else 0.0
    )
    practitioner_bonus = (
        0.18
        if profile.technical_role == "technical_practitioner"
        and pair in role_bonus_pairs["technical_practitioner"]
        else 0.0
    )
    innovation_bonus = (
        0.24
        if profile.innovation_role in {"creative_originator", "early_builder"}
        and pair in role_bonus_pairs["innovation_builder"]
        else 0.0
    )
    # Profile demand projected onto the observed task space; the profile's raw
    # task_demand_weights live in the curated task vocabulary, which real
    # workflow task_types never match.
    demand = float(demand_weights.get(observation.task_type, 0.0))
    failure_salience = 0.12 if not observation.success else 0.0
    stable_jitter = (
        int(
            stable_hash(
                {
                    "agent_id": agent.id,
                    "observation_id": observation.observation_id,
                }
            )[:8],
            16,
        )
        / 0xFFFFFFFF
    )
    return (
        demand
        + technical_bonus
        + practitioner_bonus
        + innovation_bonus
        + failure_salience
        + 0.05 * stable_jitter
    )


def _dominant_task_type(
    *,
    agent: AgentState,
    artifact: Artifact,
    rng: SeededRandom,
    observations: tuple[RealProductExecutionObservation, ...],
) -> str:
    observed_tasks = tuple(dict.fromkeys(item.task_type for item in observations))
    if not observed_tasks:
        return sample_user_task(agent=agent, artifact=artifact, rng=rng)
    demand_weights = project_task_demands(agent=agent, task_space=observed_tasks)
    return max(
        observed_tasks,
        key=lambda task: (
            demand_weights.get(task, 0.0),
            artifact.task_fit_distribution.get(task, 0.0),
            task,
        ),
    )


def _weighted_mean(values) -> float:
    values = tuple(values)
    total_weight = sum(max(0.0, weight) for _, weight in values)
    if total_weight <= 0:
        return 0.0
    return clamp01(
        sum(clamp01(value) * max(0.0, weight) for value, weight in values)
        / total_weight
    )


def _mean(values) -> float:
    values = tuple(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _looks_like_repo_digest_artifact(artifact: Artifact) -> bool:
    haystack = " ".join(
        (
            artifact.id,
            artifact.artifact_kind,
            *artifact.capability_profile,
            *artifact.reliability_profile,
            *artifact.task_fit_distribution,
            *artifact.failure_modes,
            *artifact.evidence_claims,
            *artifact.public_claims,
        )
    ).lower()
    return any(
        token in haystack
        for token in ("gitingest", "repo_digest", "repository digest", "submodule")
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
