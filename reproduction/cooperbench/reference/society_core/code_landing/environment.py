"""Workspace environment bootstrap for Code-Max runs."""

from __future__ import annotations

import json
import os
import re
import shlex
import stat
import sys
import tomllib
from importlib import metadata
from pathlib import Path

from society_core.execution import CommandExecutor
from society_core.hashing import stable_hash
from society_core.repository_paths import ignored_repository_path
from society_core.safe_files import (
    UnsafeRegularFileError,
    read_regular_file_bytes,
    read_regular_file_text,
    regular_file_fingerprint,
)
from society_core.workspace_transaction import CandidateWorkspace
from society_core.workspace_update import run_workspace_verification

from .schemas import WorkspaceExecutionProfile


def build_workspace_execution_profile(
    workspace_root: Path,
    *,
    run_smoke: bool = False,
    timeout_seconds: float = 20.0,
    executor: CommandExecutor | None = None,
) -> WorkspaceExecutionProfile:
    root = workspace_root.resolve()
    package_manager = _detect_package_manager(root)
    language_stack = _detect_language_stack(root)
    test_commands, lint_commands, build_commands = _detect_commands(
        root, package_manager
    )
    smoke_commands = _smoke_commands(root, language_stack, package_manager)
    setup_status = "ready" if package_manager or language_stack else "partial"
    blocked_reason = (
        None if setup_status == "ready" else "no_manifest_or_language_files"
    )
    missing_declared_dependencies = _missing_declared_dependencies(root)
    if setup_status == "ready" and missing_declared_dependencies:
        setup_status = "partial"
        blocked_reason = "declared_dependencies_missing:" + ",".join(
            missing_declared_dependencies[:12]
        )
    flaky_commands: tuple[str, ...] = ()
    if run_smoke and smoke_commands:
        with CandidateWorkspace.create(root) as smoke_workspace:
            results = run_workspace_verification(
                smoke_workspace.root,
                smoke_commands,
                timeout_seconds=timeout_seconds,
                max_commands=4,
                executor=executor,
            )
        failed = tuple(
            result.command for result in results if result.status != "passed"
        )
        if failed:
            setup_status = "partial"
            blocked_reason = "smoke_command_failed:" + ",".join(failed[:3])
    os_image = _execution_image(executor)
    profile_payload = {
        "root": root.name,
        "os_image": os_image,
        "package_manager": package_manager,
        "language_stack": language_stack,
        "test_commands": test_commands,
        "lint_commands": lint_commands,
        "build_commands": build_commands,
        "smoke_commands": smoke_commands,
        "setup_status": setup_status,
        "blocked_reason": blocked_reason,
        "missing_declared_dependencies": missing_declared_dependencies,
    }
    return WorkspaceExecutionProfile(
        repo_hash=stable_hash(_repo_hash_inputs(root)),
        os_image=os_image,
        language_stack=language_stack,
        package_manager=package_manager,
        install_commands=_install_commands(package_manager),
        setup_status=setup_status,
        blocked_reason=blocked_reason,
        discovered_test_commands=test_commands,
        discovered_lint_commands=lint_commands,
        discovered_build_commands=build_commands,
        smoke_commands=smoke_commands,
        env_vars_required=(),
        services_required=(),
        flaky_commands=flaky_commands,
        command_timeouts={
            command: int(timeout_seconds)
            for command in (*test_commands, *smoke_commands)
        },
        setup_log_hash=stable_hash(profile_payload),
        missing_declared_dependencies=missing_declared_dependencies,
    )


def _repo_hash_inputs(root: Path) -> dict[str, object]:
    files: list[tuple[object, ...]] = []
    for path in sorted(root.rglob("*")):
        if _ignored(path, root):
            continue
        relative = path.relative_to(root).as_posix()
        try:
            metadata = path.lstat()
        except OSError:
            files.append((relative, "unreadable_entry"))
            continue
        if stat.S_ISLNK(metadata.st_mode):
            try:
                target = os.readlink(path)
            except OSError:
                target = "unreadable"
            files.append((relative, "symlink", target, metadata.st_mode))
            continue
        if stat.S_ISREG(metadata.st_mode):
            try:
                mode, digest = regular_file_fingerprint(root, path)
            except (OSError, UnsafeRegularFileError):
                files.append(
                    (
                        relative,
                        "unreadable_file",
                        metadata.st_size,
                        metadata.st_mtime_ns,
                        metadata.st_mode,
                    )
                )
                continue
            files.append((relative, "file", metadata.st_size, mode, digest))
            continue
        entry_type = "directory" if stat.S_ISDIR(metadata.st_mode) else "other"
        files.append((relative, entry_type, metadata.st_mode))
    return {
        "files": files,
        "dependency_state": _dependency_state_inputs(root),
    }


def _dependency_state_inputs(root: Path) -> tuple[tuple[object, ...], ...]:
    entries: list[tuple[object, ...]] = []
    for name in ("node_modules", ".venv"):
        dependency_root = root / name
        if dependency_root.is_symlink() or not dependency_root.is_dir():
            continue
        for path in sorted(dependency_root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            try:
                metadata = path.lstat()
            except OSError:
                continue
            if path.is_symlink():
                try:
                    target = path.readlink().as_posix()
                except OSError:
                    target = "unreadable"
                entries.append((relative, "symlink", target, metadata.st_mtime_ns))
            elif path.is_file():
                entries.append(
                    (
                        relative,
                        "file",
                        metadata.st_size,
                        metadata.st_mtime_ns,
                        metadata.st_mode,
                    )
                )
            elif path.is_dir():
                entries.append((relative, "directory", metadata.st_mode))
    return tuple(entries)


def _detect_package_manager(root: Path) -> str | None:
    if (root / "pnpm-lock.yaml").exists():
        return "pnpm"
    if (root / "yarn.lock").exists():
        return "yarn"
    if (root / "package.json").exists():
        return "npm"
    if (root / "pyproject.toml").exists() or (root / "setup.py").exists():
        return "python"
    return None


def _detect_language_stack(root: Path) -> tuple[str, ...]:
    languages: list[str] = []
    if any(root.rglob("*.py")):
        languages.append("python")
    if any(root.rglob("*.js")) or any(root.rglob("*.mjs")) or any(root.rglob("*.cjs")):
        languages.append("javascript")
    if any(root.rglob("*.ts")) or any(root.rglob("*.tsx")):
        languages.append("typescript")
    if any(root.rglob("*.json")):
        languages.append("json")
    return tuple(dict.fromkeys(languages))


def _detect_commands(
    root: Path,
    package_manager: str | None,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if (root / "package.json").exists():
        try:
            package = json.loads(
                read_regular_file_text(
                    root,
                    root / "package.json",
                    max_bytes=2_000_000,
                )
            )
        except (json.JSONDecodeError, UnicodeDecodeError, UnsafeRegularFileError):
            package = {}
        scripts = package.get("scripts", {}) if isinstance(package, dict) else {}
        prefix = "npm run"
        if package_manager == "yarn":
            prefix = "yarn"
        elif package_manager == "pnpm":
            prefix = "pnpm run"
        tests = (_script_command(prefix, "test", scripts),) if "test" in scripts else ()
        lint = (_script_command(prefix, "lint", scripts),) if "lint" in scripts else ()
        build = (
            (_script_command(prefix, "build", scripts),) if "build" in scripts else ()
        )
        return (
            tuple(filter(None, tests)),
            tuple(filter(None, lint)),
            tuple(filter(None, build)),
        )
    if (root / "pyproject.toml").exists() or (root / "setup.py").exists():
        project = _read_pyproject(root)
        tool = project.get("tool", {}) if isinstance(project, dict) else {}
        tool = tool if isinstance(tool, dict) else {}

        tests: list[str] = []
        if (root / "tests" / "runtests.py").is_file():
            tests.append("python tests/runtests.py")
        elif (root / "runtests.py").is_file():
            tests.append("python runtests.py")
        elif (
            isinstance(tool.get("pytest"), dict)
            or (root / "pytest.ini").is_file()
            or (root / "tests").is_dir()
            or (root / "testing").is_dir()
        ):
            tests.append("python -m pytest")

        lint: list[str] = []
        if isinstance(tool.get("ruff"), dict) or any(
            (root / name).is_file() for name in ("ruff.toml", ".ruff.toml")
        ):
            lint.append("ruff check .")
        if isinstance(tool.get("black"), dict):
            lint.append("black --check .")
        if isinstance(tool.get("mypy"), dict):
            lint.append("mypy .")
        if isinstance(tool.get("pyright"), dict):
            lint.append("pyright")

        return tuple(tests), tuple(lint), ()
    return (), (), ()


def _read_pyproject(root: Path) -> dict[str, object]:
    path = root / "pyproject.toml"
    if not path.is_file() or path.is_symlink():
        return {}
    try:
        parsed = tomllib.loads(read_regular_file_text(root, path, max_bytes=2_000_000))
    except (
        OSError,
        tomllib.TOMLDecodeError,
        UnicodeDecodeError,
        UnsafeRegularFileError,
    ):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _execution_image(executor: CommandExecutor | None) -> str:
    policy = getattr(executor, "policy", None)
    if getattr(policy, "backend", None) == "docker":
        platform_name = str(
            getattr(policy, "container_platform", None) or "unspecified"
        )
        image = str(getattr(policy, "container_image", None) or "unconfigured")
        return f"docker:{platform_name}:{image}"
    return (
        f"local:{sys.platform}:python{sys.version_info.major}.{sys.version_info.minor}"
    )


def _script_command(prefix: str, script: str, scripts: object) -> str | None:
    if not isinstance(scripts, dict) or script not in scripts:
        return None
    if prefix == "yarn":
        return f"yarn {script}"
    return f"{prefix} {script}"


def _smoke_commands(
    root: Path,
    language_stack: tuple[str, ...],
    package_manager: str | None,
) -> tuple[str, ...]:
    commands: list[str] = []
    if "python" in language_stack:
        commands.append(f"{shlex.quote(sys.executable)} -m compileall -q .")
    if "javascript" in language_stack or "typescript" in language_stack:
        main_file = _main_js_file(root)
        if main_file:
            commands.append(f"node --check {main_file}")
    if package_manager == "npm" and not commands:
        commands.append(
            "node -e \"JSON.parse(require('fs').readFileSync('package.json','utf8'))\""
        )
    return tuple(commands[:4])


def _main_js_file(root: Path) -> str | None:
    package_path = root / "package.json"
    if package_path.exists():
        try:
            package = json.loads(
                read_regular_file_text(root, package_path, max_bytes=2_000_000)
            )
            main = package.get("main")
            if isinstance(main, str) and (root / main).exists():
                return main
        except (json.JSONDecodeError, UnicodeDecodeError, UnsafeRegularFileError):
            pass
    for candidate in ("src/index.js", "index.js", "dist/index.js"):
        if (root / candidate).exists():
            return candidate
    return None


def _install_commands(package_manager: str | None) -> tuple[str, ...]:
    if package_manager == "npm":
        return ("npm install",)
    if package_manager == "pnpm":
        return ("pnpm install",)
    if package_manager == "yarn":
        return ("yarn install",)
    if package_manager == "python":
        return ("python -m pip install -e .",)
    return ()


def _missing_declared_dependencies(root: Path) -> tuple[str, ...]:
    missing: list[str] = []
    for name in _declared_runtime_dependencies(root):
        try:
            metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
    return tuple(dict.fromkeys(missing))


def _declared_runtime_dependencies(root: Path) -> tuple[str, ...]:
    declarations: list[str] = []
    pyproject = root / "pyproject.toml"
    if pyproject.is_file() and not pyproject.is_symlink():
        try:
            project = tomllib.loads(
                read_regular_file_text(root, pyproject, max_bytes=2_000_000)
            ).get("project", {})
        except (
            OSError,
            tomllib.TOMLDecodeError,
            UnicodeDecodeError,
            UnsafeRegularFileError,
        ):
            project = {}
        dependencies = (
            project.get("dependencies", ()) if isinstance(project, dict) else ()
        )
        if isinstance(dependencies, list):
            declarations.extend(
                value for value in dependencies if isinstance(value, str)
            )
    requirements = root / "requirements.txt"
    if requirements.is_file() and not requirements.is_symlink():
        try:
            declarations.extend(
                read_regular_file_text(
                    root,
                    requirements,
                    max_bytes=2_000_000,
                ).splitlines()
            )
        except (OSError, UnicodeDecodeError, UnsafeRegularFileError):
            pass
    names: list[str] = []
    for declaration in declarations:
        value = declaration.strip()
        if not value or value.startswith(("#", "-", ".", "/", "git+")) or ";" in value:
            continue
        match = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)", value)
        if match:
            names.append(match.group(1))
    return tuple(dict.fromkeys(names))


def _ignored(path: Path, root: Path) -> bool:
    return ignored_repository_path(path.relative_to(root))
