"""Materialize the in-world OSS product state and run its declared smoke test.

`export_product_repo` writes each non-issue artifact's `content` to `dest/<linked_file_path>`,
producing an actual repository on disk. Release gates call ``export_and_smoke``;
the workload manifest must declare the command that validates that repository.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


def _file_text(a: Any, prefer_mainline: bool) -> str:
    if prefer_mainline:
        return getattr(a, "mainline_content", "") or getattr(a, "content", "") or ""
    return getattr(a, "content", "") or ""


def _safe_export_path(root: Path, raw: Any) -> tuple[str, Path]:
    normalized = str(raw or "").replace("\\", "/")
    relative = Path(normalized)
    if (
        not normalized
        or relative.is_absolute()
        or ".." in relative.parts
    ):
        raise ValueError(f"unsafe_product_artifact_path:{normalized}")
    target = root.joinpath(relative)
    resolved_parent = target.parent.resolve()
    try:
        resolved_parent.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"product_artifact_path_escapes_export:{normalized}"
        ) from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(
                f"product_artifact_path_contains_symlink:{normalized}"
            )
    return relative.as_posix(), target


def export_product_repo(
    world: Any,
    dest: str,
    prefer_mainline: bool = False,
    overrides: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Write each non-issue artifact's real file text to `dest`. `prefer_mainline` exports the
    merged mainline tree (for releases); otherwise the working tree (artifact.content).

    ``overrides`` replaces individual artifacts' text by artifact id, which is how
    a merge candidate is built: the mainline tree plus one pull request's own
    changes and nothing else.
    """
    arts = getattr(world, "product_artifacts", {}) or {}
    root = Path(dest).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    written = []
    empties = []
    for a in arts.values():
        if getattr(a, "artifact_type", "") == "issue":
            continue
        raw_path = getattr(a, "linked_file_path", None)
        if not raw_path:
            continue
        path, full = _safe_export_path(root, raw_path)
        content = (overrides or {}).get(getattr(a, "artifact_id", ""))
        if content is None:
            content = _file_text(a, prefer_mainline)
        full.parent.mkdir(parents=True, exist_ok=True)
        with full.open("w", encoding="utf-8") as fh:
            fh.write(content)
        written.append(path)
        if not content.strip():
            empties.append(path)
    return {"dir": str(root), "files": sorted(written),
            "file_count": len(written), "empty_files": sorted(empties)}


def _safe_product_environment(
    repo_dir: str,
    extra: Optional[Dict[str, str]],
) -> Dict[str, str]:
    allowed = (
        "PATH",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
        "SYSTEMROOT",
    )
    environment = {
        key: os.environ[key]
        for key in allowed
        if os.environ.get(key)
    }
    environment.update(
        {
            "HOME": os.path.join(repo_dir, ".product-home"),
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": os.pathsep.join(
                (
                    repo_dir,
                    os.path.join(repo_dir, "src"),
                )
            ),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "HTTP_PROXY": "http://127.0.0.1:9",
            "HTTPS_PROXY": "http://127.0.0.1:9",
            "ALL_PROXY": "http://127.0.0.1:9",
            "NO_PROXY": "",
            "no_proxy": "",
        }
    )
    os.makedirs(environment["HOME"], exist_ok=True)
    if extra:
        raise ValueError("product smoke does not accept injected environment values")
    return environment


def _formal_product_executor():
    if str(os.environ.get("ORG_OSS_MODE") or "").lower() != "formal":
        return None
    from relic.evaluation.execution import ExecutionPolicy, build_command_executor

    values = {
        "backend": os.environ.get("RELIC_EVALUATOR_BACKEND"),
        "container_image": os.environ.get("RELIC_EVALUATOR_CONTAINER_IMAGE"),
        "container_platform": os.environ.get("RELIC_EVALUATOR_CONTAINER_PLATFORM"),
    }
    missing = sorted(key for key, value in values.items() if not value)
    if missing:
        raise RuntimeError(
            "formal_product_runtime_binding_missing:"
            + ",".join(missing)
        )
    return build_command_executor(
        ExecutionPolicy(
            trust_level="untrusted",
            backend=str(values["backend"]),
            container_image=str(values["container_image"]),
            container_platform=str(values["container_platform"]),
            clear_container_entrypoint=(
                str(
                    os.environ.get("RELIC_EVALUATOR_CLEAR_ENTRYPOINT") or ""
                ).strip()
                == "1"
            ),
            network_enabled=False,
            memory_limit_mb=2048,
            cpu_limit=2.0,
            pids_limit=128,
        )
    )


def _container_command(command: List[str]) -> tuple[str, ...]:
    normalized = list(command)
    if normalized and (
        normalized[0] in {"python", "python3"}
        or Path(normalized[0]).name.startswith("python")
    ):
        normalized[0] = "python"
    return tuple(normalized)


_WORKSPACE_PROBE = (
    "python",
    "-c",
    "import os,sys;sys.stdout.write(str(len(os.listdir('.'))))",
)


def _workspace_visibility_error(executor: Any, repo_dir: str) -> Optional[str]:
    """Return an infrastructure error when the executor cannot see the exported tree.

    A containerized executor mounts ``repo_dir`` into the sandbox. When the host
    path lies outside the container runtime's shared-filesystem set (on macOS,
    Docker Desktop shares neither the system TMPDIR under ``/var/folders`` nor
    ``/tmp`` by default), the bind mount SUCCEEDS and yields an EMPTY directory.
    Every command then fails with an import/file error that looks exactly like a
    broken product, which is how a 336-tick formal run recorded 172/172
    ``ci_contract_break`` while the product was fine.

    The invariant checked here is substrate-independent: the sandbox must see the
    same non-empty workspace the host just wrote.
    """
    host_entries = 0
    try:
        host_entries = len(os.listdir(repo_dir))
    except OSError as exc:
        return f"workspace_unreadable_on_host:{exc}"
    if host_entries == 0:
        return "workspace_export_empty"
    try:
        outcome = executor.run(
            root=Path(repo_dir),
            argv=_WORKSPACE_PROBE,
            timeout_seconds=60.0,
        )
    except Exception as exc:  # executor/runtime unavailable is infra, not product
        return f"workspace_probe_failed:{exc!r}"
    if outcome.status in {"blocked", "timeout", "infra_error"}:
        return f"workspace_probe_{outcome.status}:{outcome.blocked_reason or ''}"
    if outcome.exit_code != 0:
        return f"workspace_probe_exit_{outcome.exit_code}"
    seen = (outcome.stdout_tail or "").strip()
    if seen.isdigit() and int(seen) == 0:
        return (
            "workspace_mount_empty:host_has_"
            f"{host_entries}_entries_sandbox_sees_0 "
            f"(host path {repo_dir!r} is not shared with the container runtime; "
            "export to a shared location or configure ORG_PRODUCT_SMOKE_ROOT)"
        )
    return None


def _execute_smoke_command(
    *,
    command: List[str],
    repo_dir: str,
    timeout: int,
    environment: Dict[str, str],
    executor: Any,
    image_workspace: str | None = None,
) -> tuple[int | None, str, str, str | None]:
    if executor is not None:
        if image_workspace:
            runner = getattr(executor, "run_in_image_workspace", None)
            if runner is None:
                return None, "", "", "executor_image_workspace_unsupported"
            outcome = runner(
                root=Path(repo_dir),
                argv=_container_command(command),
                timeout_seconds=float(timeout),
                image_workspace=image_workspace,
            )
        else:
            visibility_error = _workspace_visibility_error(executor, repo_dir)
            if visibility_error:
                return None, "", "", visibility_error
            outcome = executor.run(
                root=Path(repo_dir),
                argv=_container_command(command),
                timeout_seconds=float(timeout),
            )
        error = (
            outcome.blocked_reason
            if outcome.status in {"blocked", "timeout", "infra_error"}
            else None
        )
        return (
            outcome.exit_code,
            outcome.stdout_tail,
            outcome.stderr_tail,
            error,
        )
    # A manifest says `python` because that is what the command means inside the
    # evaluator image. Run locally there may be no bare `python` on PATH (a venv
    # commonly exposes only `python3`), and OSError then reads as "the suite is
    # broken" when nothing about the product was ever executed. The container
    # branch above keeps the manifest spelling; only the host run is rebound.
    local_command = list(command)
    if local_command and local_command[0] in ("python", "python3"):
        local_command[0] = sys.executable
    try:
        process = subprocess.run(
            local_command,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, "", "", f"smoke timed out after {timeout}s"
    except OSError as exc:
        return None, "", "", str(exc)
    return (
        process.returncode,
        process.stdout or "",
        process.stderr or "",
        None,
    )


def run_product_smoke(
    repo_dir: str,
    timeout: int = 25,
    env: Dict[str, str] = None,
    command: Optional[List[str]] = None,
    executor: Any = None,
    image_workspace: str | None = None,
) -> Dict[str, Any]:
    """Run the product's declared smoke command inside the exported repo.

    A missing command is a malformed workload, not permission to guess a
    product-specific entry point.
    """
    if command is None:
        return {"ok": False, "error": "workload manifest declares no smoke command"}
    smoke_command = list(command)
    if not smoke_command:
        return {"ok": False, "error": "workload manifest smoke command is empty"}
    if smoke_command and smoke_command[0] in ("python", "python3"):
        smoke_command[0] = sys.executable
    formal_executor = executor if executor is not None else _formal_product_executor()
    run_env = _safe_product_environment(repo_dir, env)
    returncode, stdout, stderr, launch_error = _execute_smoke_command(
        command=smoke_command,
        repo_dir=repo_dir,
        timeout=timeout,
        environment=run_env,
        executor=formal_executor,
        image_workspace=image_workspace,
    )
    if launch_error:
        return {"ok": False, "error": launch_error}
    metrics, summary = None, None
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith("{") and "metrics" in s:
            try:
                summary = json.loads(s)
                metrics = summary.get("metrics")
                break
            except Exception:
                pass
    result = {"ok": returncode == 0, "returncode": returncode, "metrics": metrics,
              "summary": summary, "stdout_tail": stdout[-2000:],
              "stderr_tail": stderr[-1500:]}
    result["command"] = smoke_command
    return result


def declared_public_test_command(world: Any) -> Optional[List[str]]:
    """The substrate's OWN public test command, if it declares one.

    These are the tests that ship inside the agent-visible starter repo (the
    manifest lists hidden tests separately under private_evaluator_only), so
    running them leaks nothing: an agent can already read the files. Exposing
    them is what lets the organization verify its own fix, which is the whole
    difference between patching blindly and converging on a correct fix.
    """
    from environments.org_env.product.substrates.eval_assets import oss_eval_assets

    assets = oss_eval_assets(world) or {}
    manifest = assets.get("manifest") or {}
    command = (manifest.get("public_tests") or {}).get("command")
    return list(command) if isinstance(command, list) and command else None


def declared_public_test_files(world: Any) -> List[str]:
    """Repo paths the declared public test command actually runs.

    Only these files can produce evidence: a test written anywhere else is never
    executed, so the organization would author it and learn nothing. Derived
    from the command itself rather than a separate manifest key, so the two can
    never disagree about what the suite covers.

    Options and the runner invocation are skipped; what remains are path-shaped
    arguments. Substrate-independent.
    """
    command = declared_public_test_command(world) or []
    files: List[str] = []
    for token in command:
        text = str(token)
        if text.startswith("-") or "/" not in text.replace("\\", "/"):
            continue
        files.append(text.replace("\\", "/"))
    return files


def public_test_target_files(world: Any) -> List[str]:
    """Concrete repo files the agent can actually edit to add a public test.

    declared_public_test_files reports what the command names, and for most
    packs that is a DIRECTORY — `pytest tests/public` runs a tree, not a file.
    Nothing in the world is an artifact called "tests/public", so a candidate
    keyed on that path found no artifact and was dropped: of thirteen packs only
    one names a file directly, so writing a regression test was unreachable in
    twelve of them. That is the discipline the whole red-then-green contract
    depends on, silently unavailable wherever the command happens to be written
    against a directory.

    Directory tokens expand to the test files that exist under them; file tokens
    pass through. Only paths that exist as artifacts are returned, because a
    path the agent cannot open is not a place it can write a test.
    """
    declared = declared_public_test_files(world)
    if not declared:
        return []
    artifact_paths = {
        path
        for artifact in (getattr(world, "product_artifacts", {}) or {}).values()
        if (path := str(getattr(artifact, "linked_file_path", "") or ""))
    }
    out: List[str] = []
    for token in declared:
        if token in artifact_paths:
            out.append(token)
            continue
        prefix = token.rstrip("/") + "/"
        out.extend(sorted(p for p in artifact_paths if p.startswith(prefix)))
    # dict.fromkeys: a file named both directly and via its directory is one
    # place to write, not two.
    return list(dict.fromkeys(out))


def declared_public_test_dependencies(world: Any) -> List[str]:
    """Extra modules the public test command needs (pytest plugins, fixtures).

    Separate from runtime_dependencies (what the PRODUCT imports) and from
    evaluator_dependencies (what the hidden oracles need): a test plugin is
    required only to RUN the suite. Declared per substrate so nothing is
    hardcoded.
    """
    from environments.org_env.product.substrates.eval_assets import oss_eval_assets

    assets = oss_eval_assets(world) or {}
    manifest = assets.get("manifest") or {}
    deps = (manifest.get("public_tests") or {}).get("dependencies")
    return [str(d) for d in deps] if isinstance(deps, list) else []


def public_test_sandbox_readiness(world: Any, repo_dir: str, executor: Any) -> Optional[str]:
    """Verify the declared test dependencies are importable WHERE THE SUITE RUNS.

    The formal preflight checks dependencies with importlib on the HOST, but the
    public suite executes inside the sandbox: a host that has pytest_asyncio
    proves nothing about the container. Checking in the wrong place is why a
    missing test plugin surfaced only as an unexplained red suite.
    """
    deps = declared_public_test_dependencies(world)
    if not deps or executor is None:
        return None
    # A requirement names a distribution; `find_spec` wants a module. They differ
    # for almost every hyphenated package -- typing-extensions imports as
    # typing_extensions, pytest-xdist as xdist -- so asking find_spec for the
    # distribution name reports installed packages as missing. cattrs declares
    # four such, two of which were present, and its agent-visible suite was
    # refused fifty-three times across a 336-tick run: the organization could
    # never once see its own tests.
    #
    # A marker is honoured too. cattrs asks for exceptiongroup only below Python
    # 3.11; demanding it on 3.12 blocks a suite over a dependency the suite does
    # not want.
    requirements = [str(d).strip() for d in deps if str(d).strip()]
    if not requirements:
        return None
    probe = (
        "import sys\n"
        "from importlib.metadata import distribution, PackageNotFoundError\n"
        "try:\n"
        "    from packaging.requirements import Requirement\n"
        "except Exception:\n"
        "    Requirement = None\n"
        f"missing = []\n"
        f"for raw in {requirements!r}:\n"
        "    name, marker = raw, None\n"
        "    if Requirement is not None:\n"
        "        try:\n"
        "            req = Requirement(raw)\n"
        "            name, marker = req.name, req.marker\n"
        "        except Exception:\n"
        "            pass\n"
        "    else:\n"
        "        name = raw.split(';')[0].split('[')[0]\n"
        "        for sep in ('>=', '<=', '==', '!=', '~=', '>', '<'):\n"
        "            name = name.split(sep)[0]\n"
        "        name = name.strip()\n"
        "    if marker is not None and not marker.evaluate():\n"
        "        continue\n"
        "    try:\n"
        "        distribution(name)\n"
        "    except PackageNotFoundError:\n"
        "        import importlib.util as u\n"
        "        if u.find_spec(name.replace('-', '_')) is None:\n"
        "            missing.append(name)\n"
        "    except Exception:\n"
        "        pass\n"
        "sys.stdout.write(','.join(missing))\n"
    )
    try:
        outcome = executor.run(
            root=Path(repo_dir), argv=("python", "-c", probe), timeout_seconds=60.0
        )
    except Exception as exc:
        return f"public_test_dependency_probe_failed:{exc!r}"
    missing = (outcome.stdout_tail or "").strip()
    if outcome.exit_code == 0 and missing:
        return (
            "public_test_dependencies_missing_in_sandbox:" + missing
            + " (declare them under public_tests.dependencies and install them in"
            " the evaluator image; the host having them proves nothing)"
        )
    return None


def run_public_tests(world: Any, timeout: int = 120) -> Dict[str, Any]:
    """Run the substrate's declared public test suite against the WORKING tree.

    Returns ``{ok, available, returncode, failed_tests, summary, error}``.
    ``available`` is False when the substrate declares no public tests, so a
    caller can distinguish "no suite here" from "suite failed" — the two must
    never be conflated into a single falsy verdict.
    """
    command = declared_public_test_command(world)
    if not command:
        return {"ok": False, "available": False, "summary": "substrate declares no public tests"}
    cache = world.__dict__.setdefault("_public_test_cache", {})
    key = _repo_hash(world, prefer_mainline=False)
    if key in cache:
        return cache[key]
    dest = os.path.join(_product_smoke_root(), "public_tests_" + key[:12])
    export_product_repo(world, dest, prefer_mainline=False)
    executor = _formal_product_executor()
    image_workspace = str(
        getattr(world, "__dict__", {}).get("_cooperbench_image_workspace") or ""
    ).strip() or None
    readiness_error = public_test_sandbox_readiness(world, dest, executor)
    if readiness_error:
        result = {"ok": False, "available": True, "returncode": None,
                  "failed_tests": [], "summary": "", "error": readiness_error}
        cache[key] = result
        return result
    environment = _safe_product_environment(dest, {})
    returncode, stdout, stderr, launch_error = _execute_smoke_command(
        command=list(command),
        repo_dir=dest,
        timeout=timeout,
        environment=environment,
        executor=executor,
        image_workspace=image_workspace,
    )
    if launch_error:
        result = {"ok": False, "available": True, "returncode": None,
                  "failed_tests": [], "summary": "", "error": str(launch_error)}
        cache[key] = result
        return result
    combined = f"{stdout}\n{stderr}"
    lines = combined.splitlines()
    failed = sorted({
        line.split("::")[0].strip() + "::" + line.split("::")[1].split()[0].strip()
        for line in lines
        if line.startswith("FAILED ") and "::" in line
    })
    summary = ""
    for line in reversed(lines):
        stripped = line.strip()
        if any(tok in stripped for tok in (" passed", " failed", " error", "no tests ran")):
            summary = stripped
            break
    harness_error = _test_harness_launch_error(returncode, combined, summary)
    if harness_error:
        # The suite never ran: reporting this as "tests failed" would send the org
        # after a code defect that does not exist - the same misattribution that
        # turned an empty container mount into 172 phantom contract breaks.
        result = {"ok": False, "available": True, "returncode": returncode,
                  "failed_tests": [], "summary": summary[:300], "error": harness_error}
        cache[key] = result
        return result
    result = {"ok": returncode == 0, "available": True, "returncode": returncode,
              "failed_tests": failed[:20], "summary": summary[:300], "error": None,
              "collected": _collected_test_count(summary)}
    cache[key] = result
    return result


_OUTCOME_COUNT_RE = re.compile(
    r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed)\b")


def _collected_test_count(summary: str) -> int:
    """How many tests the runner actually reported on, from its summary line.

    The suite is agent-editable, so the organization can make a red suite green
    by deleting or weakening tests instead of fixing the product. A pass rate
    cannot distinguish those; the size of the suite can. Counting every reported
    outcome - not just passes - is what makes a shrunk suite visible.

    Returns 0 when no outcome counts appear, which callers must read as "no
    measurement", never as "the suite is empty".
    """
    return sum(int(n) for n, _ in _OUTCOME_COUNT_RE.findall(str(summary or "")))


_HARNESS_LAUNCH_MARKERS = (
    "error importing plugin",
    "no module named",
    "importerror",
    "modulenotfounderror",
    "error: unrecognized arguments",
    "usage error",
    "internal error",
)


def _test_harness_launch_error(
    returncode: int | None,
    output: str,
    summary: str,
) -> Optional[str]:
    """Distinguish "the test runner could not start" from "tests failed".

    A missing test plugin, an unimportable conftest or a usage error means the
    suite never executed, so there is no evidence about the product at all.
    Substrate-independent signal: no per-test outcome summary was produced AND
    the output carries a harness-level import/usage failure.
    """
    if returncode == 0:
        return None
    produced_outcomes = bool(summary) and any(
        token in summary for token in (" passed", " failed", " error")
    )
    if produced_outcomes:
        return None
    lowered = output.lower()
    for marker in _HARNESS_LAUNCH_MARKERS:
        if marker in lowered:
            for line in reversed(output.splitlines()):
                if marker in line.lower():
                    return f"public_test_harness_failed:{line.strip()[:180]}"
            return f"public_test_harness_failed:{marker}"
    # Fundamental criterion, independent of how the runner phrases its error: a
    # non-zero exit that produced NO per-test outcome at all means the suite never
    # ran. Enumerating error shapes is endless - a duplicate-plugin registration
    # surfaces as a bare ValueError with empty stdout and matches no marker.
    if not produced_outcomes:
        detail = ""
        for line in reversed(output.splitlines()):
            stripped = line.strip()
            if stripped and not stripped.startswith(("File \"", "  ", "{")):
                detail = stripped
                break
        return (
            "public_test_harness_produced_no_outcomes:"
            + (detail[:180] or f"exit={returncode}")
        )
    return None


def _declared_smoke_command(world: Any) -> Optional[List[str]]:
    """Return an OSS manifest smoke command without exposing evaluator-only paths or content."""
    from environments.org_env.product.substrates.eval_assets import (
        oss_eval_assets,
    )

    assets = oss_eval_assets(world) or {}
    manifest = assets.get("manifest") or {}
    command = ((manifest.get("entrypoints") or {}).get("smoke") or {}).get("command")
    return list(command) if isinstance(command, list) and command else None


def pr_public_test_command(world: Any, pr: Any) -> Optional[List[str]]:
    """Narrow a PR's public CI to the issues that request actually carries.

    A pack may expose one acceptance module per public issue.  Running the whole
    directory on every PR makes independent work impossible: a correct fix for
    issue A remains red until B..N are implemented too.  Running only a generic
    smoke is the opposite failure -- any edit turns CI green.  The merge
    candidate therefore runs the pack's top-level smoke modules plus the
    acceptance modules for ``pr.linked_issue_ids``.

    Packs without issue-level acceptance modules retain their declared command.
    No evaluator-only path or hidden test is consulted here.
    """
    command = _declared_smoke_command(world)
    if not command or pr is None:
        return command

    linked = list(getattr(pr, "linked_issue_ids", None) or [])
    fallback = str(getattr(pr, "linked_issue", "") or "")
    if fallback and fallback not in linked:
        linked.append(fallback)
    if not linked:
        return command

    artifact_paths = {
        str(getattr(artifact, "linked_file_path", "") or "").replace("\\", "/")
        for artifact in (getattr(world, "product_artifacts", {}) or {}).values()
    }
    selected = []
    for issue_id in linked:
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(issue_id)).strip("_")
        path = f"tests/public/acceptance/test_{safe}.py"
        if path in artifact_paths:
            selected.append(path)
    if not selected:
        return command

    generic = sorted(
        path
        for path in artifact_paths
        if path.startswith("tests/public/")
        and path.endswith(".py")
        and "/acceptance/" not in path
        and path.count("/") == 2
    )
    targets = [*generic, *dict.fromkeys(selected)]

    narrowed: List[str] = []
    replaced = False
    for token in command:
        normalized = str(token).replace("\\", "/").rstrip("/")
        if normalized == "tests/public":
            narrowed.extend(targets)
            replaced = True
        else:
            narrowed.append(str(token))
    return narrowed if replaced else command


def smoke_error_brief(sm: Dict[str, Any]) -> str:
    """A short, actionable error from a failed smoke run: '<file>:<line> | <ErrorType: msg>'.
    Fed into the release-blocker issue + the code editor so agents fix the real break."""
    import re
    if not sm or sm.get("ok"):
        return ""
    if sm.get("error"):
        return str(sm["error"])
    stdout = str(sm.get("stdout_tail") or "")
    stderr = str(sm.get("stderr_tail") or "")
    # Pytest writes assertion failures and source locations to stdout, while
    # plugin warnings go to stderr.  Reading stderr alone made every red TG CI
    # say only "PytestDeprecationWarning", so agents re-ran the same check
    # without learning which product file was broken.
    for stream in (stdout, stderr):
        locations = []
        for line in stream.splitlines():
            match = re.match(
                r"^([A-Za-z0-9_./\\-]+\.py):(\d+):\s*(.+)$",
                line.strip(),
            )
            if not match:
                continue
            path = match.group(1).replace("\\", "/")
            if path.startswith("tests/") or "site-packages/" in path or "/.venv/" in path:
                continue
            locations.append((path, match.group(2), match.group(3)))
        if locations:
            path, line_number, detail = locations[-1]
            return f"{path}:{line_number} | {detail}"[:300]

    lines = [ln.rstrip() for ln in (stdout + "\n" + stderr).splitlines() if ln.strip()]
    failures = [line.strip()[7:] for line in lines if line.strip().startswith("FAILED ")]
    if failures:
        return failures[0][:300]

    useful = [
        line
        for line in lines
        if "PytestDeprecationWarning" not in line
        and not line.lstrip().startswith("warnings.warn(")
    ]
    if not useful:
        return f"smoke exited rc={sm.get('returncode')}"
    frame = ""
    for ln in useful:
        if ln.lstrip().startswith("File "):
            frame = ln.strip()
    final = useful[-1]

    # Where the steps stack, the earliest failing one is the one to fix: the
    # rest fail because it does. Taking the last line named the last step
    # instead, so a run whose step 3 returned a malformed manifest was told
    # about step 5, and eight people spread their edits over three modules for
    # two hundred ticks without one of them going back to step 3.
    steps = []
    for ln in lines:
        m = re.search(r"\bstep (\d+)\b.*\bfail", ln, re.IGNORECASE)
        if m:
            steps.append((int(m.group(1)), ln.strip()))
    if steps:
        first, said = min(steps, key=lambda s: s[0])
        rest = sorted({n for n, _ in steps} - {first})
        trailing = (f"  (steps {', '.join(str(n) for n in rest)} fail behind it)"
                    if rest else "")
        return (said + trailing)[:300]


    m = re.search(r'File ".*[\\/]([^"\\/]+)", line (\d+)', frame)
    loc = f"{m.group(1)}:{m.group(2)} | " if m else ""
    return (loc + final)[:300]


# v14b: tolerance for evaluator self-consistency. A correct eval has
# `unsupported_claim_rate ≈ 1 - claim_evidence_coverage`. If the two disagree by more than this,
# the EVALUATOR is internally inconsistent (not the product) — see smoke_eval_inconsistency.
_EVAL_CONSISTENCY_TOL = 0.34


def _smoke_metrics(sm: Dict[str, Any]):
    """Return (summary, metrics) dicts from a smoke result, or (summary, None)."""
    summary = sm.get("summary") if isinstance(sm.get("summary"), dict) else {}
    metrics = sm.get("metrics")
    if not isinstance(metrics, dict):
        metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    return summary, (metrics if isinstance(metrics, dict) else None)


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def smoke_quality_issue(sm: Dict[str, Any]) -> str:
    """v11 §8 red light: a smoke run that DOESN'T CRASH can still be semantically broken
    (the v10/v11 finding: `credibility_score=0` + `eval_failed` both "passed"). Enforce —
    ONLY when the smoke itself reports them — that the product actually WORKS:
      * eval runs cleanly        (summary.eval_result has no 'error')
      * claims are grounded       (metrics.credibility_score / claim_evidence_coverage > 0)
    Returns a brief that LEADS with the suspected '<file>.py' so the localizer + debugging
    loop target the right module; '' when clean or when the run doesn't report these signals
    (we never block the early prototype before the feature exists).

    v14b (anti false-negative): this gate must judge the PRODUCT, not trust a possibly-buggy
    evaluator. If claims are source-matched (coverage / credibility > 0) yet the eval ALSO
    reports `unsupported_claim_rate=1.0`, that contradiction is an evaluator bug — we do NOT
    block a working product on it (and `smoke_eval_inconsistency` localizes it to eval_stub).
    This is exactly what stalled v14: a t232 eval patch set unsupported=1.0 while coverage
    stayed 1.0, the old gate believed it, blocked every release, and mislocalized to
    claim_tracker.py — so the org optimized the wrong module for two weeks."""
    if not sm or not sm.get("ok"):
        return ""   # an outright crash is already covered by smoke_error_brief
    summary, metrics = _smoke_metrics(sm)
    ev = summary.get("eval_result")
    if isinstance(ev, dict) and ev.get("error"):
        return (f"eval/eval_stub.py | eval did not run cleanly (eval_result.error={ev.get('error')}): "
                "run_eval raised on the real claims/sources — fix eval/eval_stub.py")
    if metrics is None:
        return ""

    # primary grounding signal: agent-added credibility_score (semantic) if present, else the
    # seed's structural claim_evidence_coverage. This is the TRUTH about whether claims are tied
    # to sources; the all-unsupported red light only fires when it does NOT contradict this.
    cred = _num(metrics.get("credibility_score"))
    cov = _num(metrics.get("claim_evidence_coverage"))
    grounded = cred if cred is not None else cov

    if grounded is not None and grounded <= 0:
        return ("tools/claim_tracker.py | grounding=0 (no claim is linked to a source) — make "
                "research_loop.py/claim_tracker.py attach source_ids so claims are supported")
    unsup = _num(metrics.get("unsupported_claim_rate"))
    # only treat "every claim unsupported" as a PRODUCT failure when grounding doesn't say
    # otherwise — a high unsupported_rate alongside high coverage/credibility is an evaluator
    # contradiction (surfaced non-blocking via smoke_eval_inconsistency), not a product break.
    if unsup is not None and unsup >= 1.0 and not (grounded is not None and grounded > 0):
        return ("tools/claim_tracker.py | unsupported_claim_rate=1.0 (every claim is unsupported) — "
                "link claims to sources in research_loop.py/claim_tracker.py")
    # v14 P2 (anti spec-gaming): synthetic / fallback source attachment does NOT count as real
    # grounding. v13b passed coverage=1.0 with fallback_attached_count=4 (every claim auto-attached
    # to a generated source) — structurally grounded, semantically empty. If the REAL grounding
    # (claims tied to a non-fallback source) is <= 0, the gate fails.
    n = _num(metrics.get("n_claims"))
    fb = _num(metrics.get("fallback_attached_count"))
    if n and n > 0 and fb is not None and fb > 0:
        real = (cov if cov is not None else (grounded or 0)) * n - fb
        if real <= 0:
            return ("tools/claim_tracker.py | grounding is FALLBACK-ONLY (" + str(int(fb)) + " synthetic "
                    "source attachments) — link claims to REAL retrieved sources, not auto-generated ones")
    return ""


def smoke_eval_inconsistency(sm: Dict[str, Any]) -> str:
    """v14b: detect a SELF-CONTRADICTORY evaluator and localize it to eval/eval_stub.py.

    A correct eval satisfies `unsupported_claim_rate ≈ 1 - claim_evidence_coverage`. When claims
    are source-matched (coverage / credibility high) yet the eval reports them all-unsupported, the
    metric is internally inconsistent — the bug is in `run_eval`, NOT in claim tracking. Returns an
    eval_stub-localized brief (so CI / the debugging loop fix the evaluator); '' when consistent.
    This is informational: it must NOT by itself block a product that is actually grounded."""
    if not sm or not sm.get("ok"):
        return ""
    _summary, metrics = _smoke_metrics(sm)
    if metrics is None:
        return ""
    cred = _num(metrics.get("credibility_score"))
    cov = _num(metrics.get("claim_evidence_coverage"))
    unsup = _num(metrics.get("unsupported_claim_rate"))
    gr = _num(metrics.get("grounding_rate"))
    if unsup is not None:
        if cov is not None and (cov - (1.0 - unsup)) > _EVAL_CONSISTENCY_TOL:
            return (f"eval/eval_stub.py | metric self-contradiction: claim_evidence_coverage={cov} but "
                    f"unsupported_claim_rate={unsup} (should be ~{round(1.0 - cov, 2)}) — fix the "
                    "supported/unsupported logic in run_eval (eval/eval_stub.py), not claim_tracker.py")
        if cred is not None and cred > 0 and unsup >= 1.0:
            return (f"eval/eval_stub.py | metric self-contradiction: credibility_score={cred} but "
                    "unsupported_claim_rate=1.0 — fix the supported/unsupported logic in run_eval "
                    "(eval/eval_stub.py), not claim_tracker.py")
    # v14c: a dedicated SEMANTIC grounding metric must not flatly contradict the STRUCTURAL one.
    # v14b shipped with claim_evidence_coverage=1.0 yet grounding_rate=0.0 — incoherent quality
    # signals that let "0 real grounding" sail through. Force the eval to reconcile them.
    base = cov if cov is not None else cred
    if gr is not None and base is not None and (base - gr) > _EVAL_CONSISTENCY_TOL:
        return (f"eval/eval_stub.py | metric self-contradiction: grounding_rate={gr} but "
                f"claim_evidence_coverage/credibility={base} — reconcile the structural vs semantic "
                "grounding metrics in run_eval (eval/eval_stub.py); don't report both 0 and 1")
    return ""


def export_and_smoke(world: Any, dest: str, timeout: int = 25) -> Dict[str, Any]:
    exp = export_product_repo(world, dest)
    smoke = run_product_smoke(
        dest,
        timeout=timeout,
        command=_declared_smoke_command(world),
    )
    return {"export": exp, "smoke": smoke}


def _repo_hash(world: Any, prefer_mainline: bool = True) -> str:
    arts = getattr(world, "product_artifacts", {}) or {}
    h = hashlib.sha1()
    for a in sorted(arts.values(), key=lambda x: (getattr(x, "linked_file_path", "") or x.artifact_id)):
        if getattr(a, "artifact_type", "") == "issue":
            continue
        p = getattr(a, "linked_file_path", None)
        if not p:
            continue
        h.update(p.encode("utf-8")); h.update(b"\0")
        h.update(_file_text(a, prefer_mainline).encode("utf-8")); h.update(b"\0")
    return h.hexdigest()


def _product_smoke_root() -> str:
    """Directory that exported product trees are written to before a smoke run.

    Defaults to the system temp dir, which is correct for a plain subprocess
    smoke. Containerized (formal) runs need a path the container runtime is
    allowed to bind-mount: on macOS, Docker Desktop shares neither the system
    TMPDIR (``/var/folders/...``) nor ``/tmp`` by default, so a temp-dir export
    mounts as an EMPTY workspace and every command fails as if the product were
    broken. Set ``RELIC_PRODUCT_SMOKE_ROOT`` to a shared path, or rely on the
    repo-local fallback used whenever a container executor is configured.
    """
    configured = (os.environ.get("RELIC_PRODUCT_SMOKE_ROOT") or "").strip()
    if configured:
        root = os.path.abspath(os.path.expanduser(configured))
        os.makedirs(root, exist_ok=True)
        return root
    containerized = (
        str(os.environ.get("ORG_OSS_MODE") or "").lower() == "formal"
        and (os.environ.get("RELIC_EVALUATOR_BACKEND") or "").strip()
    )
    if containerized:
        root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))),
            ".product_smoke_workspaces",
        )
        os.makedirs(root, exist_ok=True)
        return root
    return tempfile.gettempdir()


def merge_candidate_text(world: Any, pr: Any) -> Dict[str, str]:
    """The tree a merge of ``pr`` would produce: mainline, plus this PR's patches.

    CI used to judge the whole working tree, which fails a pull request for code
    it does not carry. On a pack whose steps stack that is fatal rather than
    merely strict: an organization that started step 3 could never land its
    correct step 1, because the same working tree held the unfinished step 3. The
    mainline then stays empty however much correct work exists, and the
    evaluation — which scores the mainline — reports nothing. A real CI builds
    the merge candidate, so this does too.

    Isolation is only as strong as the substrate's mainline: an artifact with no
    mainline text is exported from the working copy so the product is not emptied
    out, and such a file therefore still shows whatever is on the desk. An OSS
    time-machine Pack seeds its starter into mainline_content and has no such
    gap; the synthetic substrate leaves it unset until something merges.
    """
    arts = getattr(world, "product_artifacts", {}) or {}
    patches = getattr(world, "patches", {}) or {}
    out: Dict[str, str] = {}
    try:
        carried = world.repo_system.merged_commit_patches(pr)
    except Exception:  # noqa: BLE001  a PR with no readable commits carries nothing
        return out
    for patch_id, artifact_id in carried:
        art = arts.get(artifact_id) if artifact_id else None
        if art is None:
            continue
        patch = patches.get(patch_id)
        promoted = getattr(patch, "new_content", "") if patch else ""
        # Mirror apply_merged_pr: a patch without stored text lands the working copy.
        out[artifact_id] = promoted or (getattr(art, "content", "") or "")
    return out


def _cacheable_smoke_result(result: Any) -> bool:
    """Whether a smoke result is determined by candidate content."""

    if not isinstance(result, dict):
        return False
    error = str(result.get("error") or "")
    return (
        not error
        or error == "no smoke_check.py in exported repo"
        or result.get("cli_returncode") is not None
    )


def release_smoke(
    world: Any,
    prefer_mainline: bool = True,
    timeout: int = 25,
    overrides: Dict[str, str] | None = None,
    command: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Export the (mainline) tree and run its declared smoke, cached by content hash
    so the release gate can call it every sweep without re-spawning a subprocess each tick."""
    cache = world.__dict__.setdefault("_smoke_cache", {})
    key = _repo_hash(world, prefer_mainline)
    if overrides:
        # A merge candidate is a different tree from the mainline it is based on,
        # so it must not answer from the mainline's cache entry.
        key = hashlib.sha1(
            (key + "|" + "|".join(f"{k}:{hashlib.sha1(v.encode()).hexdigest()}"
                                  for k, v in sorted(overrides.items()))).encode()
        ).hexdigest()
    smoke_command = list(command) if command is not None else _declared_smoke_command(world)
    image_workspace = str(
        getattr(world, "__dict__", {}).get("_cooperbench_image_workspace") or ""
    ).strip() or None
    if smoke_command:
        # The same candidate may be judged against different issue-level tests;
        # a result cached for issue A cannot answer issue B.
        key = hashlib.sha1(
            (key + "|command:" + json.dumps(smoke_command, separators=(",", ":"))).encode()
        ).hexdigest()
    if image_workspace:
        key = hashlib.sha1(
            (key + "|image_workspace:" + image_workspace).encode()
        ).hexdigest()
    cached = cache.get(key)
    if cached is not None:
        if _cacheable_smoke_result(cached):
            return cached
        cache.pop(key, None)
    dest = os.path.join(_product_smoke_root(), "product_smoke_" + key[:12])
    export_product_repo(world, dest, prefer_mainline=prefer_mainline,
                        overrides=overrides)
    res = run_product_smoke(
        dest,
        timeout=timeout,
        command=smoke_command,
        image_workspace=image_workspace,
    )
    res["repo_dir"] = os.path.abspath(dest)
    if _cacheable_smoke_result(res):
        cache[key] = res
    return res


def export_release_snapshot(world: Any, version: str, base: str = None) -> Dict[str, Any]:
    """Materialize the MAINLINE tree as a real, persistent release snapshot directory."""
    base = base or os.path.join("docs", "product_exports", "releases")
    dest = os.path.join(base, "v" + str(version))
    exp = export_product_repo(world, dest, prefer_mainline=True)
    smoke = run_product_smoke(dest, command=_declared_smoke_command(world))
    return {"export": exp, "smoke": smoke, "version": str(version)}


__all__ = ["export_product_repo", "run_product_smoke", "export_and_smoke",
           "declared_public_test_command", "declared_public_test_dependencies",
           "pr_public_test_command",
           "public_test_sandbox_readiness", "run_public_tests",
           "release_smoke", "export_release_snapshot", "smoke_error_brief", "smoke_quality_issue",
           "smoke_eval_inconsistency", "merge_candidate_text"]
