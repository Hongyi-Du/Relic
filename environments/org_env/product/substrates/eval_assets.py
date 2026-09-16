"""Evaluator-private assets for OSS time-machine substrates (brief §7/§8).

The reference repo, hidden behavior tests, and held-out issues live in a process-local evaluator
vault. The serializable world carries only an opaque vault binding and a public dataset identifier.
Perception, search, and snapshots read explicit fields and never index the private asset payload, so
evaluator material does not leak into agent-visible surfaces or checkpoints.

``run_oss_hidden_tests`` runs the withheld tests against a *materialized* product repo on the
evaluator path only, and returns a structured numeric summary — the hidden-test SOURCE is never
written into world events / messages / docs (brief §8).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional


def run_bounded(
    command: List[str],
    *,
    cwd: str,
    env: Optional[Dict[str, str]] = None,
    timeout: float,
    stdin_text: Optional[str] = None,
) -> tuple[int, str, str]:
    """Run a command in its own process group, and take the GROUP down on timeout.

    ``subprocess.run(timeout=...)`` kills the process it started and nothing that
    process started in turn. The suites here run product code whose defect IS
    non-termination -- soupsieve's seeded issues are an attribute scan and a
    selector cap that never finish -- and pytest runs each such case in a child
    of its own. Killing pytest on the bound therefore left those grandchildren
    orphaned and spinning: twenty-three of them accumulated over an afternoon,
    each holding a core-share on a four-core box at load average 27, while the
    eight simulation arms they were stealing from crawled at a third of their
    rate. The timeout has to end the work, not just stop waiting for it.
    """
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE if stdin_text is not None else None,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(stdin_text, timeout=timeout)
        return process.returncode, stdout or "", stderr or ""
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            process.kill()
        stdout, stderr = process.communicate()
        return 124, stdout or "", (stderr or "") + f"\ntimeout after {timeout}s"


_EVAL_KEY = "_oss_time_machine_eval"
_EVAL_BINDING_KEY = "_oss_time_machine_eval_binding"
_EVALUATOR_PYTEST_CONFIG = "tests/hidden/.evaluator_pytest.ini"
_EVALUATOR_ASSET_VAULT: Dict[str, Dict[str, Any]] = {}


def attach_oss_eval_assets(
    world: Any, spec: Any, heldout_issues: List[Any], hidden_test_specs: List[Any]
) -> Dict[str, Any]:
    """Register evaluator-only state outside the serializable world."""
    assets = {
        "dataset_id": spec.project_id,
        "project_id": spec.project_id,
        "product_name": spec.product_name,
        "manifest": spec.manifest,
        "reference_repo_dir": spec.reference_repo_dir,
        "hidden_tests_dir": spec.hidden_tests_dir,
        "public_tests_dir": spec.public_tests_dir,
        "heldout_issues": list(heldout_issues),
        "hidden_test_specs": list(hidden_test_specs),
    }
    token = secrets.token_hex(32)
    _EVALUATOR_ASSET_VAULT[token] = assets
    world.__dict__.pop(_EVAL_KEY, None)
    world.__dict__[_EVAL_BINDING_KEY] = {
        "schema_version": "oss_evaluator_vault_binding_v1",
        "vault_token": token,
        "dataset_id": spec.project_id,
        "project_id": spec.project_id,
        "product_name": spec.product_name,
    }
    return assets


def oss_eval_assets(world: Any) -> Optional[Dict[str, Any]]:
    legacy = world.__dict__.get(_EVAL_KEY)
    if isinstance(legacy, dict):
        return legacy
    binding = world.__dict__.get(_EVAL_BINDING_KEY)
    if not isinstance(binding, dict):
        return None
    if binding.get("schema_version") != "oss_evaluator_vault_binding_v1":
        raise RuntimeError("invalid OSS evaluator vault binding")
    token = str(binding.get("vault_token") or "")
    cached = _EVALUATOR_ASSET_VAULT.get(token)
    if cached is not None:
        return cached
    dataset_id = str(binding.get("dataset_id") or "")
    if not dataset_id:
        raise RuntimeError("OSS evaluator vault binding has no dataset")
    from environments.org_env.product.substrates import loader

    spec = loader.load_oss_substrate_spec(dataset_id)
    restored = {
        "dataset_id": spec.project_id,
        "project_id": spec.project_id,
        "product_name": spec.product_name,
        "manifest": spec.manifest,
        "reference_repo_dir": spec.reference_repo_dir,
        "hidden_tests_dir": spec.hidden_tests_dir,
        "public_tests_dir": spec.public_tests_dir,
        "heldout_issues": loader.load_heldout_issues(spec),
        "hidden_test_specs": loader.load_hidden_test_specs(spec),
    }
    _EVALUATOR_ASSET_VAULT[token] = restored
    return restored


def oss_evaluator_config(world: Any) -> Dict[str, Any]:
    """The scenario-level evaluator config stashed at seed time (``_oss_evaluator_config``).
    Formal runs populate it so the core OSS metrics don't silently depend on a manually-exported
    env var (brief review §3/§4)."""
    return world.__dict__.get("_oss_evaluator_config") or {}


def oss_eval_enabled(world: Any, key: str, env_var: str) -> bool:
    """Whether a rollout-time evaluator feature is enabled.

    Formal runs can set ``hidden_feedback_forbidden`` so no environment variable
    can accidentally expose hidden results during development."""
    config = oss_evaluator_config(world)
    if key == "run_oss_hidden_tests" and config.get("hidden_feedback_forbidden"):
        return False
    if (os.environ.get(env_var, "") or "").lower() in ("1", "true", "yes", "on"):
        return True
    return bool(config.get(key))


def is_oss_substrate(world: Any) -> bool:
    ps = getattr(world, "product", None)
    return getattr(ps, "substrate_type", "") == "oss_time_machine" or any(
        key in getattr(world, "__dict__", {}) for key in (_EVAL_KEY, _EVAL_BINDING_KEY)
    )


def oss_issue_has_hidden_test(world: Any, issue_id: str) -> bool:
    """True if a hidden behavior test is mapped to ``issue_id``. Server-side ONLY (used by the backlog
    retry loop to decide whether the org's own ``close_issue`` is authoritative for an issue that has
    no evaluator signal). Returns a boolean about EXISTENCE — never the hidden-test source (no leak)."""
    a = oss_eval_assets(world)
    if not a:
        return False
    for s in a.get("hidden_test_specs") or []:
        if issue_id in (getattr(s, "issue_ids", []) or []):
            return True
    return False


def oss_eval_public_summary(world: Any) -> Optional[Dict[str, Any]]:
    """A COUNTS-ONLY summary safe for a debug/omniscient snapshot (brief §7.2) — never content."""
    a = oss_eval_assets(world)
    if not a:
        return None
    return {
        "dataset_id": a.get("dataset_id"),
        "project_id": a.get("project_id"),
        "hidden_tests_count": len(a.get("hidden_test_specs") or []),
        "heldout_issues_count": len(a.get("heldout_issues") or []),
        "has_reference_repo": bool(
            a.get("reference_repo_dir")
            and os.path.isdir(a.get("reference_repo_dir") or "")
        ),
    }


def _run_env(world: Any, repo_dir: str) -> Dict[str, str]:
    # Deliberately do not inherit controller/model/provider credentials.
    env = {
        key: os.environ[key]
        for key in (
            "PATH",
            "TMPDIR",
            "TEMP",
            "TMP",
            "LANG",
            "LC_ALL",
            "SYSTEMROOT",
        )
        if os.environ.get(key)
    }
    # make the materialized product importable regardless of pytest import mode. Support BOTH a
    # flat layout (package at repo root) and a src-layout (package under src/, e.g. real gitingest).
    roots = [repo_dir]
    src_dir = os.path.join(repo_dir, "src")
    if os.path.isdir(src_dir):
        roots.append(src_dir)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(roots + ([existing] if existing else []))
    # keep subprocess pytest hermetic/fast: don't autoload the project's pytest plugins/conftest
    # chain (which can hang or pull in unrelated deps) — brief review §8.
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.setdefault("PY_COLORS", "0")
    env["HOME"] = os.path.join(repo_dir, "tests", "hidden", ".evaluator-home")
    os.makedirs(env["HOME"], exist_ok=True)
    env["HTTP_PROXY"] = "http://127.0.0.1:9"
    env["HTTPS_PROXY"] = "http://127.0.0.1:9"
    env["ALL_PROXY"] = "http://127.0.0.1:9"
    env["NO_PROXY"] = "127.0.0.1,localhost,::1"
    env["no_proxy"] = env["NO_PROXY"]
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _spec_rel_path(spec) -> str:
    rel = getattr(spec, "rel_path", "") or ""
    if rel:
        return ("tests/hidden/" + rel) if not rel.startswith("tests/") else rel
    # derive from the command (last pytest path arg)
    for tok in reversed(list(getattr(spec, "command", []) or [])):
        if tok.endswith(".py"):
            return tok
    return ""


_HIDDEN_PYTEST_WRAPPER = r"""
import hashlib
import hmac
import json
import os
import socket
import sys
import pytest

secret = bytes.fromhex(sys.stdin.buffer.readline().decode("ascii").strip())
sys.stdin.close()
report_path = sys.argv[1]
roots = sys.argv[2].split(os.pathsep)
sys.path[:0] = [root for root in roots if root]

_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex


def _loopback_address(address):
    if not isinstance(address, tuple) or not address:
        return True
    host = str(address[0]).strip("[]").lower()
    return host in {"127.0.0.1", "::1", "localhost"}


def _guarded_connect(sock, address):
    if not _loopback_address(address):
        raise OSError("evaluator network disabled")
    return _original_connect(sock, address)


def _guarded_connect_ex(sock, address):
    if not _loopback_address(address):
        return 101
    return _original_connect_ex(sock, address)


socket.socket.connect = _guarded_connect
socket.socket.connect_ex = _guarded_connect_ex


class EvaluatorPlugin:
    def __init__(self, signing_key):
        self._signing_key = signing_key
        self.records = []

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            status = (
                "passed"
                if report.passed
                else ("infra_error" if report.skipped else "failed")
            )
            self.records.append({"path": report.nodeid, "status": status})
        elif report.when in ("setup", "teardown") and not report.passed:
            self.records.append(
                {"path": report.nodeid, "status": "infra_error"}
            )

    def pytest_collectreport(self, report):
        if report.failed:
            self.records.append(
                {"path": report.nodeid, "status": "infra_error"}
            )

    def pytest_sessionfinish(self, session, exitstatus):
        payload = json.dumps(
            self.records,
            sort_keys=True,
            separators=(",", ":"),
        )
        signature = hmac.new(
            self._signing_key,
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        envelope = {"reports": self.records, "signature": signature}
        descriptor = os.open(
            report_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(envelope, handle, separators=(",", ":"))


plugin = EvaluatorPlugin(secret)
raise SystemExit(pytest.main(sys.argv[3:], plugins=[plugin]))
"""


def _run_hidden_pytest(paths, repo_dir, env, timeout):
    """Run trusted pytest and authenticate its out-of-band result report."""
    roots = [repo_dir]
    src_dir = os.path.join(repo_dir, "src")
    if os.path.isdir(src_dir):
        roots.append(src_dir)
    config_path = ""
    report_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".ini",
            prefix="oss_evaluator_pytest_",
            encoding="utf-8",
            delete=False,
        ) as config:
            config.write("[pytest]\naddopts =\n")
            config_path = config.name
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="oss_evaluator_report_",
            encoding="utf-8",
            delete=False,
        ) as report:
            report_path = report.name
        os.unlink(report_path)
        secret = secrets.token_bytes(32)
        command = [
            sys.executable,
            "-I",
            "-c",
            _HIDDEN_PYTEST_WRAPPER,
            report_path,
            os.pathsep.join(roots),
            "-c",
            config_path,
            "--rootdir",
            repo_dir,
            *paths,
            "--confcutdir=tests/hidden",
            "-rA",
            "-q",
            "--no-header",
            "--tb=no",
            "-p",
            "no:cacheprovider",
        ]
        try:
            rc, out, err = run_bounded(
                command,
                cwd=repo_dir,
                env=env,
                timeout=timeout,
                stdin_text=secret.hex() + "\n",
            )
            if rc == 124:
                return 124, "", f"timeout after {timeout}s", []
        except Exception as exc:  # pragma: no cover
            return 1, "", str(exc), []
        reports = []
        try:
            with open(report_path, encoding="utf-8") as handle:
                envelope = json.load(handle)
            candidate_reports = envelope.get("reports")
            payload = json.dumps(
                candidate_reports,
                sort_keys=True,
                separators=(",", ":"),
            )
            expected_signature = hmac.new(
                secret,
                payload.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            if hmac.compare_digest(
                str(envelope.get("signature") or ""),
                expected_signature,
            ) and isinstance(candidate_reports, list):
                reports = [
                    {
                        "path": str(report.get("path") or ""),
                        "status": str(report.get("status") or "infra_error"),
                    }
                    for report in candidate_reports
                    if isinstance(report, dict)
                ]
        except (json.JSONDecodeError, OSError, TypeError):
            reports = []
        return rc, out, err, reports
    finally:
        for temporary_path in (config_path, report_path):
            if not temporary_path:
                continue
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


def _run_specs_batched(specs, repo_dir, env, timeout):
    """Run ALL hidden specs in ONE pytest invocation and map per-file outcomes back to specs
    (brief review §8 — batch instead of one subprocess per spec). Falls back to per-spec runs if the
    batched output can't be parsed."""
    by_test: List[Dict[str, Any]] = []
    if not specs:
        return by_test, "", ""
    rel_by_path = {_spec_rel_path(s): s for s in specs if _spec_rel_path(s)}
    catalog_path_by_name = {os.path.basename(path): path for path in rel_by_path}
    rc, out, err, reports = _run_hidden_pytest(
        ["tests/hidden"],
        repo_dir,
        env,
        timeout,
    )
    # Map trusted JUnit records back to evaluator-owned spec files.
    file_status: Dict[str, str] = {}
    status_rank = {"passed": 0, "failed": 1, "timeout": 2, "infra_error": 3}
    for report in reports:
        raw_path = str(report.get("path") or "").split("::", 1)[0]
        path = catalog_path_by_name.get(os.path.basename(raw_path), raw_path)
        status = str(report.get("status") or "infra_error")
        previous = file_status.get(path, "passed")
        file_status[path] = max(
            (previous, status),
            key=status_rank.__getitem__,
        )
    if reports:
        catalog_paths = set(rel_by_path)
        observed_paths = set(file_status)
        uncatalogued_paths = observed_paths - catalog_paths
        missing_paths = catalog_paths - observed_paths
        global_infra = bool(uncatalogued_paths) or (
            rc != 0 and not any(status == "failed" for status in file_status.values())
        )
        for path, spec in rel_by_path.items():
            status = file_status.get(path, "infra_error")
            if path in missing_paths or (global_infra and status == "passed"):
                status = "infra_error"
            by_test.append(
                {
                    "test_id": getattr(spec, "test_id", ""),
                    "issue_ids": list(getattr(spec, "issue_ids", []) or []),
                    "passed": status == "passed",
                    "status": status,
                }
            )
        return by_test, out, err
    # fallback: per-spec subprocess (clear mapping if the batch couldn't be parsed)
    last_out, last_err = out, err
    for spec in specs:
        rel_path = _spec_rel_path(spec)
        if rel_path:
            r2, o2, e2, spec_reports = _run_hidden_pytest(
                [rel_path],
                repo_dir,
                env,
                timeout,
            )
        else:
            r2, o2, e2 = _run(list(spec.command), repo_dir, env, timeout)
            spec_reports = []
        if rel_path:
            statuses = [
                str(report.get("status") or "infra_error")
                for report in spec_reports
                if os.path.basename(str(report.get("path") or "").split("::", 1)[0])
                == os.path.basename(rel_path)
            ]
            status = (
                max(statuses, key=status_rank.__getitem__)
                if statuses
                else ("timeout" if r2 == 124 else "infra_error")
            )
        else:
            status = _hidden_result_status(r2, o2, e2)
        by_test.append(
            {
                "test_id": getattr(spec, "test_id", ""),
                "issue_ids": list(getattr(spec, "issue_ids", []) or []),
                "passed": status == "passed",
                "status": status,
            }
        )
        if status != "passed":
            last_out, last_err = o2, e2
    return by_test, last_out, last_err


def _hidden_result_status(returncode: int, stdout: str, stderr: str) -> str:
    if returncode == 0:
        return "passed"
    if returncode == 124:
        return "timeout"
    combined = f"{stdout}\n{stderr}"
    if (
        "ERROR collecting" in combined
        or "ImportError while loading conftest" in combined
        or "Error importing plugin" in combined
        or "INTERNALERROR>" in combined
        or "PytestConfigWarning: Unknown config option" in combined
        or "error: unrecognized arguments:" in combined
        or any(line.strip().startswith("ERROR ") for line in combined.splitlines())
    ):
        return "infra_error"
    return "failed"


def _hermetic_hidden_command(command: List[str]) -> List[str]:
    """Run evaluator tests without inheriting the target project's pytest config."""

    cmd = list(command)
    is_pytest = (len(cmd) >= 3 and cmd[1:3] == ["-m", "pytest"]) or (
        cmd and os.path.basename(cmd[0]) in {"pytest", "py.test"}
    )
    if not is_pytest:
        return cmd
    if "-c" not in cmd and "--config-file" not in cmd:
        cmd.extend(("-c", _EVALUATOR_PYTEST_CONFIG))
    if not any(item.startswith("--confcutdir=") for item in cmd):
        cmd.append("--confcutdir=tests/hidden")
    if "no:cacheprovider" not in cmd:
        cmd.extend(("-p", "no:cacheprovider"))
    if not any(item.startswith("--tb") for item in cmd):
        cmd.append("--tb=no")
    return cmd


def _run(cmd: List[str], cwd: str, env: Dict[str, str], timeout: int):
    cmd = list(cmd)
    # the manifest/specs declare commands as "python ..." but the host may only expose python3 —
    # always run with THIS interpreter so the materialized product is exercised consistently.
    if cmd and cmd[0] in ("python", "python3"):
        cmd = [sys.executable] + cmd[1:]
    try:
        return run_bounded(cmd, cwd=cwd, env=env, timeout=timeout)
    except Exception as e:  # pragma: no cover
        return 1, "", str(e)


def run_oss_hidden_tests(
    world: Any, repo_dir: str, timeout: int = 60
) -> Dict[str, Any]:
    """Run evaluator-only hidden tests against a materialized OSS repo (brief §8).

    Copies the withheld tests into ``repo_dir/tests/hidden`` (evaluator path only), runs each
    ``HiddenTestSpec`` command, and returns a STRUCTURED NUMERIC result: passed / failed / total /
    pass_rate plus per-issue fix status. Never returns hidden-test source text."""
    a = oss_eval_assets(world)
    if not a:
        return {
            "ok": False,
            "error": "not an oss_time_machine substrate",
            "total": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
        }
    return _run_oss_hidden_tests_from_assets(a, repo_dir, timeout=timeout, world=world)


def run_oss_hidden_tests_for_spec(
    spec: Any, repo_dir: str, timeout: int = 60
) -> Dict[str, Any]:
    """Run hidden tests from a resolved substrate spec without constructing a simulation world."""

    from environments.org_env.product.substrates.loader import load_hidden_test_specs

    assets = {
        "hidden_tests_dir": spec.hidden_tests_dir,
        "hidden_test_specs": load_hidden_test_specs(spec),
    }
    return _run_oss_hidden_tests_from_assets(assets, repo_dir, timeout=timeout)


def qualify_oss_hidden_tests_for_spec(
    spec: Any,
    *,
    timeout: int = 180,
) -> Dict[str, Any]:
    """Fail-closed qualification of frozen starter/reference hidden oracles."""
    from environments.org_env.product.substrates.loader import load_hidden_test_specs

    specs = tuple(load_hidden_test_specs(spec))
    test_ids = tuple(item.test_id for item in specs)
    blocking_reasons: List[str] = []
    if not specs:
        blocking_reasons.append("hidden_suite_empty")
    if len(set(test_ids)) != len(test_ids):
        blocking_reasons.append("duplicate_hidden_test_ids")

    required_releases = tuple(
        str(version)
        for version in (
            (spec.manifest.get("evaluation") or {}).get(
                "required_hidden_release_coverage", ()
            )
            or ()
        )
    )
    covered_releases = tuple(
        sorted({str(item.introduced_in) for item in specs if str(item.introduced_in)})
    )
    if required_releases:
        for item in specs:
            if not item.introduced_in:
                blocking_reasons.append(
                    f"hidden_test_missing_introduced_in:{item.test_id}"
                )
        for version in required_releases:
            if version not in covered_releases:
                blocking_reasons.append(f"hidden_release_coverage_missing:{version}")

    def run_frozen_copy(repo_dir: str) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="oss_hidden_qualification_") as tmp:
            checkout = os.path.join(tmp, "checkout")
            shutil.copytree(
                repo_dir,
                checkout,
                ignore=shutil.ignore_patterns(
                    ".git",
                    ".hg",
                    ".svn",
                    "__pycache__",
                    ".pytest_cache",
                    "*.pyc",
                    "*.pyo",
                ),
            )
            return run_oss_hidden_tests_for_spec(spec, checkout, timeout=timeout)

    baseline = run_frozen_copy(spec.starter_repo_dir)
    reference = run_frozen_copy(spec.reference_repo_dir)
    baseline_by_id = {
        str(row.get("test_id")): str(row.get("status") or "infra_error")
        for row in baseline.get("by_test") or []
    }
    reference_by_id = {
        str(row.get("test_id")): str(row.get("status") or "infra_error")
        for row in reference.get("by_test") or []
    }
    from relic.evaluation.time_machine import oracle_blocking_reasons

    oracles = []
    for item in specs:
        baseline_status = baseline_by_id.get(item.test_id, "infra_error")
        reference_status = reference_by_id.get(item.test_id, "infra_error")
        # The same rule the evaluation plan applies, from the same place. It was
        # written twice and corrected once, so this seeder went on refusing every
        # greenfield pack -- a starter whose fixtures raise -- while preflight
        # reported those packs formal-ready.
        blocking_reasons.extend(
            oracle_blocking_reasons(
                item.test_id,
                baseline_status=baseline_status,
                reference_status=reference_status,
            )
        )
        oracles.append(
            {
                "test_id": item.test_id,
                "issue_ids": list(item.issue_ids),
                "introduced_in": item.introduced_in,
                "baseline_status": baseline_status,
                "reference_status": reference_status,
            }
        )
    return {
        "dataset_id": spec.project_id,
        "formal_ready": not blocking_reasons,
        "required_release_coverage": list(required_releases),
        "covered_release_versions": list(covered_releases),
        "blocking_reasons": list(dict.fromkeys(blocking_reasons)),
        "oracle_count": len(oracles),
        "oracles": oracles,
    }


def validate_formal_oss_world(
    world: Any,
    *,
    require_formal: bool = False,
    persist_qualification: bool = True,
) -> Optional[Dict[str, Any]]:
    """Validate and re-qualify a formal world, including loaded checkpoints."""
    params = getattr(getattr(world, "scenario", None), "params", {}) or {}
    mode = str(params.get("experiment_mode") or "")
    if require_formal and mode != "formal":
        raise RuntimeError("checkpoint is not a formal OSS world")
    if mode != "formal":
        return None
    if isinstance(world.__dict__.get(_EVAL_KEY), dict):
        raise RuntimeError("formal OSS world contains legacy inline evaluator assets")

    assets = oss_eval_assets(world) or {}
    dataset_id = str(assets.get("dataset_id") or "")
    if not dataset_id:
        raise RuntimeError("formal OSS world has no evaluator dataset")
    expected_dataset = str(os.environ.get("ORG_OSS_DATASET") or "")
    if expected_dataset and expected_dataset != dataset_id:
        raise RuntimeError(
            f"checkpoint dataset mismatch: expected {expected_dataset!r}, "
            f"loaded {dataset_id!r}"
        )
    config = oss_evaluator_config(world)
    manifest = assets.get("manifest") or {}
    evaluation = manifest.get("evaluation") or {}
    manual_checks_required = bool(evaluation.get("manual_release_checks") or ())
    required_config = {
        "run_oss_hidden_tests": False,
        "run_oss_final_evaluation": True,
        "run_oss_manual_checks": manual_checks_required,
        "hidden_feedback_forbidden": True,
        "run_oss_public_tests": True,
        "prewarm_smoke": True,
    }
    for key, expected in required_config.items():
        if config.get(key) != expected:
            raise RuntimeError(
                f"formal OSS evaluator config mismatch: {key}={config.get(key)!r}"
            )

    from environments.org_env.product.substrates.loader import (
        load_oss_substrate_spec,
    )

    qualification = qualify_oss_hidden_tests_for_spec(
        load_oss_substrate_spec(dataset_id),
        timeout=int(os.environ.get("ORG_OSS_QUALIFICATION_TIMEOUT", "180")),
    )
    if not qualification["formal_ready"]:
        raise RuntimeError(
            f"formal OSS hidden suite qualification failed for {dataset_id!r}: "
            f"{qualification['blocking_reasons']}"
        )
    if persist_qualification:
        world.__dict__["_oss_hidden_qualification"] = qualification
    return qualification


def _run_oss_hidden_tests_from_assets(
    a: Dict[str, Any],
    repo_dir: str,
    *,
    timeout: int,
    world: Any | None = None,
) -> Dict[str, Any]:
    hidden_dir = a.get("hidden_tests_dir") or ""
    if not os.path.isdir(hidden_dir):
        return {
            "ok": False,
            "error": "no hidden_tests_dir",
            "total": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
        }
    hidden_parent = os.path.join(repo_dir, "tests")
    # Staging must leave the tree byte-identical, because the caller compares the
    # repository digest before and after and refuses a run that changed anything.
    # A repository without a tests/ directory of its own — celery, urllib3 and
    # gitingest all ship one — otherwise keeps the empty directory this created,
    # and the whole evaluation aborts with candidate_mutated_during_evaluation.
    created_hidden_parent = not os.path.isdir(hidden_parent)
    os.makedirs(hidden_parent, exist_ok=True)
    dest_hidden = os.path.join(hidden_parent, "hidden")
    backup_root = tempfile.mkdtemp(
        prefix=".evaluator-hidden-backup-",
        dir=hidden_parent,
    )
    backup_hidden = os.path.join(backup_root, "hidden")
    had_candidate_hidden = os.path.lexists(dest_hidden)
    try:
        if had_candidate_hidden:
            os.replace(dest_hidden, backup_hidden)
        shutil.copytree(hidden_dir, dest_hidden)
        with open(
            os.path.join(repo_dir, _EVALUATOR_PYTEST_CONFIG),
            "w",
            encoding="utf-8",
        ) as evaluator_config:
            evaluator_config.write("[pytest]\n")

        env = _run_env(world, repo_dir)
        specs = [
            s for s in (a.get("hidden_test_specs") or []) if getattr(s, "command", None)
        ]
        by_test, last_out, last_err = _run_specs_batched(
            specs,
            repo_dir,
            env,
            timeout,
        )
        issue_pass: Dict[str, bool] = {}
        for test_result in by_test:
            for issue_id in test_result.get("issue_ids") or []:
                issue_pass[issue_id] = (
                    issue_pass.get(issue_id, True) and test_result["passed"]
                )
        total = len(by_test)
        passed_n = sum(1 for result in by_test if result["passed"])
        infra_n = sum(1 for result in by_test if result.get("status") == "infra_error")
        timeout_n = sum(1 for result in by_test if result.get("status") == "timeout")
        return {
            "ok": passed_n == total and total > 0,
            "total": total,
            "passed": passed_n,
            "failed": total - passed_n,
            "infra_error": infra_n,
            "timeout": timeout_n,
            "pass_rate": round(passed_n / total, 3) if total else 0.0,
            "by_test": by_test,
            "issue_fix": issue_pass,
            "issue_fix_rate": round(
                sum(1 for passed in issue_pass.values() if passed) / len(issue_pass),
                3,
            )
            if issue_pass
            else 0.0,
            "stdout_tail": last_out[-1200:],
            "stderr_tail": last_err[-1200:],
        }
    except Exception as exc:  # pragma: no cover
        return {
            "ok": False,
            "error": f"could not stage or run hidden tests: {exc}",
            "total": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
        }
    finally:
        if os.path.lexists(dest_hidden):
            if os.path.islink(dest_hidden) or not os.path.isdir(dest_hidden):
                os.unlink(dest_hidden)
            else:
                shutil.rmtree(dest_hidden)
        if had_candidate_hidden and os.path.lexists(backup_hidden):
            os.replace(backup_hidden, dest_hidden)
        shutil.rmtree(backup_root, ignore_errors=True)
        if created_hidden_parent:
            try:
                os.rmdir(hidden_parent)          # only ever removes it while empty
            except OSError:
                pass


def run_oss_public_tests(
    world: Any, repo_dir: str, timeout: int = 60
) -> Dict[str, Any]:
    """Run the SHIPPED public tests (agent-visible) against the materialized repo."""
    a = oss_eval_assets(world)
    manifest = (a or {}).get("manifest") or {}
    declared_command = (manifest.get("public_tests") or {}).get("command")
    cmd = declared_command or ["python", "-m", "pytest", "tests/public", "-q"]
    if not declared_command and not os.path.isdir(
        os.path.join(repo_dir, "tests", "public")
    ):
        return {
            "ok": True,
            "total": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 1.0,
            "note": "no public tests in repo",
        }
    env = _run_env(world, repo_dir)
    rc, out, err = _run(list(cmd), repo_dir, env, timeout)
    return {
        "ok": rc == 0,
        "returncode": rc,
        "pass_rate": 1.0 if rc == 0 else 0.0,
        "stdout_tail": out[-1200:],
        "stderr_tail": err[-1200:],
    }


def materialize_and_run_oss_hidden_tests(
    world: Any, prefer_mainline: bool = True, timeout: int = 60
) -> Dict[str, Any]:
    """Export the (mainline) product tree to a private temp dir and run hidden tests there. The
    export is evaluator-only and never surfaced to agents."""
    from environments.org_env.product.materialize import export_product_repo

    dest = tempfile.mkdtemp(prefix="oss_hidden_")
    try:
        export_product_repo(world, dest, prefer_mainline=prefer_mainline)
        result = run_oss_hidden_tests(world, dest, timeout=timeout)
    finally:
        shutil.rmtree(dest, ignore_errors=True)
    return result


def missing_runtime_deps(world: Any) -> List[str]:
    """Declared runtime/evaluator modules that are not importable here."""
    import importlib.util

    a = oss_eval_assets(world)
    manifest = (a or {}).get("manifest") or {}
    deps = list(manifest.get("runtime_dependencies") or [])
    deps.extend(manifest.get("evaluator_dependencies") or [])
    out: List[str] = []
    for d in deps:
        name = str(d).split("[")[0].split(">")[0].split("=")[0].split("<")[0].strip()
        if name and importlib.util.find_spec(name) is None:
            out.append(name)
    return out


def oss_post_release_evaluation(world: Any, tick: int) -> Optional[Dict[str, Any]]:
    """Close the issue->fix->test->ticket loop (brief review §6): on a published release, run the
    withheld hidden tests and close ONLY the tickets whose issue is genuinely resolved (post-issue
    work + passing hidden test). Gated on ORG_OSS_HIDDEN_TESTS; de-duped per product-tree hash so it
    re-evaluates only when the mainline actually changed. Emits one release_event/oss_evaluation."""
    if not is_oss_substrate(world):
        return None
    # evaluator_config (formal default) OR ORG_OSS_HIDDEN_TESTS override (brief review §4)
    if not oss_eval_enabled(world, "run_oss_hidden_tests", "ORG_OSS_HIDDEN_TESTS"):
        return None
    seen = world.__dict__.setdefault("_oss_eval_versions", set())
    try:
        from environments.org_env.product.materialize import _repo_hash

        key = _repo_hash(world, prefer_mainline=True)
    except Exception:
        key = f"tick_{tick}"
    if key in seen:
        return None
    seen.add(key)
    from environments.org_env.product.substrates.issue_stream import (
        close_resolved_oss_tickets,
        _has_post_issue_work,
    )

    hid = materialize_and_run_oss_hidden_tests(world, prefer_mainline=True)
    closed = close_resolved_oss_tickets(world, hidden_result=hid)
    # evaluator-authoritative issue-artifact status (backlog retry loop): mark genuinely-resolved
    # issues closed, and REOPEN any hidden-test issue that was marked closed while its hidden test
    # still fails ("declared victory" via close_issue) so it re-enters the visible backlog. This is a
    # STATUS change only — the hidden-test source is never written to any agent-visible surface.
    arts = getattr(world, "product_artifacts", {}) or {}
    issue_fix = hid.get("issue_fix") or {}
    reopened: List[str] = []
    for iid, ok in issue_fix.items():
        a = arts.get(iid)
        if a is None:
            continue
        if ok and _has_post_issue_work(world, iid):
            if getattr(a, "status", "open") != "closed":
                a.status = "closed"
        elif getattr(a, "status", "open") == "closed":
            a.status = "open"
            reopened.append(iid)
    if getattr(world, "events", None) is None:
        world.events = []
    world.events.append(
        {
            "type": "release_event",
            "subtype": "oss_evaluation",
            "tick": int(tick),
            "hidden_pass_rate": hid.get("pass_rate"),
            "issue_fix_rate": hid.get("issue_fix_rate"),
            "issue_fix": hid.get("issue_fix"),
            "closed_tickets": closed,
            "reopened_issues": reopened,
            "auto": True,
        }
    )
    return {"hidden": hid, "closed": closed, "reopened": reopened}


__all__ = [
    "attach_oss_eval_assets",
    "oss_eval_assets",
    "is_oss_substrate",
    "oss_eval_public_summary",
    "oss_issue_has_hidden_test",
    "oss_evaluator_config",
    "oss_eval_enabled",
    "run_oss_hidden_tests",
    "run_oss_hidden_tests_for_spec",
    "qualify_oss_hidden_tests_for_spec",
    "validate_formal_oss_world",
    "run_oss_public_tests",
    "materialize_and_run_oss_hidden_tests",
    "missing_runtime_deps",
    "oss_post_release_evaluation",
]
