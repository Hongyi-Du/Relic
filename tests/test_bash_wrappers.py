from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASH_DIRECTORY = PROJECT_ROOT / "scripts" / "bash"
WRAPPERS = {
    "check_env.sh": ("check-env",),
    "smoke.sh": ("smoke",),
    "run_cell.sh": ("run-cell",),
    "run_main_120.sh": ("run-main-120",),
    "evaluate.sh": ("evaluate",),
    "start_inspector.sh": ("inspect",),
}


def _install_uv_stub(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "fake bin"
    bin_dir.mkdir()
    stub = bin_dir / "uv"
    stub.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s' "$PWD" > "${UV_CAPTURE}/cwd"
printf '%s\\0' "$@" > "${UV_CAPTURE}/argv"
printf '%s' "${OPENAI_API_KEY-}" > "${UV_CAPTURE}/api-key"
printf '%s' "${OPENAI_BASE_URL-}" > "${UV_CAPTURE}/quoted-value"
printf '%s' "${RELIC_OUTPUT_ROOT-}" > "${UV_CAPTURE}/literal-value"
""",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return bin_dir


def _copy_wrapper_tree(tmp_path: Path) -> Path:
    root = tmp_path / "Relic repository with spaces"
    destination = root / "scripts" / "bash"
    destination.parent.mkdir(parents=True)
    shutil.copytree(BASH_DIRECTORY, destination)
    return root


def test_bash_wrappers_are_syntactically_valid_lf_and_executable() -> None:
    assert shutil.which("bash"), "Bash is the supported runtime for these launchers"

    scripts = sorted(BASH_DIRECTORY.glob("*.sh"))
    assert {script.name for script in scripts} == {"_common.sh", *WRAPPERS}
    for script in scripts:
        assert b"\r\n" not in script.read_bytes()
        assert script.stat().st_mode & stat.S_IXUSR
        result = subprocess.run(
            ["bash", "-n", str(script)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


def test_bash_wrapper_accepts_every_assignment_in_official_env_template(tmp_path: Path) -> None:
    root = _copy_wrapper_tree(tmp_path)
    shutil.copy(PROJECT_ROOT / ".env.example", root / ".env")
    fake_bin = _install_uv_stub(tmp_path)
    capture_directory = tmp_path / "uv capture"
    capture_directory.mkdir()
    environment = os.environ.copy()
    environment.pop("BASH_ENV", None)
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment.get('PATH', '')}"
    environment["UV_CAPTURE"] = str(capture_directory)

    completed = subprocess.run(
        [str(root / "scripts" / "bash" / "smoke.sh"), "--mode", "mock"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (capture_directory / "argv").read_bytes().split(b"\0")[:-1] == [
        b"run",
        b"relic",
        b"smoke",
        b"--mode",
        b"mock",
    ]


@pytest.mark.parametrize(("wrapper", "command"), WRAPPERS.items())
def test_wrappers_proxy_exact_arguments_from_an_arbitrary_directory(
    tmp_path: Path,
    wrapper: str,
    command: tuple[str, ...],
) -> None:
    root = _copy_wrapper_tree(tmp_path)
    secret = "sk-wrapper-secret-never-logged"
    marker = tmp_path / "dotenv-command-ran"
    (root / ".env").write_text(
        "\n".join(
            (
                f'OPENAI_API_KEY="{secret}"',
                "OPENAI_BASE_URL='value with spaces'",
                f'RELIC_OUTPUT_ROOT="$(touch {marker})"',
                "",
            )
        ),
        encoding="utf-8",
    )
    fake_bin = _install_uv_stub(tmp_path)
    capture_directory = tmp_path / "uv capture"
    capture_directory.mkdir()
    arbitrary_cwd = tmp_path / "an arbitrary cwd"
    arbitrary_cwd.mkdir()
    environment = os.environ.copy()
    environment.pop("BASH_ENV", None)
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment.get('PATH', '')}"
    environment["UV_CAPTURE"] = str(capture_directory)
    arguments = ("--flag", "value with spaces", "--literal=$HOME", "")
    launcher = root / "scripts" / "bash" / wrapper
    symlink_directory = tmp_path / "external launchers"
    symlink_directory.mkdir()
    symlink = symlink_directory / wrapper
    symlink.symlink_to(launcher)

    result = subprocess.run(
        [str(symlink), *arguments],
        cwd=arbitrary_cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert secret not in result.stdout
    assert secret not in result.stderr
    assert not marker.exists(), ".env must not be evaluated as shell code"
    assert (capture_directory / "cwd").read_text(encoding="utf-8") == str(root)
    assert (capture_directory / "api-key").read_text(encoding="utf-8") == secret
    assert (capture_directory / "quoted-value").read_text(encoding="utf-8") == "value with spaces"
    assert (capture_directory / "literal-value").read_text(encoding="utf-8") == f"$(touch {marker})"
    assert (capture_directory / "argv").read_bytes().split(b"\0")[:-1] == [
        b"run",
        b"relic",
        *[part.encode() for part in command],
        *[argument.encode() for argument in arguments],
    ]


@pytest.mark.parametrize("dangerous_key", ["PATH", "LD_PRELOAD", "BASH_ENV", "PYTHONPATH"])
def test_dotenv_rejects_process_control_variables_before_uv_lookup(
    tmp_path: Path, dangerous_key: str
) -> None:
    root = _copy_wrapper_tree(tmp_path)
    fake_bin = _install_uv_stub(tmp_path)
    capture_directory = tmp_path / "uv capture"
    capture_directory.mkdir()
    (root / ".env").write_text(f"{dangerous_key}=/tmp/untrusted\n", encoding="utf-8")
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment.get('PATH', '')}"
    environment["UV_CAPTURE"] = str(capture_directory)

    completed = subprocess.run(
        [str(root / "scripts" / "bash" / "smoke.sh"), "--mode", "mock"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "Unsupported variable" in completed.stderr
    assert not (capture_directory / "argv").exists()
