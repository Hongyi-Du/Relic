"""Static contract tests for the Windows-to-WSL launcher boundary."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POWERSHELL_ROOT = PROJECT_ROOT / "scripts" / "powershell"
HELPER = POWERSHELL_ROOT / "_RelicWsl.ps1"
LAUNCHERS = {
    "check_env": "check_env",
    "smoke": "smoke",
    "run_cell": "run_cell",
    "run_main_120": "run_main_120",
    "evaluate": "evaluate",
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_powershell_wrapper_files_exist() -> None:
    assert HELPER.is_file()
    assert (POWERSHELL_ROOT / "check_wsl.ps1").is_file()
    for launcher in LAUNCHERS:
        assert (POWERSHELL_ROOT / f"{launcher}.ps1").is_file()


def test_helper_requires_explicit_wsl_target_and_linux_repository() -> None:
    helper = _read(HELPER)
    assert "Get-Command wsl.exe" in helper
    assert "RELIC_WSL_DISTRIBUTION" in helper
    assert "RELIC_WSL_REPO" in helper
    assert "StartsWith('/')" in helper
    assert "^/mnt(?:/|$)" in helper
    assert "--list', '--quiet" in helper
    assert "--distribution" in helper
    assert "/proc/sys/kernel/osrelease" in helper
    assert "microsoft-standard|wsl2" in helper


def test_check_wsl_uses_the_shared_prerequisite_check() -> None:
    contents = _read(POWERSHELL_ROOT / "check_wsl.ps1")
    assert "_RelicWsl.ps1" in contents
    assert "Get-RelicWslConfiguration" in contents


def test_launchers_forward_arguments_to_fixed_bash_entrypoints() -> None:
    for launcher, bash_entrypoint in LAUNCHERS.items():
        contents = _read(POWERSHELL_ROOT / f"{launcher}.ps1")
        assert "ValueFromRemainingArguments = $true" in contents
        assert "Invoke-RelicWslBashScript" in contents
        assert f"-ScriptName '{bash_entrypoint}'" in contents
        assert "-ScriptArguments $ScriptArguments" in contents
        assert "$result.Output | Write-Output" in contents
        assert "exit $result.ExitCode" in contents


def test_helper_uses_argument_arrays_without_native_runtime_or_drive_paths() -> None:
    contents = _read(HELPER)
    all_scripts = [HELPER, *(POWERSHELL_ROOT / f"{name}.ps1" for name in LAUNCHERS)]
    assert "$wslArguments = @(" in contents
    assert "if ($null -ne $ScriptArguments)" in contents
    assert "$wslArguments += @($ScriptArguments)" in contents
    assert "$commandOutput = @(& wsl.exe @wslArguments)" in contents
    assert "ExitCode = [int]$commandExitCode" in contents
    assert "bash -c" not in contents
    assert "Invoke-Expression" not in contents
    for script in all_scripts:
        text = _read(script)
        assert not re.search(r"(?i)\b(?:python|py)\b", text)
        assert not re.search(r"(?i)\b[A-Z]:[\\/]", text)


@pytest.mark.skipif(
    shutil.which("pwsh") is None and shutil.which("powershell") is None,
    reason="PowerShell is not installed; static wrapper contracts remain covered above.",
)
def test_powershell_scripts_parse_when_available() -> None:
    executable = shutil.which("pwsh") or shutil.which("powershell")
    assert executable is not None
    parser = (
        "$tokens = $null; $errors = $null; "
        "[System.Management.Automation.Language.Parser]::ParseFile("
        "$args[0], [ref]$tokens, [ref]$errors) | Out-Null; "
        "if ($errors.Count) { $errors | ForEach-Object { Write-Error $_ }; exit 1 }"
    )
    for script in [HELPER, POWERSHELL_ROOT / "check_wsl.ps1", *(POWERSHELL_ROOT / f"{name}.ps1" for name in LAUNCHERS)]:
        completed = subprocess.run(
            [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", parser, str(script)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
