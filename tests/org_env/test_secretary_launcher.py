from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _launcher_module():
    path = Path(__file__).parents[2] / "packaging" / "hci" / "secretary_launcher.py"
    spec = importlib.util.spec_from_file_location("secretary_launcher_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_embedded_c_command_reassembles_fragmented_program(monkeypatch) -> None:
    launcher = _launcher_module()
    marker = "SECRETARY_FRAGMENTED_C_PROBE"
    monkeypatch.delenv(marker, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "Secretary.exe",
            "-c",
            "import",
            f"os;os.environ['{marker}']='passed'",
        ],
    )

    assert launcher.run_embedded_python() == 0
    assert os.environ[marker] == "passed"


def test_launcher_exposes_distinct_p2_and_p3_entry_points() -> None:
    launcher = _launcher_module()

    assert launcher.interface_mode([]) == "p3"
    assert launcher.interface_mode(["--p3"]) == "p3"
    assert launcher.interface_mode(["--p2"]) == "p2"
    assert launcher.interface_path("p2") == "/org/seat"
    assert launcher.interface_path("p3") == "/org/liaison"
