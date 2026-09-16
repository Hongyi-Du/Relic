from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from relic.release_checks import check_environment


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.release
def test_wheel_configuration_includes_every_inspector_asset() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    included = project["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]

    for name in ("index.html", "app.css", "app.js"):
        path = f"relic/inspector/static/{name}"
        assert included[path] == path


@pytest.mark.release
def test_core_environment_check_validates_inspector_assets() -> None:
    report = check_environment(scope="core")
    inspector = next(check for check in report["checks"] if check["name"] == "inspector")

    assert inspector["status"] == "pass"
    assert inspector["code"] == "inspector_available"
    assert inspector["details"]["assets"] == ["app.css", "app.js", "index.html"]
    assert inspector["details"]["trace_schema"] == "relic-trace-v1"
