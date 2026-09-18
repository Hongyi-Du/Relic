from __future__ import annotations

from pathlib import Path
import tomllib

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.release
def test_docker_release_files_define_a_non_root_canonical_cli_image() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    assert "FROM python:3.12-slim-bookworm@sha256:" in dockerfile
    assert "FROM ghcr.io/astral-sh/uv:0.12.15@sha256:" in dockerfile
    assert "uv sync --frozen --no-dev --no-editable" in dockerfile
    assert "USER relic" in dockerfile
    assert 'ENTRYPOINT ["relic"]' in dockerfile
    assert "COPY . " not in dockerfile
    assert "VOLUME" not in dockerfile
    assert "chown -R relic:relic /app" not in dockerfile
    assert "/.env" in ignored
    assert "/.env.*" in ignored
    assert "/artifacts" in ignored
    assert "/outputs" in ignored
    assert "/reproduction" in ignored
    assert "/traces" in ignored
    assert "artifacts" not in dockerfile
    assert "reproduction" not in dockerfile
    assert "COPY tools/run_org_baselines.py ./tools/run_org_baselines.py" in dockerfile
    assert "COPY tools/__init__.py ./tools/__init__.py" in dockerfile
    assert "!/tools/run_org_baselines.py" in ignored
    assert "!/tools/__init__.py" in ignored
    included = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert included["tools/run_org_baselines.py"] == "tools/run_org_baselines.py"


@pytest.mark.release
def test_compose_services_keep_runtime_data_outside_the_read_only_image() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    runtime = compose["services"]["relic"]
    inspector = compose["services"]["relic-inspector"]

    for service in (runtime, inspector):
        assert service["read_only"] is True
        assert service["init"] is True
        assert service["user"] == "${RELIC_UID:-1000}:${RELIC_GID:-1000}"
        assert service["cap_drop"] == ["ALL"]
        assert service["security_opt"] == ["no-new-privileges:true"]
        assert any(str(entry).startswith("/tmp:") for entry in service["tmpfs"])

    assert runtime["command"] == ["check-env", "--scope", "core"]
    assert "./outputs:/data/outputs" in runtime["volumes"]
    assert "./cache:/data/cache" in runtime["volumes"]
    assert "./traces:/data/traces:ro" in runtime["volumes"]

    assert inspector["profiles"] == ["inspector"]
    assert inspector["ports"] == ["127.0.0.1:${RELIC_INSPECTOR_PORT:-8765}:8765"]
    assert inspector["command"][0] == "inspect"
    assert inspector["command"][-1] == "--allow-remote"
    assert inspector["volumes"] == ["./traces:/data/traces:ro"]
    research_only_environment = {
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "RELIC_OPENAI_DEFAULT_HEADERS_JSON",
        "RELIC_OPENAI_DISABLE_RESPONSE_STORAGE",
        "RELIC_EVALUATOR_BACKEND",
        "RELIC_EVALUATOR_CONTAINER_IMAGE",
        "RELIC_EVALUATOR_CONTAINER_PLATFORM",
        "ORG_OSS_QUALIFICATION_TIMEOUT",
        "RELIC_CLAUDE_OPUS_4_6_MODEL",
    }
    assert research_only_environment.isdisjoint(inspector["environment"])
    assert research_only_environment <= runtime["environment"].keys()
    assert runtime["environment"]["OPENAI_API_KEY"] == "${OPENAI_API_KEY:-}"


@pytest.mark.release
def test_compose_does_not_mount_the_host_docker_socket_or_repository() -> None:
    compose_text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "/var/run/docker.sock" not in compose_text
    assert ".:/app" not in compose_text
    assert "privileged:" not in compose_text
