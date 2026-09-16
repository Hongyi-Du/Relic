import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "relic" / "inspector" / "static"


@pytest.mark.release
def test_frontend_is_local_dom_safe_and_exposes_required_panels() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    combined = f"{html}\n{javascript}\n{css}".lower()

    for panel in (
        "overview",
        "members",
        "tasks",
        "timeline",
        "episodes",
        "reflections",
        "proposals",
        "governance",
        "protocols",
        "artifacts",
        "repo",
        "evidence",
        "decisions",
    ):
        assert f'data-panel="{panel}"' in html
    for unsafe_sink in ("innerhtml", "outerhtml", "insertadjacenthtml", "document.write"):
        assert unsafe_sink not in combined
    assert "http://" not in combined
    assert "https://" not in combined
    assert "private reflection text and memory are never exported" in javascript.lower()
    assert "absence is not interpreted as zero" in javascript.lower()
    assert "do not establish causal attribution" in javascript.lower()
    assert "not a powered participant evaluation" in javascript.lower()
    assert ".event:focus-visible" in css
    assert ".table-scroller" in css


@pytest.mark.release
def test_frontend_diff_and_live_cursor_helpers_are_executable() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    before = {
        "tick": 1,
        "organization": {
            "organization_id": "org",
            "name": "Org",
            "tick": 1,
            "agents": [],
            "tasks": [{"task_id": "task_1", "status": "open"}],
            "proposals": [],
            "protocols": [],
            "artifacts": [{"artifact_id": "artifact_old", "title": "Old"}],
        },
        "events": [],
        "episodes": [],
        "decisions": [],
        "governance_events": [],
        "repo_state": {
            "repository_id": "repo_1",
            "pull_requests": [{"pr_id": "pr_1", "status": "open"}],
        },
    }
    after = {
        **before,
        "tick": 2,
        "organization": {
            **before["organization"],
            "tick": 2,
            "tasks": [{"task_id": "task_1", "status": "done"}],
            "artifacts": [{"artifact_id": "artifact_new", "title": "New"}],
        },
        "repo_state": {"repository_id": "repo_1", "pull_requests": []},
    }
    app_path = STATIC / "app.js"
    script = "\n".join(
        [
            f"const app = require({json.dumps(str(app_path))});",
            f"const before = {json.dumps(before)};",
            f"const after = {json.dumps(after)};",
            "const diff = app.computeFrameDiff(before, after, {organization_id: 'org'});",
            "const cursor = [",
            "  app.nextFrameIndexAfterRefresh(0, 0, 5),",
            "  app.nextFrameIndexAfterRefresh(4, 5, 8),",
            "  app.nextFrameIndexAfterRefresh(2, 5, 8),",
            "];",
            "process.stdout.write(JSON.stringify({diff, cursor}));",
        ]
    )
    completed = subprocess.run(
        [node, "-e", script],
        check=True,
        text=True,
        capture_output=True,
        timeout=30,
    )
    result = json.loads(completed.stdout)
    by_id = {item["id"]: item for item in result["diff"]}

    assert by_id["task_1"]["type"] == "changed"
    assert by_id["artifact_old"]["type"] == "removed"
    assert by_id["artifact_new"]["type"] == "added"
    assert by_id["pr_1"]["type"] == "removed"
    assert result["cursor"] == [4, 7, 2]
