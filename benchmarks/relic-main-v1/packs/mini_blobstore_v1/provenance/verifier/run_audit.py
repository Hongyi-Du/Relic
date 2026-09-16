#!/usr/bin/env python3
"""Generic cumulative Starter/Oracle/mutant/isolation audit for Python tasks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


TASK = Path(__file__).resolve().parents[1]
STEPS = sorted(path for path in (TASK / "steps").iterdir() if path.is_dir())


def apply_solution(step: Path, workspace: Path) -> None:
    shutil.copytree(step / "solution/files", workspace, dirs_exist_ok=True)


def run_verifier(step: Path, workspace: Path, reward: Path) -> dict:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME"}
    }
    env.update(
        {
            "TDF_WORKSPACE": str(workspace),
            "TDF_TESTS_DIR": str(step / "tests"),
            "TDF_REWARD_DIR": str(reward),
            "TDF_VERIFIER_PROCESS": "1",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-I", str(step / "tests/verifier.py")],
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )
    scores = json.loads((reward / "reward.json").read_text())
    evidence = json.loads((reward / "evidence.json").read_text())
    return {
        "returncode": completed.returncode,
        "scores": scores,
        "summary": evidence["summary"],
        "stderr_tail": completed.stderr[-1200:],
    }


def main() -> int:
    config = json.loads((TASK / "verifier/audit.json").read_text())
    report: dict = {"task": TASK.name, "steps": [], "mutants": [], "isolation": {}}
    with tempfile.TemporaryDirectory(prefix=f"{TASK.name}-audit-") as raw:
        root = Path(raw)
        workspace = root / "workspace"
        shutil.copytree(TASK / "environment/codebase", workspace)
        for number, step in enumerate(STEPS, start=1):
            starter = run_verifier(step, workspace, root / f"starter-{number}")
            apply_solution(step, workspace)
            oracle = run_verifier(step, workspace, root / f"oracle-{number}")
            report["steps"].append(
                {
                    "step": step.name,
                    "starter": starter,
                    "oracle": oracle,
                    "passed": starter["scores"]["release_pass"] == 0.0
                    and oracle["scores"]["release_pass"] == 1.0,
                }
            )

        marker = root / "sitecustomize-marker"
        (workspace / "sitecustomize.py").write_text(
            "import os\nfrom pathlib import Path\n"
            "p=os.environ.get('TDF_ISOLATION_MARKER')\n"
            "Path(p).write_text('imported') if p and "
            "os.environ.get('TDF_VERIFIER_PROCESS')=='1' else None\n"
        )
        isolation_env = os.environ.copy()
        isolation_env.update(
            {
                "PYTHONPATH": str(workspace),
                "TDF_WORKSPACE": str(workspace),
                "TDF_TESTS_DIR": str(STEPS[-1] / "tests"),
                "TDF_REWARD_DIR": str(root / "isolation-reward"),
                "TDF_VERIFIER_PROCESS": "1",
                "TDF_ISOLATION_MARKER": str(marker),
            }
        )
        isolation = subprocess.run(
            [sys.executable, "-I", str(STEPS[-1] / "tests/verifier.py")],
            env=isolation_env,
            text=True,
            capture_output=True,
            timeout=180,
        )
        report["isolation"] = {
            "passed": not marker.exists()
            and (root / "isolation-reward/reward.json").is_file(),
            "returncode": isolation.returncode,
        }
        (workspace / "sitecustomize.py").unlink(missing_ok=True)

        for mutant in config["mutants"]:
            mutant_workspace = root / f"mutant-{mutant['name']}"
            shutil.copytree(workspace, mutant_workspace)
            completed = subprocess.run(
                [sys.executable, str(TASK / "verifier/anti_cheat" / mutant["script"])],
                cwd=mutant_workspace,
                text=True,
                capture_output=True,
                timeout=30,
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr or completed.stdout)
            result = run_verifier(
                STEPS[int(mutant["step"]) - 1],
                mutant_workspace,
                root / f"mutant-reward-{mutant['name']}",
            )
            report["mutants"].append(
                {
                    "name": mutant["name"],
                    "step": STEPS[int(mutant["step"]) - 1].name,
                    "result": result,
                    "rejected": result["scores"]["release_pass"] == 0.0,
                }
            )
    report["passed"] = (
        all(step["passed"] for step in report["steps"])
        and len(report["mutants"]) >= 4
        and all(mutant["rejected"] for mutant in report["mutants"])
        and report["isolation"]["passed"]
    )
    output = TASK / "verifier/audit-report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
