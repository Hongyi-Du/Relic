"""Run the frozen Cooper regression suite without paid provider calls."""
from pathlib import Path
import os
import sys


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "reproduction/cooperbench/reference"
    os.chdir(root)
    sys.path.insert(0, str(root))
    import pytest

    return pytest.main([
        "-q", "-c", str(root / "pytest.ini"),
        "tests/org_env/test_cooperbench_sdl.py",
        "tests/org_env/test_cooperbench_joint_probe_replay.py",
        "tests/org_env/test_cooperbench_actor_workspace.py",
        "tests/org_env/test_thinking_model_adaptation.py",
        *sys.argv[1:],
    ])


if __name__ == "__main__":
    raise SystemExit(main())
