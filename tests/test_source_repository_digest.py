"""Conformance checks for the source final-evaluator repository digest."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from environments.org_env.experiments.provenance import (
    repository_digest as run_record_repository_digest,
)
from relic.evaluation.time_machine import _repo_digest as evaluator_repository_digest
from relic.research.repository_digest import repository_digest


def _write_source_digest_fixture(root: Path) -> None:
    (root / "outputs").mkdir(parents=True)
    (root / "tests" / "mypy" / "outputs").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "__pycache__").mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'digest-fixture'\n", encoding="utf-8"
    )
    (root / "module.py").write_text("SOURCE = 1\n", encoding="utf-8")
    (root / "outputs" / "runtime.log").write_text("ignored runtime\n", encoding="utf-8")
    (root / "tests" / "mypy" / "outputs" / "expected.txt").write_text(
        "semantic fixture\n", encoding="utf-8"
    )
    (root / ".git" / "config").write_text("ignored metadata\n", encoding="utf-8")
    (root / "__pycache__" / "module.pyc").write_text("ignored cache\n", encoding="utf-8")
    script = root / "run.sh"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    script.chmod(0o755)
    (root / "current.py").symlink_to("module.py")


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "symlink"),
    reason="the pinned source fixture captures POSIX mode and symlink semantics",
)
def test_digest_matches_the_pinned_source_repo_hash_fixture(tmp_path: Path) -> None:
    """Pinned output from SocioGenesis hci-human-seat@dda36fb5.

    The fixture covers the distinctions that the pre-source local walker got
    wrong: root-only ``outputs`` exclusion, nested semantic ``outputs``,
    ignored VCS/cache state, modes, and a symlink target.
    """

    root = tmp_path / "fixture"
    root.mkdir()
    _write_source_digest_fixture(root)

    expected = "8de29d8d374af8d218bc098721effb7d75f61682d5f8d8d4ca3ac9fd00487ca9"
    assert repository_digest(root) == expected
    assert run_record_repository_digest(root) == expected
    assert evaluator_repository_digest(root) == expected


def test_root_runtime_outputs_do_not_hide_nested_source_outputs(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    root.mkdir()
    (root / "outputs").mkdir()
    nested = root / "tests" / "mypy" / "outputs"
    nested.mkdir(parents=True)
    runtime = root / "outputs" / "runtime.log"
    expected = nested / "expected.txt"
    runtime.write_text("first\n", encoding="utf-8")
    expected.write_text("first\n", encoding="utf-8")

    baseline = repository_digest(root)
    runtime.write_text("second\n", encoding="utf-8")
    assert repository_digest(root) == baseline
    expected.write_text("second\n", encoding="utf-8")
    assert repository_digest(root) != baseline


def test_digest_rejects_a_missing_repository(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repository path is not a directory"):
        repository_digest(tmp_path / "missing")
