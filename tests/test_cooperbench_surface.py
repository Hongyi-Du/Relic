from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from environments.org_env.cooperbench.surface import (
    SurfaceBuildError,
    materialize_public_pack,
    select_public_surface,
)


def _git(repo: Path, *args: str) -> None:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "src" / "widget.py").write_text(
        "class WidgetRegistry:\n    def register_widget(self, name):\n        return name\n",
        encoding="utf-8",
    )
    (repo / "src" / "cache.py").write_text(
        "class CacheStore:\n    def invalidate_cache(self, key):\n        return key\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_widget.py").write_text(
        "def test_widget_registry():\n    pass\n", encoding="utf-8"
    )
    (repo / "pyproject.toml").write_text("[project]\nname='sample'\n", encoding="utf-8")
    (repo / "blob.bin").write_bytes(b"\xff\x00\xfe")
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    return repo


def test_surface_uses_only_public_tasks_and_maps_each_feature(tmp_path: Path):
    repo = _repo(tmp_path)
    tasks = {
        "agent1": "Extend WidgetRegistry.register_widget in src/widget.py",
        "agent2": "Make CacheStore.invalidate_cache support prefixes in src/cache.py",
    }

    plan = select_public_surface(
        repo, tasks, max_files=32, max_total_bytes=1024 * 1024
    )

    assert plan.component_map["cooper_feature_1"] == ("src/widget.py",)
    assert plan.component_map["cooper_feature_2"] == ("src/cache.py",)
    assert all("tests/test_widget.py" not in paths for paths in plan.component_map.values())
    assert plan.public_test_paths == ("tests/test_widget.py",)
    assert plan.public_test_command == (
        "python",
        "-m",
        "pytest",
        "-q",
        "tests/test_widget.py",
    )
    assert "blob.bin" not in plan.selected_paths
    assert plan.digest == plan.to_dict()["surface_digest"]


def test_untracked_native_python_extension_is_a_runtime_asset(tmp_path: Path):
    repo = _repo(tmp_path)
    package = repo / "nativepkg"
    package.mkdir()
    (package / "__init__.py").write_text(
        "from .core import encode\n", encoding="utf-8"
    )
    (package / "core.py").write_text(
        "def encode(value):\n    return value\n", encoding="utf-8"
    )
    _git(repo, "add", "nativepkg")
    _git(repo, "commit", "-m", "add public native package")
    extension = "nativepkg/_native.cpython-311-x86_64-linux-gnu.so"
    (repo / extension).write_bytes(b"\x7fELFpublic-runtime")

    plan = select_public_surface(
        repo,
        {
            "agent1": "Extend encode in nativepkg/core.py",
            "agent2": "Add batch support to encode in nativepkg/core.py",
        },
        max_files=64,
        max_total_bytes=1024 * 1024,
        source_policy="feature_independent",
    )

    assert plan.runtime_asset_paths == ("blob.bin", extension)
    assert extension not in plan.selected_paths
    assert {
        row["path"] for row in plan.runtime_assets_manifest["files"]
    } == {extension, "blob.bin"}


def test_setuptools_scm_generated_version_module_is_a_runtime_asset(tmp_path: Path):
    repo = _repo(tmp_path)
    package = repo / "publicpkg"
    package.mkdir()
    (package / "__init__.py").write_text(
        "from ._version import __version__\n", encoding="utf-8"
    )
    (repo / "pyproject.toml").write_text(
        "[project]\nname='sample'\n"
        "[tool.setuptools_scm]\nwrite_to='publicpkg/_version.py'\n",
        encoding="utf-8",
    )
    (repo / ".gitignore").write_text("*_version.py\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "declare generated public version module")
    generated = "publicpkg/_version.py"
    (repo / generated).write_text("__version__ = '1.0.0'\n", encoding="utf-8")

    plan = select_public_surface(
        repo,
        {
            "agent1": "Extend publicpkg/__init__.py",
            "agent2": "Add metadata to publicpkg/__init__.py",
        },
        max_files=64,
        max_total_bytes=1024 * 1024,
        source_policy="feature_independent",
    )

    assert generated in plan.runtime_asset_paths
    assert generated not in plan.selected_paths


def test_publicly_declared_new_implementation_paths_reach_the_owner_surface(
    tmp_path: Path,
):
    repo = _repo(tmp_path)
    tasks = {
        "agent1": (
            "Add path-value support.\n\n## Files Modified\n\n"
            "- `src/widget.py`\n"
            "- `src/path_value.py`\n"
            "- `tests/test_path_value.py`\n"
            "- `hidden/gold.py`\n"
        ),
        "agent2": "Change CacheStore. Files Modified: src/cache.py",
    }

    plan = select_public_surface(
        repo,
        tasks,
        max_files=32,
        max_total_bytes=1024 * 1024,
        source_policy="feature_independent",
    )

    assert plan.component_map["cooper_feature_1"] == (
        "src/widget.py",
        "src/path_value.py",
    )
    assert "src/path_value.py" not in plan.selected_paths
    pack = materialize_public_pack(
        repo,
        tmp_path / "pack_new_path",
        tasks,
        plan,
        project_id="cooper_new_path",
        private_features=True,
    )
    assert not (pack / "starter_repo" / "src" / "path_value.py").exists()
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["component_map"] == {}


def test_go_surface_uses_the_repository_native_public_gate(tmp_path: Path):
    repo = tmp_path / "go_repo"
    repo.mkdir()
    (repo / "go.mod").write_text(
        "module example.com/sample\n\ngo 1.22\n", encoding="utf-8"
    )
    (repo / "mux.go").write_text(
        "package sample\n\ntype Mux struct{}\n", encoding="utf-8"
    )
    (repo / "metrics.go").write_text(
        "package sample\n\ntype MetricsCollector interface{}\n", encoding="utf-8"
    )
    (repo / "mux_test.go").write_text(
        "package sample\n\nfunc TestMux(t *testing.T) {}\n", encoding="utf-8"
    )
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")

    plan = select_public_surface(
        repo,
        {
            "agent1": "Change Mux in mux.go",
            "agent2": "Add MetricsCollector in metrics.go",
        },
        max_files=32,
        max_total_bytes=1024 * 1024,
        source_policy="feature_independent",
    )

    assert plan.public_test_paths == ("mux_test.go",)
    assert plan.public_test_command == ("go", "test", "-p=1", "./...")


def test_rust_surface_compiles_the_public_test_package(tmp_path: Path):
    repo = tmp_path / "rust_repo"
    (repo / "crates" / "engine" / "src").mkdir(parents=True)
    (repo / "tests" / "src").mkdir(parents=True)
    (repo / "Cargo.toml").write_text(
        '[workspace]\nmembers=["crates/engine", "tests"]\n', encoding="utf-8"
    )
    (repo / "crates" / "engine" / "Cargo.toml").write_text(
        '[package]\nname="engine"\nversion="0.1.0"\n', encoding="utf-8"
    )
    (repo / "crates" / "engine" / "src" / "lib.rs").write_text(
        "pub fn render() {}\n", encoding="utf-8"
    )
    (repo / "tests" / "Cargo.toml").write_text(
        '[package]\nname="engine-tests"\nversion="0.1.0"\n',
        encoding="utf-8",
    )
    (repo / "tests" / "src" / "lib.rs").write_text(
        "pub fn fixture() {}\n", encoding="utf-8"
    )
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")

    plan = select_public_surface(
        repo,
        {
            "agent1": "Change render in crates/engine/src/lib.rs",
            "agent2": "Add layout support in crates/engine/src/layout.rs",
        },
        max_files=32,
        max_total_bytes=1024 * 1024,
        source_policy="feature_independent",
    )

    assert plan.public_test_command == (
        "cargo",
        "test",
        "-p",
        "engine-tests",
        "--no-run",
        "-j",
        "1",
    )


def test_typescript_surface_prefers_declared_test_script(tmp_path: Path):
    repo = tmp_path / "typescript_repo"
    (repo / "src").mkdir(parents=True)
    (repo / "package.json").write_text(
        json.dumps(
            {
                "name": "sample",
                "packageManager": "pnpm@9.1.0",
                "scripts": {"type": "tsc --noEmit", "test": "jest"},
            }
        ),
        encoding="utf-8",
    )
    (repo / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\n", encoding="utf-8")
    (repo / "tsconfig.json").write_text("{}\n", encoding="utf-8")
    (repo / "src" / "form.ts").write_text(
        "export const form = 1;\n", encoding="utf-8"
    )
    (repo / "src" / "state.ts").write_text(
        "export const state = 1;\n", encoding="utf-8"
    )
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")

    plan = select_public_surface(
        repo,
        {
            "agent1": "Change form in src/form.ts",
            "agent2": "Change state in src/state.ts",
        },
        max_files=32,
        max_total_bytes=1024 * 1024,
        source_policy="feature_independent",
    )

    assert plan.public_test_command == ("pnpm", "run", "test")


def test_typescript_surface_uses_typecheck_when_no_test_exists(tmp_path: Path):
    repo = tmp_path / "typescript_typecheck_repo"
    (repo / "src").mkdir(parents=True)
    (repo / "package.json").write_text(
        json.dumps(
            {
                "name": "sample",
                "packageManager": "npm@10.0.0",
                "scripts": {"typecheck": "tsc --noEmit"},
            }
        ),
        encoding="utf-8",
    )
    (repo / "tsconfig.json").write_text("{}\n", encoding="utf-8")
    (repo / "src" / "form.ts").write_text(
        "export const form = 1;\n", encoding="utf-8"
    )
    (repo / "src" / "state.ts").write_text(
        "export const state = 1;\n", encoding="utf-8"
    )
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")

    plan = select_public_surface(
        repo,
        {
            "agent1": "Change form in src/form.ts",
            "agent2": "Change state in src/state.ts",
        },
        max_files=32,
        max_total_bytes=1024 * 1024,
        source_policy="feature_independent",
    )
    pack = materialize_public_pack(
        repo, tmp_path / "pack_typecheck", {
            "agent1": "Change form in src/form.ts",
            "agent2": "Change state in src/state.ts",
        }, plan, project_id="typecheck_only"
    )
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))

    assert plan.public_test_command == ("npm", "run", "typecheck")
    assert manifest["cooperbench"]["public_validation_strength"] == "typecheck_only"
    assert manifest["entrypoints"]["smoke"]["scope"] == "selected_public_validation"


def test_materialized_pack_has_two_public_issues_and_no_private_assets(tmp_path: Path):
    repo = _repo(tmp_path)
    tasks = {
        "agent1": "Change WidgetRegistry in src/widget.py",
        "agent2": "Change CacheStore in src/cache.py",
    }
    plan = select_public_surface(
        repo, tasks, max_files=32, max_total_bytes=1024 * 1024
    )
    pack = materialize_public_pack(
        repo, tmp_path / "pack", tasks, plan, project_id="cooper_test"
    )

    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    issue_files = sorted((pack / "issues" / "public").glob("*.json"))
    assert len(issue_files) == 2
    assert "private_evaluator_only" not in manifest
    assert manifest["cooperbench"]["evaluator_assets_in_pack"] is False
    assert manifest["cooperbench"]["internal_smoke_is_functional_evaluator"] is False
    assert manifest["entrypoints"]["smoke"]["command"] == list(
        plan.public_test_command
    )
    assert manifest["public_tests"]["command"] == list(plan.public_test_command)
    assert manifest["public_tests"]["files"] == ["tests/test_widget.py"]
    assert manifest["cooperbench"]["public_validation_strength"] == "public_regression"
    assert not (pack / "tests" / "hidden").exists()
    assert not (pack / "reference_repo").exists()


def test_materialized_pack_builds_as_nonexplicit_two_member_b3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo = _repo(tmp_path)
    tasks = {
        "agent1": "Change WidgetRegistry in src/widget.py",
        "agent2": "Change CacheStore in src/cache.py",
    }
    plan = select_public_surface(
        repo, tasks, max_files=32, max_total_bytes=1024 * 1024
    )
    pack = materialize_public_pack(
        repo, tmp_path / "pack", tasks, plan, project_id="cooper_test"
    )
    for key in (
        "ORG_EXPERIMENT_CONDITION",
        "ORG_MECHANISM_ABLATIONS",
        "ORG_TRANSFER_ARM",
        "ORG_TRANSFER_SOURCE_BUNDLE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ORG_EXECUTION_PROFILE", "native")
    monkeypatch.setenv("ORG_PRODUCT_SUBSTRATE", "oss_time_machine")
    monkeypatch.setenv("ORG_OSS_DATASET", str(pack.resolve()))
    monkeypatch.setenv("ORG_OSS_REPOSITORY_ID", "cooper_test")
    monkeypatch.setenv("ORG_OSS_MODE", "dev")
    monkeypatch.setenv("ORG_OSS_CONTROL", "none")
    monkeypatch.setenv("ORG_OSS_ANONYMIZE", "0")

    from environments.org_env.runtime_adapter.live import OrgInspectorSession

    session = OrgInspectorSession(
        seed=6101,
        n_agents=2,
        member_ids=("sean", "calvin"),
        load_llm=False,
        experiment_condition="b3_full_sociogenesis",
        noncanonical_roster_variant="cooperbench_two_person_b3",
        max_frames=1,
    )

    assert session.world.experiment_condition == "b3_full_sociogenesis"
    assert session.world.experiment_condition_explicit is True
    assert tuple(session.world.agents) == ("sean", "calvin")
    assert set(session.world.tasks) == {
        "task_oss_cooper_feature_1",
        "task_oss_cooper_feature_2",
    }
    assert session.world.condition_spec.institutionalization_enabled is True


def test_surface_fails_instead_of_falling_back_to_whole_repository(tmp_path: Path):
    repo = _repo(tmp_path)

    with pytest.raises(SurfaceBuildError, match="no_task_relevant_editable_path"):
        select_public_surface(
            repo,
            {
                "agent1": "zzzzunrelatedalpha zzzzunrelatedbeta",
                "agent2": "qqqqunrelatedgamma qqqqunrelateddelta",
            },
            max_files=32,
            max_total_bytes=1024 * 1024,
        )


def test_test_module_slots_cannot_be_consumed_by_initializers_and_helpers(tmp_path: Path):
    repo = _repo(tmp_path)
    (repo / "src" / "__init__.py").write_text("from .cache import CacheStore\n", encoding="utf-8")
    for index in range(12):
        support = repo / "tests" / f"support{index}"
        support.mkdir()
        for name in ("__init__.py", "utils.py"):
            (support / name).write_text("# cache namespace configure_cache\n", encoding="utf-8")
    (repo / "tests" / "test_cache.py").write_text(
        "def test_cache():\n    assert True\n", encoding="utf-8")
    (repo / "tests" / "test_unrelated_integration.py").write_text(
        "# namespace cache configure_cache\ndef test_other():\n    assert True\n", encoding="utf-8")
    _git(repo, "add", ".")
    plan = select_public_surface(repo, {
        "agent1": "Add namespace caching in src/cache.py and src/__init__.py",
        "agent2": "Add compression in src/cache.py and src/__init__.py",
    }, max_files=64, max_total_bytes=1024 * 1024)
    assert plan.public_test_paths == ("tests/test_cache.py",)
    assert all(Path(path).name.startswith("test_") for path in plan.public_test_paths)
