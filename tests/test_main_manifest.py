import json
from pathlib import Path

import pytest

from relic import manifest as manifest_module
from relic.cell_spec import compile_cell_spec
from relic.manifest import build_main_manifest, recommended_parallelism, write_manifest


@pytest.mark.unit
def test_single_model_manifest_is_the_canonical_120_cells(tmp_path: Path) -> None:
    manifest = build_main_manifest(
        model="gpt-5.6-terra",
        output_root=tmp_path,
        max_parallel=1,
    )

    assert manifest["total_cells"] == 120
    assert {cell["arm"] for cell in manifest["cells"]} == {"B0", "B1", "B2", "B3"}
    assert {cell["workload"] for cell in manifest["cells"]} == {
        f"W{number:02d}" for number in range(1, 11)
    }
    assert {cell["seed"] for cell in manifest["cells"]} == {1401, 2711, 4013}
    assert len({cell["cell_id"] for cell in manifest["cells"]}) == 120
    assert manifest["workloads"]["W01"]["scoring_units"] == {
        "seeded": 5,
        "exposed": 35,
        "held_out": 0,
        "contracts": 5,
        "cases": 35,
    }
    assert manifest["workloads"]["W10"]["scoring_units"] == {
        "seeded": 12,
        "exposed": 12,
        "held_out": 4,
        "contracts": 16,
        "cases": 16,
    }
    assert manifest["arms"]["B0"]["private_memory_and_appraisal"] is True
    assert manifest["arms"]["B0"]["shared_workspace_channels_meetings"] is False
    assert manifest["arms"]["B3"]["organization_reflection_to_institution_path"] is True


@pytest.mark.unit
def test_manifest_paths_are_the_canonical_cellspec_layout(tmp_path: Path) -> None:
    """The plan must point at exactly the directory a worker will freeze."""
    manifest = build_main_manifest(
        model="gpt-5.6-terra",
        output_root=tmp_path,
        max_parallel=1,
    )

    for cell in manifest["cells"]:
        spec = compile_cell_spec(
            model=cell["model"],
            workload=cell["workload"],
            arm=cell["arm"],
            seed=cell["seed"],
            output_root=tmp_path,
            verify_pack=False,
        )
        assert Path(cell["output_path"]) == spec.cell_dir
        assert Path(cell["output_path"]).is_relative_to(tmp_path)


@pytest.mark.unit
def test_write_manifest_keeps_previous_document_if_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "run_manifest.json"
    previous = {"schema_version": "previous", "cells": []}
    destination.write_text(json.dumps(previous), encoding="utf-8")

    def replace_fails(_source: Path, _destination: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(manifest_module.os, "replace", replace_fails)
    with pytest.raises(OSError, match="simulated replace failure"):
        write_manifest({"schema_version": "next", "cells": ["new"]}, destination)

    assert json.loads(destination.read_text(encoding="utf-8")) == previous
    assert list(tmp_path.glob(".run_manifest.json.*.tmp")) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("memory_gib", "expected"),
    [(16, 1), (31.9, 1), (32, 2), (64, 4), (100, 8), (128, 8)],
)
def test_memory_policy(memory_gib: float, expected: int) -> None:
    assert recommended_parallelism(memory_gib) == expected


@pytest.mark.unit
def test_unknown_model_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown main-study model"):
        build_main_manifest(model="old-placeholder", output_root=tmp_path)
