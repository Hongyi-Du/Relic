from pathlib import Path

import pytest

from relic.manifest import build_main_manifest, recommended_parallelism


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

