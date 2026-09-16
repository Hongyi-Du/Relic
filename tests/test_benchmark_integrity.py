import pytest

from relic.benchmark import load_benchmark_manifest, verify_benchmark


@pytest.mark.release
def test_relic_main_v1_contains_exactly_ten_verified_workloads() -> None:
    manifest = load_benchmark_manifest()

    assert [row["id"] for row in manifest["workloads"]] == [
        f"w{number:02d}" for number in range(1, 11)
    ]
    assert verify_benchmark() == []

