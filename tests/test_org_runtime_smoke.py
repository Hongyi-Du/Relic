from __future__ import annotations

from environments.org_env.backend.simulation.world import OrgWorld
from environments.org_env.config.scenarios import oss_time_machine_formal
from environments.org_env.product.substrates.eval_assets import (
    validate_formal_oss_world,
)
from environments.org_env.product.materialize import run_product_smoke
from environments.org_env.runtime_adapter.checkpoint import (
    load_world_checkpoint,
    save_world_checkpoint,
)


def _mini_blobstore_world(*, seed: int) -> OrgWorld:
    scenario = oss_time_machine_formal(
        seed=seed,
        dataset_id="mini_blobstore_v1",
    )
    return OrgWorld(scenario).build()


def test_formal_world_builds_and_advances_one_mock_tick() -> None:
    world = _mini_blobstore_world(seed=7)

    assert world.datasets == {}
    assert world.benchmarks == {}
    assert world.repo_system.repo.name == "mini-blobstore"
    assert world.repo_system.repo.modules
    assert "research_loop" not in world.repo_system.repo.modules

    result = world.step()

    assert world.world_tick == 1
    assert result["current_tick"] == 1
    assert world.__dict__["_oss_hidden_qualification"]["formal_ready"] is True


def test_formal_checkpoint_requalifies_and_resumes(tmp_path) -> None:
    checkpoint = tmp_path / "world.pkl"
    world = _mini_blobstore_world(seed=11)
    world.step()
    save_world_checkpoint(world, str(checkpoint))

    resumed, metadata = load_world_checkpoint(
        str(checkpoint),
        expected_seed=11,
    )
    qualification = validate_formal_oss_world(resumed, require_formal=True)
    resumed.step()

    assert metadata["tick"] == 1
    assert qualification is not None
    assert qualification["formal_ready"] is True
    assert resumed.world_tick == 2


def test_product_smoke_requires_a_manifest_command(tmp_path) -> None:
    result = run_product_smoke(str(tmp_path), command=None)

    assert result == {
        "ok": False,
        "error": "workload manifest declares no smoke command",
    }
