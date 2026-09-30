"""Verify the public text inventory and bind runtime assets before any smoke.

This initialization seam is deliberately separate from late worker desk setup:
the native substrate can export a product during its own construction.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .runtime_assets import (
    RuntimeAssetError, assert_no_runtime_asset_conflicts,
    initialize_runtime_assets, read_runtime_assets,
)


def verify_public_text_inventory(metadata: Mapping, source_files: Mapping[str, str]) -> None:
    """Require exact path and original UTF-8 bytes, both before and after seeding."""
    expected = metadata.get("public_text_files")
    if not isinstance(expected, list) or not expected:
        raise RuntimeAssetError("public_baseline_text_inventory_missing")
    actual = [
        {"path": path, "size_bytes": len(content.encode("utf-8")),
         "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}
        for path, content in sorted(source_files.items())
    ]
    if actual != expected:
        raise RuntimeAssetError("public_baseline_text_inventory_mismatch")


def initialize_public_baseline(
    world: Any, pack_dir: str | Path, metadata: Mapping, source_files: Mapping[str, str],
) -> None:
    """Fail closed on omission, normalization or tampering; no source fallback."""
    if metadata.get("source_policy") != "feature_independent":
        raise RuntimeAssetError("feature_independent_baseline_required")
    if metadata.get("text_encoding_policy") != "utf8_preserve_newlines":
        raise RuntimeAssetError("public_baseline_text_encoding_policy_mismatch")
    verify_public_text_inventory(metadata, source_files)
    asset_metadata = metadata.get("runtime_assets")
    if not isinstance(asset_metadata, dict) or not asset_metadata.get("asset_set_digest"):
        raise RuntimeAssetError("public_runtime_assets_manifest_missing")
    assets = read_runtime_assets(pack_dir, expected_digest=asset_metadata["asset_set_digest"])
    if assets.manifest() != asset_metadata:
        raise RuntimeAssetError("public_runtime_assets_manifest_mismatch")
    assert_no_runtime_asset_conflicts(assets, source_files)
    if set(assets.source_modes) != set(source_files):
        raise RuntimeAssetError("public_baseline_source_modes_inventory_mismatch")
    initialize_runtime_assets(world, assets)
    world.__dict__["_cooperbench_source_policy"] = "feature_independent"
    world.__dict__["_cooperbench_public_baseline_digest"] = metadata.get("surface_digest")
