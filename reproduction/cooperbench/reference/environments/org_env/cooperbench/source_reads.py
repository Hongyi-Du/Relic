"""Explicit, immutable source access for actor-private prompt consumers.

This is a read projection, not a replacement world or artifact. Only canonical
path metadata comes from the registry; source bytes and visibility come from one
fixed actor desk. In particular, an empty visible file is not a missing file.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from environments.org_env.cooperbench import source_views as sv


@dataclass(frozen=True)
class ActorSourceReadView:
    actor_id: str
    snapshot: Any
    catalog: Mapping[str, str]
    files: Mapping[str, str]

    def text(self, artifact: Any, *, creates_file: bool = False) -> str:
        artifact_id = str(getattr(artifact, "artifact_id", "") or "")
        path = str(getattr(artifact, "linked_file_path", "") or "")
        if creates_file:
            if path in self.files:
                raise sv.SourceViewError("actor_create_path_exists")
            return ""
        if self.catalog.get(artifact_id) != path or path not in self.files:
            raise sv.SourceViewError("actor_source_target_not_visible")
        return self.files[path]

    def assert_current(self, world: Any) -> None:
        sv.assert_actor_desk_current(world, self.actor_id, self.snapshot)


def actor_source_read_view(world: Any, actor_id: str | None, *,
                           source_snapshot: Any = None) -> ActorSourceReadView | None:
    """Legacy returns None; opted-in missing identity/state always fails closed."""
    if not sv.actor_desks_enabled(world):
        return None
    if not isinstance(actor_id, str) or not actor_id:
        raise sv.SourceViewError("actor_source_reader_missing")
    if source_snapshot is None:
        # actor_desk_snapshot has just reconstructed and saved the exact current
        # desk. Rebuilding it immediately in assert_actor_desk_current used to
        # hash a whole monorepo twice before a single source read.
        snapshot = sv.actor_desk_snapshot(world, actor_id)
    else:
        snapshot = source_snapshot
        sv.assert_actor_desk_current(world, actor_id, snapshot)
    files = dict(snapshot._file_entries)
    catalog = {
        artifact_id: path
        for artifact_id, path in snapshot._artifact_entries
        if path in files
    }
    return ActorSourceReadView(
        actor_id,
        snapshot,
        MappingProxyType(catalog),
        MappingProxyType(files),
    )


def visible_source_artifacts(world: Any, view: ActorSourceReadView | None):
    """Yield original metadata objects only for files present in this view.

    Consumers must still use view.text(), never an object's mutable source or
    derived summary/gap/history fields, when the private stage is active.
    """
    for artifact_id, artifact in (getattr(world, "product_artifacts", {}) or {}).items():
        if view is None or artifact_id in view.catalog:
            yield artifact


__all__ = ["ActorSourceReadView", "actor_source_read_view", "visible_source_artifacts"]
