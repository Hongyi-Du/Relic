from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator

from .objects import ObjectStore


SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")


class ManifestConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class Manifest:
    name: str
    objects: tuple[str, ...]
    metadata: dict


class ManifestStore:
    def __init__(self, root: str | os.PathLike[str], objects: ObjectStore):
        self.root = Path(root)
        self.directory = self.root / "manifests"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.objects = objects

    @staticmethod
    def validate_name(name: str) -> None:
        path = PurePosixPath(name)
        if (
            not name
            or name.startswith("/")
            or "//" in name
            or any(part in {"", ".", ".."} or not SEGMENT.fullmatch(part) for part in path.parts)
        ):
            raise ValueError("invalid manifest name")

    def path_for(self, name: str) -> Path:
        self.validate_name(name)
        return self.directory.joinpath(*PurePosixPath(name).parts).with_suffix(".json")

    @staticmethod
    def _encoded(name: str, object_ids: list[str], metadata: dict) -> bytes:
        payload = {
            "metadata": metadata,
            "name": name,
            "objects": object_ids,
            "version": 1,
        }
        return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()

    def create(
        self, name: str, object_ids: list[str], metadata: dict | None = None
    ) -> Manifest:
        self.validate_name(name)
        if not isinstance(metadata or {}, dict):
            raise TypeError("metadata must be a dictionary")
        for object_id in object_ids:
            if not self.objects.exists(object_id) or not self.objects.verify(object_id):
                raise ValueError(f"unavailable object: {object_id}")
        encoded = self._encoded(name, object_ids, dict(metadata or {}))
        path = self.path_for(name)
        if path.exists():
            if path.read_bytes() == encoded:
                return self.get(name)
            raise ManifestConflictError(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return self.get(name)

    def get(self, name: str) -> Manifest:
        raw = json.loads(self.path_for(name).read_text(encoding="utf-8"))
        if raw.get("version") != 1 or raw.get("name") != name:
            raise ValueError("invalid manifest")
        object_ids = tuple(raw["objects"])
        for object_id in object_ids:
            self.objects.validate_id(object_id)
        if not isinstance(raw["metadata"], dict):
            raise ValueError("invalid metadata")
        return Manifest(name, object_ids, raw["metadata"])

    def delete(self, name: str) -> bool:
        path = self.path_for(name)
        if not path.exists():
            return False
        path.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return True

    def iter_names(self) -> Iterator[str]:
        names = []
        for path in self.directory.rglob("*.json"):
            relative = path.relative_to(self.directory).with_suffix("")
            names.append(relative.as_posix())
        yield from sorted(names)

    def referenced_ids(self) -> set[str]:
        return {
            object_id
            for name in self.iter_names()
            for object_id in self.get(name).objects
        }
