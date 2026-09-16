from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path

from .manifests import ManifestStore
from .objects import ObjectStore


class UploadConflictError(RuntimeError):
    pass


class UploadStore:
    def __init__(
        self,
        root: str | os.PathLike[str],
        objects: ObjectStore,
        manifests: ManifestStore,
    ):
        self.root = Path(root)
        self.directory = self.root / "uploads"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.objects = objects
        self.manifests = manifests

    def _path(self, upload_id: str) -> Path:
        if len(upload_id) != 32 or any(ch not in "0123456789abcdef" for ch in upload_id):
            raise ValueError("invalid upload id")
        return self.directory / upload_id / "state.json"

    def _load(self, upload_id: str) -> dict:
        raw = json.loads(self._path(upload_id).read_text(encoding="utf-8"))
        if raw.get("version") != 1 or raw.get("upload_id") != upload_id:
            raise ValueError("invalid upload state")
        return raw

    @staticmethod
    def _publish(path: Path, state: dict) -> None:
        encoded = (
            json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
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

    def begin(self, target: str, metadata: dict | None = None) -> str:
        self.manifests.validate_name(target)
        upload_id = uuid.uuid4().hex
        state = {
            "completed": None,
            "metadata": dict(metadata or {}),
            "parts": {},
            "target": target,
            "upload_id": upload_id,
            "version": 1,
        }
        self._publish(self._path(upload_id), state)
        return upload_id

    def upload_part(self, upload_id: str, number: int, data: bytes) -> str:
        if number <= 0:
            raise ValueError("part number must be positive")
        state = self._load(upload_id)
        if state["completed"] is not None:
            raise UploadConflictError("upload already completed")
        object_id = self.objects.put_bytes(data)
        key = str(number)
        previous = state["parts"].get(key)
        if previous is not None and previous != object_id:
            raise UploadConflictError("part bytes changed")
        state["parts"][key] = object_id
        self._publish(self._path(upload_id), state)
        return object_id

    def complete(self, upload_id: str, part_count: int) -> dict:
        if part_count <= 0:
            raise ValueError("part_count must be positive")
        state = self._load(upload_id)
        if state["completed"] is not None:
            if state["completed"]["part_count"] != part_count:
                raise UploadConflictError("part count changed")
            return state["completed"]
        expected = [str(number) for number in range(1, part_count + 1)]
        if sorted(state["parts"], key=int) != expected:
            raise ValueError("parts are not contiguous")
        fd, temporary = tempfile.mkstemp(prefix=".assembled-", dir=self.directory)
        try:
            with os.fdopen(fd, "wb") as output:
                for number in expected:
                    output.write(self.objects.get(state["parts"][number]))
                output.flush()
                os.fsync(output.fileno())
            with open(temporary, "rb") as source:
                object_id = self.objects.put_stream(source)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self.manifests.create(state["target"], [object_id], state["metadata"])
        state["completed"] = {"object_id": object_id, "part_count": part_count}
        self._publish(self._path(upload_id), state)
        return state["completed"]

    def incomplete_object_ids(self) -> set[str]:
        result = set()
        for path in sorted(self.directory.glob("*/state.json")):
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("completed") is None:
                result.update(state.get("parts", {}).values())
        return result
