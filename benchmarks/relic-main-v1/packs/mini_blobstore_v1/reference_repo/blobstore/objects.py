from __future__ import annotations

import hashlib
import io
import os
import re
import tempfile
from pathlib import Path
from typing import BinaryIO, Iterator


OBJECT_ID = re.compile(r"^[0-9a-f]{64}$")


class InvalidObjectId(ValueError):
    pass


class CorruptObjectError(RuntimeError):
    pass


class ObjectStore:
    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.objects.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_id(object_id: str) -> None:
        if not OBJECT_ID.fullmatch(object_id):
            raise InvalidObjectId(object_id)

    def path_for(self, object_id: str) -> Path:
        self.validate_id(object_id)
        return self.objects / object_id[:2] / object_id[2:]

    def put_bytes(self, data: bytes) -> str:
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        return self.put_stream(io.BytesIO(data))

    def put_stream(self, source: BinaryIO) -> str:
        self.objects.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".incoming-", dir=self.objects)
        digest = hashlib.sha256()
        try:
            with os.fdopen(fd, "wb") as target:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    if not isinstance(block, bytes):
                        raise TypeError("stream must return bytes")
                    digest.update(block)
                    target.write(block)
                target.flush()
                os.fsync(target.fileno())
            object_id = digest.hexdigest()
            destination = self.path_for(object_id)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if not self.verify(object_id):
                    raise CorruptObjectError(object_id)
                return object_id
            os.replace(temporary, destination)
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            return object_id
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def get(self, object_id: str) -> bytes:
        path = self.path_for(object_id)
        if not path.is_file():
            raise FileNotFoundError(object_id)
        return path.read_bytes()

    def exists(self, object_id: str) -> bool:
        return self.path_for(object_id).is_file()

    def verify(self, object_id: str) -> bool:
        path = self.path_for(object_id)
        if not path.is_file():
            return False
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest() == object_id

    def iter_ids(self) -> Iterator[str]:
        found = []
        for directory in self.objects.iterdir():
            if not directory.is_dir() or not re.fullmatch(r"[0-9a-f]{2}", directory.name):
                continue
            for path in directory.iterdir():
                candidate = directory.name + path.name
                if path.is_file() and OBJECT_ID.fullmatch(candidate):
                    found.append(candidate)
        yield from sorted(found)
