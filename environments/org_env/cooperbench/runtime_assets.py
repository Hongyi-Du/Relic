"""Immutable public runtime resources, separate from editable/prompt source.

Callers select public paths; this module never discovers benchmark assets. Raw
bytes live only in the explicit pack/checkpoint payload and export. Ordinary
world/snapshot consumers receive a validated frozen set or metadata receipt.
"""
from __future__ import annotations

import base64
from bisect import bisect_left
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any, Iterable, Mapping

from environments.org_env.product.repo_paths import normalize_repo_relative_path


SCHEMA = "cooperbench_public_runtime_assets_v1"
MANIFEST_FILENAME = "cooperbench_runtime_assets.json"
ASSET_DIRECTORY = "cooperbench_runtime_assets"
CONTAINER_ASSET_ROOT = "/cooperbench-runtime-assets"
DEFAULT_MAX_FILES = 32_768
DEFAULT_MAX_TOTAL_BYTES = 1024 * 1024 ** 2
DEFAULT_MAX_FILE_BYTES = 16 * 1024 ** 2
INLINE_MAX_TOTAL_BYTES = 32 * 1024 ** 2
MANIFEST_MAX_BYTES = 64 * 1024 ** 2
_STORE = "_cooperbench_runtime_assets"
_PRIVATE_PARTS = frozenset({"hidden", "private", "gold", "solution", "solutions"})


class RuntimeAssetError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__("cooperbench_runtime_assets:" + code)


def _path(raw: Any) -> str:
    try:
        path = normalize_repo_relative_path(raw)
    except ValueError as error:
        raise RuntimeAssetError("unsafe_path") from error
    if _PRIVATE_PARTS.intersection(part.casefold() for part in path.split("/")):
        raise RuntimeAssetError("private_asset_path")
    return path


def _paths(raw: Iterable[str]) -> tuple[str, ...]:
    if isinstance(raw, (str, bytes)):
        raise RuntimeAssetError("path_list_required")
    paths = tuple(_path(item) for item in raw)
    seen = set()
    for path in sorted(paths, key=str.casefold):
        folded = path.casefold()
        if folded in seen:
            raise RuntimeAssetError("duplicate_or_aliased_path:" + path)
        if any("/".join(folded.split("/")[:i]) in seen for i in range(1, len(folded.split("/")))):
            raise RuntimeAssetError("file_directory_collision:" + path)
        seen.add(folded)
    return tuple(sorted(paths))


def _limits(max_files: int, max_total_bytes: int, max_file_bytes: int) -> None:
    if any(type(value) is not int or value < 0 for value in (max_files, max_total_bytes, max_file_bytes)):
        raise RuntimeAssetError("invalid_budget")


def _mode(raw: Any) -> int:
    if type(raw) is not int or raw not in (0o644, 0o755):
        raise RuntimeAssetError("invalid_normalized_mode")
    return raw


def _mode_pairs(raw: Mapping[str, int] | None) -> tuple[tuple[str, int], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, Mapping):
        raise RuntimeAssetError("mode_mapping_required")
    paths = _paths(raw)
    values = {_path(path): _mode(mode) for path, mode in raw.items()}
    return tuple((path, values[path]) for path in paths)


def _metadata(entries: tuple[tuple[str, bytes], ...], asset_modes: tuple[tuple[str, int], ...],
              source_modes: tuple[tuple[str, int], ...]) -> dict:
    modes = dict(asset_modes)
    files = [{"path": path, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "mode": modes[path]}
             for path, raw in entries]
    body = {"schema_version": SCHEMA, "file_count": len(files),
            "total_bytes": sum(row["size_bytes"] for row in files), "files": files,
            "source_modes": dict(source_modes)}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    return {**body, "asset_set_digest": digest}


def _metadata_from_rows(
    rows: Iterable[tuple[str, int, str, int]],
    source_modes: tuple[tuple[str, int], ...],
) -> dict:
    files = [
        {"path": path, "size_bytes": size, "sha256": digest, "mode": mode}
        for path, size, digest, mode in rows
    ]
    body = {
        "schema_version": SCHEMA,
        "file_count": len(files),
        "total_bytes": sum(row["size_bytes"] for row in files),
        "files": files,
        "source_modes": dict(source_modes),
    }
    digest = hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()
    return {**body, "asset_set_digest": digest}


def _stream_digest(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    with os.fdopen(descriptor, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def runtime_asset_backing_root(assets: "RuntimeAssetSet") -> Path | None:
    """Return the already-verified large-asset store without exposing bytes.

    ``read_runtime_assets`` hashes every body at the persistence boundary. The
    returned directory is suitable only for a read-only sandbox mount; callers
    must never give the candidate a writable view of it.
    """

    _verify_set(assets)
    if assets._backing_root is None:
        return None
    root = _root(assets._backing_root)
    for path, size, _digest, mode in assets._rows:
        target = _file(root, path)
        if target.stat().st_size != size:
            raise RuntimeAssetError("asset_content_changed:" + path)
        _verify_mode(target, mode, path)
    return root


def link_runtime_assets_for_container(
    assets: "RuntimeAssetSet",
    dest: str | Path,
    *,
    container_root: str = CONTAINER_ASSET_ROOT,
) -> dict:
    """Create cheap links whose targets are supplied by a read-only mount.

    The links are deliberately broken on the host and resolve only inside the
    isolated container where ``runtime_asset_backing_root`` is mounted at the
    fixed ``container_root``. This avoids copying hundreds of megabytes for
    every public-test/review head while preventing candidate code from writing
    back into the frozen asset pack.
    """

    backing = runtime_asset_backing_root(assets)
    if backing is None:
        raise RuntimeAssetError("directory_backed_assets_required")
    if (
        os.name != "posix"
        or not re.fullmatch(r"/[A-Za-z0-9_./-]+", container_root)
        or ".." in PurePosixPath(container_root).parts
    ):
        raise RuntimeAssetError("invalid_container_asset_root")
    root = _root(dest, create=True)
    for path in assets.paths:
        target = root.joinpath(*path.split("/"))
        for parent in reversed(target.parents):
            if parent == root or root in parent.parents:
                if _unsafe_link(parent):
                    raise RuntimeAssetError("asset_parent_symlink_or_junction:" + path)
                if parent.exists() and not parent.is_dir():
                    raise RuntimeAssetError("asset_parent_not_directory:" + path)
        target.parent.mkdir(parents=True, exist_ok=True)
        link_target = str(PurePosixPath(container_root).joinpath(*path.split("/")))
        if os.path.lexists(target):
            if target.is_symlink() and os.readlink(target) == link_target:
                continue
            raise RuntimeAssetError("asset_link_destination_exists:" + path)
        target.symlink_to(link_target)
        if not target.is_symlink() or os.readlink(target) != link_target:
            raise RuntimeAssetError("asset_link_creation_failed:" + path)
    return {**assets.manifest(), "verified": True}


@dataclass(frozen=True, init=False)
class RuntimeAssetSet:
    """Bytes/tuples are immutable; construction always computes the identity."""
    _entries: tuple[tuple[str, bytes], ...]
    _rows: tuple[tuple[str, int, str, int], ...]
    _asset_modes: tuple[tuple[str, int], ...]
    _source_modes: tuple[tuple[str, int], ...]
    _backing_root: str | None
    asset_set_digest: str
    total_bytes: int

    def __init__(self, entries: Iterable[tuple[str, bytes]] = (), *,
                 asset_modes: Mapping[str, int] | None = None,
                 source_modes: Mapping[str, int] | None = None):
        entries = tuple(entries)
        if any(not isinstance(item, tuple) or len(item) != 2 or type(item[1]) is not bytes for item in entries):
            raise RuntimeAssetError("immutable_bytes_required")
        paths = _paths(path for path, _ in entries)
        by_path = {_path(path): raw for path, raw in entries}
        canonical = tuple((path, by_path[path]) for path in paths)
        supplied_modes = dict(_mode_pairs(asset_modes))
        if set(supplied_modes) - set(paths):
            raise RuntimeAssetError("asset_mode_paths_mismatch")
        normalized_modes = tuple((path, supplied_modes.get(path, 0o644)) for path in paths)
        source_mode_pairs = _mode_pairs(source_modes)
        # Source modes are only metadata; they must not introduce aliases or
        # turn an asset path into an editable source path.
        _paths((*paths, *(path for path, _ in source_mode_pairs)))
        metadata = _metadata(canonical, normalized_modes, source_mode_pairs)
        object.__setattr__(self, "_entries", canonical)
        object.__setattr__(self, "_rows", tuple((row["path"], row["size_bytes"], row["sha256"], row["mode"]) for row in metadata["files"]))
        object.__setattr__(self, "_asset_modes", normalized_modes)
        object.__setattr__(self, "_source_modes", source_mode_pairs)
        object.__setattr__(self, "_backing_root", None)
        object.__setattr__(self, "asset_set_digest", metadata["asset_set_digest"])
        object.__setattr__(self, "total_bytes", metadata["total_bytes"])

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(row[0] for row in self._rows)

    def manifest(self) -> dict:
        """Cheap detached metadata; deliberately excludes raw/base64 bytes."""
        return {"schema_version": SCHEMA, "asset_set_digest": self.asset_set_digest,
                "file_count": len(self._rows), "total_bytes": self.total_bytes,
                "files": [{"path": path, "size_bytes": size, "sha256": digest, "mode": mode}
                          for path, size, digest, mode in self._rows], "source_modes": dict(self._source_modes)}

    @property
    def source_modes(self) -> dict[str, int]:
        """Detached mode metadata only; no source bytes are stored here."""
        return dict(self._source_modes)

    def bytes_for(self, path: str) -> bytes:
        canonical = _path(path)
        if self._backing_root is not None:
            row = next((item for item in self._rows if item[0] == canonical), None)
            if row is None:
                raise RuntimeAssetError("asset_missing:" + canonical)
            return _read(Path(self._backing_root), canonical, row[1])
        for name, raw in self._entries:
            if name == canonical:
                return raw
        raise RuntimeAssetError("asset_missing:" + canonical)

    def to_payload(self) -> dict:
        _verify_set(self)
        if self._backing_root is not None:
            raise RuntimeAssetError("directory_backed_inline_payload_prohibited")
        return {**self.manifest(), "content_base64": {path: base64.b64encode(raw).decode("ascii") for path, raw in self._entries}}

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        relative_paths: Iterable[str],
        *,
        source_paths: Iterable[str] = (),
        max_files: int = DEFAULT_MAX_FILES,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    ) -> "RuntimeAssetSet":
        """Hash assets once while keeping large immutable bytes on disk."""

        _limits(max_files, max_total_bytes, max_file_bytes)
        paths, sources = _paths(relative_paths), _paths(source_paths)
        _paths((*paths, *sources))
        if len(paths) + len(sources) > max_files:
            raise RuntimeAssetError("file_budget_exceeded")
        directory = _root(root)
        rows: list[tuple[str, int, str, int]] = []
        total = 0
        for path in paths:
            target = _file(directory, path)
            info = target.lstat()
            size = int(info.st_size)
            if size > max_file_bytes:
                raise RuntimeAssetError("file_budget_exceeded:" + path)
            total += size
            if total > max_total_bytes:
                raise RuntimeAssetError("total_byte_budget_exceeded")
            rows.append(
                (path, size, _stream_digest(target), _normalized_file_mode(info, path))
            )
        source_modes = tuple(
            (path, _normalized_file_mode(_file(directory, path).lstat(), path))
            for path in sources
        )
        return cls._from_directory_manifest(directory, rows, source_modes)

    @classmethod
    def _from_directory_manifest(
        cls,
        root: str | Path,
        rows: Iterable[tuple[str, int, str, int]],
        source_modes: Iterable[tuple[str, int]],
        *,
        expected_digest: str | None = None,
    ) -> "RuntimeAssetSet":
        directory = _root(root)
        canonical_rows = tuple(sorted(tuple(row) for row in rows))
        paths = _paths(row[0] for row in canonical_rows)
        if tuple(row[0] for row in canonical_rows) != paths:
            raise RuntimeAssetError("manifest_entries_invalid")
        canonical_source_modes = _mode_pairs(dict(source_modes))
        _paths((*paths, *(path for path, _ in canonical_source_modes)))
        for path, size, digest, mode in canonical_rows:
            if type(size) is not int or size < 0 or not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
                raise RuntimeAssetError("manifest_entries_invalid")
            _mode(mode)
            target = _file(directory, path)
            if target.stat().st_size != size:
                raise RuntimeAssetError("asset_content_changed:" + path)
        metadata = _metadata_from_rows(canonical_rows, canonical_source_modes)
        if expected_digest is not None and metadata["asset_set_digest"] != expected_digest:
            raise RuntimeAssetError("asset_set_digest_mismatch")
        result = cls()
        object.__setattr__(result, "_rows", canonical_rows)
        object.__setattr__(result, "_asset_modes", tuple((row[0], row[3]) for row in canonical_rows))
        object.__setattr__(result, "_source_modes", canonical_source_modes)
        object.__setattr__(result, "_backing_root", str(directory))
        object.__setattr__(result, "asset_set_digest", metadata["asset_set_digest"])
        object.__setattr__(result, "total_bytes", metadata["total_bytes"])
        return result

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], *, expected_digest: str | None = None,
                     max_files: int = DEFAULT_MAX_FILES, max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
                     max_file_bytes: int = DEFAULT_MAX_FILE_BYTES) -> "RuntimeAssetSet":
        _limits(max_files, max_total_bytes, max_file_bytes)
        keys = {"schema_version", "asset_set_digest", "file_count", "total_bytes", "files", "content_base64", "source_modes"}
        if not isinstance(payload, Mapping) or set(payload) != keys or payload.get("schema_version") != SCHEMA:
            raise RuntimeAssetError("manifest_schema_invalid")
        rows, blobs = payload.get("files"), payload.get("content_base64")
        if not isinstance(rows, list) or not isinstance(blobs, dict):
            raise RuntimeAssetError("manifest_entries_invalid")
        if (type(payload["file_count"]) is not int or type(payload["total_bytes"]) is not int
                or not isinstance(payload["asset_set_digest"], str)):
            raise RuntimeAssetError("manifest_identity_invalid")
        source_modes = _mode_pairs(payload.get("source_modes"))
        if len(rows) + len(source_modes) > max_files:
            raise RuntimeAssetError("file_budget_exceeded")
        if any(not isinstance(row, dict) or set(row) != {"path", "size_bytes", "sha256", "mode"} for row in rows):
            raise RuntimeAssetError("manifest_entries_invalid")
        paths = _paths(row["path"] for row in rows)
        if set(blobs) != set(paths):
            raise RuntimeAssetError("manifest_blob_paths_mismatch")
        entries, total = [], 0
        for row in rows:
            path, size = _path(row["path"]), row["size_bytes"]
            if type(size) is not int or size < 0 or size > max_file_bytes:
                raise RuntimeAssetError("file_budget_exceeded:" + path)
            total += size
            if total > max_total_bytes:
                raise RuntimeAssetError("total_byte_budget_exceeded")
            encoded = blobs[path]
            if not isinstance(encoded, str) or len(encoded) != 4 * ((size + 2) // 3):
                raise RuntimeAssetError("encoded_size_mismatch:" + path)
            try:
                raw = base64.b64decode(encoded, validate=True)
            except (ValueError, UnicodeError) as error:
                raise RuntimeAssetError("invalid_base64:" + path) from error
            if len(raw) != size or hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise RuntimeAssetError("asset_content_changed:" + path)
            entries.append((path, raw))
        assets = cls(entries, asset_modes={_path(row["path"]): row["mode"] for row in rows}, source_modes=dict(source_modes))
        if assets.manifest() != {key: payload[key] for key in keys - {"content_base64"}}:
            raise RuntimeAssetError("manifest_identity_mismatch")
        _expected(assets, expected_digest)
        return assets

    def __reduce__(self):
        # Checkpoint restoration revalidates immutable data once; ordinary
        # reads need not re-hash a large resource bundle every tick.
        if self._backing_root is not None:
            return (
                _restore_directory_asset_set,
                (self._backing_root, self._rows, self._source_modes, self.asset_set_digest),
            )
        return (_restore_asset_set, (self._entries, self._asset_modes, self._source_modes, self.asset_set_digest))


def _restore_asset_set(entries, asset_modes, source_modes, digest):
    assets = RuntimeAssetSet(entries, asset_modes=dict(asset_modes), source_modes=dict(source_modes))
    _expected(assets, digest)
    return assets


def _restore_directory_asset_set(root, rows, source_modes, digest):
    return RuntimeAssetSet._from_directory_manifest(
        root, rows, source_modes, expected_digest=digest
    )


def _verify_set(assets: RuntimeAssetSet) -> None:
    if type(assets) is not RuntimeAssetSet:
        raise RuntimeAssetError("validated_asset_set_required")
    if assets._backing_root is not None:
        metadata = _metadata_from_rows(assets._rows, assets._source_modes)
        if (
            metadata != assets.manifest()
            or not Path(assets._backing_root).is_dir()
        ):
            raise RuntimeAssetError("asset_set_modified")
        return
    if RuntimeAssetSet(assets._entries, asset_modes=dict(assets._asset_modes), source_modes=dict(assets._source_modes)).manifest() != assets.manifest():
        raise RuntimeAssetError("asset_set_modified")


def _expected(assets: RuntimeAssetSet, expected_digest: str | None) -> None:
    if expected_digest is not None and assets.asset_set_digest != expected_digest:
        raise RuntimeAssetError("asset_set_digest_mismatch")


def _unsafe_link(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def _normalized_file_mode(info: os.stat_result, path: str) -> int:
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX):
        raise RuntimeAssetError("unsafe_file_mode:" + path)
    return 0o755 if info.st_mode & 0o111 else 0o644


def _verify_mode(target: Path, expected: int, path: str) -> None:
    info = target.lstat()
    normalized = _normalized_file_mode(info, path)
    # NTFS does not expose POSIX execute bits. Never claim an executable was
    # preserved there; native Linux export is required for such assets.
    if os.name == "nt" and expected == 0o755:
        raise RuntimeAssetError("executable_mode_unsupported_on_host:" + path)
    if normalized != expected or (os.name != "nt" and stat.S_IMODE(info.st_mode) != expected):
        raise RuntimeAssetError("exported_file_mode_changed:" + path)


def _root(raw: str | Path, *, create: bool = False) -> Path:
    root = Path(os.path.abspath(raw))
    for part in (*reversed(root.parents), root):
        if _unsafe_link(part):
            raise RuntimeAssetError("root_symlink_or_junction")
        if part.exists() and not part.is_dir():
            raise RuntimeAssetError("root_not_directory")
    if create:
        root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise RuntimeAssetError("root_missing")
    return root


def _file(root: Path, path: str, *, missing_ok: bool = False) -> Path:
    canonical = _path(path)
    target = root.joinpath(*canonical.split("/"))
    for current in (*reversed(target.parents), target):
        if current == root or root in current.parents:
            if _unsafe_link(current):
                raise RuntimeAssetError("asset_symlink_or_junction:" + canonical)
            if current != target and current.exists() and not current.is_dir():
                raise RuntimeAssetError("asset_parent_not_directory:" + canonical)
    try:
        target.resolve(strict=False).relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeAssetError("asset_path_escaped:" + canonical) from error
    if not target.exists():
        if missing_ok:
            return target
        raise RuntimeAssetError("asset_file_missing:" + canonical)
    info = target.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RuntimeAssetError("asset_not_regular_or_hardlinked:" + canonical)
    return target


def _read(root: Path, path: str, maximum: int, *, include_mode: bool = False) -> bytes | tuple[bytes, int]:
    target = _file(root, path)
    before = target.lstat()
    mode = _normalized_file_mode(before, path)
    if before.st_size > maximum:
        raise RuntimeAssetError("file_budget_exceeded:" + path)
    descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino) or not stat.S_ISREG(opened.st_mode):
            raise RuntimeAssetError("asset_changed_during_read:" + path)
        raw = handle.read(maximum + 1)
        after = os.fstat(handle.fileno())
    _file(root, path)
    current = target.lstat()
    if (len(raw) > maximum or (after.st_size, after.st_mtime_ns, after.st_ino, after.st_mode) !=
            (before.st_size, before.st_mtime_ns, before.st_ino, before.st_mode) or
            (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_mode) !=
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_mode)):
        raise RuntimeAssetError("asset_changed_during_read:" + path)
    return (raw, mode) if include_mode else raw


def collect_runtime_assets(repo_root: str | Path, relative_paths: Iterable[str], *,
                           source_paths: Iterable[str] = (),
                           max_files: int = DEFAULT_MAX_FILES, max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
                           max_file_bytes: int = DEFAULT_MAX_FILE_BYTES) -> RuntimeAssetSet:
    _limits(max_files, max_total_bytes, max_file_bytes)
    assets = RuntimeAssetSet.from_directory(
        repo_root,
        relative_paths,
        source_paths=source_paths,
        max_files=max_files,
        max_total_bytes=max_total_bytes,
        max_file_bytes=max_file_bytes,
    )
    if assets.total_bytes > INLINE_MAX_TOTAL_BYTES:
        return assets
    # Keep small synthetic and ordinary packs self-contained in memory; large
    # repositories use the content-addressed directory backend below so their
    # binaries are not base64-expanded into every checkpoint.
    return RuntimeAssetSet(
        tuple((path, assets.bytes_for(path)) for path in assets.paths),
        asset_modes=dict(assets._asset_modes),
        source_modes=assets.source_modes,
    )


def _write(root: Path, path: str, raw: bytes, *, mode: int = 0o644) -> None:
    _mode(mode)
    if os.name == "nt" and mode == 0o755:
        raise RuntimeAssetError("executable_mode_unsupported_on_host:" + path)
    target = _file(root, path, missing_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    _file(root, path, missing_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".runtime-asset-", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        if _file(root, path, missing_ok=True).exists():
            raise RuntimeAssetError("asset_destination_appeared:" + path)
        os.replace(temporary, target)
        _verify_mode(target, mode, path)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_backed_asset(
    source_root: Path,
    target_root: Path,
    path: str,
    *,
    size: int,
    digest: str,
    mode: int,
) -> None:
    """Copy one immutable backing file atomically without loading it in RAM."""

    source = _file(source_root, path)
    if source.stat().st_size != size or _stream_digest(source) != digest:
        raise RuntimeAssetError("asset_content_changed:" + path)
    target = _file(target_root, path, missing_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    _file(target_root, path, missing_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".runtime-asset-", dir=target.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        # copy2 may use copy_file_range/reflinks where available and never
        # creates a hardlink that an untrusted test could mutate backwards.
        shutil.copyfile(source, temporary)
        os.chmod(temporary, mode)
        if temporary.stat().st_size != size or _stream_digest(temporary) != digest:
            raise RuntimeAssetError("asset_copy_changed:" + path)
        if _file(target_root, path, missing_ok=True).exists():
            raise RuntimeAssetError("asset_destination_appeared:" + path)
        os.replace(temporary, target)
        _verify_mode(target, mode, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_runtime_assets(assets: RuntimeAssetSet, pack_dir: str | Path) -> Path:
    _verify_set(assets)
    root = _root(pack_dir, create=True)
    target = _file(root, MANIFEST_FILENAME, missing_ok=True)
    if target.exists():
        read_runtime_assets(root, expected_digest=assets.asset_set_digest)
        return target
    if assets._backing_root is None:
        payload = assets.to_payload()
    else:
        asset_root = _root(root / ASSET_DIRECTORY, create=True)
        source_root = _root(assets._backing_root)
        for path, size, digest, mode in assets._rows:
            _copy_backed_asset(
                source_root, asset_root, path,
                size=size, digest=digest, mode=mode,
            )
        payload = {
            **assets.manifest(),
            "storage": {
                "type": "content_addressed_directory_v1",
                "directory": ASSET_DIRECTORY,
            },
        }
    _write(root, MANIFEST_FILENAME, json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    return target


def read_runtime_assets(pack_dir: str | Path, *, expected_digest: str | None = None,
                        max_files: int = DEFAULT_MAX_FILES, max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
                        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES) -> RuntimeAssetSet:
    _limits(max_files, max_total_bytes, max_file_bytes)
    root = _root(pack_dir)
    raw = _read(
        root,
        MANIFEST_FILENAME,
        min(MANIFEST_MAX_BYTES, 2 * max_total_bytes + 1024 * max_files + 4096),
    )
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeAssetError("manifest_duplicate_key")
            result[key] = value
        return result
    try:
        payload = json.loads(raw, object_pairs_hook=unique_object)
    except (ValueError, UnicodeError) as error:
        raise RuntimeAssetError("manifest_json_invalid") from error
    if not isinstance(payload, Mapping):
        raise RuntimeAssetError("manifest_schema_invalid")
    if "storage" not in payload:
        return RuntimeAssetSet.from_payload(payload, expected_digest=expected_digest, max_files=max_files,
                                            max_total_bytes=max_total_bytes, max_file_bytes=max_file_bytes)
    keys = {
        "schema_version", "asset_set_digest", "file_count", "total_bytes",
        "files", "source_modes", "storage",
    }
    storage = payload.get("storage")
    if set(payload) != keys or storage != {
        "type": "content_addressed_directory_v1",
        "directory": ASSET_DIRECTORY,
    }:
        raise RuntimeAssetError("manifest_schema_invalid")
    rows = payload.get("files")
    source_modes = payload.get("source_modes")
    if not isinstance(rows, list) or not isinstance(source_modes, Mapping) or any(
        not isinstance(row, dict)
        or set(row) != {"path", "size_bytes", "sha256", "mode"}
        for row in rows
    ):
        raise RuntimeAssetError("manifest_entries_invalid")
    if (
        type(payload.get("file_count")) is not int
        or payload["file_count"] != len(rows)
        or type(payload.get("total_bytes")) is not int
        or payload["total_bytes"] > max_total_bytes
        or len(rows) + len(source_modes) > max_files
        or any(
            type(row["size_bytes"]) is not int
            or row["size_bytes"] < 0
            or row["size_bytes"] > max_file_bytes
            for row in rows
        )
    ):
        raise RuntimeAssetError("manifest_identity_invalid")
    assets = RuntimeAssetSet._from_directory_manifest(
        root / ASSET_DIRECTORY,
        tuple(
            (row["path"], row["size_bytes"], row["sha256"], row["mode"])
            for row in rows
        ),
        tuple(source_modes.items()),
        expected_digest=expected_digest or str(payload.get("asset_set_digest") or ""),
    )
    if assets.manifest() != {key: payload[key] for key in keys - {"storage"}}:
        raise RuntimeAssetError("manifest_identity_mismatch")
    # A pack is an external persistence boundary. Hash every body once while
    # loading it; later metadata reads remain cheap and exports re-check source
    # bytes as they copy.
    for path, size, digest, mode in assets._rows:
        source = _file(Path(assets._backing_root), path)
        if source.stat().st_size != size or _stream_digest(source) != digest:
            raise RuntimeAssetError("asset_content_changed:" + path)
        _verify_mode(source, mode, path)
    return assets


def initialize_runtime_assets(world: Any, assets: RuntimeAssetSet) -> RuntimeAssetSet:
    _verify_set(assets)
    previous = runtime_assets_for_world(world)
    if previous is not None:
        _expected(assets, previous.asset_set_digest)
        return previous
    world.__dict__[_STORE] = assets
    return assets


def runtime_assets_enabled(world: Any) -> bool:
    return _STORE in getattr(world, "__dict__", {})


def runtime_assets_for_world(world: Any) -> RuntimeAssetSet | None:
    if not runtime_assets_enabled(world):
        return None
    assets = world.__dict__[_STORE]
    if type(assets) is not RuntimeAssetSet:
        raise RuntimeAssetError("validated_asset_set_required")
    return assets


def runtime_assets_digest(world: Any) -> str | None:
    assets = runtime_assets_for_world(world)
    return None if assets is None else assets.asset_set_digest


def runtime_asset_receipt(world: Any) -> dict | None:
    assets = runtime_assets_for_world(world)
    return None if assets is None else assets.manifest()


def assert_no_runtime_asset_conflicts(assets: RuntimeAssetSet, source_paths: Iterable[str]) -> None:
    if type(assets) is not RuntimeAssetSet:
        raise RuntimeAssetError("validated_asset_set_required")
    sources = _paths(source_paths)
    folded_sources = sorted(source.casefold() for source in sources)
    source_set = set(folded_sources)
    for path in assets.paths:
        folded = path.casefold()
        parts = folded.split("/")
        if any("/".join(parts[:index]) in source_set
               for index in range(1, len(parts) + 1)):
            raise RuntimeAssetError("editable_source_conflict:" + path)
        descendant_prefix = folded + "/"
        index = bisect_left(folded_sources, descendant_prefix)
        if (index < len(folded_sources)
                and folded_sources[index].startswith(descendant_prefix)):
            raise RuntimeAssetError("editable_source_conflict:" + path)


def verify_runtime_assets(assets: RuntimeAssetSet, root: str | Path, *, source_paths: Iterable[str],
                          expected_digest: str | None = None) -> dict:
    _verify_set(assets)
    _expected(assets, expected_digest)
    assert_no_runtime_asset_conflicts(assets, source_paths)
    directory = _root(root)
    modes = dict(assets._asset_modes)
    if assets._backing_root is None:
        rows = tuple(
            (path, len(raw), hashlib.sha256(raw).hexdigest(), raw)
            for path, raw in assets._entries
        )
    else:
        rows = tuple((path, size, digest, None) for path, size, digest, _mode_value in assets._rows)
    for path, size, digest, inline in rows:
        target = _file(directory, path)
        if target.stat().st_size != size or _stream_digest(target) != digest:
            raise RuntimeAssetError("exported_asset_content_changed:" + path)
        if inline is not None and _read(directory, path, size) != inline:
            raise RuntimeAssetError("exported_asset_content_changed:" + path)
        _verify_mode(target, modes[path], path)
    return {**assets.manifest(), "verified": True}


def export_runtime_assets(assets: RuntimeAssetSet, dest: str | Path, *, source_paths: Iterable[str],
                          expected_digest: str | None = None) -> dict:
    _verify_set(assets)
    _expected(assets, expected_digest)
    source_paths = tuple(source_paths)
    assert_no_runtime_asset_conflicts(assets, source_paths)
    modes = dict(assets._asset_modes)
    if os.name == "nt" and 0o755 in modes.values():
        raise RuntimeAssetError("executable_mode_unsupported_on_host")
    root = _root(dest, create=True)
    # Validate the entire destination BEFORE writing the first file. Never
    # repair mutated existing assets or overwrite source under another alias.
    missing = []
    if assets._backing_root is None:
        rows = tuple(
            (path, len(raw), hashlib.sha256(raw).hexdigest(), mode, raw)
            for (path, raw), (_mode_path, mode) in zip(assets._entries, assets._asset_modes)
        )
    else:
        rows = tuple(
            (path, size, digest, mode, None)
            for path, size, digest, mode in assets._rows
        )
    for path, size, digest, mode, inline in rows:
        target = _file(root, path, missing_ok=True)
        if target.exists():
            if target.stat().st_size != size or _stream_digest(target) != digest:
                raise RuntimeAssetError("exported_asset_content_changed:" + path)
            _verify_mode(target, modes[path], path)
        else:
            missing.append((path, size, digest, mode, inline))
    for path, size, digest, mode, inline in missing:
        if inline is not None:
            _write(root, path, inline, mode=mode)
        else:
            _copy_backed_asset(
                _root(assets._backing_root), root, path,
                size=size, digest=digest, mode=mode,
            )
    return verify_runtime_assets(assets, root, source_paths=source_paths, expected_digest=assets.asset_set_digest)


def apply_runtime_source_modes(assets: RuntimeAssetSet, dest: str | Path, *,
                               source_paths: Iterable[str], expected_digest: str | None = None) -> dict:
    """Apply only mode metadata to explicitly exported, current source paths.

    Call with the snapshot's actual file paths, NOT the baseline inventory.
    Deleted paths are never created; new paths default to normal 0644. This
    function does not read, replace or copy any source contents.
    """
    _verify_set(assets)
    _expected(assets, expected_digest)
    paths = _paths(source_paths)
    assert_no_runtime_asset_conflicts(assets, paths)
    modes = dict(assets._source_modes)
    expected = {path: modes.get(path, 0o644) for path in paths}
    if os.name == "nt" and 0o755 in expected.values():
        raise RuntimeAssetError("executable_mode_unsupported_on_host")
    root = _root(dest)
    # Validate every target before changing the first permission. No symlink,
    # hardlink, special mode or absent source may be silently repaired.
    for path in paths:
        _normalized_file_mode(_file(root, path).lstat(), path)
    for path, mode in expected.items():
        target = _file(root, path)
        os.chmod(target, mode)
        _verify_mode(target, mode, path)
    return {"asset_set_digest": assets.asset_set_digest, "source_modes": expected, "verified": True}


__all__ = ["SCHEMA", "MANIFEST_FILENAME", "ASSET_DIRECTORY", "CONTAINER_ASSET_ROOT",
           "RuntimeAssetError", "RuntimeAssetSet", "runtime_asset_backing_root",
           "link_runtime_assets_for_container",
           "collect_runtime_assets", "write_runtime_assets", "read_runtime_assets", "initialize_runtime_assets",
           "runtime_assets_enabled", "runtime_assets_for_world", "runtime_assets_digest", "runtime_asset_receipt",
           "assert_no_runtime_asset_conflicts", "export_runtime_assets", "verify_runtime_assets", "apply_runtime_source_modes"]
