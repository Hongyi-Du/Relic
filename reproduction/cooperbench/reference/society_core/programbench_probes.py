"""Declarative, bounded public behavior probes for ProgramBench candidates.

A probe definition command may emit JSON, but this module accepts only the JSON
bytes and never executes that command. Each case is then executed independently
against a reference command and a candidate command through caller-injected
boundaries. The included host subprocess backend is only for trusted local test
CLIs; formal evaluation must inject separate pinned sandboxes.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO, Literal, Mapping, Protocol, Sequence


PUBLIC_PROBE_SCHEMA_VERSION = "programbench_public_probe_cases_v1"
PUBLIC_PROBE_RUNNER = "society_core.programbench_probes.evaluate_public_probes"
PublicProbeMode = Literal["differential", "reference_only"]
PublicProbeComparisonPolicy = Literal["exact_v1", "go_standard_log_clock_v1"]
PUBLIC_PROBE_ENV_ALLOWLIST = frozenset(
    {"COLUMNS", "LANG", "LC_ALL", "LINES", "NO_COLOR", "TERM", "TZ"}
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PINNED_IMAGE = re.compile(r"(?:@sha256:[0-9a-fA-F]{64}|^sha256:[0-9a-fA-F]{64})\Z")
_GO_STANDARD_LOG_TIMESTAMP = re.compile(
    r"(?m)^\d{4}/(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01]) "
    r"(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d "
)

_PROBE_EXECUTION_BOOTSTRAP = r'''from __future__ import annotations

import json
from pathlib import Path

import probe_runtime


def emit(value, budget):
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("ascii")
    if len(payload) > budget:
        payload = json.dumps(
            {
                "status": "infra_error",
                "reason": "probe_bridge_json_exceeds_executor_tail_budget",
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    import sys
    sys.stdout.buffer.write(payload)


request = json.loads(Path("/workspace/request.json").read_text(encoding="utf-8"))
budget = int(request["bridge_budget"])
try:
    limits = probe_runtime.ProbeLimits(**request["limits"])
    document = json.dumps(
        {
            "schema_version": probe_runtime.PUBLIC_PROBE_SCHEMA_VERSION,
            "cases": [request["case"]],
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    case = probe_runtime.load_probe_cases(document, limits=limits)[0]
    result = probe_runtime._execute_case(tuple(request["command"]), case, limits=limits)
except BaseException as error:
    result = {
        "status": "infra_error",
        "reason": "probe_bootstrap_failed:" + type(error).__name__,
    }
emit(result, budget)
'''

_PROBE_DOCUMENT_BOOTSTRAP = r'''from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path

import probe_runtime


def emit(value, budget):
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("ascii")
    if len(payload) > budget:
        payload = json.dumps(
            {
                "status": "infra_error",
                "reason": "probe_document_bridge_json_exceeds_executor_tail_budget",
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    sys.stdout.buffer.write(payload)


request = json.loads(Path("/workspace/request.json").read_text(encoding="utf-8"))
budget = int(request["bridge_budget"])
try:
    limits = probe_runtime.ProbeLimits(**request["limits"])
    case = probe_runtime.ProbeCase(argv=(), stdin="", input_files=(), env=())
    result = probe_runtime._execute_case(
        (sys.executable, request["script"]), case, limits=limits
    )
    if result.get("status") != "completed":
        envelope = result
    elif result["exit"]["timed_out"]:
        envelope = {"status": "definition_error", "reason": "probe_document_timeout"}
    elif result["exit"]["code"] != 0:
        envelope = {"status": "definition_error", "reason": "probe_document_nonzero_exit"}
    elif result["stdout"]["truncated"]:
        envelope = {"status": "definition_error", "reason": "probe_document_stdout_truncated"}
    else:
        payload = result["stdout"]["text"].encode("utf-8")
        if len(payload) != result["stdout"]["bytes"]:
            envelope = {"status": "definition_error", "reason": "probe_document_not_utf8"}
        elif hashlib.sha256(payload).hexdigest() != result["stdout"]["sha256"]:
            envelope = {"status": "infra_error", "reason": "probe_document_hash_mismatch"}
        elif not payload:
            envelope = {"status": "definition_error", "reason": "probe_document_empty"}
        else:
            try:
                probe_runtime.load_probe_cases(payload, limits=limits)
            except probe_runtime.ProgramBenchProbeError as error:
                envelope = {"status": "definition_error", "reason": str(error)}
            else:
                envelope = {
                    "status": "ok",
                    "payload_base64": base64.b64encode(payload).decode("ascii"),
                }
except BaseException as error:
    envelope = {
        "status": "infra_error",
        "reason": "probe_document_bootstrap_failed:" + type(error).__name__,
    }
emit(envelope, budget)
'''


class ProgramBenchProbeError(ValueError):
    """A public probe definition or execution request is invalid."""


@dataclass(frozen=True)
class ProbeLimits:
    max_cases: int = 128
    max_argv_items: int = 64
    max_argument_bytes: int = 4096
    max_stdin_bytes: int = 64 * 1024
    max_input_files: int = 16
    max_input_file_bytes: int = 64 * 1024
    max_input_bytes: int = 256 * 1024
    max_env_items: int = 8
    timeout_seconds: float = 10.0
    max_output_bytes: int = 1024 * 1024
    max_json_bytes: int = 1024 * 1024
    max_effect_entries: int = 1024
    max_effect_file_bytes: int = 4 * 1024 * 1024
    max_effect_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        integer_limits = {
            "max_cases": self.max_cases,
            "max_argv_items": self.max_argv_items,
            "max_argument_bytes": self.max_argument_bytes,
            "max_stdin_bytes": self.max_stdin_bytes,
            "max_input_files": self.max_input_files,
            "max_input_file_bytes": self.max_input_file_bytes,
            "max_input_bytes": self.max_input_bytes,
            "max_env_items": self.max_env_items,
            "max_output_bytes": self.max_output_bytes,
            "max_json_bytes": self.max_json_bytes,
            "max_effect_entries": self.max_effect_entries,
            "max_effect_file_bytes": self.max_effect_file_bytes,
            "max_effect_bytes": self.max_effect_bytes,
        }
        for name, value in integer_limits.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name}_must_be_positive_integer")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds_must_be_positive_finite")
        if self.max_input_file_bytes > self.max_input_bytes:
            raise ValueError("max_input_file_bytes_exceeds_total")
        if self.max_effect_file_bytes > self.max_effect_bytes:
            raise ValueError("max_effect_file_bytes_exceeds_total")


@dataclass(frozen=True)
class ProbeInputFile:
    path: str
    content: bytes


@dataclass(frozen=True)
class ProbeCase:
    argv: tuple[str, ...]
    stdin: str
    input_files: tuple[ProbeInputFile, ...]
    env: tuple[tuple[str, str], ...]


class ProbeExecutor(Protocol):
    """Execution boundary supplied by the evaluator integration."""

    def execute(
        self,
        *,
        command: tuple[str, ...],
        case: ProbeCase,
        limits: ProbeLimits,
    ) -> Mapping[str, Any]: ...


def public_probe_schema() -> dict[str, Any]:
    """Return the JSON Schema shipped in ProgramBench starter repositories."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": PUBLIC_PROBE_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "cases"],
        "properties": {
            "schema_version": {"const": PUBLIC_PROBE_SCHEMA_VERSION},
            "cases": {
                "type": "array",
                "maxItems": ProbeLimits().max_cases,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["argv", "stdin", "input_files", "env"],
                    "properties": {
                        "argv": {
                            "type": "array",
                            "maxItems": ProbeLimits().max_argv_items,
                            "items": {"type": "string"},
                            # The runner executes (*command, *argv), where
                            # command already names the compiled executable.
                            # POSIX argv[0] is the program, so an author
                            # following that convention writes the program name
                            # here and every such case degenerates into the same
                            # unknown-argument output: one run authored 84
                            # distinct inputs and got 2 distinct behaviours out
                            # of the reference. Saying so is cheaper than
                            # letting each organization rediscover it.
                            "description": (
                                "Arguments passed after the program. The runner "
                                "supplies the compiled executable, so this array "
                                "must NOT begin with the program name: probe "
                                "'prog --help' as [\"--help\"]."
                            ),
                        },
                        "stdin": {"type": "string"},
                        "input_files": {
                            "type": "array",
                            "maxItems": ProbeLimits().max_input_files,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["path", "content_base64"],
                                "properties": {
                                    "path": {"type": "string"},
                                    "content_base64": {"type": "string"},
                                },
                            },
                        },
                        "env": {
                            "type": "object",
                            "propertyNames": {
                                "enum": sorted(PUBLIC_PROBE_ENV_ALLOWLIST)
                            },
                            "additionalProperties": {"type": "string"},
                            "maxProperties": ProbeLimits().max_env_items,
                        },
                    },
                },
            },
        },
    }


def _strict_json(payload: bytes, *, limits: ProbeLimits) -> Any:
    if len(payload) > limits.max_json_bytes:
        raise ProgramBenchProbeError("probe_json_too_large")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ProgramBenchProbeError("probe_json_not_utf8") from error

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ProgramBenchProbeError("probe_json_duplicate_key")
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=pairs)
    except ProgramBenchProbeError:
        raise
    except (json.JSONDecodeError, RecursionError) as error:
        raise ProgramBenchProbeError("probe_json_invalid") from error


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ProgramBenchProbeError(f"{label}_keys_invalid")


def _canonical_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ProgramBenchProbeError("input_file_path_invalid")
    path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise ProgramBenchProbeError("input_file_path_not_canonical")
    if value != path.as_posix():
        raise ProgramBenchProbeError("input_file_path_not_canonical")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ProgramBenchProbeError("input_file_path_not_canonical")
    return value


def _validate_probe_argument(value: str) -> None:
    """Reject arguments that can name evaluator/runtime-owned filesystem paths.

    Probe authors control argv, while the trusted runner supplies the executable.
    Absolute paths, traversal, file URLs, and response-file spellings would let
    a file-reading CLI turn the read-only evaluator mount into an oracle-byte
    disclosure channel.  Validate both a whole token and the value side of the
    conventional ``--option=value`` spelling.
    """
    candidates = [value]
    if "=" in value:
        candidates.append(value.split("=", 1)[1])
    for candidate in candidates:
        token = candidate.lstrip("@").strip()
        normalized = token.replace("\\", "/")
        lowered = normalized.casefold()
        windows = PureWindowsPath(token)
        if lowered.startswith(("file://", "file:/")):
            raise ProgramBenchProbeError("probe_argument_reserved_path")
        if (
            normalized.startswith("/")
            or windows.is_absolute()
            or bool(windows.drive)
        ):
            raise ProgramBenchProbeError("probe_argument_reserved_path")
        if ".." in PurePosixPath(normalized).parts:
            raise ProgramBenchProbeError("probe_argument_reserved_path")


def load_probe_cases(
    payload: bytes | str, *, limits: ProbeLimits = ProbeLimits()
) -> tuple[ProbeCase, ...]:
    """Parse the exact public-probe JSON contract into immutable cases."""

    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    root = _strict_json(payload, limits=limits)
    if not isinstance(root, dict):
        raise ProgramBenchProbeError("probe_json_not_object")
    _exact_keys(root, {"schema_version", "cases"}, label="probe_json")
    if root["schema_version"] != PUBLIC_PROBE_SCHEMA_VERSION:
        raise ProgramBenchProbeError("probe_schema_unsupported")
    rows = root["cases"]
    if not isinstance(rows, list) or len(rows) > limits.max_cases:
        raise ProgramBenchProbeError("probe_cases_invalid")

    cases: list[ProbeCase] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ProgramBenchProbeError("probe_case_not_object")
        _exact_keys(row, {"argv", "stdin", "input_files", "env"}, label="probe_case")
        argv = row["argv"]
        if not isinstance(argv, list) or len(argv) > limits.max_argv_items:
            raise ProgramBenchProbeError("probe_argv_invalid")
        checked_argv: list[str] = []
        for argument in argv:
            if (
                not isinstance(argument, str)
                or "\x00" in argument
                or len(argument.encode("utf-8")) > limits.max_argument_bytes
            ):
                raise ProgramBenchProbeError("probe_argument_invalid")
            _validate_probe_argument(argument)
            checked_argv.append(argument)

        stdin = row["stdin"]
        if not isinstance(stdin, str) or len(stdin.encode("utf-8")) > limits.max_stdin_bytes:
            raise ProgramBenchProbeError("probe_stdin_invalid")

        input_rows = row["input_files"]
        if not isinstance(input_rows, list) or len(input_rows) > limits.max_input_files:
            raise ProgramBenchProbeError("probe_input_files_invalid")
        input_files: list[ProbeInputFile] = []
        seen_paths: set[str] = set()
        total_bytes = 0
        for input_row in input_rows:
            if not isinstance(input_row, dict):
                raise ProgramBenchProbeError("probe_input_file_not_object")
            _exact_keys(input_row, {"path", "content_base64"}, label="probe_input_file")
            path = _canonical_relative_path(input_row["path"])
            if path in seen_paths:
                raise ProgramBenchProbeError("probe_input_file_duplicate")
            if any(
                path.startswith(existing + "/") or existing.startswith(path + "/")
                for existing in seen_paths
            ):
                raise ProgramBenchProbeError("probe_input_file_ancestor_collision")
            encoded = input_row["content_base64"]
            if not isinstance(encoded, str):
                raise ProgramBenchProbeError("probe_input_file_content_invalid")
            try:
                content = base64.b64decode(encoded, validate=True)
            except (ValueError, UnicodeEncodeError) as error:
                raise ProgramBenchProbeError("probe_input_file_content_invalid") from error
            if len(content) > limits.max_input_file_bytes:
                raise ProgramBenchProbeError("probe_input_file_too_large")
            total_bytes += len(content)
            if total_bytes > limits.max_input_bytes:
                raise ProgramBenchProbeError("probe_input_files_too_large")
            seen_paths.add(path)
            input_files.append(ProbeInputFile(path=path, content=content))

        env = row["env"]
        if not isinstance(env, dict) or len(env) > limits.max_env_items:
            raise ProgramBenchProbeError("probe_env_invalid")
        checked_env: list[tuple[str, str]] = []
        for key, value in env.items():
            if key not in PUBLIC_PROBE_ENV_ALLOWLIST:
                raise ProgramBenchProbeError("probe_env_key_forbidden")
            if not isinstance(value, str) or "\x00" in value or len(value.encode("utf-8")) > 4096:
                raise ProgramBenchProbeError("probe_env_value_invalid")
            checked_env.append((key, value))

        cases.append(
            ProbeCase(
                argv=tuple(checked_argv),
                stdin=stdin,
                input_files=tuple(input_files),
                env=tuple(sorted(checked_env)),
            )
        )
    return tuple(cases)


def _base_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    # PATH is runner-owned, never supplied by a probe case.  Windows requires a
    # few additional controller values to start a process reliably.
    inherited = {"PATH": os.environ.get("PATH", "")}
    for key in ("COMSPEC", "PATHEXT", "SystemRoot", "WINDIR"):
        if key in os.environ:
            inherited[key] = os.environ[key]
    inherited.update({"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TZ": "UTC"})
    if extra:
        inherited.update(extra)
    return inherited


def _validate_command(command: Sequence[str], *, label: str) -> tuple[str, ...]:
    if isinstance(command, (str, bytes)) or not command:
        raise ProgramBenchProbeError(f"{label}_command_invalid")
    checked = tuple(command)
    if any(not isinstance(item, str) or not item or "\x00" in item for item in checked):
        raise ProgramBenchProbeError(f"{label}_command_invalid")
    return checked


def _output_summary(handle: BinaryIO, *, limit: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    prefix = bytearray()
    handle.flush()
    handle.seek(0)
    for chunk in iter(lambda: handle.read(64 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
        if len(prefix) < limit:
            prefix.extend(chunk[: limit - len(prefix)])
    return {
        "text": bytes(prefix).decode("utf-8", errors="replace"),
        "bytes": size,
        "sha256": digest.hexdigest(),
        "truncated": size > limit,
    }


def _stream_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _workspace_snapshot(
    root: Path, *, limits: ProbeLimits
) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if len(snapshot) >= limits.max_effect_entries:
            raise ProgramBenchProbeError("probe_effect_entry_limit_exceeded")
        relative = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            target = os.readlink(path)
            target_bytes = target.encode("utf-8")
            total_bytes += len(target_bytes)
            snapshot[relative] = {
                "kind": "symlink",
                "target": target,
                "sha256": hashlib.sha256(target_bytes).hexdigest(),
            }
        elif stat.S_ISREG(mode):
            size = path.stat().st_size
            if size > limits.max_effect_file_bytes:
                raise ProgramBenchProbeError("probe_effect_file_too_large")
            total_bytes += size
            snapshot[relative] = {
                "kind": "file",
                "size": size,
                "sha256": _stream_file_hash(path),
            }
        elif stat.S_ISDIR(mode):
            snapshot[relative] = {"kind": "directory"}
        else:
            raise ProgramBenchProbeError("probe_workspace_special_file_forbidden")
        if total_bytes > limits.max_effect_bytes:
            raise ProgramBenchProbeError("probe_effect_bytes_limit_exceeded")
    return snapshot


def _filesystem_effects(
    before: Mapping[str, Mapping[str, Any]], after: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    before_paths = set(before)
    after_paths = set(after)
    return {
        "created": [
            {"path": path, **after[path]} for path in sorted(after_paths - before_paths)
        ],
        "modified": [
            {"path": path, "before": before[path], "after": after[path]}
            for path in sorted(before_paths & after_paths)
            if before[path] != after[path]
        ],
        "deleted": [
            {"path": path, **before[path]} for path in sorted(before_paths - after_paths)
        ],
    }


def _canonical_output_summary(
    value: Any, *, label: str, limits: ProbeLimits
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProgramBenchProbeError(f"probe_executor_{label}_invalid")
    _exact_keys(value, {"text", "bytes", "sha256", "truncated"}, label=label)
    text = value["text"]
    size = value["bytes"]
    digest = value["sha256"]
    truncated = value["truncated"]
    if not isinstance(text, str):
        raise ProgramBenchProbeError(f"probe_executor_{label}_text_invalid")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ProgramBenchProbeError(f"probe_executor_{label}_bytes_invalid")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ProgramBenchProbeError(f"probe_executor_{label}_sha256_invalid")
    if not isinstance(truncated, bool):
        raise ProgramBenchProbeError(f"probe_executor_{label}_truncated_invalid")
    if truncated != (size > limits.max_output_bytes):
        raise ProgramBenchProbeError(f"probe_executor_{label}_truncated_mismatch")
    if len(text.encode("utf-8")) > limits.max_output_bytes * 3:
        raise ProgramBenchProbeError(f"probe_executor_{label}_text_too_large")
    return {
        "text": text,
        "bytes": size,
        "sha256": digest,
        "truncated": truncated,
    }


def _canonical_effect_metadata(
    value: Any, *, label: str, limits: ProbeLimits
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProgramBenchProbeError(f"probe_executor_{label}_metadata_invalid")
    kind = value.get("kind")
    if kind == "file":
        _exact_keys(value, {"kind", "size", "sha256"}, label=label)
        size = value["size"]
        digest = value["sha256"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ProgramBenchProbeError(f"probe_executor_{label}_size_invalid")
        if size > limits.max_effect_file_bytes:
            raise ProgramBenchProbeError(f"probe_executor_{label}_file_too_large")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ProgramBenchProbeError(f"probe_executor_{label}_sha256_invalid")
        return {"kind": "file", "size": size, "sha256": digest}
    if kind == "symlink":
        _exact_keys(value, {"kind", "target", "sha256"}, label=label)
        target = value["target"]
        digest = value["sha256"]
        if not isinstance(target, str) or "\x00" in target:
            raise ProgramBenchProbeError(f"probe_executor_{label}_target_invalid")
        target_bytes = target.encode("utf-8")
        if len(target_bytes) > limits.max_effect_file_bytes:
            raise ProgramBenchProbeError(f"probe_executor_{label}_target_too_large")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ProgramBenchProbeError(f"probe_executor_{label}_sha256_invalid")
        if digest != hashlib.sha256(target_bytes).hexdigest():
            raise ProgramBenchProbeError(f"probe_executor_{label}_sha256_mismatch")
        return {"kind": "symlink", "target": target, "sha256": digest}
    if kind == "directory":
        _exact_keys(value, {"kind"}, label=label)
        return {"kind": "directory"}
    raise ProgramBenchProbeError(f"probe_executor_{label}_kind_invalid")


def _canonical_effect_path(value: Any, *, label: str) -> str:
    try:
        return _canonical_relative_path(value)
    except ProgramBenchProbeError as error:
        raise ProgramBenchProbeError(f"probe_executor_{label}_path_invalid") from error


def _canonical_effects(value: Any, *, limits: ProbeLimits) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProgramBenchProbeError("probe_executor_filesystem_effects_invalid")
    _exact_keys(value, {"created", "modified", "deleted"}, label="filesystem_effects")
    normalized: dict[str, list[dict[str, Any]]] = {
        "created": [],
        "modified": [],
        "deleted": [],
    }
    seen: set[str] = set()
    before_bytes = 0
    after_bytes = 0

    def metadata_bytes(metadata: Mapping[str, Any]) -> int:
        if metadata["kind"] == "file":
            return int(metadata["size"])
        if metadata["kind"] == "symlink":
            return len(str(metadata["target"]).encode("utf-8"))
        return 0

    for category in ("created", "modified", "deleted"):
        rows = value[category]
        if not isinstance(rows, list):
            raise ProgramBenchProbeError(
                f"probe_executor_filesystem_{category}_invalid"
            )
        for row in rows:
            if not isinstance(row, Mapping):
                raise ProgramBenchProbeError(
                    f"probe_executor_filesystem_{category}_entry_invalid"
                )
            if category == "modified":
                _exact_keys(row, {"path", "before", "after"}, label=category)
                path = _canonical_effect_path(row["path"], label=category)
                item = {
                    "path": path,
                    "before": _canonical_effect_metadata(
                        row["before"], label="modified_before", limits=limits
                    ),
                    "after": _canonical_effect_metadata(
                        row["after"], label="modified_after", limits=limits
                    ),
                }
                if item["before"] == item["after"]:
                    raise ProgramBenchProbeError(
                        "probe_executor_modified_effect_unchanged"
                    )
            else:
                if "path" not in row:
                    raise ProgramBenchProbeError(
                        f"probe_executor_filesystem_{category}_path_missing"
                    )
                path = _canonical_effect_path(row["path"], label=category)
                metadata = {key: item for key, item in row.items() if key != "path"}
                item = {
                    "path": path,
                    **_canonical_effect_metadata(
                        metadata, label=category, limits=limits
                    ),
                }
            if path in seen:
                raise ProgramBenchProbeError("probe_executor_effect_path_duplicate")
            seen.add(path)
            normalized[category].append(item)
            if category == "created":
                after_bytes += metadata_bytes(item)
            elif category == "deleted":
                before_bytes += metadata_bytes(item)
            else:
                before_bytes += metadata_bytes(item["before"])
                after_bytes += metadata_bytes(item["after"])
        if [row["path"] for row in normalized[category]] != sorted(
            row["path"] for row in normalized[category]
        ):
            raise ProgramBenchProbeError(
                f"probe_executor_filesystem_{category}_not_canonical"
            )
    if len(seen) > limits.max_effect_entries:
        raise ProgramBenchProbeError("probe_executor_effect_entry_limit_exceeded")
    if before_bytes > limits.max_effect_bytes or after_bytes > limits.max_effect_bytes:
        raise ProgramBenchProbeError("probe_executor_effect_bytes_limit_exceeded")
    return normalized


def _canonical_execution_result(
    value: Any, *, limits: ProbeLimits
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProgramBenchProbeError("probe_executor_result_invalid")
    status = value.get("status")
    if status == "infra_error":
        _exact_keys(value, {"status", "reason"}, label="executor_infra")
        reason = value["reason"]
        if not isinstance(reason, str) or not reason or len(reason) > 512:
            raise ProgramBenchProbeError("probe_executor_infra_reason_invalid")
        return {"status": "infra_error", "reason": reason}
    if status != "completed":
        raise ProgramBenchProbeError("probe_executor_status_invalid")
    _exact_keys(
        value,
        {"status", "stdout", "stderr", "exit", "filesystem_effects"},
        label="executor_result",
    )
    exit_summary = value["exit"]
    if not isinstance(exit_summary, Mapping):
        raise ProgramBenchProbeError("probe_executor_exit_invalid")
    _exact_keys(
        exit_summary, {"code", "timed_out", "spawn_error"}, label="executor_exit"
    )
    code = exit_summary["code"]
    timed_out = exit_summary["timed_out"]
    spawn_error = exit_summary["spawn_error"]
    if code is not None and (isinstance(code, bool) or not isinstance(code, int)):
        raise ProgramBenchProbeError("probe_executor_exit_code_invalid")
    if not isinstance(timed_out, bool):
        raise ProgramBenchProbeError("probe_executor_timeout_invalid")
    if spawn_error is not None:
        raise ProgramBenchProbeError("probe_executor_completed_has_spawn_error")
    if code is None and not timed_out:
        raise ProgramBenchProbeError("probe_executor_exit_code_missing")
    return {
        "status": "completed",
        "stdout": _canonical_output_summary(
            value["stdout"], label="stdout", limits=limits
        ),
        "stderr": _canonical_output_summary(
            value["stderr"], label="stderr", limits=limits
        ),
        "exit": {"code": code, "timed_out": timed_out, "spawn_error": None},
        "filesystem_effects": _canonical_effects(
            value["filesystem_effects"], limits=limits
        ),
    }


def _comparison_execution_result(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a comparison-only view with Go's volatile log clock removed.

    Go's standard logger prefixes stderr lines with ``YYYY/MM/DD HH:MM:SS``.
    Reference and candidate are deliberately executed in separate sandboxes, so
    otherwise-identical behavior can cross a wall-clock second and mismatch. Raw
    evidence remains untouched; this normalization is deliberately restricted to
    complete, internally consistent stderr summaries and that exact line prefix.
    """

    result = dict(value)
    stderr = value.get("stderr")
    if not isinstance(stderr, Mapping) or stderr.get("truncated") is not False:
        return result
    text = stderr.get("text")
    if not isinstance(text, str):
        return result
    encoded = text.encode("utf-8")
    if (
        stderr.get("bytes") != len(encoded)
        or stderr.get("sha256") != hashlib.sha256(encoded).hexdigest()
    ):
        return result
    def remove_valid_clock(match: re.Match[str]) -> str:
        try:
            datetime.strptime(match.group(0).rstrip(), "%Y/%m/%d %H:%M:%S")
        except ValueError:
            return match.group(0)
        return ""

    normalized = _GO_STANDARD_LOG_TIMESTAMP.sub(remove_valid_clock, text)
    if normalized == text:
        return result
    normalized_bytes = normalized.encode("utf-8")
    result["stderr"] = {
        "text": normalized,
        "bytes": len(normalized_bytes),
        "sha256": hashlib.sha256(normalized_bytes).hexdigest(),
        "truncated": False,
    }
    return result


def _execution_results_match(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    comparison_policy: PublicProbeComparisonPolicy,
) -> bool:
    if reference == candidate:
        return True
    if comparison_policy == "exact_v1":
        return False
    if comparison_policy != "go_standard_log_clock_v1":
        raise ProgramBenchProbeError("probe_comparison_policy_invalid")
    return _comparison_execution_result(reference) == _comparison_execution_result(
        candidate
    )


def _divergent_observation_fields(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    comparison_policy: PublicProbeComparisonPolicy,
) -> tuple[str, ...]:
    """Name the observable fields that differ under the comparison policy.

    Reported through the same normalization the differential will later use, so
    the named field is a field that would actually have failed rather than one
    the policy is already designed to absorb.
    """

    if comparison_policy == "go_standard_log_clock_v1":
        left = _comparison_execution_result(first)
        right = _comparison_execution_result(second)
    else:
        left, right = dict(first), dict(second)
    fields = {"status", "stdout", "stderr", "exit", "filesystem_effects"}
    return tuple(
        sorted(name for name in fields if left.get(name) != right.get(name))
    )


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":
        subprocess.run(
            ("taskkill", "/PID", str(process.pid), "/T", "/F"),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.kill()
    process.communicate()


def _execute_case(
    command: tuple[str, ...], case: ProbeCase, *, limits: ProbeLimits
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="programbench-public-probe-") as raw_root:
        root = Path(raw_root)
        for input_file in case.input_files:
            destination = root.joinpath(*PurePosixPath(input_file.path).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(input_file.content)
        before = _workspace_snapshot(root, limits=limits)
        timed_out = False
        spawn_error: str | None = None
        exit_code: int | None = None
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            try:
                process = subprocess.Popen(
                    (*command, *case.argv),
                    cwd=root,
                    env=_base_environment(dict(case.env)),
                    stdin=subprocess.PIPE,
                    stdout=stdout,
                    stderr=stderr,
                    creationflags=(
                        subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                    ),
                    start_new_session=os.name != "nt",
                )
            except OSError as error:
                spawn_error = type(error).__name__
            else:
                try:
                    process.communicate(
                        case.stdin.encode("utf-8"), timeout=limits.timeout_seconds
                    )
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _terminate_process_tree(process)
                exit_code = process.returncode
            if spawn_error is not None:
                return {
                    "status": "infra_error",
                    "reason": f"process_start_failed:{spawn_error}",
                }
            after = _workspace_snapshot(root, limits=limits)
            return {
                "status": "completed",
                "stdout": _output_summary(stdout, limit=limits.max_output_bytes),
                "stderr": _output_summary(stderr, limit=limits.max_output_bytes),
                "exit": {
                    "code": exit_code,
                    "timed_out": timed_out,
                    "spawn_error": spawn_error,
                },
                "filesystem_effects": _filesystem_effects(before, after),
            }


class LocalSubprocessProbeExecutor:
    """Host subprocess backend for focused tests and trusted local CLIs only.

    Formal ProgramBench materialization must inject a pinned sandbox executor
    instead. In particular, it must never use this backend for agent-authored
    candidates or references/oracles.
    """

    def execute(
        self,
        *,
        command: tuple[str, ...],
        case: ProbeCase,
        limits: ProbeLimits,
    ) -> Mapping[str, Any]:
        return _execute_case(command, case, limits=limits)


def _validated_sandbox_executor(
    executor: Any, *, require_execute_only: bool = False
) -> int:
    policy = getattr(executor, "policy", None)
    if policy is None or getattr(policy, "trust_level", None) != "untrusted":
        raise ValueError("probe_sandbox_requires_untrusted_policy")
    backend = getattr(policy, "backend", None)
    if backend not in {"docker", "apptainer"}:
        raise ValueError("probe_sandbox_requires_container_backend")
    image = str(getattr(policy, "container_image", "") or "")
    if _PINNED_IMAGE.search(image) is None:
        raise ValueError("probe_sandbox_requires_digest_pinned_image")
    if getattr(policy, "network_enabled", None) is not False:
        raise ValueError("probe_sandbox_requires_network_disabled")
    for name in (
        "memory_limit_mb",
        "cpu_limit",
        "pids_limit",
        "max_output_bytes",
    ):
        value = getattr(policy, name, None)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"probe_sandbox_{name}_invalid")
    if not callable(getattr(executor, "run_with_read_only_evaluator", None)):
        raise ValueError("probe_sandbox_requires_read_only_mount_api")
    if require_execute_only and not callable(
        getattr(executor, "run_with_execute_only_evaluator", None)
    ):
        raise ValueError("probe_sandbox_requires_execute_only_mount_api")
    tail_budget = getattr(policy, "output_tail_bytes", None)
    if (
        isinstance(tail_budget, bool)
        or not isinstance(tail_budget, int)
        or tail_budget < 256
    ):
        raise ValueError("probe_sandbox_output_tail_budget_invalid")
    if tail_budget > getattr(policy, "max_output_bytes"):
        raise ValueError("probe_sandbox_tail_budget_exceeds_output_limit")
    return tail_budget


def _path_is_junction(path: Path) -> bool:
    checker = getattr(path, "is_junction", None)
    if not callable(checker):
        return False
    try:
        return bool(checker())
    except OSError:
        return True


def _path_or_parent_is_junction(path: Path) -> bool:
    return any(_path_is_junction(item) for item in (path, *path.parents))


def _trusted_relative_file(
    root: str | Path, relative: Any, *, label: str, executable: bool
) -> tuple[Path, str]:
    original_root = Path(root)
    if (
        original_root.is_symlink()
        or _path_or_parent_is_junction(original_root)
        or not original_root.is_dir()
    ):
        raise ProgramBenchProbeError(f"{label}_root_invalid")
    try:
        normalized = _canonical_relative_path(relative)
    except ProgramBenchProbeError as error:
        raise ProgramBenchProbeError(f"{label}_path_invalid") from error
    resolved_root = original_root.resolve()
    cursor = resolved_root
    parts = PurePosixPath(normalized).parts
    for index, part in enumerate(parts):
        cursor = cursor / part
        if _path_is_junction(cursor):
            raise ProgramBenchProbeError(f"{label}_path_junction_forbidden")
        if cursor.is_symlink():
            if index != len(parts) - 1 or not executable:
                raise ProgramBenchProbeError(f"{label}_path_symlink_forbidden")
            link_target = os.readlink(cursor)
            if (
                PurePosixPath(link_target).is_absolute()
                or PureWindowsPath(link_target).is_absolute()
                or PureWindowsPath(link_target).drive
            ):
                raise ProgramBenchProbeError(
                    f"{label}_path_absolute_symlink_forbidden"
                )
    try:
        resolved_file = cursor.resolve(strict=True)
        relative_resolved = resolved_file.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise ProgramBenchProbeError(f"{label}_path_escape") from error
    resolved_cursor = resolved_root
    for part in relative_resolved.parts[:-1]:
        resolved_cursor = resolved_cursor / part
        if _path_is_junction(resolved_cursor):
            raise ProgramBenchProbeError(f"{label}_path_junction_forbidden")
    if not resolved_file.is_file():
        raise ProgramBenchProbeError(f"{label}_file_missing")
    if executable and os.name != "nt" and not (
        stat.S_IMODE(resolved_file.stat().st_mode)
        & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    ):
        raise ProgramBenchProbeError(f"{label}_file_not_executable")
    return resolved_root, normalized


def _probe_case_row(case: ProbeCase) -> dict[str, Any]:
    return {
        "argv": list(case.argv),
        "stdin": case.stdin,
        "input_files": [
            {
                "path": item.path,
                "content_base64": base64.b64encode(item.content).decode("ascii"),
            }
            for item in case.input_files
        ],
        "env": dict(case.env),
    }


def _write_bridge_root(
    root: Path, *, bootstrap: str, request: Mapping[str, Any]
) -> None:
    root.joinpath("probe_runtime.py").write_bytes(Path(__file__).read_bytes())
    root.joinpath("launcher.py").write_text(bootstrap, encoding="utf-8", newline="\n")
    root.joinpath("request.json").write_text(
        json.dumps(request, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


def _bridge_mapping(payload: bytes, *, max_bytes: int) -> dict[str, Any]:
    if len(payload) > max_bytes:
        raise ProgramBenchProbeError("probe_bridge_json_too_large")

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ProgramBenchProbeError("probe_bridge_json_duplicate_key")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("ascii"), object_pairs_hook=pairs)
    except ProgramBenchProbeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ProgramBenchProbeError("probe_bridge_json_invalid") from error
    if not isinstance(value, dict):
        raise ProgramBenchProbeError("probe_bridge_json_not_object")
    return value


def _bridge_payload_from_outcome(outcome: Any, *, tail_budget: int) -> bytes:
    if getattr(outcome, "status", None) != "passed":
        raise ProgramBenchProbeError(
            "probe_bridge_executor_" + str(getattr(outcome, "status", "invalid"))
        )
    if getattr(outcome, "exit_code", None) != 0:
        raise ProgramBenchProbeError("probe_bridge_executor_nonzero_exit")
    stdout = getattr(outcome, "stdout_tail", None)
    stderr = getattr(outcome, "stderr_tail", None)
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        raise ProgramBenchProbeError("probe_bridge_executor_output_invalid")
    if stderr:
        raise ProgramBenchProbeError("probe_bridge_executor_stderr_not_empty")
    payload = stdout.encode("ascii", errors="strict")
    if len(payload) > tail_budget:
        raise ProgramBenchProbeError("probe_bridge_executor_tail_overflow")
    if getattr(outcome, "stdout_hash", None) != hashlib.sha256(payload).hexdigest():
        raise ProgramBenchProbeError("probe_bridge_executor_stdout_truncated")
    if getattr(outcome, "stderr_hash", None) != hashlib.sha256(b"").hexdigest():
        raise ProgramBenchProbeError("probe_bridge_executor_stderr_hash_invalid")
    return payload


def _bridge_infra(reason: str) -> dict[str, str]:
    return {"status": "infra_error", "reason": reason[:512]}


def _run_bridge_launcher(run: Any, /, **kwargs: Any) -> Any:
    """Run the trusted bridge with the image's preferred Python command.

    Language-specific ProgramBench images are only required to provide
    ``python3`` for the trusted bridge.  Retrying with the legacy ``python``
    spelling is limited to an outer launcher command failure: failures from the
    evaluated program are serialized by the launcher and therefore never
    trigger a second execution.
    """

    outcome = run(argv=("python3", "/workspace/launcher.py"), **kwargs)
    if getattr(outcome, "status", None) == "failed" and getattr(
        outcome, "exit_code", None
    ) in {126, 127}:
        outcome = run(argv=("python", "/workspace/launcher.py"), **kwargs)
    return outcome


class SandboxedCommandProbeExecutor:
    """Run one program through an existing digest-pinned untrusted container.

    The trusted launcher root is mounted read-only at ``/workspace`` and the
    program is exposed separately at ``/evaluator``. Candidate programs use a
    read-only bind. Reference programs set ``execute_only=True`` and require the
    command executor to provide a verified execute-only inode boundary (rather
    than treating a Windows read-only bind as unreadable). ``_execute_case``
    creates the case workspace under the container's ``/tmp``. Separate
    instances therefore never mount candidate and reference roots together.
    """

    def __init__(
        self,
        *,
        command_executor: Any,
        program_root: str | Path,
        execute_only: bool = False,
    ) -> None:
        self.command_executor = command_executor
        self.execute_only = bool(execute_only)
        self.bridge_budget = _validated_sandbox_executor(
            command_executor, require_execute_only=self.execute_only
        )
        root = Path(program_root)
        if root.is_symlink() or _path_or_parent_is_junction(root) or not root.is_dir():
            raise ValueError("probe_program_root_invalid")
        self.program_root = root.resolve()

    def execute(
        self,
        *,
        command: tuple[str, ...],
        case: ProbeCase,
        limits: ProbeLimits,
    ) -> Mapping[str, Any]:
        checked = _validate_command(command, label="sandbox_probe")
        program_root, executable = _trusted_relative_file(
            self.program_root,
            checked[0],
            label="probe_program",
            executable=True,
        )
        mounted_command = (f"/evaluator/{executable}", *checked[1:])
        request = {
            "bridge_budget": self.bridge_budget,
            "case": _probe_case_row(case),
            "command": list(mounted_command),
            "limits": asdict(limits),
        }
        try:
            with tempfile.TemporaryDirectory(prefix="programbench-probe-launcher-") as raw:
                launcher_root = Path(raw)
                _write_bridge_root(
                    launcher_root,
                    bootstrap=_PROBE_EXECUTION_BOOTSTRAP,
                    request=request,
                )
                if self.execute_only:
                    outcome = _run_bridge_launcher(
                        self.command_executor.run_with_execute_only_evaluator,
                        root=launcher_root,
                        evaluator_root=program_root,
                        executable_relative_path=executable,
                        timeout_seconds=limits.timeout_seconds + 10.0,
                    )
                else:
                    outcome = _run_bridge_launcher(
                        self.command_executor.run_with_read_only_evaluator,
                        root=launcher_root,
                        evaluator_root=program_root,
                        timeout_seconds=limits.timeout_seconds + 10.0,
                    )
            payload = _bridge_payload_from_outcome(
                outcome, tail_budget=self.bridge_budget
            )
            result = _bridge_mapping(payload, max_bytes=self.bridge_budget)
            return _canonical_execution_result(result, limits=limits)
        except Exception as error:
            reason = (
                str(error)
                if isinstance(error, ProgramBenchProbeError)
                else "probe_bridge_failed:" + type(error).__name__
            )
            return _bridge_infra(reason)


def capture_probe_document_in_sandbox(
    *,
    command_executor: Any,
    candidate_repo: str | Path,
    script_path: str,
    limits: ProbeLimits = ProbeLimits(),
) -> bytes:
    """Capture one explicitly selected agent script without mounting an oracle."""

    bridge_budget = _validated_sandbox_executor(command_executor)
    repo_root, script = _trusted_relative_file(
        candidate_repo,
        script_path,
        label="probe_definition",
        executable=False,
    )
    request = {
        "bridge_budget": bridge_budget,
        "limits": asdict(limits),
        "script": f"/evaluator/{script}",
    }
    try:
        with tempfile.TemporaryDirectory(prefix="programbench-probe-definition-") as raw:
            launcher_root = Path(raw)
            _write_bridge_root(
                launcher_root,
                bootstrap=_PROBE_DOCUMENT_BOOTSTRAP,
                request=request,
            )
            outcome = _run_bridge_launcher(
                command_executor.run_with_read_only_evaluator,
                root=launcher_root,
                evaluator_root=repo_root,
                timeout_seconds=limits.timeout_seconds + 10.0,
            )
        bridge = _bridge_mapping(
            _bridge_payload_from_outcome(outcome, tail_budget=bridge_budget),
            max_bytes=bridge_budget,
        )
    except Exception as error:
        reason = (
            str(error)
            if isinstance(error, ProgramBenchProbeError)
            else "probe_document_bridge_failed:" + type(error).__name__
        )
        raise ProgramBenchProbeError(
            "probe_document_capture_failed:" + reason
        ) from error
    status = bridge.get("status")
    if status == "definition_error":
        _exact_keys(bridge, {"status", "reason"}, label="probe_document_definition")
        reason = bridge["reason"]
        if not isinstance(reason, str) or not reason:
            raise ProgramBenchProbeError("probe_document_definition_reason_invalid")
        raise ProgramBenchProbeError(reason)
    if status == "infra_error":
        _exact_keys(bridge, {"status", "reason"}, label="probe_document_infra")
        reason = bridge["reason"]
        if not isinstance(reason, str) or not reason:
            raise ProgramBenchProbeError("probe_document_infra_reason_invalid")
        raise ProgramBenchProbeError(reason)
    if status != "ok":
        raise ProgramBenchProbeError("probe_document_bridge_status_invalid")
    _exact_keys(bridge, {"status", "payload_base64"}, label="probe_document_bridge")
    encoded = bridge["payload_base64"]
    if not isinstance(encoded, str):
        raise ProgramBenchProbeError("probe_document_payload_invalid")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, UnicodeEncodeError) as error:
        raise ProgramBenchProbeError("probe_document_payload_invalid") from error
    if not payload:
        raise ProgramBenchProbeError("probe_document_empty")
    load_probe_cases(payload, limits=limits)
    return payload


def run_public_probe_cases(
    cases: Sequence[ProbeCase],
    *,
    reference_command: Sequence[str],
    candidate_command: Sequence[str] | None,
    reference_executor: ProbeExecutor,
    candidate_executor: ProbeExecutor | None,
    probe_mode: PublicProbeMode = "differential",
    comparison_policy: PublicProbeComparisonPolicy = "exact_v1",
    limits: ProbeLimits = ProbeLimits(),
) -> dict[str, Any]:
    """Run cases through explicitly injected, isolated execution boundaries.

    ``reference_only`` is the exploration mode: it records bounded reference
    observations without validating, executing, or otherwise touching a
    candidate command.  The default remains the original differential path.

    The authoring path executes the reference **twice** per case and compares
    the two observations under the same ``comparison_policy`` the differential
    will use later.  A case whose reference does not reproduce cannot be
    satisfied by any candidate, so it is not an observation but a defect in the
    stimulus, and it is recorded as ``reference_nondeterministic`` instead of
    ``reference_observed``.  Finding that out here costs one extra sandbox
    execution per authored case; finding it out later cost a 336-tick run,
    because a nonzero public differential blocks the explicit transactional
    merge, so the mainline stayed at the starter and the arm could not be
    scored at all.  The differential path deliberately does not pay this cost:
    by then the corpus has already been vetted.
    """

    if len(cases) > limits.max_cases or any(not isinstance(case, ProbeCase) for case in cases):
        raise ProgramBenchProbeError("probe_cases_invalid")
    if probe_mode not in {"differential", "reference_only"}:
        raise ProgramBenchProbeError("probe_mode_invalid")
    if comparison_policy not in {"exact_v1", "go_standard_log_clock_v1"}:
        raise ProgramBenchProbeError("probe_comparison_policy_invalid")
    reference = _validate_command(reference_command, label="reference")
    candidate: tuple[str, ...] | None = None
    if probe_mode == "differential":
        if candidate_command is None or candidate_executor is None:
            raise ProgramBenchProbeError("candidate_boundary_required")
        candidate = _validate_command(candidate_command, label="candidate")
    reports: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        reference_result = _canonical_execution_result(
            reference_executor.execute(command=reference, case=case, limits=limits),
            limits=limits,
        )
        if reference_result["status"] == "infra_error":
            reports.append(
                {
                    "case_index": index,
                    "status": "infra_error",
                    "infra_side": "reference",
                    "reference": reference_result,
                    "candidate": {
                        "status": "skipped",
                        "reason": "reference_infrastructure_failure",
                    },
                    "matched": None,
                }
            )
            continue
        if probe_mode == "reference_only":
            repeat_result = _canonical_execution_result(
                reference_executor.execute(
                    command=reference, case=case, limits=limits
                ),
                limits=limits,
            )
            if repeat_result["status"] == "infra_error":
                reports.append(
                    {
                        "case_index": index,
                        "status": "infra_error",
                        "infra_side": "reference",
                        "reference": reference_result,
                        "candidate": {
                            "status": "skipped",
                            "reason": "reference_infrastructure_failure",
                        },
                        "matched": None,
                    }
                )
                continue
            reproducible = _execution_results_match(
                reference_result,
                repeat_result,
                comparison_policy=comparison_policy,
            )
            if not reproducible:
                reports.append(
                    {
                        "case_index": index,
                        "status": "reference_nondeterministic",
                        "infra_side": None,
                        "reference": reference_result,
                        "candidate": {
                            "status": "skipped",
                            "reason": "reference_nondeterministic",
                        },
                        "matched": None,
                        # Both observations are retained, so the claim "this
                        # stimulus is not reproducible" is auditable from the
                        # receipt rather than only assertable.
                        "reference_repeat": repeat_result,
                        "reference_divergent_fields": list(
                            _divergent_observation_fields(
                                reference_result,
                                repeat_result,
                                comparison_policy=comparison_policy,
                            )
                        ),
                    }
                )
                continue
            reports.append(
                {
                    "case_index": index,
                    "status": "reference_observed",
                    "infra_side": None,
                    "reference": reference_result,
                    "candidate": {
                        "status": "skipped",
                        "reason": "reference_only_mode",
                    },
                    "matched": None,
                }
            )
            continue
        assert candidate is not None
        assert candidate_executor is not None
        candidate_result = _canonical_execution_result(
            candidate_executor.execute(command=candidate, case=case, limits=limits),
            limits=limits,
        )
        if candidate_result["status"] == "infra_error":
            reports.append(
                {
                    "case_index": index,
                    "status": "infra_error",
                    "infra_side": "candidate",
                    "reference": reference_result,
                    "candidate": candidate_result,
                    "matched": None,
                }
            )
            continue
        reports.append(
            {
                "case_index": index,
                "status": "compared",
                "infra_side": None,
                "reference": reference_result,
                "candidate": candidate_result,
                "matched": _execution_results_match(
                    reference_result,
                    candidate_result,
                    comparison_policy=comparison_policy,
                ),
            }
        )
    compared = [row for row in reports if row["status"] == "compared"]
    return {
        "schema_version": "programbench_public_probe_report_v2",
        "probe_schema_version": PUBLIC_PROBE_SCHEMA_VERSION,
        "mode": probe_mode,
        "comparison_policy": comparison_policy,
        "case_count": len(reports),
        "compared_case_count": len(compared),
        "reference_observed_case_count": sum(
            row["status"] == "reference_observed" for row in reports
        ),
        "reference_nondeterministic_case_count": sum(
            row["status"] == "reference_nondeterministic" for row in reports
        ),
        "matched_case_count": sum(row["matched"] is True for row in compared),
        "mismatched_case_count": sum(row["matched"] is False for row in compared),
        "infra_error_count": sum(
            row["status"] == "infra_error" for row in reports
        ),
        "cases": reports,
    }


def evaluate_public_probes(
    *,
    probe_document: bytes | str,
    reference_command: Sequence[str],
    candidate_command: Sequence[str] | None = None,
    reference_executor: ProbeExecutor,
    candidate_executor: ProbeExecutor | None = None,
    probe_mode: PublicProbeMode = "differential",
    comparison_policy: PublicProbeComparisonPolicy = "exact_v1",
    limits: ProbeLimits = ProbeLimits(),
) -> dict[str, Any]:
    """Validate declarative JSON, then evaluate it across injected sandboxes.

    This API never executes an eval script. If an agent-authored ``eval/*.py``
    emits the document, that script must run in an untrusted candidate sandbox
    with no readable reference/oracle root before its stdout crosses this API.
    """

    cases = load_probe_cases(probe_document, limits=limits)
    return run_public_probe_cases(
        cases,
        reference_command=reference_command,
        candidate_command=candidate_command,
        reference_executor=reference_executor,
        candidate_executor=candidate_executor,
        probe_mode=probe_mode,
        comparison_policy=comparison_policy,
        limits=limits,
    )


__all__ = [
    "PUBLIC_PROBE_ENV_ALLOWLIST",
    "PUBLIC_PROBE_RUNNER",
    "PUBLIC_PROBE_SCHEMA_VERSION",
    "LocalSubprocessProbeExecutor",
    "ProbeCase",
    "ProbeExecutor",
    "ProbeInputFile",
    "ProbeLimits",
    "ProgramBenchProbeError",
    "PublicProbeMode",
    "SandboxedCommandProbeExecutor",
    "capture_probe_document_in_sandbox",
    "evaluate_public_probes",
    "load_probe_cases",
    "public_probe_schema",
    "run_public_probe_cases",
]
