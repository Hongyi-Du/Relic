"""Build a bounded public source surface from a Cooper image.

CooperBench supplies a public feature description and a task image.  It does
not give agents the feature tests or gold patches. The feature-independent
policy preserves the complete bounded tracked baseline and runtime assets.
Private descriptions affect only their owner's host-side target map, never
the physical baseline or test sample. Evaluator assets never enter the pack.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


SURFACE_SCHEMA_VERSION = "orgenv_cooperbench_public_surface_v9"
FEATURE_INDEPENDENT_SOURCE_POLICY = "feature_independent"
LEGACY_SOURCE_POLICY = "task_relevant_union"
_MAX_EDITABLE_FILE_BYTES = 200_000
_TEXT_CONTEXT_FILE_BYTES = 512_000
_TARGETS_PER_FEATURE = 12
_CONTEXT_PER_FEATURE = 48

_CODE_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cs",
        ".go",
        ".h",
        ".hpp",
        ".java",
        ".js",
        ".jsx",
        ".kt",
        ".lua",
        ".php",
        ".py",
        ".rb",
        ".rs",
        ".scala",
        ".sh",
        ".sql",
        ".swift",
        ".ts",
        ".tsx",
        ".vue",
    }
)
_ROOT_CONTEXT_NAMES = frozenset(
    {
        "cargo.toml",
        "go.mod",
        "go.sum",
        "makefile",
        "package.json",
        "pnpm-workspace.yaml",
        "pyproject.toml",
        "requirements.txt",
        "setup.cfg",
        "setup.py",
        "tsconfig.json",
    }
)
_STOP_WORDS = frozenset(
    {
        "about",
        "after",
        "agent",
        "allow",
        "also",
        "before",
        "being",
        "change",
        "could",
        "feature",
        "from",
        "have",
        "implement",
        "into",
        "make",
        "must",
        "need",
        "should",
        "support",
        "task",
        "than",
        "that",
        "their",
        "there",
        "these",
        "this",
        "using",
        "when",
        "where",
        "which",
        "with",
        "would",
    }
)
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")
_TEST_SEGMENT_RE = re.compile(r"(^|/)(tests?|__tests__|spec|testdata)(/|$)", re.I)
_TEST_FILE_RE = re.compile(
    r"(^|/)(test_[^/]+|[^/]+_test\.[A-Za-z0-9]+|[^/]+\.(test|spec)\.[A-Za-z0-9]+)$",
    re.I,
)
_DECLARED_FILE_SECTION_RE = re.compile(
    r"\bfiles?\s+(?:modified|added|created|changed|to\s+modify)\b",
    re.I,
)
_CODE_PATH_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])([A-Za-z0-9_.+-]+(?:[/\\][A-Za-z0-9_.+-]+)*)"
)
_PRIVATE_PATH_PARTS = frozenset(
    {"hidden", "private", "gold", "solution", "solutions"}
)


class SurfaceBuildError(RuntimeError):
    """The public task surface could not be built without guessing broadly."""


@dataclass(frozen=True)
class SurfaceFile:
    path: str
    size_bytes: int
    sha256: str
    score_by_agent: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "score_by_agent": dict(self.score_by_agent),
        }


@dataclass(frozen=True)
class SurfacePlan:
    schema_version: str
    files: tuple[SurfaceFile, ...]
    component_map: Mapping[str, tuple[str, ...]]
    public_test_paths: tuple[str, ...]
    public_test_command: tuple[str, ...]
    task_digests: Mapping[str, str]
    total_bytes: int
    tracked_file_count: int
    skipped_binary_or_non_utf8: int
    skipped_oversize: int
    source_policy: str = LEGACY_SOURCE_POLICY
    runtime_asset_paths: tuple[str, ...] = ()
    runtime_assets_manifest: Mapping[str, Any] | None = None

    @property
    def selected_paths(self) -> tuple[str, ...]:
        return tuple(item.path for item in self.files)

    @property
    def digest(self) -> str:
        payload = self.to_dict(include_digest=False)
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    @property
    def baseline_digest(self) -> str:
        """Public physical input identity excludes private task/routing scores."""
        if self.source_policy == LEGACY_SOURCE_POLICY:
            return self.digest
        material = {
            "policy": self.source_policy,
            "files": [{"path": row.path, "size_bytes": row.size_bytes, "sha256": row.sha256}
                      for row in self.files],
            "runtime_assets": self.runtime_assets_manifest,
            "public_test_paths": self.public_test_paths,
            "public_test_command": self.public_test_command,
        }
        return hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "files": [item.to_dict() for item in self.files],
            "component_map": {
                key: list(value) for key, value in self.component_map.items()
            },
            "public_test_paths": list(self.public_test_paths),
            "public_test_command": list(self.public_test_command),
            "task_digests": dict(self.task_digests),
            "total_bytes": self.total_bytes,
            "tracked_file_count": self.tracked_file_count,
            "skipped_binary_or_non_utf8": self.skipped_binary_or_non_utf8,
            "skipped_oversize": self.skipped_oversize,
        }
        if include_digest:
            payload["surface_digest"] = self.digest
        if self.source_policy != LEGACY_SOURCE_POLICY:
            payload.update(source_policy=self.source_policy,
                           runtime_asset_paths=list(self.runtime_asset_paths),
                           runtime_assets_manifest=self.runtime_assets_manifest)
        return payload


def _tracked_paths(repo_root: Path) -> list[str]:
    raw = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo_root}",
            "-C",
            str(repo_root),
            "ls-files",
            "-z",
        ],
        capture_output=True,
        check=False,
    )
    if raw.returncode != 0:
        detail = raw.stderr.decode("utf-8", errors="replace")[:500]
        raise SurfaceBuildError(f"git_ls_files_failed:{detail}")
    decoded = raw.stdout.decode("utf-8", errors="strict")
    paths = [item for item in decoded.split("\0") if item]
    if not paths:
        raise SurfaceBuildError("task_image_repository_has_no_tracked_files")
    return paths


def _safe_file(repo_root: Path, rel_path: str) -> Path | None:
    pure = PurePosixPath(rel_path)
    if pure.is_absolute() or ".." in pure.parts:
        raise SurfaceBuildError(f"unsafe_tracked_path:{rel_path}")
    candidate = repo_root.joinpath(*pure.parts)
    try:
        metadata = candidate.lstat()
    except OSError:
        return None
    if not stat.S_ISREG(metadata.st_mode) or candidate.is_symlink():
        return None
    try:
        candidate.resolve().relative_to(repo_root.resolve())
    except (OSError, ValueError) as error:
        raise SurfaceBuildError(f"tracked_path_escapes_repository:{rel_path}") from error
    return candidate


def _untracked_python_extension_assets(
    repo_root: Path, tracked_paths: list[str]
) -> list[str]:
    """Keep built extension modules beside public Python packages runnable.

    Cooper task images may build a native module after checkout, so the file
    is absent from ``git ls-files`` even though the untouched public tests
    import it from the repository package.  Admit only regular native-module
    files immediately beside a tracked package ``__init__.py``; other
    untracked image contents remain outside the public surface.
    """

    tracked = set(tracked_paths)
    package_dirs = {
        PurePosixPath(path).parent
        for path in tracked_paths
        if PurePosixPath(path).name == "__init__.py"
        and not {
            part.casefold() for part in PurePosixPath(path).parts
        }.intersection(_PRIVATE_PATH_PARTS)
    }
    extensions: list[str] = []
    for package_dir in sorted(package_dirs, key=str):
        directory = repo_root.joinpath(*package_dir.parts)
        try:
            children = sorted(directory.iterdir(), key=lambda path: path.name)
        except OSError:
            continue
        for child in children:
            name = child.name.casefold()
            if not (name.endswith(".so") or name.endswith(".pyd")):
                continue
            relative = child.relative_to(repo_root).as_posix()
            if relative in tracked or _safe_file(repo_root, relative) is None:
                continue
            extensions.append(relative)
    return extensions


def _setuptools_scm_generated_python_assets(
    repo_root: Path, tracked_paths: list[str]
) -> list[str]:
    """Keep the generated version module declared by public build metadata."""

    if "pyproject.toml" not in tracked_paths:
        return []
    try:
        config = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return []
    write_to = (
        ((config.get("tool") or {}).get("setuptools_scm") or {}).get("write_to")
    )
    if not isinstance(write_to, str):
        return []
    pure = PurePosixPath(write_to)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or pure.suffix.casefold() != ".py"
        or {part.casefold() for part in pure.parts}.intersection(_PRIVATE_PATH_PARTS)
    ):
        return []
    relative = pure.as_posix()
    if relative in tracked_paths or _safe_file(repo_root, relative) is None:
        return []
    return [relative]


def _tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_RE.findall(text)
        if token.casefold() not in _STOP_WORDS
    }


def _is_test_path(path: str) -> bool:
    return bool(_TEST_SEGMENT_RE.search(path) or _TEST_FILE_RE.search(path))


def _is_public_test_path(path: str) -> bool:
    folded_parts = {part.casefold() for part in PurePosixPath(path).parts}
    pure = PurePosixPath(path)
    if pure.suffix.casefold() == ".py" and not (
        pure.name.startswith("test_") or pure.name.endswith("_test.py")
    ):
        # Helpers and initializers belong in the materialized support surface,
        # not in the eight executable test-module slots.
        return False
    return _is_test_path(path) and not folded_parts.intersection(
        _PRIVATE_PATH_PARTS
    )


def _declared_code_paths(task_text: str, *, file_sections_only: bool = False) -> tuple[str, ...]:
    """Return safe implementation paths explicitly named by the public brief.

    Existing source selection already prefers tracked paths whose complete name
    appears in the request. A feature may also publicly declare files that it
    expects the implementation to create, so those paths have no tracked file
    to match. Admit only code-like paths from a Files Modified/Added/Created
    section or an exact Markdown code span; never infer names from prose tokens.
    """

    from environments.org_env.product.repo_paths import (
        InvalidRepoPath,
        normalize_repo_relative_path,
    )

    text = str(task_text or "").replace("\\", "/")
    candidate_fragments = [] if file_sections_only else re.findall(r"`([^`\r\n]{1,500})`", text)
    in_file_section = False
    for line in text.splitlines():
        if _DECLARED_FILE_SECTION_RE.search(line):
            in_file_section = True
        elif in_file_section and re.match(r"^\s*#{1,6}\s+", line):
            in_file_section = False
        if in_file_section:
            candidate_fragments.append(line)

    paths: list[str] = []
    seen: set[str] = set()
    for fragment in candidate_fragments:
        for match in _CODE_PATH_TOKEN_RE.finditer(fragment):
            raw = match.group(1).replace("\\", "/")
            try:
                path = normalize_repo_relative_path(raw)
            except InvalidRepoPath:
                continue
            pure = PurePosixPath(path)
            if (
                pure.suffix.casefold() not in _CODE_SUFFIXES
                or _is_test_path(path)
                or {part.casefold() for part in pure.parts}
                & _PRIVATE_PATH_PARTS
            ):
                continue
            folded = path.casefold()
            if folded not in seen:
                seen.add(folded)
                paths.append(path)
    return tuple(paths)


def _resolve_declared_code_path(
    path: str, known_paths: Mapping[str, str], *, declared_paths: tuple[str, ...] = (),
    explicit_file_paths: tuple[str, ...] = (),
) -> str:
    """Resolve public abbreviations only against explicit, existing paths."""

    exact = known_paths.get(path.casefold())
    if exact is not None:
        return exact
    if "/" not in path and path not in explicit_file_paths:
        # A prose reference to `types.py` is not a request to create another
        # root file when the same public brief names its full existing path.
        # Do not guess from repository-wide basenames or ambiguous matches.
        matches = {
            known_paths[candidate.casefold()]
            for candidate in declared_paths
            if candidate.casefold().endswith("/" + path.casefold())
            and candidate.casefold() in known_paths
        }
        if len(matches) == 1:
            return next(iter(matches))
    if re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\.py", path):
        module_path = path[:-3].replace(".", "/") + ".py"
        return known_paths.get(module_path.casefold(), path)
    return path


def _record_text(
    records: Mapping[str, tuple[bytes, str, Mapping[str, int]]],
    wanted_path: str,
) -> str | None:
    wanted = wanted_path.casefold()
    for path, record in records.items():
        if path.casefold() == wanted:
            return record[1]
    return None


def _public_test_command(
    paths: list[str],
    selected_paths: set[str],
    records: Mapping[str, tuple[bytes, str, Mapping[str, int]]],
) -> tuple[str, ...]:
    python_tests = [path for path in paths if PurePosixPath(path).suffix.casefold() == ".py"]
    if python_tests:
        return ("python", "-m", "pytest", "-q", *python_tests)
    suffixes = {
        PurePosixPath(path).suffix.casefold() for path in selected_paths
    }
    folded_paths = {path.casefold() for path in selected_paths}
    if "go.mod" in folded_paths and ".go" in suffixes:
        return ("go", "test", "-p=1", "./...")
    if "cargo.toml" in folded_paths and ".rs" in suffixes:
        test_packages: list[tuple[str, str]] = []
        for path in selected_paths:
            if path.casefold() == "cargo.toml" or not path.casefold().endswith(
                "/cargo.toml"
            ):
                continue
            content = _record_text(records, path)
            if content is None:
                continue
            try:
                package = tomllib.loads(content).get("package") or {}
            except (tomllib.TOMLDecodeError, TypeError):
                continue
            name = str(package.get("name") or "").strip()
            if name and (
                name.casefold().endswith("-tests")
                or PurePosixPath(path).parent.name.casefold() == "tests"
            ):
                test_packages.append((path, name))
        if test_packages:
            package = sorted(test_packages, key=lambda item: item[0])[0][1]
            return ("cargo", "test", "-p", package, "--no-run", "-j", "1")
        return ("cargo", "test", "--workspace", "--no-run", "-j", "1")
    package_json = _record_text(records, "package.json")
    if package_json is not None and suffixes.intersection(
        {".js", ".jsx", ".ts", ".tsx", ".vue"}
    ):
        try:
            package = json.loads(package_json)
        except json.JSONDecodeError:
            package = {}
        scripts = package.get("scripts") if isinstance(package, dict) else {}
        if isinstance(scripts, dict):
            script = next(
                (
                    name
                    for name in ("test", "typecheck", "type", "check")
                    if isinstance(scripts.get(name), str)
                    and str(scripts[name]).strip()
                ),
                None,
            )
            if script:
                declared_manager = str(package.get("packageManager") or "").split(
                    "@", 1
                )[0]
                manager = (
                    declared_manager
                    if declared_manager in {"npm", "pnpm", "yarn"}
                    else "pnpm"
                    if "pnpm-lock.yaml" in folded_paths
                    else "yarn"
                    if "yarn.lock" in folded_paths
                    else "npm"
                )
                return (manager, "run", script)
    if ".py" in suffixes:
        return ("python", "-m", "compileall", "-q", ".")
    return ()


def public_validation_capability(
    command: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Classify what a successful native command actually demonstrates."""

    raw = [str(item) for item in command]
    folded = [item.casefold() for item in raw]
    if folded[:4] == ["python", "-m", "pytest", "-q"]:
        return {
            "functional_public_regression_available": True,
            "public_validation_strength": "public_regression",
        }
    if len(folded) >= 2 and folded[:2] == ["go", "test"]:
        return {
            "functional_public_regression_available": True,
            "public_validation_strength": "public_regression",
        }
    if len(folded) >= 2 and folded[:2] == ["cargo", "test"]:
        functional = "--no-run" not in folded
        return {
            "functional_public_regression_available": functional,
            "public_validation_strength": (
                "public_regression" if functional else "compile_only"
            ),
        }
    if (
        len(folded) >= 3
        and folded[0] in {"npm", "pnpm", "yarn"}
        and folded[1] == "run"
    ):
        functional = folded[2] == "test"
        return {
            "functional_public_regression_available": functional,
            "public_validation_strength": (
                "public_regression" if functional else "typecheck_only"
            ),
        }
    if folded[:3] == ["python", "-m", "compileall"]:
        return {
            "functional_public_regression_available": False,
            "public_validation_strength": "syntax_only",
        }
    return {
        "functional_public_regression_available": False,
        "public_validation_strength": (
            "build_only" if raw else "source_snapshot_integrity"
        ),
    }


def _is_editable_path(path: str, size_bytes: int) -> bool:
    pure = PurePosixPath(path)
    return (
        size_bytes <= _MAX_EDITABLE_FILE_BYTES
        and pure.suffix.casefold() in _CODE_SUFFIXES
        and not _is_test_path(path)
    )


def _score(path: str, content: str, task_text: str, task_tokens: set[str]) -> int:
    path_folded = path.casefold()
    task_folded = task_text.casefold()
    pure = PurePosixPath(path)
    basename = pure.name.casefold()
    stem = pure.stem.casefold()
    score = 0
    if path_folded in task_folded:
        score += 20_000
    if basename and basename in task_folded:
        score += 1_000
    if stem and len(stem) >= 4 and stem in task_tokens:
        score += 400
    path_tokens = _tokens(path_folded)
    score += 80 * len(path_tokens & task_tokens)
    content_tokens = _tokens(content)
    overlap = content_tokens & task_tokens
    score += 3 * min(40, len(overlap))
    for token in sorted(overlap, key=len, reverse=True)[:12]:
        if re.search(
            rf"\b(class|def|func|function|interface|struct|trait|type)\s+{re.escape(token)}\b",
            content,
            re.I,
        ):
            score += 120
    if _is_test_path(path):
        score -= 20
    return max(0, score)


def select_public_surface(
    repo_root: str | os.PathLike[str],
    tasks: Mapping[str, str],
    *,
    max_files: int,
    max_total_bytes: int,
    source_policy: str = LEGACY_SOURCE_POLICY,
) -> SurfacePlan:
    """Build a fixed baseline, or the explicitly requested legacy task union."""

    root = Path(repo_root).resolve()
    if not (root / ".git").exists():
        raise SurfaceBuildError("task_image_repository_git_metadata_required")
    if len(tasks) != 2 or any(not str(value).strip() for value in tasks.values()):
        raise SurfaceBuildError("exactly_two_public_tasks_required")
    if max_files < 24 or max_total_bytes < 256 * 1024:
        raise SurfaceBuildError("surface_limits_too_small")
    if source_policy not in {LEGACY_SOURCE_POLICY, FEATURE_INDEPENDENT_SOURCE_POLICY}:
        raise SurfaceBuildError("unknown_public_source_policy")
    independent = source_policy == FEATURE_INDEPENDENT_SOURCE_POLICY

    tracked = _tracked_paths(root)
    task_tokens = {agent: _tokens(text) for agent, text in tasks.items()}
    records: dict[str, tuple[bytes, str, dict[str, int]]] = {}
    skipped_binary = 0
    skipped_oversize = 0
    runtime_asset_paths = (
        [
            *_untracked_python_extension_assets(root, tracked),
            *_setuptools_scm_generated_python_assets(root, tracked),
        ]
        if independent
        else []
    )
    physical_bytes = sum(
        root.joinpath(*PurePosixPath(path).parts).stat().st_size
        for path in runtime_asset_paths
    )
    for path in tracked:
        source = _safe_file(root, path)
        if source is None:
            # Gitlinks are tracked entries but not regular files in an
            # uninitialized task-image checkout (Pillow ships one). They have
            # no bytes to place in a public baseline; account for and omit the
            # entry rather than making the entire repository unrunnable.
            skipped_binary += 1
            continue
        size = source.stat().st_size
        physical_bytes += size
        if size > _TEXT_CONTEXT_FILE_BYTES:
            skipped_oversize += 1
            if independent:
                runtime_asset_paths.append(path)
            continue
        raw = source.read_bytes()
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            skipped_binary += 1
            if independent:
                runtime_asset_paths.append(path)
            continue
        if independent and b"\x00" in raw:
            skipped_binary += 1
            runtime_asset_paths.append(path)
            continue
        scores = {
            agent: _score(path, content, tasks[agent], task_tokens[agent])
            for agent in tasks
        }
        records[path] = (raw, content, scores)

    if not records:
        raise SurfaceBuildError("task_image_has_no_bounded_utf8_source_surface")
    if independent and (len(tracked) > max_files or physical_bytes > max_total_bytes):
        raise SurfaceBuildError(
            f"public_baseline_capacity_exceeded:files={len(tracked)}/{max_files}:bytes={physical_bytes}/{max_total_bytes}"
        )

    component_map: dict[str, tuple[str, ...]] = {}
    required: set[str] = set()
    ranked_context: list[str] = []
    for index, agent in enumerate(tasks, start=1):
        ranked = sorted(
            records,
            key=lambda path: (
                -records[path][2][agent],
                len(path),
                path,
            ),
        )
        relevant = [path for path in ranked if records[path][2][agent] > 0]
        editable = [
            path
            for path in relevant
            if _is_editable_path(path, len(records[path][0]))
        ]
        public_declared = _declared_code_paths(tasks[agent])
        explicit_file_paths = _declared_code_paths(tasks[agent], file_sections_only=True)
        tracked_by_folded = {path.casefold(): path for path in tracked}
        tracked_exact = set(tracked)
        declared = [
            _resolve_declared_code_path(path, tracked_by_folded, declared_paths=public_declared,
                                        explicit_file_paths=explicit_file_paths)
            for path in public_declared
        ]
        declared_targets = [
            path
            for path in declared
            if path not in tracked_exact or path in editable
        ]
        if not editable and not declared_targets:
            raise SurfaceBuildError(f"no_task_relevant_editable_path:{agent}")
        # A public feature description often names the intended repository
        # path explicitly (for example in a "Files Modified" section).  That
        # is stronger task evidence than token overlap with a dozen nearby
        # modules.  Keep those named tracked/editable paths as the feature's
        # implementation boundary; only fall back to relevance ranking when
        # the user-facing task does not name a path.  Context selection below
        # remains broad enough for the editor to understand surrounding APIs.
        normalized_task = str(tasks[agent]).replace("\\", "/").casefold()
        explicit_paths = [
            path
            for path in editable
            if path.casefold() in normalized_task
        ]
        ordered_targets = list(
            dict.fromkeys([*declared_targets, *explicit_paths])
        )
        if len(ordered_targets) > _TARGETS_PER_FEATURE:
            raise SurfaceBuildError(
                f"declared_feature_paths_exceed_limit:{agent}:"
                f"{len(ordered_targets)}/{_TARGETS_PER_FEATURE}"
            )
        targets = tuple(ordered_targets or editable[:_TARGETS_PER_FEATURE])
        component_map[f"cooper_feature_{index}"] = targets
        required.update(targets)
        ranked_context.extend(relevant[:_CONTEXT_PER_FEATURE])

    if independent:
        # A fixed repository smoke sample, never ranked by either private task.
        # Feature correctness is checked separately by owner-authorized probes.
        # Do not fall back to a different sample when a baseline test fails.
        public_tests = sorted(
            (path for path in records if _is_public_test_path(path)
             and PurePosixPath(path).suffix.casefold() in _CODE_SUFFIXES),
            key=lambda path: (len(PurePosixPath(path).parts), path),
        )[:8]
        selected = tuple(sorted(records))
        from .runtime_assets import collect_runtime_assets
        assets = collect_runtime_assets(
            root, sorted(runtime_asset_paths), source_paths=selected,
            max_files=max_files, max_total_bytes=max_total_bytes,
        )
        return SurfacePlan(
            schema_version=SURFACE_SCHEMA_VERSION,
            files=tuple(SurfaceFile(path, len(records[path][0]), hashlib.sha256(records[path][0]).hexdigest(), {})
                        for path in selected),
            component_map=component_map, public_test_paths=tuple(public_tests),
            public_test_command=_public_test_command(public_tests, set(selected), records),
            task_digests={agent: hashlib.sha256(text.encode()).hexdigest() for agent, text in tasks.items()},
            total_bytes=sum(len(records[path][0]) for path in selected), tracked_file_count=len(tracked),
            skipped_binary_or_non_utf8=skipped_binary, skipped_oversize=skipped_oversize,
            source_policy=source_policy, runtime_asset_paths=tuple(sorted(runtime_asset_paths)),
            runtime_assets_manifest=assets.manifest(),
        )

    # The task image's tracked tests are public repository evidence. Select a
    # small task-relevant subset, never evaluator-only/gold paths, and make it a
    # required part of the generated pack so internal CI observes behavior
    # rather than accepting any syntactically valid edit.
    target_stems = {
        PurePosixPath(path).stem.casefold()
        for paths in component_map.values()
        for path in paths
        if len(PurePosixPath(path).stem) >= 4
        and not PurePosixPath(path).stem.startswith("__")
    }

    def test_matches_target(path: str) -> bool:
        content = records[path][1].casefold()
        folded_path = path.casefold()
        return any(stem in folded_path or stem in content for stem in target_stems)

    eligible_tests = [
        path
        for path in records
        if _is_public_test_path(path)
        and (max(records[path][2].values()) > 0 or test_matches_target(path))
    ]
    direct_tests = [
        path for path in eligible_tests
        if any(PurePosixPath(path).stem.casefold() in {f"test_{stem}", f"{stem}_test"}
               for stem in target_stems)
    ]
    # Once the repository supplies tests for the named component, those are
    # stronger evidence than broad token matches in unrelated integration tests.
    public_test_paths = sorted(
        direct_tests or eligible_tests,
        key=lambda path: (
            not any(stem in PurePosixPath(path).stem.casefold() for stem in target_stems),
            not test_matches_target(path),
            -max(records[path][2].values()),
            len(path),
            path,
        ),
    )[:8]
    required.update(public_test_paths)
    # Pytest discovers conftest.py and package initializers outside the named
    # test module. Keep tracked ancestors in the same bounded public surface so
    # a harness import failure is not manufactured by the adapter itself.
    for test_path in public_test_paths:
        parent = PurePosixPath(test_path).parent
        for directory in (PurePosixPath("."), *parent.parents, parent):
            prefix = "" if str(directory) == "." else directory.as_posix() + "/"
            for support_name in ("conftest.py", "__init__.py"):
                support_path = prefix + support_name
                if support_path in records:
                    required.add(support_path)

    root_context = [
        path
        for path in records
        if "/" not in path and PurePosixPath(path).name.casefold() in _ROOT_CONTEXT_NAMES
    ]
    if len(required) > max_files:
        raise SurfaceBuildError("required_task_surface_exceeds_file_limit")
    ordered_candidates: list[str] = []
    seen: set[str] = set()
    for path in [*sorted(required), *ranked_context, *sorted(root_context)]:
        if path not in seen:
            seen.add(path)
            ordered_candidates.append(path)
    for path in sorted(
        records,
        key=lambda item: (-max(records[item][2].values()), item),
    ):
        if max(records[path][2].values()) <= 0 or path in seen:
            continue
        seen.add(path)
        ordered_candidates.append(path)

    chosen: list[SurfaceFile] = []
    total_bytes = 0
    for path in ordered_candidates:
        raw, _content, scores = records[path]
        required_path = path in required
        if len(chosen) >= max_files and not required_path:
            continue
        if total_bytes + len(raw) > max_total_bytes and not required_path:
            continue
        if total_bytes + len(raw) > max_total_bytes:
            raise SurfaceBuildError("required_task_surface_exceeds_byte_limit")
        chosen.append(
            SurfaceFile(
                path=path,
                size_bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                score_by_agent=dict(scores),
            )
        )
        total_bytes += len(raw)
        if len(chosen) >= max_files and required.issubset(
            {item.path for item in chosen}
        ):
            break

    selected = {item.path for item in chosen}
    if not required.issubset(selected):
        raise SurfaceBuildError("required_task_surface_truncated")
    return SurfacePlan(
        schema_version=SURFACE_SCHEMA_VERSION,
        files=tuple(chosen),
        component_map=component_map,
        public_test_paths=tuple(path for path in public_test_paths if path in selected),
        public_test_command=_public_test_command(public_test_paths, selected, records),
        task_digests={
            agent: hashlib.sha256(tasks[agent].encode("utf-8")).hexdigest()
            for agent in tasks
        },
        total_bytes=total_bytes,
        tracked_file_count=len(tracked),
        skipped_binary_or_non_utf8=skipped_binary,
        skipped_oversize=skipped_oversize,
    )


def materialize_public_pack(
    repo_root: str | os.PathLike[str],
    dataset_dir: str | os.PathLike[str],
    tasks: Mapping[str, str],
    plan: SurfacePlan,
    *,
    project_id: str,
    private_features: bool = False,
) -> Path:
    """Write starter source plus public issue text or strict metadata only."""

    source_root = Path(repo_root).resolve()
    target_root = Path(dataset_dir).resolve()
    starter = target_root / "starter_repo"
    issues = target_root / "issues" / "public"
    starter.mkdir(parents=True, exist_ok=False)
    issues.mkdir(parents=True, exist_ok=True)
    selected = set(plan.selected_paths)
    if plan.source_policy == FEATURE_INDEPENDENT_SOURCE_POLICY:
        from .runtime_assets import collect_runtime_assets, write_runtime_assets
        assets = collect_runtime_assets(source_root, plan.runtime_asset_paths, source_paths=plan.selected_paths)
        if assets.manifest() != plan.runtime_assets_manifest:
            raise SurfaceBuildError("public_runtime_assets_changed_after_selection")
        write_runtime_assets(assets, target_root)
    source_rows = {item.path: item for item in plan.files}
    for path in plan.selected_paths:
        source = _safe_file(source_root, path)
        if source is None:
            raise SurfaceBuildError(f"selected_surface_file_disappeared:{path}")
        if plan.source_policy == FEATURE_INDEPENDENT_SOURCE_POLICY:
            row = source_rows[path]
            if source.stat().st_size != row.size_bytes or hashlib.sha256(source.read_bytes()).hexdigest() != row.sha256:
                raise SurfaceBuildError(f"public_baseline_source_changed_after_selection:{path}")
        destination = starter.joinpath(*PurePosixPath(path).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination, follow_symlinks=False)

    validation_capability = public_validation_capability(
        plan.public_test_command
    )
    manifest = {
        "schema_version": "org_env_oss_time_machine_pack_v2",
        "project_id": project_id,
        "source_project_name": "CooperBench task image",
        "anonymized_product_name": "CooperBench two-feature task",
        "product_summary": (
            "Two CooperBench feature requests assigned to one two-member "
            "B3 organization. CooperBench remains the authoritative evaluator."
        ),
        "agent_visible": {
            "starter_repo": "starter_repo",
            "public_issues": "issues/public",
        },
        # CooperBench's private feature evaluator stays outside the pack. The
        # tracked public repository tests are a legitimate feedback channel and
        # become the internal merge/freeze gate when a relevant subset exists.
        "entrypoints": {
            "smoke": {
                "command": list(plan.public_test_command),
                "scope": (
                    "selected_public_regression_tests"
                    if validation_capability[
                        "functional_public_regression_available"
                    ]
                    else "selected_public_validation"
                    if plan.public_test_command
                    else "selected_public_source_surface"
                ),
            }
        },
        "public_tests": {
            "command": list(plan.public_test_command),
            "files": list(plan.public_test_paths),
            "dependencies": [],
        },
        "component_map": {} if private_features else {
            key: [path for path in paths if path in selected]
            for key, paths in plan.component_map.items()
        },
        "issue_stream": {"first_tick": 0},
        "cooperbench": {
            "schema_version": SURFACE_SCHEMA_VERSION,
            "surface_digest": plan.baseline_digest,
            "source_policy": plan.source_policy,
            "text_encoding_policy": "utf8_preserve_newlines" if plan.source_policy == FEATURE_INDEPENDENT_SOURCE_POLICY else "legacy_text",
            "feature_count": 2,
            "feature_visibility": "owner_private_until_explicit_share" if private_features else "legacy_public",
            "evaluator_assets_in_pack": False,
            "internal_smoke_is_functional_evaluator": False,
            "public_validation_strength": validation_capability[
                "public_validation_strength"
            ],
        },
    }
    if plan.runtime_assets_manifest is not None:
        manifest["cooperbench"]["runtime_assets"] = plan.runtime_assets_manifest
    if plan.source_policy == FEATURE_INDEPENDENT_SOURCE_POLICY:
        manifest["cooperbench"]["public_text_files"] = [
            {"path": row.path, "size_bytes": row.size_bytes, "sha256": row.sha256}
            for row in plan.files
        ]
    (target_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for index, (agent, task) in enumerate(tasks.items(), start=1):
        component = f"cooper_feature_{index}"
        record = {
            "issue_id": component,
            "title": f"CooperBench feature {index}",
            "body": "Feature brief assigned privately to its owner." if private_features else task,
            "text_rewritten": "Feature brief assigned privately to its owner." if private_features else task,
            "issue_type": "feature",
            "severity": "major",
            "component": component,
            "candidate_path_hints": [] if private_features else list(plan.component_map[component]),
            "created_tick": 0,
            "source": "cooperbench_private_feature_metadata" if private_features else "cooperbench_public_feature",
            "hidden": False,
            "labels": ["cooperbench", agent],
        }
        (issues / f"{component}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return target_root


__all__ = [
    "SURFACE_SCHEMA_VERSION",
    "FEATURE_INDEPENDENT_SOURCE_POLICY",
    "LEGACY_SOURCE_POLICY",
    "SurfaceBuildError",
    "SurfaceFile",
    "SurfacePlan",
    "materialize_public_pack",
    "public_validation_capability",
    "select_public_surface",
]
