"""Withheld contract: an atomic write accepts a filesystem path object as its target.

Only the public atomic-write entry point is used.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from boltons.fileutils import atomic_save


def test_a_path_object_names_the_destination() -> None:
    with tempfile.TemporaryDirectory() as base:
        target = Path(base) / "report.bin"

        with atomic_save(target) as handle:
            handle.write(b"payload")

        assert target.read_bytes() == b"payload"
        assert sorted(os.listdir(base)) == ["report.bin"]


def test_path_objects_work_in_text_mode_and_when_replacing_a_file() -> None:
    with tempfile.TemporaryDirectory() as base:
        target = Path(base) / "notes.txt"
        target.write_text("stale", encoding="utf-8")

        with atomic_save(target, text_mode=True) as handle:
            handle.write("fresh")

        assert target.read_text(encoding="utf-8") == "fresh"
        assert sorted(os.listdir(base)) == ["notes.txt"]


def test_string_destinations_and_failure_cleanup_are_unchanged() -> None:
    with tempfile.TemporaryDirectory() as base:
        target = os.path.join(base, "plain.bin")

        with atomic_save(target) as handle:
            handle.write(b"plain")

        with open(target, "rb") as handle:
            assert handle.read() == b"plain"

        doomed = os.path.join(base, "doomed.bin")
        try:
            with atomic_save(doomed) as handle:
                handle.write(b"partial")
                raise RuntimeError("interrupted")
        except RuntimeError:
            pass

        assert sorted(os.listdir(base)) == ["plain.bin"]
