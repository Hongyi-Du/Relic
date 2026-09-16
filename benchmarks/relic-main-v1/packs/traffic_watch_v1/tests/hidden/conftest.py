"""Shared fixtures and fakes for the traffic-violation test suite."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pytest

# Ensure /solution is importable when tests run outside Docker too.
ROOT = Path(__file__).resolve().parents[1]
SOLUTION_DIR = ROOT / "solution"
if SOLUTION_DIR.exists() and str(SOLUTION_DIR) not in sys.path:
    sys.path.insert(0, str(SOLUTION_DIR))

from traffic_violation_system import (  # noqa: E402
    BoundingBox,
    CameraConfig,
    Detection,
    FlowDirection,
    HelmetDetector,
    PlateOCR,
    SeatbeltDetector,
    SQLiteViolationStore,
    SystemConfig,
    VehicleDetector,
)


class ScriptedDetector(VehicleDetector):
    """Returns pre-scripted detections, one list per frame."""

    def __init__(self, frames: List[List[Detection]]):
        self._frames = frames
        self._idx = 0

    def detect(self, frame):
        if self._idx >= len(self._frames):
            return []
        out = self._frames[self._idx]
        self._idx += 1
        return out


class DictOCR(PlateOCR):
    """Maps each plate bbox (rounded) to a fixed string. None ⇒ unreadable."""

    def __init__(self, mapping: Dict[tuple, Optional[str]], default: Optional[str] = None):
        self._mapping = mapping
        self._default = default

    def read(self, frame, bbox):
        key = (round(bbox.x1, 2), round(bbox.y1, 2), round(bbox.x2, 2), round(bbox.y2, 2))
        return self._mapping.get(key, self._default)


class ConstHelmet(HelmetDetector):
    def __init__(self, has: bool):
        self._has = has

    def has_helmet(self, frame, bbox):
        return self._has


class ConstSeatbelt(SeatbeltDetector):
    def __init__(self, has: bool):
        self._has = has

    def has_seatbelt(self, frame, bbox):
        return self._has


@pytest.fixture
def system_config():
    return SystemConfig(
        database_url="sqlite:///:memory:",
        webhook_url=None,
        snapshot_dir=str(Path("/tmp/tvs-snapshots")),
        iou_match_threshold=0.3,
        track_max_missed_frames=10,
    )


@pytest.fixture
def camera():
    # Stop line is the horizontal line y=100 between x=0..200.
    return CameraConfig(
        camera_id="cam-01",
        location="Main & 1st",
        stop_line=((0.0, 100.0), (200.0, 100.0)),
        allowed_flow=FlowDirection.SOUTH,
        speed_limit_kmh=60.0,
        pixels_per_meter=10.0,
        fps=10.0,
        no_change_zone=(
            ((0.0, 50.0), (200.0, 50.0)),
            ((0.0, 90.0), (200.0, 90.0)),
        ),
        lane_dividers=(((100.0, 0.0), (100.0, 200.0)),),
    )


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "violations.db"
    s = SQLiteViolationStore(db_path=str(db))
    yield s
    s.close()


@pytest.fixture
def now():
    return datetime(2026, 5, 28, 12, 0, 0)


@pytest.fixture
def scripted_detector_factory():
    return ScriptedDetector


@pytest.fixture
def dict_ocr_factory():
    return DictOCR


@pytest.fixture
def const_helmet_factory():
    return ConstHelmet


@pytest.fixture
def const_seatbelt_factory():
    return ConstSeatbelt


@pytest.fixture
def bbox():
    return BoundingBox


@pytest.fixture
def detection():
    return Detection
