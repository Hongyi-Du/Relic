"""Configuration models for cameras and the overall system."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple

class FlowDirection(str, Enum):
    NORTH = 'north'
    SOUTH = 'south'
    EAST = 'east'
    WEST = 'west'
Point = Tuple[float, float]
Line = Tuple[Point, Point]

@dataclass(frozen=True)
class CameraConfig:
    """Per-camera geometry and rule configuration.

    Coordinates are in pixel space of the source frames.
    """
    camera_id: str
    location: str
    stop_line: Line
    allowed_flow: FlowDirection
    speed_limit_kmh: float
    pixels_per_meter: float
    fps: float
    no_change_zone: Tuple[Line, Line] | None = None
    lane_dividers: Tuple[Line, ...] = field(default_factory=tuple)
    reckless_window_frames: int = 30
    reckless_min_distinct: int = 3

    def __post_init__(self) -> None:
        raise NotImplementedError('__post_init__ is not implemented yet')

@dataclass(frozen=True)
class SystemConfig:
    """Global system configuration."""
    database_url: str = 'sqlite:///violations.db'
    webhook_url: str | None = None
    snapshot_dir: str = 'snapshots'
    iou_match_threshold: float = 0.3
    track_max_missed_frames: int = 15
