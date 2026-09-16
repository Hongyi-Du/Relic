"""Violation records and types."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional

class ViolationType(str, Enum):
    RED_LIGHT = 'red_light'
    WRONG_WAY = 'wrong_way'
    OVER_SPEED = 'over_speed'
    LANE_CHANGE = 'lane_change'
    NO_HELMET = 'no_helmet'
    NO_SEATBELT = 'no_seatbelt'
    RECKLESS_DRIVING = 'reckless_driving'

class UnreadablePlateError(Exception):
    """Raised when OCR cannot decode a plate; the record stores plate_text=None."""

@dataclass
class ViolationRecord:
    violation_id: str
    track_id: int
    violation_type: ViolationType
    timestamp: datetime
    camera_id: str
    location: str
    plate_text: Optional[str]
    snapshot_path: str
    measured_value: Optional[float] = None

    def to_dict(self) -> dict:
        raise NotImplementedError('to_dict is not implemented yet')
