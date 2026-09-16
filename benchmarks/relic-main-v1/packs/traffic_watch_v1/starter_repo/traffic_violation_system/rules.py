"""Violation rules. Each rule inspects tracked vehicles + frame context
and returns the ViolationType to fire (or None)."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Sequence
from .config import CameraConfig, FlowDirection
from .detection import BoundingBox, Detection
from .tracking import TrackedVehicle
from .violations import ViolationType

def _segment_side(p1, p2, q) -> float:
    """>0 if q is left of line p1->p2, <0 if right, 0 on line."""
    raise NotImplementedError('_segment_side is not implemented yet')

def crossed_line(prev_center, curr_center, line) -> bool:
    """True iff the segment prev->curr crosses the geometric line."""
    raise NotImplementedError('crossed_line is not implemented yet')

def red_light_violation(track: TrackedVehicle, camera: CameraConfig, light_is_red: bool) -> Optional[ViolationType]:
    """Fires when the vehicle crosses the stop line while the light is red.

    Requires at least two history points to detect a crossing.
    """
    raise NotImplementedError('red_light_violation is not implemented yet')

def _direction_vector(direction: FlowDirection) -> tuple[float, float]:
    """Image coordinates: y axis grows downward."""
    raise NotImplementedError('_direction_vector is not implemented yet')

def wrong_way_violation(track: TrackedVehicle, camera: CameraConfig) -> Optional[ViolationType]:
    """Fires when the vehicle's net motion opposes the allowed flow."""
    raise NotImplementedError('wrong_way_violation is not implemented yet')

def measure_speed_kmh(track: TrackedVehicle, camera: CameraConfig) -> Optional[float]:
    """Maximum single-step speed (km/h) across consecutive samples in history."""
    raise NotImplementedError('measure_speed_kmh is not implemented yet')

def overspeed_violation(track: TrackedVehicle, camera: CameraConfig) -> tuple[Optional[ViolationType], Optional[float]]:
    raise NotImplementedError('overspeed_violation is not implemented yet')

def _point_in_zone(point, zone) -> bool:
    """A zone is two parallel lines; the point lies between them iff the
    signs of (line, point) differ between the two lines."""
    raise NotImplementedError('_point_in_zone is not implemented yet')

def lane_violation(track: TrackedVehicle, camera: CameraConfig) -> Optional[ViolationType]:
    raise NotImplementedError('lane_violation is not implemented yet')
