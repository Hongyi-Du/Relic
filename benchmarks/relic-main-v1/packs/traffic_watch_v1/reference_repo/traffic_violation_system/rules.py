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
    return (p2[0] - p1[0]) * (q[1] - p1[1]) - (p2[1] - p1[1]) * (q[0] - p1[0])


def crossed_line(prev_center, curr_center, line) -> bool:
    """True iff the segment prev->curr crosses the geometric line."""
    a, b = line
    s_prev = _segment_side(a, b, prev_center)
    s_curr = _segment_side(a, b, curr_center)
    return s_prev == 0.0 or s_curr == 0.0 or (s_prev * s_curr < 0)


def red_light_violation(
    track: TrackedVehicle, camera: CameraConfig, light_is_red: bool
) -> Optional[ViolationType]:
    """Fires when the vehicle crosses the stop line while the light is red.

    Requires at least two history points to detect a crossing.
    """
    if not light_is_red:
        return None
    if len(track.history) < 2:
        return None
    prev = track.history[-2][1].center
    curr = track.history[-1][1].center
    if crossed_line(prev, curr, camera.stop_line):
        return ViolationType.RED_LIGHT
    return None


def _direction_vector(direction: FlowDirection) -> tuple[float, float]:
    """Image coordinates: y axis grows downward."""
    return {
        FlowDirection.NORTH: (0.0, -1.0),
        FlowDirection.SOUTH: (0.0, 1.0),
        FlowDirection.EAST: (1.0, 0.0),
        FlowDirection.WEST: (-1.0, 0.0),
    }[direction]

def wrong_way_violation(
    track: TrackedVehicle, camera: CameraConfig
) -> Optional[ViolationType]:
    """Fires when the vehicle's net motion opposes the allowed flow."""
    if len(track.history) < 2:
        return None
    
    prev = track.history[-2][1].center
    curr = track.history[-1][1].center
    dx = curr[0] - prev[0]
    dy = curr[1] - prev[1]

    flow = camera.allowed_flow
    if flow == FlowDirection.SOUTH and dy < 0:
        return ViolationType.WRONG_WAY
    if flow == FlowDirection.NORTH and dy > 0:
        return ViolationType.WRONG_WAY
    if flow == FlowDirection.EAST and dx < 0:
        return ViolationType.WRONG_WAY
    if flow == FlowDirection.WEST and dx > 0:
        return ViolationType.WRONG_WAY
    
    return None


def measure_speed_kmh(track: TrackedVehicle, camera: CameraConfig) -> Optional[float]:
    """Maximum single-step speed (km/h) across consecutive samples in history."""
    if len(track.history) < 2:
        return None
    best: Optional[float] = None
    for (f0, b0), (f1, b1) in zip(track.history, track.history[1:]):
        if f1 <= f0:
            continue
        dx = b1.center[0] - b0.center[0]
        dy = b1.center[1] - b0.center[1]
        dist_m = math.hypot(dx, dy) / camera.pixels_per_meter
        dt_s = (f1 - f0) / camera.fps
        if dt_s <= 0:
            continue
        step = (dist_m / dt_s) * 3.6
        if best is None or step > best:
            best = step
    return best


def overspeed_violation(
    track: TrackedVehicle, camera: CameraConfig
) -> tuple[Optional[ViolationType], Optional[float]]:
    speed = measure_speed_kmh(track, camera)
    if speed is None:
        return None, None
    speed = round(speed, 1)
    if speed > camera.speed_limit_kmh:
        return ViolationType.OVER_SPEED, speed
    return None, speed


def _point_in_zone(point, zone) -> bool:
    """A zone is two parallel lines; the point lies between them iff the
    signs of (line, point) differ between the two lines."""
    line_a, line_b = zone
    sa = _segment_side(line_a[0], line_a[1], point)
    sb = _segment_side(line_b[0], line_b[1], point)
    return sa * sb < 0


def lane_violation(
    track: TrackedVehicle, camera: CameraConfig
) -> Optional[ViolationType]:
    if camera.no_change_zone is None or not camera.lane_dividers:
        return None
    if len(track.history) < 2:
        return None
    prev = track.history[-2][1].center
    curr = track.history[-1][1].center
    if not (_point_in_zone(prev, camera.no_change_zone)
            and _point_in_zone(curr, camera.no_change_zone)):
        return None
    for divider in camera.lane_dividers:
        if crossed_line(prev, curr, divider):
            return ViolationType.LANE_CHANGE
    return None
