"""Multi-frame vehicle tracker.

Greedy association: match each new detection to the existing track with
highest IoU above the threshold; for non-overlapping detections fall back
to centroid distance bounded by the track's box size so a fast-moving
vehicle still keeps its ID. Unmatched detections spawn new tracks; tracks
that go missing for more than `max_missed_frames` consecutive frames are
evicted.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from math import hypot
from typing import Dict, List, Optional, Sequence, Tuple
from .detection import BoundingBox, Detection

@dataclass
class TrackedVehicle:
    track_id: int
    label: str
    bbox: BoundingBox
    history: List[Tuple[int, BoundingBox]] = field(default_factory=list)
    missed_frames: int = 0
    plate_text: Optional[str] = None

    def update(self, frame_idx: int, bbox: BoundingBox, label: str) -> None:
        raise NotImplementedError('update is not implemented yet')

class VehicleTracker:

    def __init__(self, iou_threshold: float=0.3, max_missed_frames: int=15) -> None:
        raise NotImplementedError('__init__ is not implemented yet')

    @property
    def tracks(self) -> Dict[int, TrackedVehicle]:
        raise NotImplementedError('tracks is not implemented yet')

    def _max_centroid_dist(self, track: TrackedVehicle) -> float:
        """Centroid-distance gate scaled by the track's recent box size."""
        raise NotImplementedError('_max_centroid_dist is not implemented yet')

    def update(self, frame_idx: int, detections: Sequence[Detection]) -> List[Tuple[TrackedVehicle, Detection]]:
        raise NotImplementedError('update is not implemented yet')
