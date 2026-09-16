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
        self.bbox = bbox
        self.label = label
        self.history.append((frame_idx, bbox))
        self.missed_frames = 0


class VehicleTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed_frames: int = 15) -> None:
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0,1]")
        self._iou = iou_threshold
        self._max_missed = max_missed_frames
        self._next_id = 1
        self._tracks: Dict[int, TrackedVehicle] = {}

    @property
    def tracks(self) -> Dict[int, TrackedVehicle]:
        return self._tracks

    def _max_centroid_dist(self, track: TrackedVehicle) -> float:
        """Centroid-distance gate scaled by the track's recent box size."""
        diag = hypot(track.bbox.width, track.bbox.height)
        return max(diag * 2.5, 50.0)

    def update(
        self, frame_idx: int, detections: Sequence[Detection]
    ) -> List[Tuple[TrackedVehicle, Detection]]:
        assignments: List[Tuple[TrackedVehicle, Detection]] = []
        unmatched_track_ids = set(self._tracks.keys())
        unmatched_det_indices = list(range(len(detections)))

        # Score every (track, det) pair. IoU-based pairs first (sorted desc);
        # then centroid-distance fallbacks bounded by track box diagonal.
        iou_pairs: List[Tuple[float, int, int]] = []
        dist_pairs: List[Tuple[float, int, int]] = []
        for tid, track in self._tracks.items():
            for di, det in enumerate(detections):
                if det.label != track.label:
                    # Don't merge across vehicle classes.
                    continue
                iou = track.bbox.iou(det.bbox)
                if iou >= self._iou:
                    iou_pairs.append((iou, tid, di))
                else:
                    tc = track.bbox.center
                    dc = det.bbox.center
                    d = hypot(tc[0] - dc[0], tc[1] - dc[1])
                    if d <= self._max_centroid_dist(track):
                        dist_pairs.append((d, tid, di))

        iou_pairs.sort(key=lambda x: x[0], reverse=True)
        dist_pairs.sort(key=lambda x: x[0])

        used_tracks: set = set()
        used_dets: set = set()

        def consume(pairs, score_is_distance: bool):
            for score, tid, di in pairs:
                if tid in used_tracks or di in used_dets:
                    continue
                track = self._tracks[tid]
                det = detections[di]
                track.update(frame_idx, det.bbox, det.label)
                assignments.append((track, det))
                used_tracks.add(tid)
                used_dets.add(di)

        consume(iou_pairs, score_is_distance=False)
        consume(dist_pairs, score_is_distance=True)

        for tid in used_tracks:
            unmatched_track_ids.discard(tid)
        for di in used_dets:
            unmatched_det_indices.remove(di)

        for di in unmatched_det_indices:
            det = detections[di]
            track = TrackedVehicle(
                track_id=self._next_id,
                label=det.label,
                bbox=det.bbox,
            )
            track.history.append((frame_idx, det.bbox))
            self._tracks[self._next_id] = track
            self._next_id += 1
            assignments.append((track, det))

        evict: List[int] = []
        for tid in unmatched_track_ids:
            self._tracks[tid].missed_frames += 1
            if self._tracks[tid].missed_frames > self._max_missed:
                evict.append(tid)
        for tid in evict:
            del self._tracks[tid]

        return assignments
