"""Frame-level orchestration: detection → tracking → rules → persistence."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .alerts import Alert, AlertDispatcher
from .config import CameraConfig, SystemConfig
from .detection import (
    Detection,
    HelmetDetector,
    PlateOCR,
    SeatbeltDetector,
    VehicleDetector,
)
from .rules import (
    lane_violation,
    overspeed_violation,
    red_light_violation,
    wrong_way_violation,
)
from .storage import ViolationStore
from .tracking import TrackedVehicle, VehicleTracker
from .violations import ViolationRecord, ViolationType


@dataclass
class FrameContext:
    """Per-frame context passed by the caller."""

    frame_idx: int
    timestamp: datetime
    camera: CameraConfig
    light_is_red: bool = False


# Emission precedence: declaration order of ViolationType.
_VIOLATION_PRECEDENCE: Dict[ViolationType, int] = {
    v: i for i, v in enumerate(ViolationType)
}


def normalize_plate(text: Optional[str]) -> Optional[str]:
    """Upper-case, strip whitespace/hyphens, fold OCR confusions; empty -> None."""
    if text is None:
        return None
    cleaned = re.sub(r"[\s\-]+", "", text).upper()
    cleaned = cleaned.replace("O", "0").replace("I", "1")
    return cleaned or None


def _save_snapshot(frame, bbox, snapshot_dir: str, violation_id: str) -> str:
    """Save a crop. We use a permissive policy:
    - if frame is a numpy ndarray with cv2 available, write a JPEG;
    - otherwise write a placeholder text file (used in tests).
    """
    Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
    path = Path(snapshot_dir) / f"{violation_id}.jpg"
    try:  # pragma: no cover - exercised only when cv2 + ndarray present
        import numpy as np  # type: ignore
        import cv2  # type: ignore

        if isinstance(frame, np.ndarray):
            x1, y1, x2, y2 = (
                max(0, int(bbox.x1)),
                max(0, int(bbox.y1)),
                int(bbox.x2),
                int(bbox.y2),
            )
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                cv2.imwrite(str(path), crop)
                return str(path)
    except Exception:
        pass
    # Fallback: write a stub file so the path is real.
    path.write_text(f"snapshot:{violation_id}")
    return str(path)


class ViolationPipeline:
    """Coordinates per-frame detection, tracking, rules, persistence, alerts."""

    def __init__(
        self,
        system_config: SystemConfig,
        detector: VehicleDetector,
        ocr: PlateOCR,
        helmet_detector: HelmetDetector,
        seatbelt_detector: SeatbeltDetector,
        store: ViolationStore,
        alerts: AlertDispatcher,
        tracker: Optional[VehicleTracker] = None,
    ) -> None:
        self.system_config = system_config
        self.detector = detector
        self.ocr = ocr
        self.helmet_detector = helmet_detector
        self.seatbelt_detector = seatbelt_detector
        self.store = store
        self.alerts = alerts
        self.tracker = tracker or VehicleTracker(
            iou_threshold=system_config.iou_match_threshold,
            max_missed_frames=system_config.track_max_missed_frames,
        )
        # Each track may only emit a given violation type once.
        self._fired: set[tuple[int, ViolationType]] = set()
        # Frame index at which each (track, type) first fired (for the
        # reckless-driving sliding window).
        self._fired_at: Dict[tuple[int, ViolationType], int] = {}

    def _emit(
        self,
        frame,
        track: TrackedVehicle,
        det: Detection,
        vtype: ViolationType,
        ctx: FrameContext,
        measured_value: Optional[float] = None,
    ) -> Optional[ViolationRecord]:
        key = (track.track_id, vtype)
        if key in self._fired:
            return None
        self._fired.add(key)
        self._fired_at[key] = ctx.frame_idx

        # OCR plate text. None when unreadable.
        plate_text = track.plate_text
        if plate_text is None and det.plate_bbox is not None:
            plate_text = normalize_plate(self.ocr.read(frame, det.plate_bbox))
            track.plate_text = plate_text

        vid = uuid.uuid4().hex
        snapshot_path = _save_snapshot(
            frame, det.bbox, self.system_config.snapshot_dir, vid
        )
        record = ViolationRecord(
            violation_id=vid,
            track_id=track.track_id,
            violation_type=vtype,
            timestamp=ctx.timestamp,
            camera_id=ctx.camera.camera_id,
            location=ctx.camera.location,
            plate_text=plate_text,
            snapshot_path=snapshot_path,
            measured_value=measured_value,
        )
        self.store.save(record)
        self.alerts.dispatch(record)
        return record

    def process_frame(self, frame, ctx: FrameContext) -> List[ViolationRecord]:
        detections = self.detector.detect(frame)
        assignments = self.tracker.update(ctx.frame_idx, detections)
        emitted: List[ViolationRecord] = []

        for track, det in assignments:
            label = (det.label or "").lower()

            v = red_light_violation(track, ctx.camera, ctx.light_is_red)
            if v is not None:
                rec = self._emit(frame, track, det, v, ctx)
                if rec is not None:
                    emitted.append(rec)

            v = wrong_way_violation(track, ctx.camera)
            if v is not None:
                rec = self._emit(frame, track, det, v, ctx)
                if rec is not None:
                    emitted.append(rec)

            vtype, speed = overspeed_violation(track, ctx.camera)
            if vtype is not None:
                rec = self._emit(frame, track, det, vtype, ctx, measured_value=speed)
                if rec is not None:
                    emitted.append(rec)

            v = lane_violation(track, ctx.camera)
            if v is not None:
                rec = self._emit(frame, track, det, v, ctx)
                if rec is not None:
                    emitted.append(rec)

            if label in ("motorcycle", "motorbike", "scooter"):
                if not self.helmet_detector.has_helmet(frame, det.bbox):
                    rec = self._emit(
                        frame, track, det, ViolationType.NO_HELMET, ctx
                    )
                    if rec is not None:
                        emitted.append(rec)

            if label == "car":
                if not self.seatbelt_detector.has_seatbelt(frame, det.bbox):
                    rec = self._emit(
                        frame, track, det, ViolationType.NO_SEATBELT, ctx
                    )
                    if rec is not None:
                        emitted.append(rec)

            # Reckless-driving escalation, evaluated after this frame's primary
            # violations are recorded so they count toward the window.
            window = ctx.camera.reckless_window_frames
            distinct = {
                vtype
                for (tid, vtype), fired_idx in self._fired_at.items()
                if tid == track.track_id
                and vtype is not ViolationType.RECKLESS_DRIVING
                and ctx.frame_idx - fired_idx <= window
            }
            if len(distinct) >= ctx.camera.reckless_min_distinct:
                rec = self._emit(
                    frame, track, det, ViolationType.RECKLESS_DRIVING, ctx,
                    measured_value=float(len(distinct)),
                )
                if rec is not None:
                    emitted.append(rec)

        emitted.sort(
            key=lambda r: (r.track_id, _VIOLATION_PRECEDENCE[r.violation_type])
        )
        return emitted
