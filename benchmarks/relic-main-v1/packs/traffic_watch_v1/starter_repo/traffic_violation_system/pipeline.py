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
from .detection import Detection, HelmetDetector, PlateOCR, SeatbeltDetector, VehicleDetector
from .rules import lane_violation, overspeed_violation, red_light_violation, wrong_way_violation
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
_VIOLATION_PRECEDENCE: Dict[ViolationType, int] = {v: i for i, v in enumerate(ViolationType)}

def normalize_plate(text: Optional[str]) -> Optional[str]:
    """Upper-case, strip whitespace/hyphens, fold OCR confusions; empty -> None."""
    raise NotImplementedError('normalize_plate is not implemented yet')

def _save_snapshot(frame, bbox, snapshot_dir: str, violation_id: str) -> str:
    """Save a crop. We use a permissive policy:
    - if frame is a numpy ndarray with cv2 available, write a JPEG;
    - otherwise write a placeholder text file (used in tests).
    """
    raise NotImplementedError('_save_snapshot is not implemented yet')

class ViolationPipeline:
    """Coordinates per-frame detection, tracking, rules, persistence, alerts."""

    def __init__(self, system_config: SystemConfig, detector: VehicleDetector, ocr: PlateOCR, helmet_detector: HelmetDetector, seatbelt_detector: SeatbeltDetector, store: ViolationStore, alerts: AlertDispatcher, tracker: Optional[VehicleTracker]=None) -> None:
        raise NotImplementedError('__init__ is not implemented yet')

    def _emit(self, frame, track: TrackedVehicle, det: Detection, vtype: ViolationType, ctx: FrameContext, measured_value: Optional[float]=None) -> Optional[ViolationRecord]:
        raise NotImplementedError('_emit is not implemented yet')

    def process_frame(self, frame, ctx: FrameContext) -> List[ViolationRecord]:
        raise NotImplementedError('process_frame is not implemented yet')
