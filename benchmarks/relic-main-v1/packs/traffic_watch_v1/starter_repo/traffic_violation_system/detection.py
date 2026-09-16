"""Detector and OCR seams.

These are Protocol-like base classes so tests can inject deterministic
fakes; the production wiring would plug in a YOLO/TensorFlow/PyTorch
detector and a Tesseract/EasyOCR plate reader.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        raise NotImplementedError('__post_init__ is not implemented yet')

    @property
    def width(self) -> float:
        raise NotImplementedError('width is not implemented yet')

    @property
    def height(self) -> float:
        raise NotImplementedError('height is not implemented yet')

    @property
    def center(self) -> tuple[float, float]:
        raise NotImplementedError('center is not implemented yet')

    @property
    def area(self) -> float:
        raise NotImplementedError('area is not implemented yet')

    def iou(self, other: 'BoundingBox') -> float:
        raise NotImplementedError('iou is not implemented yet')

@dataclass(frozen=True)
class Detection:
    """A detected object on a frame."""
    bbox: BoundingBox
    label: str
    confidence: float
    plate_bbox: Optional[BoundingBox] = None

class VehicleDetector:
    """Base detector. Subclasses implement `detect`."""

    def detect(self, frame) -> List[Detection]:
        raise NotImplementedError('detect is not implemented yet')

class PlateOCR:
    """Base OCR reader. Subclasses implement `read`."""

    def read(self, frame, bbox: BoundingBox) -> Optional[str]:
        raise NotImplementedError('read is not implemented yet')

class HelmetDetector:
    """Returns True iff the rider in the given bbox is wearing a helmet."""

    def has_helmet(self, frame, bbox: BoundingBox) -> bool:
        raise NotImplementedError('has_helmet is not implemented yet')

class SeatbeltDetector:
    """Returns True iff the driver in the given bbox is wearing a seatbelt."""

    def has_seatbelt(self, frame, bbox: BoundingBox) -> bool:
        raise NotImplementedError('has_seatbelt is not implemented yet')
