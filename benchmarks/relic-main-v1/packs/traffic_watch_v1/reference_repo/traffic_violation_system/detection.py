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
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("BoundingBox: x2/y2 must be >= x1/y1")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def iou(self, other: "BoundingBox") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        if union <= 0:
            return 0.0
        return inter / union


@dataclass(frozen=True)
class Detection:
    """A detected object on a frame."""

    bbox: BoundingBox
    label: str
    confidence: float
    plate_bbox: Optional[BoundingBox] = None


class VehicleDetector:
    """Base detector. Subclasses implement `detect`."""

    def detect(self, frame) -> List[Detection]:  # pragma: no cover - abstract
        raise NotImplementedError


class PlateOCR:
    """Base OCR reader. Subclasses implement `read`."""

    def read(self, frame, bbox: BoundingBox) -> Optional[str]:  # pragma: no cover
        raise NotImplementedError


class HelmetDetector:
    """Returns True iff the rider in the given bbox is wearing a helmet."""

    def has_helmet(self, frame, bbox: BoundingBox) -> bool:  # pragma: no cover
        raise NotImplementedError


class SeatbeltDetector:
    """Returns True iff the driver in the given bbox is wearing a seatbelt."""

    def has_seatbelt(self, frame, bbox: BoundingBox) -> bool:  # pragma: no cover
        raise NotImplementedError
