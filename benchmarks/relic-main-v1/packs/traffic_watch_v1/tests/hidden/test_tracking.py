"""Vehicle tracking: stable IDs across frames."""

from __future__ import annotations

from traffic_violation_system import BoundingBox, Detection, VehicleTracker


def test_track_id_stable_across_frames():
    tracker = VehicleTracker(iou_threshold=0.3, max_missed_frames=5)
    det0 = Detection(BoundingBox(10, 10, 50, 50), "car", 0.9)
    det1 = Detection(BoundingBox(12, 12, 52, 52), "car", 0.9)
    a0 = tracker.update(0, [det0])
    a1 = tracker.update(1, [det1])
    assert a0[0][0].track_id == a1[0][0].track_id


def test_new_detection_far_away_spawns_new_track():
    tracker = VehicleTracker(iou_threshold=0.3, max_missed_frames=5)
    a0 = tracker.update(0, [Detection(BoundingBox(10, 10, 50, 50), "car", 0.9)])
    a1 = tracker.update(1, [Detection(BoundingBox(500, 500, 540, 540), "car", 0.9)])
    assert a0[0][0].track_id != a1[0][0].track_id


def test_missing_tracks_are_evicted():
    tracker = VehicleTracker(iou_threshold=0.3, max_missed_frames=2)
    tracker.update(0, [Detection(BoundingBox(10, 10, 50, 50), "car", 0.9)])
    for f in range(1, 5):
        tracker.update(f, [])
    assert tracker.tracks == {}


def test_two_detections_assigned_to_distinct_tracks():
    tracker = VehicleTracker(iou_threshold=0.3, max_missed_frames=5)
    dets = [
        Detection(BoundingBox(10, 10, 50, 50), "car", 0.9),
        Detection(BoundingBox(200, 10, 240, 50), "car", 0.9),
    ]
    a = tracker.update(0, dets)
    ids = {t.track_id for t, _ in a}
    assert len(ids) == 2
