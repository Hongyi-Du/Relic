"""Over-speed, wrong-way, and lane-change violations."""

from __future__ import annotations

from datetime import timedelta

from traffic_violation_system import (
    AlertDispatcher,
    BoundingBox,
    Detection,
    FrameContext,
    ViolationPipeline,
    ViolationType,
)


def make_pipeline(system_config, store, det, ocr, helmet, seatbelt):
    return ViolationPipeline(
        system_config=system_config,
        detector=det,
        ocr=ocr,
        helmet_detector=helmet,
        seatbelt_detector=seatbelt,
        store=store,
        alerts=AlertDispatcher(),
    )


def test_overspeed_fires_above_limit(
    system_config, store, camera, now,
    scripted_detector_factory, dict_ocr_factory,
    const_helmet_factory, const_seatbelt_factory,
):
    # pixels_per_meter=10, fps=10 -> 1 px / frame = 1 m/s = 3.6 km/h.
    # limit = 60 km/h. Move 30 px in 1 frame -> 30 m/s -> 108 km/h.
    plate_box = BoundingBox(0, 0, 10, 10)
    frames = [
        [Detection(BoundingBox(0, 110, 40, 150), "car", 0.9, plate_bbox=plate_box)],
        [Detection(BoundingBox(0, 140, 40, 180), "car", 0.9, plate_bbox=plate_box)],
    ]
    det = scripted_detector_factory(frames)
    ocr = dict_ocr_factory({})
    pipe = make_pipeline(system_config, store, det, ocr,
                         const_helmet_factory(True), const_seatbelt_factory(True))
    pipe.process_frame(None, FrameContext(0, now, camera))
    out = pipe.process_frame(None, FrameContext(1, now, camera))
    speeders = [r for r in out if r.violation_type is ViolationType.OVER_SPEED]
    assert len(speeders) == 1
    assert speeders[0].measured_value is not None
    assert speeders[0].measured_value > 60.0


def test_speed_not_fired_under_limit(
    system_config, store, camera, now,
    scripted_detector_factory, dict_ocr_factory,
    const_helmet_factory, const_seatbelt_factory,
):
    # 1 px / frame -> 3.6 km/h, well under 60.
    plate_box = BoundingBox(0, 0, 10, 10)
    frames = [
        [Detection(BoundingBox(0, 110, 40, 150), "car", 0.9, plate_bbox=plate_box)],
        [Detection(BoundingBox(0, 111, 40, 151), "car", 0.9, plate_bbox=plate_box)],
    ]
    det = scripted_detector_factory(frames)
    ocr = dict_ocr_factory({})
    pipe = make_pipeline(system_config, store, det, ocr,
                         const_helmet_factory(True), const_seatbelt_factory(True))
    pipe.process_frame(None, FrameContext(0, now, camera))
    out = pipe.process_frame(None, FrameContext(1, now, camera))
    assert all(r.violation_type is not ViolationType.OVER_SPEED for r in out)


def test_wrong_way_fires_against_flow(
    system_config, store, camera, now,
    scripted_detector_factory, dict_ocr_factory,
    const_helmet_factory, const_seatbelt_factory,
):
    # Allowed flow = south (+y). Move north (-y): y from 150 to 50.
    # Stop line is at y=100; this motion crosses it but the light is GREEN so
    # red-light doesn't fire. We still expect WRONG_WAY.
    plate_box = BoundingBox(150, 140, 160, 150)
    frames = [
        [Detection(BoundingBox(150, 140, 190, 180), "car", 0.9, plate_bbox=plate_box)],
        [Detection(BoundingBox(150, 40, 190, 80), "car", 0.9, plate_bbox=plate_box)],
    ]
    det = scripted_detector_factory(frames)
    ocr = dict_ocr_factory({})
    pipe = make_pipeline(system_config, store, det, ocr,
                         const_helmet_factory(True), const_seatbelt_factory(True))
    pipe.process_frame(None, FrameContext(0, now, camera))
    out = pipe.process_frame(None, FrameContext(1, now, camera))
    assert any(r.violation_type is ViolationType.WRONG_WAY for r in out)


def test_lane_change_fires_inside_no_change_zone(
    system_config, store, camera, now,
    scripted_detector_factory, dict_ocr_factory,
    const_helmet_factory, const_seatbelt_factory,
):
    # Vehicle in no-change zone (50<y<90), crosses x=100 divider.
    plate_box = BoundingBox(80, 60, 90, 70)
    frames = [
        [Detection(BoundingBox(80, 60, 95, 80), "car", 0.9, plate_bbox=plate_box)],
        [Detection(BoundingBox(105, 60, 120, 80), "car", 0.9, plate_bbox=plate_box)],
    ]
    det = scripted_detector_factory(frames)
    ocr = dict_ocr_factory({})
    pipe = make_pipeline(system_config, store, det, ocr,
                         const_helmet_factory(True), const_seatbelt_factory(True))
    pipe.process_frame(None, FrameContext(0, now, camera))
    out = pipe.process_frame(None, FrameContext(1, now, camera))
    assert any(r.violation_type is ViolationType.LANE_CHANGE for r in out)
