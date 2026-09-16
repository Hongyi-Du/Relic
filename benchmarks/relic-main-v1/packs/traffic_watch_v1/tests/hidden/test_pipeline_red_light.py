"""Red-light violation: vehicle crosses the stop line while the light is red."""

from __future__ import annotations

from traffic_violation_system import (
    AlertDispatcher,
    BoundingBox,
    Detection,
    FrameContext,
    ViolationPipeline,
    ViolationType,
)


def make_pipeline(system_config, store, det, ocr, helmet, seatbelt, alerts=None):
    return ViolationPipeline(
        system_config=system_config,
        detector=det,
        ocr=ocr,
        helmet_detector=helmet,
        seatbelt_detector=seatbelt,
        store=store,
        alerts=alerts or AlertDispatcher(),
    )


def test_red_light_violation_fires_when_red(
    system_config, store, camera, now,
    scripted_detector_factory, dict_ocr_factory,
    const_helmet_factory, const_seatbelt_factory,
):
    # Vehicle center y=80 then y=120 -> crosses stop line at y=100.
    plate_box = BoundingBox(20, 70, 60, 80)
    frames = [
        [Detection(BoundingBox(20, 60, 60, 100), "car", 0.9, plate_bbox=plate_box)],
        [Detection(BoundingBox(20, 100, 60, 140), "car", 0.9, plate_bbox=plate_box)],
    ]
    det = scripted_detector_factory(frames)
    ocr = dict_ocr_factory({(20.0, 70.0, 60.0, 80.0): "ABC123"})
    pipe = make_pipeline(system_config, store, det, ocr,
                         const_helmet_factory(True), const_seatbelt_factory(True))

    pipe.process_frame(None, FrameContext(0, now, camera, light_is_red=True))
    out = pipe.process_frame(None, FrameContext(1, now, camera, light_is_red=True))

    types = [r.violation_type for r in out]
    assert ViolationType.RED_LIGHT in types
    rec = next(r for r in out if r.violation_type is ViolationType.RED_LIGHT)
    assert rec.plate_text == "ABC123"
    assert rec.camera_id == "cam-01"
    assert rec.location == "Main & 1st"


def test_red_light_fires_when_centroid_lands_on_line(
    system_config, store, camera, now,
    scripted_detector_factory, dict_ocr_factory,
    const_helmet_factory, const_seatbelt_factory,
):
    # Centroid moves from y=80 to exactly y=100 (on the stop line). Landing on
    # the line counts as a crossing, so RED_LIGHT must fire.
    frames = [
        [Detection(BoundingBox(20, 60, 60, 100), "car", 0.9)],   # center y=80
        [Detection(BoundingBox(20, 80, 60, 120), "car", 0.9)],   # center y=100
    ]
    det = scripted_detector_factory(frames)
    ocr = dict_ocr_factory({})
    pipe = make_pipeline(system_config, store, det, ocr,
                         const_helmet_factory(True), const_seatbelt_factory(True))

    pipe.process_frame(None, FrameContext(0, now, camera, light_is_red=True))
    out = pipe.process_frame(None, FrameContext(1, now, camera, light_is_red=True))
    assert any(r.violation_type is ViolationType.RED_LIGHT for r in out)


def test_red_light_violation_not_fired_when_green(
    system_config, store, camera, now,
    scripted_detector_factory, dict_ocr_factory,
    const_helmet_factory, const_seatbelt_factory,
):
    plate_box = BoundingBox(20, 70, 60, 80)
    frames = [
        [Detection(BoundingBox(20, 60, 60, 100), "car", 0.9, plate_bbox=plate_box)],
        [Detection(BoundingBox(20, 100, 60, 140), "car", 0.9, plate_bbox=plate_box)],
    ]
    det = scripted_detector_factory(frames)
    ocr = dict_ocr_factory({})
    pipe = make_pipeline(system_config, store, det, ocr,
                         const_helmet_factory(True), const_seatbelt_factory(True))

    pipe.process_frame(None, FrameContext(0, now, camera, light_is_red=False))
    out = pipe.process_frame(None, FrameContext(1, now, camera, light_is_red=False))
    assert all(r.violation_type is not ViolationType.RED_LIGHT for r in out)
