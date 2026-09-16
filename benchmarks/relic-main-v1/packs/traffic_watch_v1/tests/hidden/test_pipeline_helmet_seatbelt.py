"""Helmet/seatbelt rule firing."""

from __future__ import annotations

from traffic_violation_system import (
    AlertDispatcher,
    BoundingBox,
    Detection,
    FrameContext,
    ViolationPipeline,
    ViolationType,
)


def make(system_config, store, det, ocr, helmet, seatbelt):
    return ViolationPipeline(
        system_config=system_config,
        detector=det,
        ocr=ocr,
        helmet_detector=helmet,
        seatbelt_detector=seatbelt,
        store=store,
        alerts=AlertDispatcher(),
    )


def test_no_helmet_on_motorcycle(
    system_config, store, camera, now,
    scripted_detector_factory, dict_ocr_factory,
    const_helmet_factory, const_seatbelt_factory,
):
    plate_box = BoundingBox(80, 60, 90, 70)
    frames = [
        [Detection(BoundingBox(80, 60, 100, 80), "motorcycle", 0.9, plate_bbox=plate_box)],
    ]
    det = scripted_detector_factory(frames)
    ocr = dict_ocr_factory({(80.0, 60.0, 90.0, 70.0): "BIKE99"})
    pipe = make(system_config, store, det, ocr,
                const_helmet_factory(False), const_seatbelt_factory(True))
    out = pipe.process_frame(None, FrameContext(0, now, camera))
    helm = [r for r in out if r.violation_type is ViolationType.NO_HELMET]
    assert len(helm) == 1
    # "BIKE99" -> upper -> I folded to 1 -> "B1KE99".
    assert helm[0].plate_text == "B1KE99"


def test_helmet_present_no_violation(
    system_config, store, camera, now,
    scripted_detector_factory, dict_ocr_factory,
    const_helmet_factory, const_seatbelt_factory,
):
    frames = [[Detection(BoundingBox(80, 60, 100, 80), "motorcycle", 0.9)]]
    det = scripted_detector_factory(frames)
    pipe = make(system_config, store, det, dict_ocr_factory({}),
                const_helmet_factory(True), const_seatbelt_factory(True))
    out = pipe.process_frame(None, FrameContext(0, now, camera))
    assert all(r.violation_type is not ViolationType.NO_HELMET for r in out)


def test_no_seatbelt_on_car(
    system_config, store, camera, now,
    scripted_detector_factory, dict_ocr_factory,
    const_helmet_factory, const_seatbelt_factory,
):
    frames = [[Detection(BoundingBox(80, 60, 100, 80), "car", 0.9)]]
    det = scripted_detector_factory(frames)
    pipe = make(system_config, store, det, dict_ocr_factory({}),
                const_helmet_factory(True), const_seatbelt_factory(False))
    out = pipe.process_frame(None, FrameContext(0, now, camera))
    assert any(r.violation_type is ViolationType.NO_SEATBELT for r in out)


def test_helmet_rule_does_not_apply_to_cars(
    system_config, store, camera, now,
    scripted_detector_factory, dict_ocr_factory,
    const_helmet_factory, const_seatbelt_factory,
):
    frames = [[Detection(BoundingBox(80, 60, 100, 80), "car", 0.9)]]
    det = scripted_detector_factory(frames)
    # has_helmet=False, but vehicle is a car -> helmet rule must not fire
    pipe = make(system_config, store, det, dict_ocr_factory({}),
                const_helmet_factory(False), const_seatbelt_factory(True))
    out = pipe.process_frame(None, FrameContext(0, now, camera))
    assert all(r.violation_type is not ViolationType.NO_HELMET for r in out)
