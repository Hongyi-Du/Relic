"""Reckless-driving escalation: distinct violation types within a window."""

from __future__ import annotations

from dataclasses import replace

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


def test_reckless_fires_on_three_distinct_types_in_window(
    system_config, store, camera, now,
    scripted_detector_factory, dict_ocr_factory,
    const_helmet_factory, const_seatbelt_factory,
):
    # Car with no seatbelt (frame 0), then drives north (wrong way) and fast
    # (over speed) on frame 1 -> three distinct types within the window.
    frames = [
        [Detection(BoundingBox(150, 140, 190, 180), "car", 0.9)],
        [Detection(BoundingBox(150, 40, 190, 80), "car", 0.9)],
    ]
    det = scripted_detector_factory(frames)
    ocr = dict_ocr_factory({})
    pipe = make_pipeline(system_config, store, det, ocr,
                         const_helmet_factory(True), const_seatbelt_factory(False))
    pipe.process_frame(None, FrameContext(0, now, camera))
    out = pipe.process_frame(None, FrameContext(1, now, camera))

    reckless = [r for r in out if r.violation_type is ViolationType.RECKLESS_DRIVING]
    assert len(reckless) == 1
    assert reckless[0].measured_value == 3.0
    # Reckless sorts last among the track's violations.
    assert out[-1].violation_type is ViolationType.RECKLESS_DRIVING


def test_reckless_does_not_fire_on_two_distinct_types(
    system_config, store, camera, now,
    scripted_detector_factory, dict_ocr_factory,
    const_helmet_factory, const_seatbelt_factory,
):
    # No seatbelt + over speed (driving with the flow) = only two distinct types.
    frames = [
        [Detection(BoundingBox(0, 0, 40, 40), "car", 0.9)],
        [Detection(BoundingBox(0, 30, 40, 70), "car", 0.9)],
    ]
    det = scripted_detector_factory(frames)
    ocr = dict_ocr_factory({})
    pipe = make_pipeline(system_config, store, det, ocr,
                         const_helmet_factory(True), const_seatbelt_factory(False))
    out0 = pipe.process_frame(None, FrameContext(0, now, camera))
    out1 = pipe.process_frame(None, FrameContext(1, now, camera))
    everything = out0 + out1
    assert not any(
        r.violation_type is ViolationType.RECKLESS_DRIVING for r in everything
    )


def test_reckless_respects_the_frame_window(
    system_config, store, camera, now,
    scripted_detector_factory, dict_ocr_factory,
    const_helmet_factory, const_seatbelt_factory,
):
    # Window shrunk to 2 frames. The seatbelt violation at frame 0 falls out of
    # the window by the time wrong-way + over-speed fire at frame 5, leaving only
    # two distinct types in-window -> no reckless.
    narrow = replace(camera, reckless_window_frames=2)
    frames = [
        [Detection(BoundingBox(150, 140, 190, 180), "car", 0.9)],  # f0: no_seatbelt
        [Detection(BoundingBox(150, 140, 190, 180), "car", 0.9)],  # f1: idle
        [Detection(BoundingBox(150, 140, 190, 180), "car", 0.9)],  # f2: idle
        [Detection(BoundingBox(150, 140, 190, 180), "car", 0.9)],  # f3: idle
        [Detection(BoundingBox(150, 140, 190, 180), "car", 0.9)],  # f4: idle
        [Detection(BoundingBox(150, 40, 190, 80), "car", 0.9)],    # f5: wrong+over
    ]
    det = scripted_detector_factory(frames)
    ocr = dict_ocr_factory({})
    pipe = make_pipeline(system_config, store, det, ocr,
                         const_helmet_factory(True), const_seatbelt_factory(False))
    seen = []
    for i in range(6):
        seen.extend(pipe.process_frame(None, FrameContext(i, now, narrow)))
    assert not any(
        r.violation_type is ViolationType.RECKLESS_DRIVING for r in seen
    )
