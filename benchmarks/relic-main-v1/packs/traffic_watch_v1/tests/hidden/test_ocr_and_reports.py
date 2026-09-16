"""OCR null-plate handling and CSV export."""

from __future__ import annotations

from pathlib import Path

from traffic_violation_system import (
    AlertDispatcher,
    BoundingBox,
    Detection,
    FrameContext,
    ReportExporter,
    ViolationPipeline,
    ViolationType,
)


def test_unreadable_plate_stored_as_null(
    system_config, store, camera, now,
    scripted_detector_factory, dict_ocr_factory,
    const_helmet_factory, const_seatbelt_factory,
):
    # OCR always returns None -> record plate_text should be None.
    plate_box = BoundingBox(20, 70, 60, 80)
    frames = [
        [Detection(BoundingBox(20, 60, 60, 100), "car", 0.9, plate_bbox=plate_box)],
        [Detection(BoundingBox(20, 100, 60, 140), "car", 0.9, plate_bbox=plate_box)],
    ]
    det = scripted_detector_factory(frames)
    ocr = dict_ocr_factory({}, default=None)
    pipe = ViolationPipeline(
        system_config=system_config,
        detector=det,
        ocr=ocr,
        helmet_detector=const_helmet_factory(True),
        seatbelt_detector=const_seatbelt_factory(True),
        store=store,
        alerts=AlertDispatcher(),
    )
    pipe.process_frame(None, FrameContext(0, now, camera, light_is_red=True))
    out = pipe.process_frame(None, FrameContext(1, now, camera, light_is_red=True))
    red = [r for r in out if r.violation_type is ViolationType.RED_LIGHT]
    assert len(red) == 1
    assert red[0].plate_text is None


def test_plate_text_normalized_before_storage(
    system_config, store, camera, now,
    scripted_detector_factory, dict_ocr_factory,
    const_helmet_factory, const_seatbelt_factory,
):
    # OCR yields a noisy plate; it must be stored upper-cased with whitespace
    # and hyphens removed: "ab 12-34" -> "AB1234".
    plate_box = BoundingBox(20, 70, 60, 80)
    frames = [
        [Detection(BoundingBox(20, 60, 60, 100), "car", 0.9, plate_bbox=plate_box)],
        [Detection(BoundingBox(20, 100, 60, 140), "car", 0.9, plate_bbox=plate_box)],
    ]
    det = scripted_detector_factory(frames)
    ocr = dict_ocr_factory({}, default="ab 12-34")
    pipe = ViolationPipeline(
        system_config=system_config,
        detector=det,
        ocr=ocr,
        helmet_detector=const_helmet_factory(True),
        seatbelt_detector=const_seatbelt_factory(True),
        store=store,
        alerts=AlertDispatcher(),
    )
    pipe.process_frame(None, FrameContext(0, now, camera, light_is_red=True))
    out = pipe.process_frame(None, FrameContext(1, now, camera, light_is_red=True))
    red = [r for r in out if r.violation_type is ViolationType.RED_LIGHT]
    assert len(red) == 1
    assert red[0].plate_text == "AB1234"


def test_plate_ocr_confusions_folded_to_digits(
    system_config, store, camera, now,
    scripted_detector_factory, dict_ocr_factory,
    const_helmet_factory, const_seatbelt_factory,
):
    # "io9" -> upper "IO9" -> O->0, I->1 -> "109".
    plate_box = BoundingBox(20, 70, 60, 80)
    frames = [
        [Detection(BoundingBox(20, 60, 60, 100), "car", 0.9, plate_bbox=plate_box)],
        [Detection(BoundingBox(20, 100, 60, 140), "car", 0.9, plate_bbox=plate_box)],
    ]
    det = scripted_detector_factory(frames)
    ocr = dict_ocr_factory({}, default="io9")
    pipe = ViolationPipeline(
        system_config=system_config,
        detector=det,
        ocr=ocr,
        helmet_detector=const_helmet_factory(True),
        seatbelt_detector=const_seatbelt_factory(True),
        store=store,
        alerts=AlertDispatcher(),
    )
    pipe.process_frame(None, FrameContext(0, now, camera, light_is_red=True))
    out = pipe.process_frame(None, FrameContext(1, now, camera, light_is_red=True))
    red = [r for r in out if r.violation_type is ViolationType.RED_LIGHT]
    assert len(red) == 1
    assert red[0].plate_text == "109"


def test_csv_export(tmp_path, store, now):
    from traffic_violation_system import ViolationRecord, ViolationType
    rec = ViolationRecord(
        violation_id="abc",
        track_id=1,
        violation_type=ViolationType.RED_LIGHT,
        timestamp=now,
        camera_id="cam-01",
        location="Main",
        plate_text="X1",
        snapshot_path="/tmp/x.jpg",
    )
    store.save(rec)
    out = tmp_path / "rep.csv"
    ReportExporter().export_csv(store.all(), out)
    assert out.exists()
    text = out.read_text()
    assert "violation_id" in text
    assert "abc" in text
    assert "red_light" in text


def test_snapshot_export(tmp_path, store, now):
    from traffic_violation_system import ViolationRecord, ViolationType
    src = tmp_path / "snap.jpg"
    src.write_text("image-bytes")
    rec = ViolationRecord(
        violation_id="snap1",
        track_id=1,
        violation_type=ViolationType.NO_HELMET,
        timestamp=now,
        camera_id="cam-01",
        location="Main",
        plate_text=None,
        snapshot_path=str(src),
    )
    store.save(rec)
    dest = tmp_path / "out"
    copied = ReportExporter().export_snapshots(store.all(), dest)
    assert len(copied) == 1
    assert copied[0].exists()

def test_plate_text_preserves_special_characters(
    system_config, store, camera, now,
    scripted_detector_factory, dict_ocr_factory,
    const_helmet_factory, const_seatbelt_factory,
):
    # O modelo SOTA costuma usar .isalnum() e apagar caracteres especiais.
    # Esta placa tem '@' e '_', que DEVEM ser mantidos segundo o Requirement.
    plate_box = BoundingBox(20, 70, 60, 80)
    frames = [
        [Detection(BoundingBox(20, 60, 60, 100), "car", 0.9, plate_bbox=plate_box)],
        [Detection(BoundingBox(20, 100, 60, 140), "car", 0.9, plate_bbox=plate_box)],
    ]
    det = scripted_detector_factory(frames)
    
    # O OCR lê a placa com espaços, hífen e caracteres especiais
    ocr = dict_ocr_factory({}, default="a_b 1@2-3!4")
    
    pipe = ViolationPipeline(
        system_config=system_config,
        detector=det,
        ocr=ocr,
        helmet_detector=const_helmet_factory(True),
        seatbelt_detector=const_seatbelt_factory(True),
        store=store,
        alerts=AlertDispatcher(),
    )
    pipe.process_frame(None, FrameContext(0, now, camera, light_is_red=True))
    out = pipe.process_frame(None, FrameContext(1, now, camera, light_is_red=True))
    
    red = [r for r in out if r.violation_type is ViolationType.RED_LIGHT]
    assert len(red) == 1
    
    # O resultado salvo tem que remover espaços e hífens, subir pra maiúscula, 
    # MAS manter o _, o @ e o !
    assert red[0].plate_text == "A_B1@23!4"
