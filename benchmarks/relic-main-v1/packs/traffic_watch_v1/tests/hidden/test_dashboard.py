"""Flask dashboard: upload, list, export."""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

from traffic_violation_system import (
    SQLiteViolationStore,
    ViolationRecord,
    ViolationType,
)
from traffic_violation_system.dashboard import create_app


def _seed(store, now):
    store.save(
        ViolationRecord(
            violation_id="v1",
            track_id=1,
            violation_type=ViolationType.RED_LIGHT,
            timestamp=now,
            camera_id="cam-01",
            location="Main",
            plate_text="X1",
            snapshot_path="/tmp/x.jpg",
        )
    )


def test_index_returns_html(tmp_path, now):
    store = SQLiteViolationStore(db_path=str(tmp_path / "v.db"))
    app = create_app(store, upload_dir=str(tmp_path / "up"))
    client = app.test_client()
    r = client.get("/")
    assert r.status_code == 200
    assert b"Traffic Violation Dashboard" in r.data


def test_upload_video(tmp_path, now):
    store = SQLiteViolationStore(db_path=str(tmp_path / "v.db"))
    app = create_app(store, upload_dir=str(tmp_path / "up"))
    client = app.test_client()
    data = {"video": (io.BytesIO(b"video-bytes"), "clip.mp4")}
    r = client.post("/upload", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "queued"
    assert (tmp_path / "up" / "clip.mp4").exists()


def test_live_camera_url(tmp_path):
    store = SQLiteViolationStore(db_path=str(tmp_path / "v.db"))
    app = create_app(store, upload_dir=str(tmp_path / "up"))
    client = app.test_client()
    r = client.post("/live", json={"camera_url": "rtsp://cam.example/0"})
    assert r.status_code == 200
    assert r.get_json()["camera_url"] == "rtsp://cam.example/0"


def test_violations_list(tmp_path, now):
    store = SQLiteViolationStore(db_path=str(tmp_path / "v.db"))
    _seed(store, now)
    app = create_app(store, upload_dir=str(tmp_path / "up"))
    client = app.test_client()
    r = client.get("/violations")
    body = r.get_json()
    assert body["count"] == 1
    assert body["violations"][0]["violation_id"] == "v1"


def test_export_csv(tmp_path, now):
    store = SQLiteViolationStore(db_path=str(tmp_path / "v.db"))
    _seed(store, now)
    app = create_app(store, upload_dir=str(tmp_path / "up"))
    client = app.test_client()
    r = client.get("/export.csv")
    assert r.status_code == 200
    assert r.mimetype == "text/csv"
    assert b"violation_id" in r.data
    assert b"v1" in r.data


def test_snapshot_route(tmp_path, now):
    snap = tmp_path / "s.jpg"
    snap.write_text("img")
    store = SQLiteViolationStore(db_path=str(tmp_path / "v.db"))
    rec = ViolationRecord(
        violation_id="snap",
        track_id=1,
        violation_type=ViolationType.NO_HELMET,
        timestamp=now,
        camera_id="cam-01",
        location="Main",
        plate_text=None,
        snapshot_path=str(snap),
    )
    store.save(rec)
    app = create_app(store, upload_dir=str(tmp_path / "up"))
    client = app.test_client()
    r = client.get("/violations/snap/snapshot")
    assert r.status_code == 200
    assert r.data == b"img"
