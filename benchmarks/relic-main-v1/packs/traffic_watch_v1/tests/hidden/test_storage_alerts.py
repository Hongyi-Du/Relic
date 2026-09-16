"""Storage, alert dispatch, and record schema."""

from __future__ import annotations

from datetime import datetime

from traffic_violation_system import (
    Alert,
    AlertDispatcher,
    SQLiteViolationStore,
    ViolationRecord,
    ViolationType,
)


def _make_record(now, vid="v1") -> ViolationRecord:
    return ViolationRecord(
        violation_id=vid,
        track_id=42,
        violation_type=ViolationType.OVER_SPEED,
        timestamp=now,
        camera_id="cam-01",
        location="Main & 1st",
        plate_text="ABC123",
        snapshot_path="/tmp/x.jpg",
        measured_value=88.0,
    )


def test_record_fields(now):
    rec = _make_record(now)
    d = rec.to_dict()
    for f in ("violation_id", "track_id", "violation_type", "timestamp",
              "camera_id", "location", "plate_text", "snapshot_path"):
        assert f in d


def test_store_round_trip(tmp_path, now):
    s = SQLiteViolationStore(db_path=str(tmp_path / "v.db"))
    s.save(_make_record(now))
    rows = s.all()
    assert len(rows) == 1
    assert rows[0].plate_text == "ABC123"
    assert rows[0].violation_type is ViolationType.OVER_SPEED
    s.close()


def test_store_accepts_null_plate(tmp_path, now):
    s = SQLiteViolationStore(db_path=str(tmp_path / "v.db"))
    rec = _make_record(now)
    rec.plate_text = None
    s.save(rec)
    rows = s.all()
    assert rows[0].plate_text is None
    s.close()


def test_alert_dispatcher_feed(now):
    d = AlertDispatcher()
    a = d.dispatch(_make_record(now))
    assert isinstance(a, Alert)
    feed = d.feed()
    assert len(feed) == 1
    assert feed[0].record.violation_id == "v1"


def test_alert_dispatcher_webhook(now):
    calls = []

    def poster(url, payload):
        calls.append((url, payload))

    d = AlertDispatcher(webhook_url="http://x.test/hook", webhook_poster=poster)
    d.dispatch(_make_record(now))
    assert len(calls) == 1
    assert calls[0][0] == "http://x.test/hook"
    assert calls[0][1]["violation_id"] == "v1"


def test_sqlite_strict_schema(tmp_path):
    from traffic_violation_system import SQLiteViolationStore
    import sqlite3
    
    db_path = str(tmp_path / "schema.db")
    store = SQLiteViolationStore(db_path=db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(violation_records)")
    columns = {row[1]: row[2].upper() for row in cursor.fetchall()}
    
    assert "violation_id" in columns
    assert "TEXT" in columns["violation_id"]
    assert columns.get("track_id") == "INTEGER"
    assert columns.get("measured_value") == "REAL"
    assert "BOOLEAN" not in columns.values(), "Native Boolean type used instead of TEXT/INTEGER"
    assert "DATETIME" not in columns.values(), "Native DateTime type used instead of TEXT"
    store.close()


def test_webhook_resilience(system_config, now):
    from traffic_violation_system import AlertDispatcher, ViolationRecord, ViolationType
    
    def crashing_poster(url, payload):
        raise RuntimeError("Simulated Network Timeout")
    
    alerts = AlertDispatcher(webhook_url="http://fake-url.com", webhook_poster=crashing_poster)
    rec = ViolationRecord(
        violation_id="v1", track_id=1, violation_type=ViolationType.NO_SEATBELT,
        timestamp=now, camera_id="c1", location="loc", plate_text="ABC", snapshot_path="path"
    )
    
    # A pipeline NÃO pode dar crash aqui
    alert_obj = alerts.dispatch(rec)
    
    # O objeto tem que retornar normalmente e o alerta tem que ir pra fila da memória
    assert alert_obj is not None
    assert len(alerts.feed()) == 1