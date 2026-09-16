"""Flask-based admin dashboard."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request, send_file, abort

from .reports import CSV_COLUMNS, ReportExporter
from .storage import ViolationStore


def create_app(
    store: ViolationStore,
    upload_dir: str | Path = "/tmp/uploads",
    exporter: Optional[ReportExporter] = None,
    live_camera_url: Optional[str] = None,
) -> Flask:
    app = Flask(__name__)
    app.config["UPLOAD_DIR"] = str(upload_dir)
    Path(upload_dir).mkdir(parents=True, exist_ok=True)
    exporter = exporter or ReportExporter()

    @app.get("/")
    def index():
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Traffic Violation Dashboard</title></head>"
            "<body><h1>Traffic Violation Dashboard</h1>"
            "<p>POST /upload, GET /violations, GET /export.csv</p>"
            "</body></html>"
        )

    @app.post("/upload")
    def upload():
        if "video" not in request.files:
            return jsonify({"error": "no video field"}), 400
        f = request.files["video"]
        if not f.filename:
            return jsonify({"error": "empty filename"}), 400
        target = Path(app.config["UPLOAD_DIR"]) / f.filename
        f.save(str(target))
        return jsonify({"status": "queued", "path": str(target)})

    @app.post("/live")
    def live():
        data = request.get_json(silent=True) or {}
        url = data.get("camera_url")
        if not url:
            return jsonify({"error": "camera_url required"}), 400
        return jsonify({"status": "tracking", "camera_url": url})

    @app.get("/violations")
    def violations():
        try:
            limit = int(request.args.get("limit", "50"))
        except ValueError:
            limit = 50
        rows = [r.to_dict() for r in store.list_recent(limit=limit)]
        return jsonify({"violations": rows, "count": len(rows)})

    @app.get("/violations/<violation_id>/snapshot")
    def snapshot(violation_id):
        for r in store.all():
            if r.violation_id == violation_id:
                p = Path(r.snapshot_path)
                if not p.exists():
                    abort(404)
                return send_file(str(p))
        abort(404)

    @app.get("/export.csv")
    def export_csv():
        rows = store.all()
        buf = io.StringIO()
        import csv

        writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            d = r.to_dict()
            writer.writerow({k: d.get(k, "") for k in CSV_COLUMNS})
        out = io.BytesIO(buf.getvalue().encode("utf-8"))
        out.seek(0)
        return send_file(
            out,
            mimetype="text/csv",
            as_attachment=True,
            download_name="violations.csv",
        )

    return app
