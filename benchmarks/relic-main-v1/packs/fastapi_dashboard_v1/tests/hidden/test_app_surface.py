"""Tests for cross-cutting app properties: prefix, error envelope, static assets."""

from __future__ import annotations

from pathlib import Path

from conftest import auth_headers


def test_api_mounted_under_versioned_prefix(client, user_token):
    # Versioned route works
    ok = client.get("/api/v1/users/me", headers=auth_headers(user_token))
    assert ok.status_code == 200
    # Unversioned route does not exist
    missing = client.get("/users/me", headers=auth_headers(user_token))
    assert missing.status_code == 404
    assert missing.json()["error"] == "not_found"


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_dashboard_root_serves_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    body = resp.text
    assert "<canvas" in body
    # Chart.js loaded from vendored static, not a CDN
    assert "/static/chart.min.js" in body
    assert "cdn" not in body.lower()


def test_chartjs_is_vendored_on_disk():
    import app

    static_dir = Path(app.__file__).resolve().parent / "static"
    asset = static_dir / "chart.min.js"
    assert asset.is_file(), "Chart.js must be vendored under solution/app/static/"
    contents = asset.read_text(encoding="utf-8")
    assert "Chart" in contents


def test_static_assets_served_by_app(client):
    resp = client.get("/static/chart.min.js")
    assert resp.status_code == 200
    assert "javascript" in resp.headers["content-type"].lower()


def test_openapi_spec_lists_versioned_paths(client):
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]
    assert "/api/v1/auth/signup" in paths
    assert "/api/v1/auth/login" in paths
    assert "/api/v1/users/me" in paths
    assert "/api/v1/charts/sales" in paths


def test_method_not_allowed_uses_envelope(client):
    resp = client.delete("/api/v1/auth/login")
    assert resp.status_code == 405
    body = resp.json()
    assert "error" in body and "detail" in body
