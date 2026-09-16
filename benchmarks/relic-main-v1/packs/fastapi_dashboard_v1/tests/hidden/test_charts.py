"""Tests for the visualization endpoints feeding the dashboard."""

from __future__ import annotations

from conftest import auth_headers


def test_sales_endpoint_returns_seeded_rows(client, user_token):
    resp = client.get("/api/v1/charts/sales", headers=auth_headers(user_token))
    assert resp.status_code == 200
    rows = resp.json()
    assert isinstance(rows, list)
    assert len(rows) >= 12  # seed: 3 categories x 4 months
    first = rows[0]
    assert set(first.keys()) == {"id", "category", "month", "amount"}


def test_sales_by_category_aggregates(client, user_token):
    resp = client.get(
        "/api/v1/charts/sales/by-category", headers=auth_headers(user_token)
    )
    assert resp.status_code == 200
    rows = resp.json()
    cats = {row["category"]: row["total"] for row in rows}
    assert set(cats.keys()) == {"books", "electronics", "clothing"}
    # books seed: 120 + 180.5 + 210 + 175.25 = 685.75
    assert abs(cats["books"] - 685.75) < 1e-6


def test_sales_by_month_aggregates(client, user_token):
    resp = client.get(
        "/api/v1/charts/sales/by-month", headers=auth_headers(user_token)
    )
    assert resp.status_code == 200
    rows = resp.json()
    months = {row["month"]: row["total"] for row in rows}
    assert set(months.keys()) == {"2024-01", "2024-02", "2024-03", "2024-04"}
    # 2024-01: 120 + 540 + 230 = 890
    assert abs(months["2024-01"] - 890.0) < 1e-6


def test_charts_endpoints_require_auth(client):
    for path in (
        "/api/v1/charts/sales",
        "/api/v1/charts/sales/by-category",
        "/api/v1/charts/sales/by-month",
    ):
        resp = client.get(path)
        assert resp.status_code == 401, path
        assert resp.json()["error"] == "unauthorized"
