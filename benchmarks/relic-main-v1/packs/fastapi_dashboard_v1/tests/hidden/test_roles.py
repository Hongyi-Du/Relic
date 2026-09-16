"""Tests for the role gate on admin-only endpoints."""

from __future__ import annotations

from conftest import auth_headers


def test_listing_users_requires_admin(client, user_token):
    resp = client.get("/api/v1/users", headers=auth_headers(user_token))
    assert resp.status_code == 403
    body = resp.json()
    assert body["error"] == "forbidden"
    assert "admin" in body["detail"].lower()


def test_admin_can_list_users(client, admin_token, user_token):  # noqa: ARG001 - ensure alice exists
    resp = client.get("/api/v1/users", headers=auth_headers(admin_token))
    assert resp.status_code == 200
    usernames = [u["username"] for u in resp.json()]
    assert "admin" in usernames
    assert "alice" in usernames


def test_anonymous_request_is_unauthorized_not_forbidden(client):
    resp = client.get("/api/v1/users")
    assert resp.status_code == 401
    assert resp.json()["error"] == "unauthorized"
