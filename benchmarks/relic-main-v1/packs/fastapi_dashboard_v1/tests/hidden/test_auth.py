"""Tests for signup, login, password hashing, and JWT issuance."""

from __future__ import annotations

from conftest import auth_headers


def test_signup_creates_user_and_hashes_password(client):
    resp = client.post(
        "/api/v1/auth/signup",
        json={
            "username": "bob",
            "email": "bob@example.com",
            "password": "bobpassword",
            "role": "user",
        },
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["username"] == "bob"
    assert body["email"] == "bob@example.com"
    assert body["role"] == "user"
    assert "id" in body
    assert "password" not in body
    assert "hashed_password" not in body

    from app.db import get_sessionmaker
    from app.models import User

    SessionLocal = get_sessionmaker()
    with SessionLocal() as session:
        user = session.query(User).filter(User.username == "bob").one()
        assert user.hashed_password != "bobpassword"
        assert user.hashed_password.startswith("$2")  # bcrypt prefix


def test_signup_rejects_duplicate_username(client):
    payload = {
        "username": "dupe",
        "email": "dupe1@example.com",
        "password": "password1",
        "role": "user",
    }
    first = client.post("/api/v1/auth/signup", json=payload)
    assert first.status_code == 201
    payload["email"] = "dupe2@example.com"
    second = client.post("/api/v1/auth/signup", json=payload)
    assert second.status_code == 409
    body = second.json()
    assert body["error"] == "conflict"
    assert "detail" in body


def test_login_returns_jwt_bearer(client, user_token):
    assert isinstance(user_token, str) and user_token.count(".") == 2


def test_login_with_wrong_password_returns_envelope(client):
    client.post(
        "/api/v1/auth/signup",
        json={
            "username": "carol",
            "email": "carol@example.com",
            "password": "carolpass1",
            "role": "user",
        },
    )
    resp = client.post(
        "/api/v1/auth/login",
        json={"username": "carol", "password": "wrong"},
    )
    assert resp.status_code == 401
    body = resp.json()
    assert body == {"error": "unauthorized", "detail": "invalid credentials"}


def test_validation_error_uses_envelope(client):
    resp = client.post(
        "/api/v1/auth/signup",
        json={"username": "x", "email": "not-an-email", "password": "short"},
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "validation_error"
    assert isinstance(body["detail"], str)


def test_oauth2_token_endpoint_accepts_form(client, user_token):  # noqa: ARG001 - just need user to exist
    resp = client.post(
        "/api/v1/auth/token",
        data={"username": "alice", "password": "alicepass1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"].count(".") == 2


def test_protected_route_requires_token(client):
    resp = client.get("/api/v1/users/me")
    assert resp.status_code == 401
    body = resp.json()
    assert body["error"] == "unauthorized"


def test_protected_route_returns_user_with_token(client, user_token):
    resp = client.get("/api/v1/users/me", headers=auth_headers(user_token))
    assert resp.status_code == 200
    body = resp.json()
    assert body["username"] == "alice"
    assert body["role"] == "user"


def test_invalid_token_returns_envelope(client):
    resp = client.get(
        "/api/v1/users/me",
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert resp.status_code == 401
    assert resp.json()["error"] == "unauthorized"
