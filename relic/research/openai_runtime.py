"""Shared OpenAI-compatible runtime configuration without secret persistence."""

from __future__ import annotations

import json
import os
import re
import signal
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from time import monotonic
from typing import Any

from .hashing import canonicalize, stable_hash

DEFAULT_HEADERS_ENV = "RELIC_OPENAI_DEFAULT_HEADERS_JSON"
DISABLE_RESPONSE_STORAGE_ENV = "RELIC_OPENAI_DISABLE_RESPONSE_STORAGE"
_HEADER_NAME = re.compile(r"^[A-Za-z0-9!#$%&'*+.^_`|~-]+$")
_FORBIDDEN_CONFIGURED_HEADERS = frozenset(
    {
        "authorization",
        "content-length",
        "host",
        "proxy-authorization",
    }
)


class OpenAICallDeadlineExceeded(TimeoutError):
    """Raised when one OpenAI call exceeds its total wall-clock budget."""


@contextmanager
def openai_call_watchdog(
    seconds: float,
    *,
    label: str = "OpenAI call",
):
    """Enforce a wall-clock deadline when the platform supports SIGALRM."""

    if seconds <= 0:
        yield
        return

    on_main_thread = threading.current_thread() is threading.main_thread()
    if (
        not hasattr(signal, "SIGALRM")
        or not hasattr(signal, "setitimer")
        or not on_main_thread
    ):
        # SIGALRM only works on the main thread, and yielding bare here left
        # every worker-thread call with NO deadline at all. Measured: the P5 arm
        # is the only condition that executes agents in parallel, so its LLM
        # calls run off-main and its watchdog was silently inert — the run sat
        # at 0% CPU for twenty minutes, twice, and resuming from a checkpoint
        # stalled at the same tick.
        #
        # A timer thread cannot interrupt a blocked socket read, so it does not
        # replace the signal path; what it does is guarantee the stall is
        # REPORTED rather than silent, which is the difference between a run
        # that looks slow and one a person can diagnose.
        timer = threading.Timer(
            seconds,
            lambda: print(
                f"[watchdog] {label} exceeded {seconds:.0f}s on a worker thread; "
                "SIGALRM cannot fire off-main, so this call is not being "
                "interrupted - the process may be stalled on a socket read",
                flush=True,
            ),
        )
        timer.daemon = True
        timer.start()
        try:
            yield
        finally:
            timer.cancel()
        return

    def _raise_timeout(signum, frame):
        del signum, frame
        raise OpenAICallDeadlineExceeded(
            f"{label} exceeded {seconds:.3f}s wall-clock deadline"
        )

    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    started_at = monotonic()
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0] > 0:
            elapsed = max(0.0, monotonic() - started_at)
            remaining = old_timer[0] - elapsed
            if remaining <= 0:
                if old_timer[1] > 0:
                    missed_intervals = int((-remaining) // old_timer[1]) + 1
                    remaining += missed_intervals * old_timer[1]
                else:
                    remaining = 1e-6
            signal.setitimer(signal.ITIMER_REAL, remaining, old_timer[1])


def validate_openai_default_headers(
    payload: Mapping[str, Any],
) -> dict[str, str]:
    """Validate provider compatibility headers without persisting credentials."""

    if not isinstance(payload, Mapping):
        raise TypeError("openai_default_headers_must_be_object")
    headers: dict[str, str] = {}
    for raw_name, raw_value in payload.items():
        name = str(raw_name).strip()
        if not name or not _HEADER_NAME.fullmatch(name):
            raise ValueError("invalid_openai_default_header_name")
        if name.lower() in _FORBIDDEN_CONFIGURED_HEADERS:
            raise ValueError(f"forbidden_openai_default_header:{name.lower()}")
        if not isinstance(raw_value, str):
            raise TypeError(
                f"openai_default_header_value_must_be_string:{name.lower()}"
            )
        value = raw_value.strip()
        if (
            not value
            or len(value) > 4_096
            or any(
                ord(character) < 0x20 or ord(character) == 0x7F for character in value
            )
        ):
            raise ValueError(f"invalid_openai_default_header_value:{name.lower()}")
        headers[name] = value
    return dict(sorted(headers.items(), key=lambda item: item[0].lower()))


def configured_openai_default_headers() -> dict[str, str]:
    """Return validated compatibility headers from a JSON environment value."""

    raw = os.environ.get(DEFAULT_HEADERS_ENV, "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid_openai_default_headers_json") from exc
    if not isinstance(payload, Mapping):
        raise TypeError("openai_default_headers_must_be_object")
    return validate_openai_default_headers(payload)


def openai_response_storage_disabled() -> bool:
    """Default to private, non-persistent Responses API calls."""

    raw = os.environ.get(DISABLE_RESPONSE_STORAGE_ENV, "true")
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("invalid_openai_disable_response_storage")


def prepare_openai_response_request(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply shared request defaults while preserving explicit caller values."""

    prepared = dict(request)
    if openai_response_storage_disabled():
        prepared.setdefault("store", False)
    return prepared


def build_openai_client(
    *,
    api_key: str | None = None,
    timeout_seconds: float | None = None,
    max_retries: int = 0,
) -> Any:
    """Build one client with identical endpoint and compatibility headers."""

    selected_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not selected_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    from openai import OpenAI

    kwargs: dict[str, Any] = {
        "api_key": selected_key,
        "max_retries": max_retries,
    }
    if timeout_seconds is not None:
        kwargs["timeout"] = timeout_seconds
    endpoint = os.environ.get("OPENAI_BASE_URL", "").strip()
    if endpoint:
        kwargs["base_url"] = endpoint
    headers = configured_openai_default_headers()
    if headers:
        kwargs["default_headers"] = headers
    return OpenAI(**kwargs)


def openai_runtime_identity() -> dict[str, Any]:
    """Return a non-secret identity for experiment provenance and resumption."""

    endpoint = os.environ.get("OPENAI_BASE_URL", "").strip()
    headers = configured_openai_default_headers()
    normalized_header_names = tuple(sorted(name.lower() for name in headers))
    return canonicalize(
        {
            "endpoint_kind": ("custom_compatible" if endpoint else "official_default"),
            "endpoint_hash": stable_hash(endpoint or "official_openai_default"),
            "default_header_names": normalized_header_names,
            "default_headers_hash": stable_hash(headers),
            "response_storage_disabled": (openai_response_storage_disabled()),
        }
    )
