"""Neutral public-evidence redaction used by optional benchmark adapters."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


_SECRET = re.compile(
    r"(?i)(?:bearer\s+)?(?:sk-(?:proj-)?[A-Za-z0-9_-]{16,}|api[_-]?key\s*[:=]\s*[^\s,;]+)"
)
_WINDOWS_ABSOLUTE = re.compile(r"(?i)(?:[a-z]:[\\/]|\\\\)[^\s\"'<>]*")
_PRIVATE_PATH = re.compile(
    r"(?i)(?:/[^\s\"'<>]*)?(?:tests[\\/]hidden|"
    r"reference[_-]?repo|private[_-]?vault|evaluator[_-]?(?:asset|root))"
    r"[^\s\"'<>]*"
)
_FORBIDDEN_KEYS = {
    "authorization",
    "api_key",
    "secret",
    "vault_token",
    "hidden_test",
    "hidden_tests",
    "hidden_result",
    "oracle",
    "oracle_bytes",
    "reference_executable",
    "evaluator_asset",
    "evaluator_assets",
}
_MAX_TEXT = 2048


class PublicEvidenceError(ValueError):
    """A value would cross the public-evidence boundary."""


def redact_public_text(
    value: Any,
    *,
    private_markers: Sequence[str] = (),
    max_chars: int = _MAX_TEXT,
) -> str:
    """Bound text and remove credentials and private/evaluator path spellings."""

    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
        raise ValueError("public_evidence_text_limit_invalid")
    text = str(value or "")
    for marker in sorted(
        {str(item) for item in private_markers if str(item)},
        key=len,
        reverse=True,
    ):
        forms = {marker, marker.replace("\\", "/"), marker.replace("/", "\\")}
        for form in forms:
            text = text.replace(form, "[redacted-path]")
    text = _SECRET.sub("[redacted-secret]", text)
    text = _WINDOWS_ABSOLUTE.sub("[redacted-path]", text)
    text = _PRIVATE_PATH.sub("[redacted-path]", text)
    return text[:max_chars]


def redact_public_evidence(
    value: Any,
    *,
    private_markers: Sequence[str] = (),
) -> Any:
    """Recursively redact JSON-like evidence and reject evaluator-only keys."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key in sorted(value, key=lambda item: str(item)):
            key = str(raw_key)
            normalized = key.casefold().replace("-", "_")
            if normalized in _FORBIDDEN_KEYS:
                raise PublicEvidenceError("public_evidence_forbidden_key:" + normalized)
            result[key] = redact_public_evidence(
                value[raw_key], private_markers=private_markers
            )
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            redact_public_evidence(item, private_markers=private_markers)
            for item in value
        ]
    if isinstance(value, str):
        return redact_public_text(value, private_markers=private_markers)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact_public_text(value, private_markers=private_markers)


__all__ = ["PublicEvidenceError", "redact_public_evidence", "redact_public_text"]
