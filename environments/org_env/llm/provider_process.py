"""Spawn-safe OpenAI provider attempt worker.

Only plain, in-memory request/configuration data crosses the process boundary.
The worker deliberately returns a small response projection instead of an SDK
object: OpenAI response models and their transports are not guaranteed to be
picklable, and the parent only needs text, usage, model, and response id.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Mapping

PROVIDER_RESULT_BUFFER_BYTES = 4 * 1024 * 1024


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _usage_projection(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    projected: dict[str, Any] = {}
    for name in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
    ):
        value = _field(usage, name)
        if value is not None:
            try:
                projected[name] = max(0, int(value))
            except (TypeError, ValueError):
                pass
    for details_name in (
        "prompt_tokens_details",
        "input_tokens_details",
    ):
        details = _field(usage, details_name)
        if details is None:
            continue
        cached_tokens = _field(details, "cached_tokens")
        if cached_tokens is not None:
            try:
                projected[details_name] = {
                    "cached_tokens": max(0, int(cached_tokens))
                }
            except (TypeError, ValueError):
                pass
    return projected


def project_openai_response(response: Any, wire_api: str) -> dict[str, Any]:
    """Return the minimal pickle-safe subset consumed by ``client.py``."""

    result: dict[str, Any] = {
        "id": str(_field(response, "id") or ""),
        "model": str(_field(response, "model") or ""),
        "usage": _usage_projection(_field(response, "usage")),
    }
    if wire_api == "chat_completions":
        choices = _field(response, "choices", ()) or ()
        message = _field(choices[0], "message") if choices else None
        chat_text = _field(message, "content", "")
        result["chat_text"] = chat_text if isinstance(chat_text, str) else ""
        # Why the model stopped. Without it a body cut off at the ceiling is
        # indistinguishable from a body the model chose to write that way: both
        # arrive as text that will not parse, and the caller can only record a
        # generic failure. This host takes the subprocess path on every call --
        # _requires_provider_process_deadline is true whenever signal.SIGALRM is
        # missing, which is always on Windows -- so dropping the field here hid
        # truncation from the entire run.
        finish_reason = _field(choices[0], "finish_reason") if choices else None
        result["finish_reason"] = (
            finish_reason if isinstance(finish_reason, str) else ""
        )
        return result

    # The Responses API says the same thing with a status and a reason.
    result["status"] = str(_field(response, "status") or "")
    incomplete = _field(response, "incomplete_details")
    result["incomplete_reason"] = str(_field(incomplete, "reason") or "")
    output_text = _field(response, "output_text")
    if isinstance(output_text, str):
        result["output_text"] = output_text
    output_projection: list[dict[str, Any]] = []
    for item in _field(response, "output", ()) or ():
        content_projection: list[dict[str, Any]] = []
        for content in _field(item, "content", ()) or ():
            text = _field(content, "text")
            if isinstance(text, str):
                content_projection.append({"text": text})
        output_projection.append({"content": content_projection})
    result["output"] = output_projection
    return result


def restore_openai_response(payload: Mapping[str, Any], wire_api: str) -> Any:
    """Recreate the attribute surface used by ``OpenAIOrgLLMClient``."""

    common = {
        "id": payload.get("id"),
        "model": payload.get("model"),
        "usage": payload.get("usage"),
    }
    if wire_api == "chat_completions":
        return SimpleNamespace(
            **common,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=payload.get("chat_text", "")),
                    finish_reason=payload.get("finish_reason", ""),
                )
            ],
        )
    output = [
        SimpleNamespace(
            content=[
                SimpleNamespace(text=content.get("text"))
                for content in item.get("content", ())
                if isinstance(content, Mapping)
            ]
        )
        for item in payload.get("output", ())
        if isinstance(item, Mapping)
    ]
    return SimpleNamespace(
        **common,
        output_text=payload.get("output_text"),
        output=output,
        status=payload.get("status", ""),
        incomplete_details=SimpleNamespace(
            reason=payload.get("incomplete_reason", "")
        ),
    )


def _phase_timeout(seconds: float) -> Any:
    try:
        import httpx

        return httpx.Timeout(
            connect=min(30.0, seconds),
            read=seconds,
            write=min(30.0, seconds),
            pool=min(30.0, seconds),
        )
    except Exception:
        return seconds


def _publish_result(
    result_buffer: Any,
    result_length: Any,
    done_event: Any,
    envelope: Mapping[str, Any],
) -> None:
    try:
        encoded = json.dumps(
            dict(envelope),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except Exception:
        encoded = b'{"kind":"provider_error","exception_type":"ProjectionError"}'
    if len(encoded) > len(result_buffer):
        encoded = (
            b'{"kind":"provider_error",'
            b'"exception_type":"ResponseProjectionTooLarge"}'
        )
    result_buffer[: len(encoded)] = encoded
    result_length.value = len(encoded)
    done_event.set()


def _read_request(request_buffer: Any, request_length: Any) -> dict[str, Any]:
    """Take the request out of shared memory rather than the spawn payload.

    Passing it as a Process argument put the whole prompt through the pipe that
    spawn uses to hand the child its arguments, and the parent's flush of that
    pipe is the one step of the attempt that no deadline covers: a child that
    dies before draining leaves the parent blocked in Popen.__init__ forever.
    Shared memory is what the result already travels through, for the same
    reason -- this makes the outbound direction match.
    """

    length = int(request_length.value)
    if length <= 0 or length > len(request_buffer):
        raise ValueError("provider_request_projection_invalid")
    return json.loads(bytes(request_buffer[:length]).decode("utf-8"))


def openai_provider_process_worker(
    ready_event: Any,
    go_event: Any,
    done_event: Any,
    result_buffer: Any,
    result_length: Any,
    wire_api: str,
    client_options: Mapping[str, Any],
    request_buffer: Any,
    request_length: Any,
    timeout_seconds: float,
) -> None:
    """Execute one SDK attempt and send one secret-free result envelope.

    This function is intentionally top-level so Windows ``spawn`` can import
    it.  It never retries: retry admission/accounting belongs to the parent.
    Exception messages are not returned because some gateways include request
    headers or endpoint credentials in transport error strings.
    """

    client = None
    try:
        from openai import OpenAI  # type: ignore

        timeout = _phase_timeout(max(1e-6, float(timeout_seconds)))
        options = dict(client_options)
        options["timeout"] = timeout
        options["max_retries"] = 0
        client = OpenAI(**options)
        if wire_api == "chat_completions":
            create = client.chat.completions.create
        elif wire_api == "responses":
            create = client.responses.create
        else:
            raise ValueError("unsupported_openai_wire_api")
        ready_event.set()
        if not go_event.wait(timeout=max(1e-6, float(timeout_seconds))):
            return
        payload = _read_request(request_buffer, request_length)
        payload["timeout"] = timeout
        response = create(**payload)
        _publish_result(
            result_buffer,
            result_length,
            done_event,
            {
                "kind": "success",
                "response": project_openai_response(response, wire_api),
            },
        )
    except Exception as exc:
        # Do not send repr/str(exc): provider exceptions can echo credentials.
        try:
            _publish_result(
                result_buffer,
                result_length,
                done_event,
                {
                    "kind": (
                        "provider_error" if ready_event.is_set() else "setup_error"
                    ),
                    "exception_type": type(exc).__name__,
                },
            )
        except Exception:
            pass
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
__all__ = [
    "PROVIDER_RESULT_BUFFER_BYTES",
    "openai_provider_process_worker",
    "project_openai_response",
    "restore_openai_response",
]
