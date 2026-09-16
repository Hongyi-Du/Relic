"""Dependency-free HTTP server for the public Relic Inspector."""

from __future__ import annotations

import json
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from relic.replay import load_trace
from relic.replay.hashing import canonical_sha256


class InspectorError(ValueError):
    """Raised when an Inspector launch request is invalid."""


def inspector_static_root() -> Path:
    """Return the packaged static root after checking the release bundle."""

    static_root = Path(__file__).resolve().parent / "static"
    for name in ("index.html", "app.css", "app.js"):
        if not (static_root / name).is_file():
            raise InspectorError(f"bundled Inspector asset is missing: {name}")
    return static_root


class TraceSource:
    """Validated replay source with append-only live-file refresh."""

    def __init__(self, path: str | Path, mode: str) -> None:
        if mode not in {"replay", "live"}:
            raise InspectorError("Inspector mode must be 'replay' or 'live'")
        self.path = Path(path).expanduser().resolve()
        self.mode = mode
        self._lock = threading.Lock()
        self._cached = load_trace(self.path)
        self._last_error: str | None = None

    @property
    def degraded(self) -> bool:
        return self._last_error is not None

    def read(self) -> dict[str, Any]:
        if self.mode == "replay":
            return self._cached
        with self._lock:
            try:
                candidate = load_trace(self.path)
                self._validate_append_only(candidate)
            except (OSError, ValueError) as exc:
                self._last_error = type(exc).__name__
                return self._cached
            self._cached = candidate
            self._last_error = None
            return self._cached

    def _validate_append_only(self, candidate: dict[str, Any]) -> None:
        immutable_keys = (set(self._cached) | set(candidate)) - {"frames", "trace_sha256"}
        for key in sorted(immutable_keys):
            if (
                key not in candidate
                or key not in self._cached
                or canonical_sha256(candidate[key]) != canonical_sha256(self._cached[key])
            ):
                raise InspectorError(f"live trace changed immutable field: {key}")
        current_frames = self._cached["frames"]
        candidate_frames = candidate["frames"]
        if len(candidate_frames) < len(current_frames):
            raise InspectorError("live trace cannot remove published frames")
        if canonical_sha256(candidate_frames[: len(current_frames)]) != canonical_sha256(
            current_frames
        ):
            raise InspectorError("live trace cannot rewrite published frames")


class InspectorServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    trace_source: TraceSource
    static_root: Path
    verbose: bool
    allow_remote: bool


class InspectorRequestHandler(BaseHTTPRequestHandler):
    server: InspectorServer
    protocol_version = "HTTP/1.1"
    server_version = "RelicInspector"
    sys_version = ""

    _STATIC_FILES = {
        "/": ("index.html", "text/html; charset=utf-8"),
        "/index.html": ("index.html", "text/html; charset=utf-8"),
        "/app.css": ("app.css", "text/css; charset=utf-8"),
        "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    }

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._dispatch(body=True)

    def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._dispatch(body=False)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._method_not_allowed()

    def do_PUT(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._method_not_allowed()

    def do_PATCH(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._method_not_allowed()

    def do_DELETE(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._method_not_allowed()

    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._method_not_allowed()

    def _dispatch(self, *, body: bool) -> None:
        if not self.server.allow_remote and not _is_loopback_host_header(self.headers.get("Host")):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"status": "invalid_host"},
                body=body,
            )
            return
        path = urlsplit(self.path).path
        if path == "/api/health":
            trace = self.server.trace_source.read()
            status = "degraded" if self.server.trace_source.degraded else "ready"
            self._send_json(
                HTTPStatus.OK,
                {
                    "schema_version": "relic-inspector-health-v1",
                    "status": status,
                    "mode": self.server.trace_source.mode,
                    "trace_schema_version": trace["schema_version"],
                    "run_id": trace["run_id"],
                    "last_verified_frame": len(trace["frames"]) - 1,
                },
                body=body,
            )
            return
        if path == "/api/trace":
            self._send_json(HTTPStatus.OK, self.server.trace_source.read(), body=body)
            return
        static = self._STATIC_FILES.get(path)
        if static is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"status": "not_found"}, body=body)
            return
        file_name, content_type = static
        try:
            payload = (self.server.static_root / file_name).read_bytes()
        except OSError:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"status": "asset_unavailable"},
                body=body,
            )
            return
        self._send_bytes(HTTPStatus.OK, payload, content_type, body=body)

    def _method_not_allowed(self) -> None:
        self._send_json(
            HTTPStatus.METHOD_NOT_ALLOWED,
            {"status": "method_not_allowed"},
            extra_headers={"Allow": "GET, HEAD"},
        )

    def _send_json(
        self,
        status: HTTPStatus,
        payload: dict[str, Any],
        *,
        body: bool = True,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self._send_bytes(
            status,
            encoded,
            "application/json; charset=utf-8",
            body=body,
            extra_headers=extra_headers,
        )

    def _send_bytes(
        self,
        status: HTTPStatus,
        payload: bytes,
        content_type: str,
        *,
        body: bool = True,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; base-uri 'none'; "
            "form-action 'none'; frame-ancestors 'none'; object-src 'none'",
        )
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if body:
            self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        if self.server.verbose:
            super().log_message(format, *args)


def _validate_host(host: str, *, allow_remote: bool) -> str:
    selected = host.strip()
    if not selected:
        raise InspectorError("Inspector host cannot be empty")
    if selected not in {"127.0.0.1", "localhost"} and not allow_remote:
        raise InspectorError(
            "non-loopback Inspector binding requires the explicit --allow-remote option"
        )
    return selected


def _is_loopback_host_header(value: str | None) -> bool:
    if not value:
        return False
    try:
        parsed = urlsplit(f"//{value}")
        _ = parsed.port
    except ValueError:
        return False
    return (
        parsed.hostname in {"127.0.0.1", "localhost"}
        and parsed.username is None
        and parsed.password is None
        and not parsed.path
        and not parsed.query
        and not parsed.fragment
    )


def create_inspector_server(
    *,
    trace_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    mode: str = "replay",
    verbose: bool = False,
    allow_remote: bool = False,
) -> InspectorServer:
    """Create a validated Inspector server without starting its event loop."""

    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise InspectorError("Inspector port must be an integer from 0 to 65535")
    selected_host = _validate_host(host, allow_remote=allow_remote)
    static_root = inspector_static_root()
    trace_source = TraceSource(trace_path, mode)
    server = InspectorServer((selected_host, port), InspectorRequestHandler)
    server.trace_source = trace_source
    server.static_root = static_root
    server.verbose = verbose
    server.allow_remote = allow_remote
    return server


def serve_inspector(
    *,
    trace_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    mode: str = "replay",
    open_browser: bool = False,
    verbose: bool = False,
    allow_remote: bool = False,
) -> None:
    """Serve one public trace until interrupted."""

    server = create_inspector_server(
        trace_path=trace_path,
        host=host,
        port=port,
        mode=mode,
        verbose=verbose,
        allow_remote=allow_remote,
    )
    actual_port = int(server.server_address[1])
    bound_host = str(server.server_address[0])
    display_host = "127.0.0.1" if bound_host == "0.0.0.0" else bound_host
    url = f"http://{display_host}:{actual_port}/"
    try:
        if bound_host not in {"127.0.0.1", "localhost"}:
            print(
                "relic: warning: Inspector is unauthenticated and bound beyond loopback",
                file=sys.stderr,
                flush=True,
            )
        if open_browser and not webbrowser.open(url):
            raise InspectorError("no browser accepted the Inspector URL")
        print(
            json.dumps(
                {
                    "schema_version": "relic-inspector-launch-v1",
                    "status": "ready",
                    "mode": mode,
                    "trace_path": str(server.trace_source.path),
                    "url": url,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


__all__ = [
    "InspectorError",
    "TraceSource",
    "create_inspector_server",
    "inspector_static_root",
    "serve_inspector",
]
