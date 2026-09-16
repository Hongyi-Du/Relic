"""Public, trace-only Relic Inspector."""

from relic.inspector.server import (
    InspectorError,
    create_inspector_server,
    inspector_static_root,
    serve_inspector,
)

__all__ = [
    "InspectorError",
    "create_inspector_server",
    "inspector_static_root",
    "serve_inspector",
]
