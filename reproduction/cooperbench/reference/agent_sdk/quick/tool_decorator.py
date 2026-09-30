"""@tool decorator — auto-generates ToolDescriptor from function signature."""
from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Optional, get_type_hints

from agent_sdk.contracts.tools import ToolDescriptor

# Python type -> JSON Schema type
_TYPE_MAP: Dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json(py_type: Any) -> str:
    """Map a Python type annotation to a JSON Schema type string."""
    origin = getattr(py_type, "__origin__", None)
    if origin is not None:
        # Handle Optional[X] -> extract X
        args = getattr(py_type, "__args__", ())
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _TYPE_MAP.get(non_none[0], "string")
    return _TYPE_MAP.get(py_type, "string")


def _build_schema(fn: Callable) -> Dict[str, Any]:
    """Build a JSON Schema 'parameters' object from function signature."""
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    properties: Dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        py_type = hints.get(name, str)
        if name == "return":
            continue
        json_type = _python_type_to_json(py_type)
        properties[name] = {"type": json_type}

        # If no default -> required
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


class ToolWrapper:
    """Wraps a function with its ToolDescriptor. Callable as the original function."""

    def __init__(self, fn: Callable, descriptor: ToolDescriptor):
        self.fn = fn
        self.descriptor = descriptor
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self.__module__ = fn.__module__

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def tool(fn: Callable) -> ToolWrapper:
    """
    Decorator that turns a Python function into a tool for Agent.

    Usage:
        @tool
        def get_weather(city: str) -> str:
            \"\"\"Get weather for a city.\"\"\"
            return "sunny"

    The decorated function gains:
        .descriptor  -- ToolDescriptor with auto-generated JSON Schema
        .fn          -- the original function
    """
    schema = _build_schema(fn)
    description = (fn.__doc__ or "").strip()
    descriptor = ToolDescriptor(
        name=fn.__name__,
        description=description,
        schema=schema,
    )
    return ToolWrapper(fn, descriptor)
