"""Persist the actual cognitive call boundary for pair-local causal audits."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4


class RecordedCooperClient:
    """Transparent client proxy; never records credentials or provider headers."""

    def __init__(self, world: Any, client: Any, path: Path):
        object.__setattr__(self, "_world", world)
        object.__setattr__(self, "_delegate", client)
        object.__setattr__(self, "_path", Path(path))
        object.__setattr__(self, "_sequence", 0)
        object.__setattr__(self, "_session_id", uuid4().hex)
        world._cooperbench_model_call_session_id = self._session_id

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    def __setattr__(self, name, value):
        if name in {"_world", "_delegate", "_path", "_sequence", "_session_id"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self._delegate, name, value)

    def _append(self, record):
        with self._path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _call(self, method, system, user, schema, kwargs):
        self._sequence += 1
        identity = {"session_id": self._session_id, "call_id": self._sequence,
                    "tick": int(getattr(self._world, "world_tick", 0)),
                    "actor_id": getattr(self._world, "_experiment_resource_actor", None),
                    "method": method, "boundary": "cognitive_client_input"}
        self._append({**identity, "phase": "request", "system_prompt": system,
                      "user_prompt": user, "schema": schema, "options": kwargs})
        try:
            # The 1000-tick pair treatment grants 500 logical queries per
            # member, with one shared 1000-query team ceiling. Charge that
            # budget at this worker-local cognitive boundary;
            # unlike the generic experiment meter, this wrapper is always
            # present in a real pair run, even when no main-matrix resource
            # ledger was configured.
            from environments.org_env.cooperbench.work_schedule import (
                reserve_model_call,
            )
            from environments.org_env.llm.client import LLMError

            try:
                reserve_model_call(self._world, identity["actor_id"])
            except RuntimeError as error:
                raise LLMError(str(error)) from None
            fn = getattr(self._delegate, method)
            response = fn(system, user, schema, **kwargs) if method == "generate_json" else fn(system, user, **kwargs)
        except Exception as error:
            self._append({**identity, "phase": "error", "error_type": type(error).__name__,
                          "error": str(error), "received_response": getattr(error, "response_text", None)})
            raise
        self._append({**identity, "phase": "response", "response": response})
        return response

    def generate_json(self, system_prompt, user_prompt, schema, **kwargs):
        return self._call("generate_json", system_prompt, user_prompt, schema, kwargs)

    def generate_text(self, system_prompt, user_prompt, **kwargs):
        return self._call("generate_text", system_prompt, user_prompt, None, kwargs)
