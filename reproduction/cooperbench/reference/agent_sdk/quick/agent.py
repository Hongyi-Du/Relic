"""Agent — simple facade for standalone LLM agent usage."""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from agent_sdk.llm import LLMCore
from .tool_decorator import ToolWrapper


class Agent:
    """
    Simple LLM agent with tool calling support.

    Usage:
        agent = Agent(model="gpt-4o", api_key="sk-...")
        reply = agent.chat("Hello!")
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: str = "together",
        system_prompt: str = "",
        tools: Optional[List[ToolWrapper]] = None,
        memory: Any = None,
        name: str = "agent",
        max_tool_rounds: int = 10,
        llm: Optional[LLMCore] = None,
    ):
        self.name = name
        self._system_prompt = system_prompt
        self._memory = memory
        self._max_tool_rounds = max_tool_rounds

        # LLM backend
        self._llm = llm or LLMCore(provider=provider, model=model, api_key=api_key)

        # Tool registry
        self._tool_fns: Dict[str, Callable] = {}
        self._tool_schemas: List[Dict[str, Any]] = []
        if tools:
            for t in tools:
                self._register_tool(t)

        # Message history
        self._messages: List[Dict[str, str]] = []
        if self._system_prompt:
            self._messages.append({"role": "system", "content": self._system_prompt})

    def _register_tool(self, t: ToolWrapper) -> None:
        """Register a @tool-decorated function."""
        self._tool_fns[t.descriptor.name] = t.fn
        self._tool_schemas.append({
            "type": "function",
            "function": {
                "name": t.descriptor.name,
                "description": t.descriptor.description,
                "parameters": t.descriptor.schema,
            },
        })

    def chat(self, message: str) -> str:
        """
        Send a message and get a reply. Handles multi-round tool calling.

        Returns the final text reply from the LLM.
        """
        self._messages.append({"role": "user", "content": message})

        for _ in range(self._max_tool_rounds):
            response = self._llm.create_completion_with_retry(
                messages=self._messages,
                tools=self._tool_schemas if self._tool_schemas else None,
                tool_choice="auto" if self._tool_schemas else "none",
            )

            if response is None:
                self._messages.append({"role": "assistant", "content": "[LLM call failed]"})
                return "[LLM call failed]"

            # Check for tool calls
            tool_call = self._llm.parse_tool_call(response) if self._tool_fns else None
            if tool_call is None:
                # No tool call — final text reply
                content = response.get("content", "") or ""
                self._messages.append({"role": "assistant", "content": content})
                return content

            # Execute tool
            fn_name = tool_call["name"]
            fn_args = tool_call["arguments"]

            # Record assistant's tool call in history
            self._messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"call_{fn_name}",
                    "type": "function",
                    "function": {"name": fn_name, "arguments": json.dumps(fn_args)},
                }],
            })

            # Execute and record result
            fn = self._tool_fns.get(fn_name)
            if fn:
                try:
                    result = str(fn(**fn_args))
                except Exception as e:
                    result = f"Error: {e}"
            else:
                result = f"Unknown tool: {fn_name}"

            self._messages.append({
                "role": "tool",
                "tool_call_id": f"call_{fn_name}",
                "content": result,
            })

        # Max rounds exceeded
        return "[Max tool call rounds exceeded]"

    def reset(self) -> None:
        """Clear conversation history."""
        self._messages = []
        if self._system_prompt:
            self._messages.append({"role": "system", "content": self._system_prompt})
