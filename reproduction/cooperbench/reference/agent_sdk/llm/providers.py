"""
LLM Provider abstraction — ABC and concrete implementations.

Environment-agnostic: no path hardcoding, no env-specific prompt loading.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import json


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    async def create_completion_async(
        self, messages: List[Dict], tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def parse_tool_call(self, response: Dict) -> Optional[Dict]:
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible provider with multi-endpoint/multi-key support.

    Three config shapes (checked in order):
      1. Multi-endpoint (per-endpoint model):
           {endpoints: [{api_key, base_url?, model?}, ...]}
      2. Multi-key pool (same base_url + model, multiple keys):
           {api_keys: [k1, k2, ...], base_url?, model}
      3. Single endpoint (legacy):
           {api_key, base_url?, model}

    Every call picks a random (client, model) for throughput + isolation.
    On per-call failure (timeout / rate-limit / 5xx / transport) the retry
    layer rotates to a different client — so a single bad key or hung
    upstream connection doesn't stall all agents.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        from openai import AsyncOpenAI
        import random as _random

        self._clients: List[Any] = []
        self._models: List[str] = []

        endpoints = config.get("endpoints")
        if endpoints:
            # Shape 1: per-endpoint model (can differ per endpoint)
            fallback_model = config.get("model", "gpt-4")
            for ep in endpoints:
                api_key = ep.get("api_key")
                if not api_key:
                    raise ValueError("OpenAI endpoint missing api_key")
                base_url = ep.get("base_url")
                if base_url:
                    self._clients.append(AsyncOpenAI(api_key=api_key, base_url=base_url))
                else:
                    self._clients.append(AsyncOpenAI(api_key=api_key))
                self._models.append(ep.get("model") or fallback_model)
        else:
            # Shape 2 or 3: single model, one or more keys.
            api_keys = config.get("api_keys")
            if not api_keys:
                single = config.get("api_key")
                if not single:
                    raise ValueError("OpenAI API key not provided")
                api_keys = [single]
            # Dedupe while preserving order.
            seen = set(); api_keys = [k for k in api_keys if k and not (k in seen or seen.add(k))]
            base_url = config.get("base_url", None)
            model = config.get("model", "gpt-4")
            for k in api_keys:
                if base_url:
                    self._clients.append(AsyncOpenAI(api_key=k, base_url=base_url))
                else:
                    self._clients.append(AsyncOpenAI(api_key=k))
                self._models.append(model)

        # Back-compat aliases. Legacy code paths read .clients / .client / .model.
        self.clients = self._clients
        self.client = self._clients[0]
        self.model = self._models[0]
        self._rr_idx = 0
        self._rng = _random.Random()
        self._rng.seed()

        self.temperature = config.get("temperature", 0.9)
        # max_tokens: None / <= 0 / sentinel "unlimited" → omit the
        # parameter so the model uses its natural output ceiling. Used
        # by evolver/aggregator/deliberation paths where structured JSON
        # responses can exceed any small cap and silent truncation
        # produces unparseable output.
        _mt = config.get("max_tokens", 1000)
        if isinstance(_mt, str) and _mt.lower() in {"none", "null", "unlimited", ""}:
            _mt = None
        self.max_tokens = None if (_mt is None or (isinstance(_mt, (int, float)) and _mt <= 0)) else int(_mt)
        self.timeout = config.get("timeout", 60)
        self.call_timeout = float(config.get("call_timeout", self.timeout * 2))
        self.retry_max = int(config.get("retry_max", max(3, len(self.clients))))
        self.retry_backoff = float(config.get("retry_backoff", 1.0))
        # Optional sampling knobs. None = don't send (use server default).
        # top_p / presence_penalty are OpenAI-native; top_k / min_p are
        # vLLM-only and must travel via extra_body.
        self.top_p = config.get("top_p")
        self.top_k = config.get("top_k")
        self.min_p = config.get("min_p")
        self.presence_penalty = config.get("presence_penalty")
        self.verbose_tool_calls = False
        if len(self.clients) > 1:
            print(f"[OpenAIProvider] multi-client pool: {len(self.clients)} clients; call_timeout={self.call_timeout}s retry_max={self.retry_max}")

    def _pick_client(self, exclude_idx: int = -1):
        """Pick a random client, optionally excluding one (used on retry).
        Returns (client, idx). Model can be looked up via self._models[idx]."""
        if len(self.clients) <= 1:
            return self.clients[0], 0
        candidates = [i for i in range(len(self.clients)) if i != exclude_idx]
        idx = self._rng.choice(candidates)
        return self.clients[idx], idx

    def _pick_endpoint(self):
        """Round-robin to the next (client, model). asyncio is single-threaded,
        so the non-atomic counter update is safe without a lock. Used by the
        sync create_completion path (legacy probe) where retry rotation
        doesn't apply."""
        i = self._rr_idx
        self._rr_idx = (self._rr_idx + 1) % len(self._clients)
        return self._clients[i], self._models[i]

    def _build_kwargs(self, messages, tools=None, tool_choice="auto", model=None):
        """Build API kwargs, adapting to o-series and gpt-5+ which use max_completion_tokens."""
        m = model or self.model
        _newer_models = ("gpt-5", "o1", "o3", "o4")
        is_newer = any(m.startswith(p) for p in _newer_models)
        kwargs = {
            "model": m,
            "messages": messages,
            "timeout": self.timeout,
        }
        if is_newer:
            if self.max_tokens is not None:
                kwargs["max_completion_tokens"] = self.max_tokens
        else:
            kwargs["temperature"] = self.temperature
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
            if self.presence_penalty is not None:
                kwargs["presence_penalty"] = self.presence_penalty
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        # extra_body carries vLLM-only knobs (top_k, min_p) and Qwen's
        # enable_thinking flag — merged rather than overwritten so both coexist.
        extra_body: Dict[str, Any] = {}
        if "qwen" in self.model.lower():
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        if not is_newer:
            if self.top_k is not None:
                extra_body["top_k"] = self.top_k
            if self.min_p is not None:
                extra_body["min_p"] = self.min_p
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    def _guard_gemini_empty_choices(self, response):
        # Gemini's OpenAI-compat endpoint occasionally returns choices=[] on
        # safety filter, content truncation, or upstream tunnel hiccups.
        # Surface a retriable RuntimeError instead of IndexError so llm_core's
        # retry loop can schedule a new attempt with the real finish reason.
        if "gemini" in (self.model or "").lower() and not response.choices:
            feedback = getattr(response, "prompt_feedback", None)
            finish = None
            try:
                finish = response.model_dump(exclude_none=True)
            except Exception:
                finish = repr(response)[:500]
            raise RuntimeError(
                f"empty choices from {self.model} "
                f"(prompt_feedback={feedback}, response={finish})"
            )

    def create_completion(self, messages, tools=None, tool_choice="auto"):
        # Sync path: round-robin endpoint, graceful empty-choices fallback.
        client, model = self._pick_endpoint()
        kwargs = self._build_kwargs(messages, tools, tool_choice, model=model)
        response = client.chat.completions.create(**kwargs)
        if not getattr(response, "choices", None):
            # Empty choices (Gemini content filter); graceful fallback.
            return {
                "content": "",
                "tool_calls": None,
                "raw_response": response,
                "_empty_choices": True,
            }
        return {
            "content": response.choices[0].message.content,
            "tool_calls": response.choices[0].message.tool_calls,
            "raw_response": response,
        }

    async def create_completion_async(self, messages, tools=None, tool_choice="auto"):
        """Call one client with per-call wall-clock timeout; on failure rotate
        to a different client and retry (up to retry_max attempts).

        Each client has its own model (from _models[idx]), so different
        endpoints can serve different models transparently.

        Failure modes that trigger rotation:
          - asyncio.TimeoutError (hung upstream — this is R5)
          - RateLimitError / 429
          - APIStatusError 5xx
          - Any transport error (connection reset, DNS, etc.)

        Non-rotating failures (re-raised immediately):
          - Malformed request (4xx other than 429) — retrying won't help
        """
        import asyncio as _asyncio

        try:
            from openai import RateLimitError as _RateLimitError, APIStatusError as _APIStatusError
            from openai import APIConnectionError as _APIConnErr, APITimeoutError as _APITimeoutErr
        except Exception:
            _RateLimitError = _APIStatusError = _APIConnErr = _APITimeoutErr = Exception

        last_exc = None
        tried_idx = -1
        for attempt in range(self.retry_max):
            client, idx = self._pick_client(exclude_idx=tried_idx)
            tried_idx = idx
            model = self._models[idx]
            kwargs = self._build_kwargs(messages, tools, tool_choice, model=model)
            try:
                response = await _asyncio.wait_for(
                    client.chat.completions.create(**kwargs),
                    timeout=self.call_timeout,
                )
                # Gemini's safety/content filter can return response.choices=[]
                # silently. Retrying on another key is pointless (same prompt →
                # same filter trip). Return empty gracefully so caller falls
                # back to idle, avoiding a ~3× retry storm per filtered call.
                if not getattr(response, "choices", None):
                    finish_reason = None
                    try:
                        finish_reason = getattr(response, "finish_reason", None)
                    except Exception:
                        pass
                    print(f"[OpenAIProvider] empty choices on client#{idx} "
                          f"(finish={finish_reason}) — likely content filter; "
                          f"returning empty, not retrying")
                    return {
                        "content": "",
                        "tool_calls": None,
                        "raw_response": response,
                        "_empty_choices": True,
                    }
                return {
                    "content": response.choices[0].message.content,
                    "tool_calls": response.choices[0].message.tool_calls,
                    "raw_response": response,
                }
            except _asyncio.TimeoutError as exc:
                last_exc = exc
                print(f"[OpenAIProvider] timeout on client#{idx} attempt {attempt+1}/{self.retry_max} — rotating")
            except _RateLimitError as exc:
                last_exc = exc
                print(f"[OpenAIProvider] rate limit on client#{idx} attempt {attempt+1}/{self.retry_max} — rotating")
            except (_APIConnErr, _APITimeoutErr) as exc:
                last_exc = exc
                print(f"[OpenAIProvider] transport error on client#{idx} attempt {attempt+1}/{self.retry_max}: {exc.__class__.__name__} — rotating")
            except _APIStatusError as exc:
                status = getattr(exc, "status_code", None)
                if status and 500 <= status < 600:
                    last_exc = exc
                    print(f"[OpenAIProvider] {status} on client#{idx} attempt {attempt+1}/{self.retry_max} — rotating")
                else:
                    # 4xx other than 429 — caller's fault; don't retry.
                    raise
            except Exception as exc:
                last_exc = exc
                print(f"[OpenAIProvider] unknown error on client#{idx} attempt {attempt+1}/{self.retry_max}: {exc!r} — rotating")
            if attempt < self.retry_max - 1:
                await _asyncio.sleep(self.retry_backoff * (2 ** attempt))
        raise last_exc if last_exc else RuntimeError("OpenAIProvider exhausted retries")

    def parse_tool_call(self, response):
        tool_calls = response.get("tool_calls")
        if not tool_calls:
            return None
        tool_call = tool_calls[0]

        function_obj = None
        if hasattr(tool_call, "function"):
            function_obj = tool_call.function
        elif isinstance(tool_call, dict) and "function" in tool_call:
            function_obj = tool_call["function"]
        elif hasattr(tool_call, "get"):
            function_obj = tool_call.get("function")

        if not function_obj:
            return None

        if hasattr(function_obj, "name"):
            name = function_obj.name
            args = function_obj.arguments
        else:
            name = function_obj["name"]
            args = function_obj["arguments"]

        return {"name": name, "arguments": json.loads(args)}


class TogetherProvider(LLMProvider):
    """Together AI provider using Together SDK."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        try:
            from together import AsyncTogether
        except ImportError:
            raise ImportError("Together SDK not installed. Run: pip install together")

        api_key = config.get("api_key")
        if not api_key:
            raise ValueError("Together API key not provided")

        self.client = AsyncTogether(api_key=api_key)
        self.model = config.get("model", "Qwen/Qwen2.5-72B-Instruct-Turbo")
        self.temperature = config.get("temperature", 0.9)
        # max_tokens: None / <= 0 / sentinel "unlimited" → omit the
        # parameter so the model uses its natural output ceiling. Used
        # by evolver/aggregator/deliberation paths where structured JSON
        # responses can exceed any small cap and silent truncation
        # produces unparseable output.
        _mt = config.get("max_tokens", 1000)
        if isinstance(_mt, str) and _mt.lower() in {"none", "null", "unlimited", ""}:
            _mt = None
        self.max_tokens = None if (_mt is None or (isinstance(_mt, (int, float)) and _mt <= 0)) else int(_mt)
        self.timeout = config.get("timeout", 60)
        self.verbose_tool_calls = False

        print(f"[TogetherProvider] Initialized with model: {self.model}")

    async def create_completion_async(self, messages, tools=None, tool_choice="auto"):
        if self.verbose_tool_calls and tools:
            print(f"\n{'='*60}")
            print(f"[Together ASYNC API Request]")
            print(f"Model: {self.model}")
            print(f"Tool choice: {tool_choice}")
            print(f"Messages: {len(messages)} messages")
            print(f"Tools: {len(tools)} tools")
            print(f"Tool names: {[t['function']['name'] for t in tools]}")
            print(f"{'='*60}\n")

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        response = await self.client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content if response.choices else ""
        tool_calls = response.choices[0].message.tool_calls if response.choices else None

        if self.verbose_tool_calls and tools:
            print(f"\n{'='*60}")
            print(f"[Together ASYNC API Response]")
            print(f"Content: {content if content else '(empty)'}")
            print(f"Tool calls: {len(tool_calls) if tool_calls else 0}")
            if tool_calls:
                for i, tc in enumerate(tool_calls):
                    if hasattr(tc, 'function'):
                        print(f"  [{i+1}] {tc.function.name}({tc.function.arguments})")
                    elif isinstance(tc, dict):
                        print(f"  [{i+1}] {tc}")
                    else:
                        print(f"  [{i+1}] {tc} (type: {type(tc)})")
            else:
                print(f"  (No tool calls returned)")
            print(f"{'='*60}\n")

        return {
            "content": content,
            "tool_calls": tool_calls,
            "raw_response": response,
        }

    async def create_completion_stream_async(self, messages, tools=None,
                                              tool_choice="auto", stream=True):
        if not stream:
            response = await self.create_completion_async(messages, tools, tool_choice)
            yield {"type": "final", "raw": response}
            return

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        try:
            stream_iter = await self.client.chat.completions.create(**kwargs)
            last_raw = None
            async for chunk in stream_iter:
                last_raw = chunk

                choice0 = None
                try:
                    choice0 = chunk.choices[0]
                except Exception:
                    choice0 = None

                delta = getattr(choice0, "delta", None) if choice0 else None

                token = getattr(delta, "content", None) if delta else None
                if token:
                    yield {"type": "token", "content": token}

                delta_tool_calls = getattr(delta, "tool_calls", None) if delta else None
                if delta_tool_calls:
                    for tc_delta in delta_tool_calls:
                        yield {"type": "tool_call_delta", "delta": tc_delta}

            yield {"type": "final", "raw": last_raw}
        except Exception as e:
            yield {"type": "error", "error": e}

    def _accumulate_tool_calls(self, acc, delta):
        idx = None
        if hasattr(delta, "index"):
            idx = delta.index
        elif isinstance(delta, dict):
            idx = delta.get("index")
        if idx is None:
            idx = 0
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            idx = 0

        while len(acc) <= idx:
            acc.append({"id": None, "type": "function", "function": {"name": None, "arguments": ""}})

        if hasattr(delta, "id") and delta.id:
            acc[idx]["id"] = delta.id
        elif isinstance(delta, dict) and delta.get("id"):
            acc[idx]["id"] = delta.get("id")

        if hasattr(delta, "type") and delta.type:
            acc[idx]["type"] = delta.type
        elif isinstance(delta, dict) and delta.get("type"):
            acc[idx]["type"] = delta.get("type")

        func = getattr(delta, "function", None) if not isinstance(delta, dict) else delta.get("function")
        if func:
            name = getattr(func, "name", None) if not isinstance(func, dict) else func.get("name")
            if name:
                acc[idx]["function"]["name"] = name

            args_piece = getattr(func, "arguments", None) if not isinstance(func, dict) else func.get("arguments")
            if args_piece:
                acc[idx]["function"]["arguments"] = (acc[idx]["function"]["arguments"] or "") + args_piece

    def _finalize_tool_calls(self, acc):
        cleaned = []
        for tc in acc or []:
            fn = (tc or {}).get("function") or {}
            if fn.get("name") or (fn.get("arguments") or "").strip():
                cleaned.append(tc)
        return cleaned or None

    def parse_tool_call(self, response):
        try:
            tool_calls = response.get("tool_calls")
            if tool_calls:
                tool_call = tool_calls[0]
                function_obj = None
                if hasattr(tool_call, "function"):
                    function_obj = tool_call.function
                elif isinstance(tool_call, dict) and "function" in tool_call:
                    function_obj = tool_call["function"]

                if not function_obj:
                    return None

                if hasattr(function_obj, "arguments"):
                    args_str = function_obj.arguments
                    name = function_obj.name
                else:
                    args_str = function_obj.get("arguments")
                    name = function_obj.get("name")

                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError as e:
                    print(f"[TogetherProvider] Failed to parse arguments: {args_str[:200]}")
                    print(f"  Error: {e}")
                    return None

                return {"name": name, "arguments": args}

            # Fallback: Some Together models return tool calls as JSON in content field
            content = response.get("content", "")
            if content:
                content_stripped = content.strip()
                if content_stripped.startswith("["):
                    try:
                        parsed = json.loads(content_stripped)
                        if isinstance(parsed, list) and len(parsed) > 0:
                            tool_data = parsed[0]
                            if "name" in tool_data:
                                args = tool_data.get("parameters") or tool_data.get("arguments", {})
                                print(f"[TogetherProvider] Parsed tool from content: {tool_data['name']}")
                                return {"name": tool_data["name"], "arguments": args}
                    except json.JSONDecodeError as e:
                        print(f"[TogetherProvider] JSON parse failed: {e}")
                        print(f"  Content: {content[:300]}")

            print(f"[TogetherProvider] No tool_calls. Content: {content[:200]}")
            return None
        except Exception as e:
            print(f"[TogetherProvider] Unexpected error in parse_tool_call: {e}")
            import traceback
            traceback.print_exc()
            return None
