"""Load the OrgEnv LLM cognitive-layer client from YAML config.

Reads ``config/llm.yaml`` (tracked template) and overlays ``config/llm.local.yaml``
(gitignored — per-user keys, per the repo .gitignore), under the ``org_env`` key:

    org_env:
      enabled: true
      provider: openai            # openai | http | mock
      model: gpt-4o-mini
      api_key: sk-...             # put real keys ONLY in llm.local.yaml
      base_url: null              # optional (custom OpenAI-compatible endpoint)
      azure_api_version: null     # set to use AzureOpenAI with base_url as endpoint
      wire_api: responses         # responses | chat_completions
      json_transport: native      # native | prompt_only
      decides_actions: false      # true = LLM also drives action decisions (more calls)
      reasoning_effort: low       # none|minimal|low|medium|high|xhigh
      max_retries: 5              # bounded transport-error retries per call (openai only)
      retry_backoff_seconds: 1.0  # exponential backoff base between retries
      request_timeout_seconds: 120
      store_responses: false
      default_headers: {}         # non-secret provider headers when required

Default is no client (rule/template paths). Never raises — a bad/missing config
just yields ``(None, False)``.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

from environments.org_env.llm.client import (
    OPENAI_JSON_TRANSPORTS,
    OPENAI_WIRE_APIS,
    REASONING_EFFORTS,
    GenericHTTPOrgLLMClient,
    MockOrgLLMClient,
    OpenAIOrgLLMClient,
    OrgLLMClient,
)
from society_core.openai_runtime import (
    configured_openai_default_headers,
    openai_response_storage_disabled,
    validate_openai_default_headers,
)


def _repo_root() -> str:
    # environments/org_env/llm/config.py -> repo root is 4 levels up
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _read_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def load_org_llm_config(root: Optional[str] = None) -> Dict[str, Any]:
    root = root or _repo_root()
    base = _read_yaml(os.path.join(root, "config", "llm.yaml")).get("org_env", {}) or {}
    local_path = os.environ.get("ORG_LLM_LOCAL_CONFIG")
    if local_path:
        local_path = os.path.abspath(os.path.expanduser(local_path))
    else:
        local_path = os.path.join(root, "config", "llm.local.yaml")
    local = _read_yaml(local_path).get("org_env", {}) or {}
    cfg = dict(base)
    cfg.update({k: v for k, v in local.items() if v is not None})
    return cfg


def load_org_llm_client(root: Optional[str] = None) -> Tuple[Optional[OrgLLMClient], bool]:
    """Return (client, decides_actions). (None, False) if disabled/unconfigured."""
    cfg = load_org_llm_config(root)

    def _boolean(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return default

    enabled_override = os.environ.get("ORG_LLM_ENABLED")
    if enabled_override is None:
        enabled_override = os.environ.get("ORG_LLM")
    enabled = _boolean(enabled_override, bool(cfg.get("enabled", False)))
    if not enabled:
        return None, False
    provider = str(
        os.environ.get("ORG_LLM_PROVIDER")
        or cfg.get("provider", "openai")
    ).lower()
    decides = bool(cfg.get("decides_actions", False))
    # per-run model override without editing the (gitignored) local config — handy for A/B'ing
    # the cognition model (e.g. ORG_LLM_MODEL=gpt-5) across experiment runs.
    model = os.environ.get("ORG_LLM_MODEL") or cfg.get("model", "gpt-4o-mini")
    # per-run reasoning-effort override (P4a measurement integrity: the effective
    # value is recorded in the experiment run record, so a fallback here is never
    # silent in the data). Invalid values fall back to the default 'low' because
    # this loader must never raise.
    effort = str(
        os.environ.get("ORG_LLM_REASONING_EFFORT")
        or cfg.get("reasoning_effort", "low")
    ).strip().lower()
    if effort not in REASONING_EFFORTS:
        effort = "low"

    def _non_negative(value: Any, default: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed >= 0 else default

    max_retries = int(
        _non_negative(
            os.environ.get("ORG_LLM_MAX_RETRIES") or cfg.get("max_retries"),
            # Five transport retries with 1s exponential backoff is ~31s worst
            # case inside the 120s call budget. Measured need: at 2 retries a
            # 336-tick run left 25 of 282 calls terminally failed (91.1%, under
            # the 95% treatment gate); most failures were intermittent 5xx that
            # a later attempt served.
            5,
        )
    )
    backoff = _non_negative(
        os.environ.get("ORG_LLM_RETRY_BACKOFF_SECONDS")
        or cfg.get("retry_backoff_seconds"),
        1.0,
    )
    request_timeout = _non_negative(
        os.environ.get("ORG_LLM_REQUEST_TIMEOUT_SECONDS")
        or cfg.get("request_timeout_seconds"),
        120.0,
    )
    if request_timeout <= 0:
        request_timeout = 120.0
    wire_api = str(
        os.environ.get("ORG_LLM_WIRE_API")
        or cfg.get("wire_api")
        or os.environ.get("OPENAI_WIRE_API")
        or "responses"
    ).strip().lower().replace("-", "_")
    if wire_api not in OPENAI_WIRE_APIS:
        return None, False
    json_transport = str(
        os.environ.get("ORG_LLM_JSON_TRANSPORT")
        or cfg.get("json_transport")
        or "native"
    ).strip().lower().replace("-", "_")
    if json_transport not in OPENAI_JSON_TRANSPORTS:
        return None, False
    base_url = (
        os.environ.get("ORG_LLM_BASE_URL")
        or cfg.get("base_url")
        or os.environ.get("OPENAI_BASE_URL")
    )
    api_key = (
        os.environ.get("ORG_LLM_API_KEY")
        or cfg.get("api_key")
        or os.environ.get("OPENAI_API_KEY")
    )
    azure_api_version = (
        os.environ.get("ORG_LLM_AZURE_API_VERSION")
        or cfg.get("azure_api_version")
    )
    store_raw = os.environ.get("ORG_LLM_STORE_RESPONSES")
    if store_raw is None:
        if "store_responses" in cfg:
            store_responses = _boolean(cfg.get("store_responses"), False)
        else:
            try:
                store_responses = not openai_response_storage_disabled()
            except ValueError:
                return None, False
    else:
        store_responses = _boolean(store_raw, False)
    try:
        headers = configured_openai_default_headers()
    except ValueError:
        return None, False
    configured_headers = cfg.get("default_headers")
    if isinstance(configured_headers, dict):
        try:
            headers.update(validate_openai_default_headers(configured_headers))
        except ValueError:
            return None, False
    env_headers = os.environ.get("ORG_LLM_DEFAULT_HEADERS_JSON")
    if env_headers:
        try:
            parsed_headers = json.loads(env_headers)
        except (TypeError, ValueError):
            parsed_headers = None
        if not isinstance(parsed_headers, dict):
            return None, False
        try:
            headers.update(validate_openai_default_headers(parsed_headers))
        except ValueError:
            return None, False
    try:
        if provider == "openai":
            client = OpenAIOrgLLMClient(model=model, api_key=api_key,
                                        base_url=base_url,
                                        reasoning_effort=effort,
                                        max_retries=max_retries,
                                        retry_backoff_seconds=backoff,
                                        wire_api=wire_api,
                                        json_transport=json_transport,
                                        request_timeout_seconds=request_timeout,
                                        store_responses=store_responses,
                                        default_headers=headers,
                                        azure_api_version=azure_api_version)
        elif provider in ("http", "generic"):
            client = GenericHTTPOrgLLMClient(endpoint=cfg["endpoint"], model=model)
        elif provider == "mock":
            client = MockOrgLLMClient()
        else:
            return None, False
    except Exception:
        return None, False
    return client, decides


__all__ = ["load_org_llm_config", "load_org_llm_client"]
