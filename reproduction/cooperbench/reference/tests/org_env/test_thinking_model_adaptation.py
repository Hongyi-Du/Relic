"""A thinking model outside OpenAI's naming must still get its reasoning budget.

Regression context (2026-08-14). `_is_reasoning_model` decided four separate
questions from one name-prefix match against gpt-5/o1/o3/o4:

  * is a reasoning budget accepted,
  * which field carries the output ceiling,
  * is a caller temperature accepted,
  * does hidden reasoning eat the answer's allowance.

`claude-sonnet-5` matched none of those prefixes, so all four were answered
"no". The visible consequences, measured against the bound endpoint:

  * ORG_LLM_REASONING_EFFORT=low was read, validated and written into the run
    record, but never sent. On an identical prompt, 3 trials each: nothing sent
    cost 843 completion tokens and 17.3s; reasoning_effort=low cost 185 tokens
    and 5.9s, for a slightly *longer* visible answer. A 144-tick run therefore
    paid 959 completion tokens per call where the comparable GPT run paid 379.
  * max_tokens=UNCAPPED_OUTPUT sent no ceiling at all. Anthropic's API requires
    the field, so the OpenAI-compatible shim chose one: a whole-file rewrite
    stopped at exactly 8192 tokens with finish_reason="length" and unparseable
    output, while the same request with an explicit ceiling finished normally.
  * finish_reason never reached the caller, because
    provider_process.project_openai_response did not project it — and this host
    takes the subprocess path on every call, since
    _requires_provider_process_deadline is true wherever signal.SIGALRM is
    absent. Truncation was therefore indistinguishable from a bad answer, which
    is why none of the above was visible in 144 ticks of telemetry.

These tests pin the four axes apart, and pin the two GPT shapes unchanged.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from environments.org_env.llm.client import (
    ACCOUNTING_OUTPUT_CEILING,
    UNCAPPED_OUTPUT,
    LLMError,
    OpenAIOrgLLMClient,
    model_profile,
)
from environments.org_env.llm.provider_process import (
    project_openai_response,
    restore_openai_response,
)


def _client(model: str, **kwargs) -> OpenAIOrgLLMClient:
    kwargs.setdefault("max_retries", 0)
    kwargs.setdefault("reasoning_effort", "low")
    return OpenAIOrgLLMClient(
        model=model,
        api_key="test",
        base_url="https://provider.invalid/v1",
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# the defect: a thinking model that never received its budget
# --------------------------------------------------------------------------- #
def test_a_thinking_model_is_sent_the_reasoning_budget_it_was_configured_with():
    """The whole 144-tick cost overrun in one assertion."""
    kwargs = _client("claude-sonnet-5")._gen_kwargs(0.2, 1200)
    assert kwargs["reasoning_effort"] == "low"


def test_the_opus_55_gateway_signature_uses_the_thinking_chat_profile():
    profile = model_profile("claude-opus-5-5")
    kwargs = _client("claude-opus-5-5")._gen_kwargs(0.2, 1200)
    assert profile.reasoning_effort is True
    assert profile.output_limit_field == "max_tokens"
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["max_tokens"] == 6000
    assert "max_completion_tokens" not in kwargs


def test_a_thinking_model_keeps_the_plain_chat_fields():
    """Sending a budget must not also switch it to the o-series field names.

    These are independent axes. Answering both from one boolean is what forced
    the old code to choose between a budget and a working request.
    """
    kwargs = _client("claude-sonnet-5")._gen_kwargs(0.3, 800)
    assert kwargs["max_tokens"] == 6000, "thinking draws on the answer's allowance"
    assert "max_completion_tokens" not in kwargs
    assert kwargs["temperature"] == 0.3


def test_the_run_record_reports_the_budget_that_was_actually_spent():
    assert _client("claude-sonnet-5").effective_reasoning_effort == "low"


def test_an_uncapped_request_names_a_ceiling_where_omitting_one_is_not_uncapped():
    """Omitting max_tokens reaches the model's maximum only where there is one
    to fall back on. Behind a shim that must invent the field, omitting it
    concedes the choice — measured at 8192, mid-rewrite."""
    client = _client("claude-sonnet-5")
    assert client.effective_output_token_limit(UNCAPPED_OUTPUT) == (
        ACCOUNTING_OUTPUT_CEILING
    )
    kwargs = client._gen_kwargs(0.2, UNCAPPED_OUTPUT)
    assert kwargs["max_tokens"] == ACCOUNTING_OUTPUT_CEILING
    # The wire and the ledger must agree, or the budget bounds nothing.
    envelope = client.request_resource_envelope(UNCAPPED_OUTPUT)
    assert envelope["output_tokens_per_attempt"] == ACCOUNTING_OUTPUT_CEILING


# --------------------------------------------------------------------------- #
# the two GPT shapes are unchanged
# --------------------------------------------------------------------------- #
def test_an_o_series_model_keeps_its_own_shape():
    kwargs = _client("gpt-5.6-terra", reasoning_effort="high")._gen_kwargs(0.3, 1200)
    assert kwargs["reasoning_effort"] == "high"
    assert kwargs["max_completion_tokens"] == 6000
    assert "temperature" not in kwargs, "the o-series rejects a caller temperature"
    assert "max_tokens" not in kwargs


def test_an_o_series_uncapped_request_still_sends_nothing():
    """That endpoint has its own maximum, so silence is genuinely uncapped."""
    client = _client("gpt-5.6-terra")
    assert client.effective_output_token_limit(UNCAPPED_OUTPUT) is None
    kwargs = client._gen_kwargs(0.2, UNCAPPED_OUTPUT)
    assert "max_completion_tokens" not in kwargs and "max_tokens" not in kwargs


def test_a_plain_chat_model_asks_for_no_budget_and_gets_no_floor():
    client = _client("gpt-4o-mini", reasoning_effort="high")
    assert client.effective_reasoning_effort is None
    kwargs = client._gen_kwargs(0.3, 800)
    assert "reasoning_effort" not in kwargs
    assert kwargs["max_tokens"] == 800, "no hidden reasoning, so no floor"
    assert client.effective_output_token_limit(UNCAPPED_OUTPUT) is None


def test_the_profile_axes_are_independent():
    """Named so a future model family is classified, not pattern-matched."""
    claude = model_profile("claude-sonnet-5")
    o_series = model_profile("gpt-5.6-terra")
    plain = model_profile("gpt-4o-mini")

    assert claude.reasoning_effort and o_series.reasoning_effort
    assert not plain.reasoning_effort
    assert claude.output_limit_field == "max_tokens"
    assert o_series.output_limit_field == "max_completion_tokens"
    assert claude.accepts_temperature and not o_series.accepts_temperature
    assert claude.thinking_shares_output_budget
    assert not plain.thinking_shares_output_budget
    assert claude.explicit_ceiling_when_uncapped
    assert not o_series.explicit_ceiling_when_uncapped


# --------------------------------------------------------------------------- #
# truncation must be visible
# --------------------------------------------------------------------------- #
class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content, finish_reason):
        self.message = _Message(content)
        self.finish_reason = finish_reason


class _Response:
    def __init__(self, content, finish_reason):
        self.id = "resp_1"
        self.model = "claude-sonnet-5"
        self.usage = None
        self.choices = [_Choice(content, finish_reason)]


def test_finish_reason_survives_the_subprocess_boundary():
    """Every call on a host without SIGALRM crosses this boundary."""
    projected = project_openai_response(
        _Response('{"a": 1}', "length"), "chat_completions"
    )
    assert projected["finish_reason"] == "length"
    restored = restore_openai_response(projected, "chat_completions")
    assert restored.choices[0].finish_reason == "length"


def test_a_truncated_answer_says_so_instead_of_reading_as_a_bad_answer():
    client = _client("claude-sonnet-5")
    truncated = _Response('{"new_content": "package main\\n', "length")
    try:
        client._decode(truncated, truncated.choices[0].message.content, {})
    except Exception as exc:
        assert "output ceiling" in str(exc), str(exc)
    else:
        raise AssertionError("a truncated body must not parse")


def test_a_genuinely_bad_answer_is_not_blamed_on_the_ceiling():
    client = _client("claude-sonnet-5")
    prose = _Response("Here is the file:\n```go\npackage main\n```", "stop")
    try:
        client._decode(prose, prose.choices[0].message.content, {})
    except Exception as exc:
        assert "output ceiling" not in str(exc), str(exc)
    else:
        raise AssertionError("prose must not parse as JSON")


def test_an_empty_responses_body_names_the_ceiling_too():
    assert "output ceiling" in OpenAIOrgLLMClient._ceiling_note(
        type("R", (), {"status": "incomplete",
                       "incomplete_details": type("D", (), {"reason":
                                                            "max_output_tokens"})})()
    )
    assert OpenAIOrgLLMClient._ceiling_note(
        type("R", (), {"status": "completed"})()
    ) == ""


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("  ok ", name)
    print("thinking-model-adaptation OK")
