"""local_llm's two wire formats, behind cfg.llm.

Flag off (llm.api: ollama, the default) must send Ollama exactly the request it
got before the flag existed. Flag on (llm.api: openai) must send the contract's
request (integration-contract.md §2-4) and map failures so a daemon can tell
"try later" (LocalLLMUnavailable) from "this request can never succeed"
(LocalLLMRequestTooLarge) — vLLM answers 400 where Ollama silently truncated.

The OpenAI path runs the real SDK against httpx.MockTransport, so the body
and headers asserted here are the bytes that would go on the wire.
"""
from __future__ import annotations

import json

import httpx
import openai
import pytest

from alfred.config import AlfredConfig
from alfred.core import local_llm
from alfred.core.local_llm import (
    LocalLLMBadRequest,
    LocalLLMOutputTruncated,
    LocalLLMRequestTooLarge,
    LocalLLMUnavailable,
    complete,
    complete_json,
)

SPARK = "http://spark.test:8000/v1"


def _chat_response(content, finish_reason="stop", status=200):
    return httpx.Response(status, json={
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "qwen3-30b",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
        }],
    })


def _fake_openai_server(monkeypatch, respond):
    """Route the SDK through `respond(request) -> httpx.Response`; return the
    list of requests it saw. No retries, so each call is one request."""
    seen: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return respond(request)

    def _client(base_url, api_key, timeout):
        return openai.OpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(_handler)),
        )

    monkeypatch.setattr(local_llm, "_openai_client", _client)
    return seen


@pytest.fixture(autouse=True)
def _no_real_spark_env(tmp_path, monkeypatch):
    """Keep the machine's real ~/.config/spark/env and SPARK_* out of every test."""
    for name in ("SPARK_API_KEY", "SPARK_BASE_URL", "SPARK_MODEL", "TEST_SPARK_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("alfred.config.SPARK_ENV_PATH", tmp_path / "no-spark-env")


def _capture_ollama(monkeypatch, content="ok"):
    seen: list[dict] = []

    def _post(url, json=None, timeout=None):
        seen.append({"url": url, "json": json, "timeout": timeout})
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": content}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", _post)
    return seen


# ── Flag off: Ollama, byte-for-byte as before ────────────────────────────────


def test_default_config_selects_todays_ollama_backend(tmp_path):
    cfg = AlfredConfig(vault_path=tmp_path, data_dir=tmp_path / "data")

    assert cfg.llm == {
        "api": "ollama",
        "base_url": cfg.ollama_base_url,
        "model": cfg.ollama_llm_model,
    }


def test_ollama_request_is_unchanged(monkeypatch):
    """The exact payload complete() sent before the flag existed (200cf02)."""
    seen = _capture_ollama(monkeypatch, content='{"type": "note"}')

    out = complete(
        "SYS", "USER",
        base_url="http://localhost:11434", model="m:latest",
        json_mode=True, max_tokens=256,
    )

    assert out == '{"type": "note"}'
    assert seen == [{
        "url": "http://localhost:11434/api/chat",
        "json": {
            "model": "m:latest",
            "stream": False,
            "think": False,
            "messages": [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "USER"},
            ],
            "options": {"num_predict": 256},
            "format": "json",
        },
        "timeout": 180.0,
    }]


def test_ollama_ignores_schema_and_keeps_format_json(monkeypatch):
    seen = _capture_ollama(monkeypatch, content='{"type": "note"}')

    parsed = complete_json(
        "SYS", "USER", base_url="http://localhost:11434", model="m",
        schema={"type": "object"},
    )

    assert parsed == {"type": "note"}
    assert seen[0]["json"]["format"] == "json"


def test_empty_system_sends_the_user_message_alone(monkeypatch):
    """What the consolidator's /api/generate calls did: no system prompt."""
    seen = _capture_ollama(monkeypatch)

    complete("", "PROMPT", base_url="http://localhost:11434", model="m")

    assert seen[0]["json"]["messages"] == [{"role": "user", "content": "PROMPT"}]


def test_ollama_400_still_means_unavailable(monkeypatch):
    """Only the OpenAI path gets the new 400 mapping; Ollama never 400s on size."""
    monkeypatch.setattr(
        httpx, "post",
        lambda url, json=None, timeout=None: httpx.Response(
            400, text="bad", request=httpx.Request("POST", url)
        ),
    )

    with pytest.raises(LocalLLMUnavailable):
        complete("s", "u", base_url="http://localhost:11434", model="m")


def test_unknown_api_is_an_error_not_a_fallback(monkeypatch):
    _capture_ollama(monkeypatch)
    with pytest.raises(ValueError, match="unknown llm api"):
        complete("s", "u", base_url="http://x", model="m", api="anthropic")


# ── Flag on: the OpenAI-compatible request ───────────────────────────────────


def test_openai_request_body_follows_the_contract(monkeypatch):
    monkeypatch.setenv("TEST_SPARK_KEY", "sk-test")
    seen = _fake_openai_server(monkeypatch, lambda r: _chat_response("hello"))

    out = complete(
        "SYS", "USER", api="openai", base_url=SPARK, model="qwen3-30b",
        api_key_env="TEST_SPARK_KEY", max_tokens=300,
    )

    assert out == "hello"
    (request,) = seen
    assert request.method == "POST"
    assert str(request.url) == f"{SPARK}/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-test"
    body = json.loads(request.content)
    assert body == {
        "model": "qwen3-30b",
        "messages": [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "USER"},
        ],
        "max_tokens": 300,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_openai_json_mode_asks_for_a_json_object(monkeypatch):
    seen = _fake_openai_server(monkeypatch, lambda r: _chat_response('{"a": 1}'))

    parsed = complete_json("s", "u", api="openai", base_url=SPARK, model="m")

    assert parsed == {"a": 1}
    assert json.loads(seen[0].content)["response_format"] == {"type": "json_object"}


def test_openai_schema_becomes_a_json_schema_response_format(monkeypatch):
    schema = {"type": "object", "properties": {"type": {"enum": ["note", "task"]}}}
    seen = _fake_openai_server(monkeypatch, lambda r: _chat_response('{"type": "task"}'))

    parsed = complete_json("s", "u", api="openai", base_url=SPARK, model="m", schema=schema)

    assert parsed == {"type": "task"}
    assert json.loads(seen[0].content)["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "reply", "schema": schema},
    }


def test_openai_key_falls_back_to_the_spark_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / "spark-env"
    env_file.write_text(
        "# spark\nSPARK_BASE_URL=https://ignored/v1\nexport SPARK_API_KEY='sk-from-file'\n"
    )
    monkeypatch.setattr("alfred.config.SPARK_ENV_PATH", env_file)
    seen = _fake_openai_server(monkeypatch, lambda r: _chat_response("ok"))

    complete("s", "u", api="openai", base_url=SPARK, model="m", api_key_env="SPARK_API_KEY")

    assert seen[0].headers["authorization"] == "Bearer sk-from-file"


def test_openai_environment_wins_over_the_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / "spark-env"
    env_file.write_text("SPARK_API_KEY=sk-from-file\n")
    monkeypatch.setattr("alfred.config.SPARK_ENV_PATH", env_file)
    monkeypatch.setenv("SPARK_API_KEY", "sk-from-env")
    seen = _fake_openai_server(monkeypatch, lambda r: _chat_response("ok"))

    complete("s", "u", api="openai", base_url=SPARK, model="m", api_key_env="SPARK_API_KEY")

    assert seen[0].headers["authorization"] == "Bearer sk-from-env"


def test_openai_without_any_key_sends_the_contracts_dummy(monkeypatch):
    seen = _fake_openai_server(monkeypatch, lambda r: _chat_response("ok"))

    complete("s", "u", api="openai", base_url=SPARK, model="m", api_key_env="SPARK_API_KEY")

    assert seen[0].headers["authorization"] == "Bearer dummy"


def test_real_client_has_a_timeout_and_at_most_one_retry():
    """Contract §7."""
    with local_llm._openai_client(SPARK, "k", 42.0) as client:
        assert client.max_retries == 1
        assert client.timeout == 42.0
        assert str(client.base_url).rstrip("/") == SPARK


@pytest.mark.parametrize("content, expected", [
    ("<think>\nplanning\n</think>\n\nThe answer.", "The answer."),
    ("<think></think>LABEL: x", "LABEL: x"),
    ("Partial <think>ran out of budget", "Partial"),
    ("<think>never closed", ""),
    ("plain", "plain"),
])
def test_openai_strips_think_blocks(monkeypatch, content, expected):
    _fake_openai_server(monkeypatch, lambda r: _chat_response(content))

    assert complete("s", "u", api="openai", base_url=SPARK, model="m") == expected


# ── Flag on: error mapping ───────────────────────────────────────────────────


def _raises(exc):
    def _respond(request):
        raise exc
    return _respond


@pytest.mark.parametrize("respond", [
    _raises(httpx.ConnectError("connection refused")),
    _raises(httpx.ReadTimeout("slow")),
    lambda r: httpx.Response(500, json={"error": {"message": "boom"}}),
    lambda r: httpx.Response(503, text="unavailable"),
    lambda r: httpx.Response(401, json={"error": "Unauthorized"}),
    lambda r: httpx.Response(404, json={"error": {"message": "model not found"}}),
    lambda r: httpx.Response(429, text="slow down"),
    lambda r: _chat_response("half", finish_reason="abort"),
    lambda r: httpx.Response(200, json={"choices": []}),
], ids=[
    "connect", "timeout", "500", "503", "401", "404", "429", "finish-abort", "no-choices",
])
def test_openai_failures_that_should_defer_raise_unavailable(monkeypatch, respond):
    _fake_openai_server(monkeypatch, respond)

    with pytest.raises(LocalLLMUnavailable):
        complete("s", "u", api="openai", base_url=SPARK, model="m")


def test_openai_400_raises_request_too_large_with_the_servers_reason(monkeypatch):
    reason = (
        "This model's maximum context length is 32768 tokens. However, you "
        "requested 33012 tokens (31000 in the messages, 2012 in the completion)."
    )
    _fake_openai_server(
        monkeypatch,
        lambda r: httpx.Response(400, json={"error": {"message": reason, "code": 400}}),
    )

    with pytest.raises(LocalLLMRequestTooLarge, match="maximum context length"):
        complete("s", "u" * 10, api="openai", base_url=SPARK, model="m")


def test_openai_400_is_not_a_kind_of_unavailable():
    """Every daemon defers on LocalLLMUnavailable. If a 400 were one, the same
    oversized request would be deferred and re-sent forever."""
    assert not issubclass(LocalLLMRequestTooLarge, LocalLLMUnavailable)


def test_openai_truncated_answer_raises_request_too_large(monkeypatch):
    """Contract §2: finish_reason must be "stop"; never ship half an answer.
    It is the OutputTruncated kind of RequestTooLarge, so the curator can
    retry it later while every other caller keeps skipping it."""
    _fake_openai_server(monkeypatch, lambda r: _chat_response('{"items": [', finish_reason="length"))

    with pytest.raises(LocalLLMOutputTruncated, match="max_tokens=64"):
        complete_json("s", "u", api="openai", base_url=SPARK, model="m", max_tokens=64)
    assert issubclass(LocalLLMOutputTruncated, LocalLLMRequestTooLarge)


@pytest.mark.parametrize("reason", [
    (
        "This model's maximum context length is 32768 tokens. However, you requested "
        "40014 tokens (38002 in the messages, 2012 in the completion)."
    ),
    (
        "'max_tokens' or 'max_completion_tokens' is too large: 2048. This model's maximum "
        "context length is 32768 tokens and your request has 31000 input tokens."
    ),
    "The prompt (40014 tokens) is longer than the model's context length (32768).",
    "Input prompt (40014 tokens) is too long and exceeds limit of 32768",
    "Request exceeds max_model_len 32768",
], ids=["vllm-classic", "vllm-max-tokens", "prompt-longer", "input-too-long", "max-model-len"])
def test_openai_context_length_400_is_request_too_large(monkeypatch, reason):
    _fake_openai_server(
        monkeypatch,
        lambda r: httpx.Response(400, json={"error": {"message": reason, "code": 400}}),
    )

    with pytest.raises(LocalLLMRequestTooLarge) as info:
        complete("s", "u", api="openai", base_url=SPARK, model="m")
    assert not isinstance(info.value, LocalLLMUnavailable)
    assert not isinstance(info.value, LocalLLMOutputTruncated)


def test_openai_context_length_error_code_alone_is_request_too_large(monkeypatch):
    """OpenAI's own shape: the code says it, the message may not."""
    _fake_openai_server(monkeypatch, lambda r: httpx.Response(400, json={"error": {
        "message": "Please reduce the length of the messages.", "code": "context_length_exceeded",
    }}))

    with pytest.raises(LocalLLMRequestTooLarge):
        complete("s", "u", api="openai", base_url=SPARK, model="m")


@pytest.mark.parametrize("reason", [
    "The provided JSON schema contains features not supported by xgrammar.",
    "Invalid JSON schema: maxLength is not supported",
    "chat_template_kwargs is not a valid argument",
    "response_format.type must be one of json_object, json_schema, text",
    "[{'type': 'missing', 'loc': ('body', 'messages'), 'msg': 'Field required'}]",
], ids=["xgrammar", "maxLength", "template-kwargs", "response-format", "validation"])
def test_openai_other_400_is_a_loud_retry_later(monkeypatch, reason):
    """A 400 caused by the setup (a rejected schema, kwarg or format) would
    fail every input alike. Reading it as "too large" made callers mark each
    input done: the whole inbox silently skipped. It must be a retry-later
    (a LocalLLMUnavailable), logged at error and counted."""
    from structlog.testing import capture_logs

    from alfred.core.failures import drain_failures

    _fake_openai_server(
        monkeypatch,
        lambda r: httpx.Response(400, json={"error": {"message": reason, "code": 400}}),
    )
    drain_failures()

    with capture_logs() as logs, pytest.raises(LocalLLMBadRequest) as info:
        complete("s", "u", api="openai", base_url=SPARK, model="m", schema={"type": "object"})

    assert isinstance(info.value, LocalLLMUnavailable)
    assert not isinstance(info.value, LocalLLMRequestTooLarge)
    (event,) = [e for e in logs if e.get("event") == "local_llm.bad_request"
                and e.get("log_level") == "error"]
    assert event["response_format"] == "json_schema"
    assert reason in event["error"]
    assert drain_failures().get("local_llm.bad_request") == 1
