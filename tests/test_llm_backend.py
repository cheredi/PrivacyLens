"""OpenAIBackend against a fake HTTP transport: provider differences are handled, never crash."""

import json

import pytest

try:  # openai>=3 ships its own HTTP library; older releases use httpx
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx

from config import Settings
from core.llm import LLMError, OpenAIBackend

OK_BODY = {
    "id": "x", "object": "chat.completion", "created": 0, "model": "m",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": '{"answer": "ok"}'}}],
}


def _backend(handler) -> tuple[OpenAIBackend, list[dict]]:
    sent: list[dict] = []

    def record(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body)
        return handler(body)

    backend = OpenAIBackend(Settings(openai_api_key="test", openai_model="m", openai_base_url="https://llm.test/v1"))
    from openai import OpenAI

    backend._client = OpenAI(api_key="test", base_url="https://llm.test/v1", max_retries=0,
                             http_client=httpx.Client(transport=httpx.MockTransport(record)))
    return backend, sent


def _error(status: int, message: str) -> httpx.Response:
    return httpx.Response(status, json={"error": {"message": message, "type": "invalid_request_error"}})


def test_json_mode_and_temperature_sent_by_default():
    backend, sent = _backend(lambda body: httpx.Response(200, json=OK_BODY))
    assert backend.complete([{"role": "user", "content": "q"}]) == '{"answer": "ok"}'
    assert sent[0]["response_format"] == {"type": "json_object"}
    assert sent[0]["temperature"] == 0


def test_rejected_json_mode_is_dropped_and_remembered():
    def handler(body):
        if "response_format" in body:
            return _error(400, "Invalid value for response_format: json_object is not supported")
        return httpx.Response(200, json=OK_BODY)

    backend, sent = _backend(handler)
    assert backend.complete([{"role": "user", "content": "q"}]) == '{"answer": "ok"}'
    backend.complete([{"role": "user", "content": "q2"}])
    assert "response_format" in sent[0]
    assert all("response_format" not in b for b in sent[1:])
    assert len(sent) == 3  # one rejected call, then no repeat of the rejected option


def test_rejected_temperature_then_json_mode():
    def handler(body):
        if "temperature" in body:
            return _error(400, "Unsupported value: temperature")
        if "response_format" in body:
            return _error(400, "response_format is not supported")
        return httpx.Response(200, json=OK_BODY)

    backend, sent = _backend(handler)
    assert backend.complete([{"role": "user", "content": "q"}]) == '{"answer": "ok"}'
    assert len(sent) == 3


@pytest.mark.parametrize(
    "status, message, expected",
    [
        (400, "Something else entirely is wrong", "rejected the request"),
        (401, "bad key", "rejected the API key"),
        (404, "model not found", "answered 'not found'"),
        (429, "quota exceeded", "rate-limited"),
    ],
)
def test_provider_errors_become_friendly_llm_errors(status, message, expected):
    backend, _ = _backend(lambda body: _error(status, message))
    with pytest.raises(LLMError, match=expected):
        backend.complete([{"role": "user", "content": "q"}])


def test_not_found_names_both_model_and_endpoint():
    backend, _ = _backend(lambda body: _error(404, "models/gemini-x is not found"))
    with pytest.raises(LLMError) as info:
        backend.complete([{"role": "user", "content": "q"}])
    message = str(info.value)
    assert "gemini" not in backend.model  # model name comes from settings, not the provider message
    assert "'m'" in message and "https://llm.test/v1" in message
    assert "OPENAI_MODEL and OPENAI_BASE_URL" in message


def test_pasted_settings_are_cleaned(monkeypatch):
    from config import Settings

    monkeypatch.setenv("OPENAI_BASE_URL", '  "https://generativelanguage.googleapis.com/v1beta/openai/" ')
    monkeypatch.setenv("OPENAI_MODEL", " gemini-3.5-flash-lite\n")
    s = Settings()
    assert s.openai_base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert s.openai_model == "gemini-3.5-flash-lite"
