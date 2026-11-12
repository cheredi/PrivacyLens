"""Thin OpenAI wrapper that returns Pydantic-validated JSON with controlled retries."""

from __future__ import annotations

import json
import logging
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from config import Settings, get_settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMUnavailableError(Exception):
    """Raised when no API key is configured."""


class LLMError(Exception):
    """Raised when the model cannot produce a valid response."""


class ChatBackend(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


class OpenAIBackend:
    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise LLMUnavailableError(
                "AI features are disabled because OPENAI_API_KEY is not set. See the README for setup."
            )
        from openai import OpenAI

        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
            timeout=settings.llm_timeout_s,
            max_retries=2,  # network-level retries (rate limits, 5xx)
        )
        self.model = settings.openai_model
        self._temperature = float(settings.openai_temperature) if settings.openai_temperature else None
        # JSON mode is requested by default. OpenAI-compatible providers (Gemini, Groq, local
        # servers) differ in what they accept, so an option a provider rejects is dropped and
        # remembered. Output is still parsed and validated against the Pydantic schemas.
        self._json_mode = True

    def _request_kwargs(self, messages: list[dict[str, str]]) -> dict:
        kwargs: dict = {"model": self.model, "messages": messages}
        if self._json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        return kwargs

    def _drop_rejected_option(self, error_text: str) -> bool:
        """Turn off an optional parameter the provider rejected. Returns True if one was dropped."""
        text = error_text.lower()
        if self._temperature is not None and "temperature" in text:
            self._temperature = None
            return True
        if self._json_mode and any(k in text for k in ("response_format", "json_object", "json mode", "response format")):
            self._json_mode = False
            return True
        return False

    def complete(self, messages: list[dict[str, str]]) -> str:
        import openai

        for _ in range(3):  # at most: original request + two option adjustments
            try:
                resp = self._client.chat.completions.create(**self._request_kwargs(messages))
                return resp.choices[0].message.content or ""
            except openai.BadRequestError as exc:
                if self._drop_rejected_option(str(exc)):
                    continue
                raise LLMError(f"The AI service rejected the request: {exc.message}") from exc
            except openai.AuthenticationError as exc:
                raise LLMError("The AI provider rejected the API key. Check OPENAI_API_KEY.") from exc
            except openai.NotFoundError as exc:
                raise LLMError(f"The AI model '{self.model}' was not found. Check OPENAI_MODEL.") from exc
            except openai.RateLimitError as exc:
                raise LLMError("The AI service is rate-limited or out of quota. Please try again shortly.") from exc
            except openai.APITimeoutError as exc:
                raise LLMError("The AI service timed out. Please try again.") from exc
            except openai.APIError as exc:
                raise LLMError(f"The AI service returned an error: {exc}") from exc
        raise LLMError("The AI service rejected the request settings.")


def _extract_json(raw: str) -> object:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("{"):] if "{" in raw else raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start:end + 1])
        raise


class LLMClient:
    def __init__(self, backend: ChatBackend, max_retries: int = 2):
        self.backend = backend
        self.max_retries = max_retries

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "LLMClient":
        settings = settings or get_settings()
        return cls(OpenAIBackend(settings), max_retries=settings.llm_max_retries)

    def complete_json(self, messages: list[dict[str, str]], schema: type[T]) -> T:
        """Call the model and validate against `schema`, retrying with the error on failure."""
        attempt_messages = list(messages)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            raw = self.backend.complete(attempt_messages)
            try:
                data = _extract_json(raw)
                return schema.model_validate(data)
            except (json.JSONDecodeError, ValidationError, TypeError) as exc:
                last_error = exc
                log.warning("Invalid structured output (attempt %d): %s", attempt + 1, exc)
                attempt_messages = list(messages) + [
                    {"role": "assistant", "content": raw[:4000]},
                    {
                        "role": "user",
                        "content": "Your previous reply was not valid for the required JSON schema. Error: "
                        f"{str(exc)[:800]}\nReturn only the corrected JSON object.",
                    },
                ]
        raise LLMError("The AI returned an invalid response format after several attempts.") from last_error
