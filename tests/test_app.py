"""End-to-end UI tests using Streamlit's AppTest (no browser, no network).

The LLM is replaced by a scripted backend; extraction, chunking, embeddings,
FAISS retrieval and evidence validation all run for real.
"""

import json
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.embeddings import EmbeddingError, load_embedder
from core.llm import LLMClient

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(autouse=True)
def _require_model():
    from config import get_settings

    try:
        load_embedder(get_settings().embedding_model)
    except EmbeddingError:
        pytest.skip("Embedding model not available offline")


def _scripted_backend():
    class Backend:
        def complete(self, messages):
            content = messages[1]["content"]
            ids = re.findall(r'<excerpt id="(chunk_\d+)"', content)
            cat = re.search(r"Category: (.+)", content)
            if cat:
                return json.dumps({
                    "category": cat.group(1), "summary": "The policy describes this category.",
                    "evidence_status": "clearly_stated",
                    "findings": [{"category": cat.group(1),
                                  "finding": "The policy describes information practices in this section.",
                                  "plain_language_explanation": "Plain-language version.",
                                  "evidence_status": "unclear", "source_ids": ids[:2]}],
                    "limitations": "No specific detail on everything.",
                })
            return json.dumps({"answer": "Advertising partners may receive device identifiers.",
                               "evidence_status": "clearly_stated", "source_ids": ids[:1], "limitations": None})

    return LLMClient(Backend(), max_retries=0)


def test_demo_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    at.button(key="demo").click().run()
    assert not at.exception
    assert at.session_state.policy is not None and at.session_state.policy.is_demo
    assert at.session_state.overview is None
    assert any("OPENAI_API_KEY" in i.value for i in at.info)


def test_demo_with_scripted_llm_and_chat(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-real")
    monkeypatch.setattr(LLMClient, "from_settings", classmethod(lambda cls, settings=None: _scripted_backend()))
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.button(key="demo").click().run()
    assert not at.exception
    overview = at.session_state.overview
    assert overview is not None and len(overview.categories) == 5
    at.button(key="suggest_0").click().run()
    assert not at.exception
    chat = at.session_state.chat
    assert [m["role"] for m in chat] == ["user", "assistant"]
    assert chat[1]["answer"].source_ids
    # Reset clears policy-specific state.
    at.button(key="reset").click().run()
    assert at.session_state.policy is None and at.session_state.chat == []


def test_short_pasted_text_shows_friendly_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.radio[0].set_value("Paste policy text").run()
    at.text_area[0].input("We collect stuff.")
    at.button(key="analyze_text").click()  # a form sends the typed text together with the click
    at.run()
    assert not at.exception
    assert at.session_state.policy is None
    assert any("too short" in e.value for e in at.error)


def test_failed_ai_request_is_not_shown_as_not_found(monkeypatch):
    """A rejected model name must read as a failure, never as 'the policy doesn't say'."""
    from core.llm import LLMError

    class FailingBackend:
        def complete(self, messages):
            raise LLMError("The AI model 'gemini-flash-latest' was not found. Check OPENAI_MODEL.")

    monkeypatch.setenv("OPENAI_API_KEY", "test-not-real")
    monkeypatch.setattr(LLMClient, "from_settings",
                        classmethod(lambda cls, settings=None: LLMClient(FailingBackend(), max_retries=0)))
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.button(key="demo").click().run()
    assert not at.exception
    page = " ".join(m.value for m in at.markdown)
    assert page.count("Analysis failed") == 5
    assert ">Not found</span></div>" not in page  # no card header claims "Not found"
    assert "was not found. Check OPENAI_MODEL" in page
    assert any("Retry 5 failed categories" in b.label for b in at.button)


def test_url_form_submits_typed_value(monkeypatch):
    """The typed URL reaches the pipeline (blocked address -> friendly SSRF error, no crash)."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_input[0].input("http://localhost/privacy")
    at.button(key="analyze_url").click()
    at.run()
    assert not at.exception
    assert at.session_state.policy is None
    assert any("Local and internal addresses" in e.value for e in at.error)


def test_no_internal_ids_tokens_or_emojis_on_screen(monkeypatch):
    """Users see 'Source N', never chunk IDs, token counts or decorative emojis — in all three tabs."""
    import re as _re

    class ChattyBackend:
        def complete(self, messages):
            content = messages[1]["content"]
            ids = re.findall(r'<excerpt id="(chunk_\d+)"', content)
            cat = re.search(r"Category: (.+)", content)
            if cat:
                return json.dumps({"category": cat.group(1), "summary": f"Summary [{ids[0]}].",
                                   "evidence_status": "clearly_stated",
                                   "findings": [{"category": cat.group(1),
                                                 "finding": f"The policy describes information practices [{ids[0]}].",
                                                 "plain_language_explanation": "Plain.", "evidence_status": "unclear",
                                                 "source_ids": ids[:1]}]})
            return json.dumps({"answer": f"Advertising partners may receive identifiers ({ids[0]}).",
                               "evidence_status": "clearly_stated", "source_ids": ids[:1]})

    monkeypatch.setenv("OPENAI_API_KEY", "test-not-real")
    monkeypatch.setattr(LLMClient, "from_settings",
                        classmethod(lambda cls, settings=None: LLMClient(ChattyBackend(), max_retries=0)))
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.button(key="demo").click().run()
    at.button(key="suggest_0").click().run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.code)
    assert "chunk_" not in text
    assert "tokens" not in text and "ID:" not in text
    assert not _re.search("[\U0001F300-\U0001FAFF⚙\U0001F5C4]", text)
    assert "Source 1" in " ".join(b.label for b in at.button) or "Source" in text
