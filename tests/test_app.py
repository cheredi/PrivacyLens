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
    at.text_area[0].input("We collect stuff.").run()
    at.button(key="analyze_text").click().run()
    assert not at.exception
    assert at.session_state.policy is None
    assert any("too short" in e.value for e in at.error)
