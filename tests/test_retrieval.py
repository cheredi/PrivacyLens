import os

import pytest

from core.analyzer import process_policy, retrieve_for_category
from core.qa import retrieval_queries
from core.retrieval import PolicyIndex


def _heading_of_top(policy, query, k=3):
    return [h.chunk.heading or "" for h in policy.index.search(query, k=k)]


def test_index_lookup_and_ids(sample_policy):
    idx = sample_policy.index
    assert len(idx) == len(sample_policy.chunks)
    assert idx.get("chunk_1") is sample_policy.chunks[0]
    assert idx.get("chunk_999") is None
    assert idx.ids == {c.id for c in sample_policy.chunks}


def test_search_returns_unique_ranked_chunks(sample_policy):
    hits = sample_policy.index.search("how long do you retain my data", k=4)
    assert 1 <= len(hits) <= 4
    assert len({h.chunk.id for h in hits}) == len(hits)
    assert [h.score for h in hits] == sorted([h.score for h in hits], reverse=True)


def test_category_retrieval_finds_relevant_sections(sample_policy):
    chunks = retrieve_for_category(sample_policy, "Data Retention", k=4)
    assert any("Retention" in (c.heading or "") for c in chunks)


def test_follow_up_queries_use_previous_question():
    history = [{"role": "user", "content": "Do they share data with advertisers?"},
               {"role": "assistant", "content": "Yes, in the free version."}]
    assert len(retrieval_queries("Can I opt out of that?", history)) == 2
    assert retrieval_queries("Can I opt out of that?", None) == ["Can I opt out of that?"]


def test_empty_index_rejected(embedder):
    with pytest.raises(ValueError):
        PolicyIndex([], embedder)


# ---------------------------------------------------------------- real model

@pytest.fixture(scope="module")
def real_policy():
    from config import SAMPLE_POLICY_PATH, Settings
    from core.embeddings import EmbeddingError, load_embedder

    settings = Settings(openai_api_key=None)
    try:
        embedder = load_embedder(settings.embedding_model)
    except EmbeddingError:
        pytest.skip("Embedding model not available offline")
    return process_policy(SAMPLE_POLICY_PATH.read_text(), embedder, settings=settings)


@pytest.mark.skipif(os.environ.get("PRIVACYLENS_SKIP_MODEL") == "1", reason="model tests disabled")
@pytest.mark.parametrize(
    "question, expected",
    [
        ("How long do you keep my data after I delete my account?", "Retention"),
        ("Do you share my information with advertisers?", "Share"),
        ("Can I opt out of marketing emails?", "Choices"),
        ("Do you track my precise GPS location?", "Collect"),
        ("What cookies does the website use?", "Cookies"),
    ],
)
def test_real_model_retrieval(real_policy, question, expected):
    assert any(expected in h for h in _heading_of_top(real_policy, question, k=3))
