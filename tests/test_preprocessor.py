import pytest

from config import Settings
from core.preprocessor import PolicyValidationError, chunk_policy, clean_text, split_sections, validate_policy_text


def test_clean_text_normalises_whitespace_and_controls():
    assert clean_text("Hello​   world\r\n\n\n\nNext\x07 line") == "Hello world\n\nNext line"


@pytest.mark.parametrize("text", ["", "   ", "Too short to be a policy."])
def test_rejects_empty_or_short_input(text, settings):
    with pytest.raises(PolicyValidationError):
        validate_policy_text(text, settings)


def test_rejects_non_prose(settings):
    junk = " ".join(["{\"a\": 1, \"b\": [2,3]} ab"] * 200)
    with pytest.raises(PolicyValidationError):
        validate_policy_text(junk, settings)


def test_rejects_oversized_input():
    with pytest.raises(PolicyValidationError, match="very long"):
        validate_policy_text("word " * 1000, Settings(max_policy_chars=100))


def test_detects_plain_text_headings():
    text = (
        "INFORMATION WE COLLECT\nWe collect your email.\n\n"
        "2. How We Share Information\nWe share with vendors.\n\n"
        "Your Choices\nYou can opt out."
    )
    headings = [s.heading for s in split_sections(text)]
    assert headings == ["Information We Collect", "2. How We Share Information", "Your Choices"]


def test_sentences_are_not_headings():
    sections = split_sections("We collect your email address when you register.\nWe keep it safe.")
    assert [s.heading for s in sections] == [None]


def test_sample_chunks_are_stable_and_sized(sample_text, settings):
    first = chunk_policy(validate_policy_text(sample_text, settings), settings)
    second = chunk_policy(validate_policy_text(sample_text, settings), settings)
    assert [c.id for c in first] == [f"chunk_{i + 1}" for i in range(len(first))]
    assert [(c.id, c.text) for c in first] == [(c.id, c.text) for c in second]
    assert all(c.approx_tokens <= 650 for c in first)
    headings = {c.heading for c in first}
    assert any(h and "Data Retention" in h for h in headings)
    assert any(h and "Information We Collect" in h for h in headings)


def test_long_section_is_split_with_overlap(settings):
    paras = [f"Paragraph {i} says we may collect {i} kinds of data from partners and devices." * 4 for i in range(30)]
    text = "## Long Section\n" + "\n".join(paras)
    chunks = chunk_policy(text, settings)
    assert len(chunks) > 1
    assert all(c.heading == "Long Section" for c in chunks)
    assert all(c.approx_tokens <= settings.chunk_target_tokens * 1.4 for c in chunks)
    # The second chunk starts with overlap carried from the first.
    assert chunks[1].text.startswith("…")
