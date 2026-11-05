"""Shared test fixtures.

Unit tests use a deterministic hashing embedder so they run offline and fast.
`test_retrieval.py` additionally runs against the real all-MiniLM-L6-v2 model
when it is available (skipped otherwise).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import SAMPLE_POLICY_PATH, Settings  # noqa: E402


class HashingEmbedder:
    """Bag-of-words hashing embedder: test double for SentenceTransformer."""

    dimension = 512

    def embed(self, texts):
        out = np.zeros((len(texts), self.dimension), dtype="float32")
        for i, text in enumerate(texts):
            for tok in re.findall(r"[a-z]{3,}", text.lower()):
                tok = tok[:6]  # crude stemming
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dimension] += 1.0
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
        return out


class FakeBackend:
    """Returns scripted responses and records the messages it received."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def complete(self, messages):
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("FakeBackend ran out of responses")
        r = self.responses.pop(0)
        if callable(r):
            return r(messages)
        return r if isinstance(r, str) else json.dumps(r)


@pytest.fixture
def settings() -> Settings:
    return Settings(openai_api_key=None)


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def sample_text() -> str:
    return SAMPLE_POLICY_PATH.read_text(encoding="utf-8")


@pytest.fixture
def sample_policy(sample_text, embedder, settings):
    from core.analyzer import process_policy

    return process_policy(sample_text, embedder, settings=settings, is_demo=True)
