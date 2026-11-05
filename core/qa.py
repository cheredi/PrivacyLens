"""Retrieval-augmented question answering over a processed policy."""

from __future__ import annotations

import re

from config import Settings, get_settings
from core.analyzer import ProcessedPolicy
from core.evidence import verify_answer
from core.llm import LLMClient
from core.preprocessor import Chunk
from models.schemas import QAAnswer, QADraft
from prompts.system_prompts import build_qa_messages

_FOLLOW_UP_HINTS = re.compile(
    r"\b(it|they|them|that|this|those|these|there|which|what about|and|also|how about|same)\b", re.I
)
MAX_QUESTION_CHARS = 1000


def retrieval_queries(question: str, history: list[dict[str, str]] | None) -> list[str]:
    """Build retrieval queries. Short or referential follow-ups borrow the previous question."""
    queries = [question]
    if history:
        prev_user = next((t["content"] for t in reversed(history) if t["role"] == "user"), None)
        if prev_user and (len(question.split()) <= 8 or _FOLLOW_UP_HINTS.search(question)):
            queries.append(f"{prev_user} {question}")
    return queries


def retrieve_for_question(policy: ProcessedPolicy, question: str, history: list[dict[str, str]] | None,
                          k: int) -> list[Chunk]:
    hits = policy.index.search_many(retrieval_queries(question, history), k=k)
    return sorted((h.chunk for h in hits), key=lambda c: c.index)


def answer_question(policy: ProcessedPolicy, question: str, llm: LLMClient,
                    history: list[dict[str, str]] | None = None, settings: Settings | None = None) -> QAAnswer:
    """Answer a question using only retrieved excerpts; citations are validated before returning."""
    settings = settings or get_settings()
    question = (question or "").strip()[:MAX_QUESTION_CHARS]
    if not question:
        raise ValueError("Please enter a question.")
    chunks = retrieve_for_question(policy, question, history, settings.qa_top_k)
    draft = llm.complete_json(build_qa_messages(question, chunks, history), QADraft)
    return verify_answer(draft, chunks)
