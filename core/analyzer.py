"""Policy ingestion and the five-category "Privacy at a Glance" analysis."""

from __future__ import annotations

import hashlib
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from config import Settings, get_settings
from core.embeddings import Embedder
from core.evidence import verify_category
from core.llm import LLMClient, LLMError
from core.preprocessor import Chunk, chunk_policy, guess_title, validate_policy_text
from core.retrieval import PolicyIndex
from models.schemas import Category, CategoryDraft, CategorySummary, EvidenceStatus, PolicyOverview
from prompts.system_prompts import CATEGORY_QUERIES, build_category_messages

log = logging.getLogger(__name__)
CATEGORIES = [c.value for c in Category]


@dataclass
class ProcessedPolicy:
    """Everything derived from one submitted policy (kept in memory only)."""

    text: str
    title: str | None
    source_url: str | None
    chunks: list[Chunk]
    index: PolicyIndex
    fingerprint: str
    is_demo: bool = False

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def process_policy(text: str, embedder: Embedder, title: str | None = None, source_url: str | None = None,
                   settings: Settings | None = None, is_demo: bool = False) -> ProcessedPolicy:
    """Validate, chunk and index a policy. Raises PolicyValidationError on bad input."""
    settings = settings or get_settings()
    cleaned = validate_policy_text(text, settings)
    chunks = chunk_policy(cleaned, settings)
    if not chunks:
        from core.preprocessor import PolicyValidationError

        raise PolicyValidationError("No usable policy text was found after cleaning.")
    index = PolicyIndex(chunks, embedder)
    return ProcessedPolicy(
        text=cleaned,
        title=title or guess_title(cleaned),
        source_url=source_url,
        chunks=chunks,
        index=index,
        fingerprint=hashlib.sha256(cleaned.encode()).hexdigest()[:16],
        is_demo=is_demo,
    )


def retrieve_for_category(policy: ProcessedPolicy, category: str, k: int) -> list[Chunk]:
    hits = policy.index.search_many(CATEGORY_QUERIES[category], k=k)
    # Present excerpts in document order so the model reads them naturally.
    return sorted((h.chunk for h in hits), key=lambda c: c.index)


def analyze_category(policy: ProcessedPolicy, category: str, llm: LLMClient, settings: Settings | None = None) -> CategorySummary:
    settings = settings or get_settings()
    chunks = retrieve_for_category(policy, category, settings.category_top_k)
    try:
        draft = llm.complete_json(build_category_messages(category, chunks), CategoryDraft)
    except LLMError as exc:
        log.warning("Category %s failed: %s", category, exc)
        return CategorySummary(
            category=category,
            summary="This category could not be analysed because the AI request failed.",
            evidence_status=EvidenceStatus.NOT_FOUND,
            findings=[],
            retrieved_ids=[c.id for c in chunks],
            error=str(exc),
        )
    return verify_category(draft, category, chunks)


def analyze_policy(policy: ProcessedPolicy, llm: LLMClient, settings: Settings | None = None,
                   max_workers: int | None = None, retry_pause_s: float | None = None) -> PolicyOverview:
    """Analyse the five categories a few at a time. Individual failures don't abort the rest.

    Categories that still fail (typically a provider "busy" answer) get one more, sequential
    attempt after a short pause, so a brief outage rarely reaches the user.
    """
    settings = settings or get_settings()
    workers = max_workers or settings.llm_max_concurrency
    pause = settings.category_retry_pause_s if retry_pause_s is None else retry_pause_s
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda c: analyze_category(policy, c, llm, settings), CATEGORIES))
    failed = [i for i, r in enumerate(results) if r.error]
    if failed:
        time.sleep(pause)
        for i in failed:
            results[i] = analyze_category(policy, CATEGORIES[i], llm, settings)
    return PolicyOverview(categories=results)
