"""FAISS-backed retrieval over policy chunks.

Each chunk (≈400–600 tokens) is embedded as several overlapping *passages*
short enough for all-MiniLM-L6-v2's 256 word-piece window. A query is scored
against every passage and a chunk's score is its best passage score, so the
whole chunk is searchable while the LLM still receives complete chunks.

The index lives only in memory (Streamlit session state); nothing derived from
user-submitted policies is written to disk.
"""

from __future__ import annotations

from dataclasses import dataclass

import faiss

from core.embeddings import Embedder
from core.preprocessor import Chunk

PASSAGE_WORDS = 140
PASSAGE_OVERLAP = 35


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


def _passages(chunk: Chunk) -> list[str]:
    words = chunk.text.split()
    prefix = f"{chunk.heading}. " if chunk.heading else ""
    if len(words) <= PASSAGE_WORDS:
        return [prefix + chunk.text]
    out = []
    step = PASSAGE_WORDS - PASSAGE_OVERLAP
    for start in range(0, len(words), step):
        out.append(prefix + " ".join(words[start:start + PASSAGE_WORDS]))
        if start + PASSAGE_WORDS >= len(words):
            break
    return out


class PolicyIndex:
    def __init__(self, chunks: list[Chunk], embedder: Embedder):
        if not chunks:
            raise ValueError("Cannot build an index with no chunks.")
        self.chunks = list(chunks)
        self.embedder = embedder
        self._by_id = {c.id: c for c in self.chunks}
        if len(self._by_id) != len(self.chunks):
            raise ValueError("Chunk IDs must be unique.")

        passage_texts: list[str] = []
        self._passage_owner: list[int] = []
        for i, chunk in enumerate(self.chunks):
            for p in _passages(chunk):
                passage_texts.append(p)
                self._passage_owner.append(i)
        vectors = embedder.embed(passage_texts)
        self.index = faiss.IndexFlatIP(vectors.shape[1])  # cosine similarity on normalised vectors
        self.index.add(vectors)

    # -- lookup -----------------------------------------------------------
    @property
    def ids(self) -> set[str]:
        return set(self._by_id)

    def get(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    def __len__(self) -> int:
        return len(self.chunks)

    # -- search -----------------------------------------------------------
    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        return self.search_many([query], k=k)

    def search_many(self, queries: list[str], k: int = 5) -> list[RetrievedChunk]:
        """Search with one or more queries; a chunk's score is its best match."""
        queries = [q for q in queries if q and q.strip()]
        if not queries:
            return []
        q_vecs = self.embedder.embed(queries)
        n_passages = self.index.ntotal
        depth = min(n_passages, max(k * 4, 20))
        scores, idxs = self.index.search(q_vecs, depth)
        best: dict[int, float] = {}
        for row_scores, row_idxs in zip(scores, idxs):
            for s, p in zip(row_scores, row_idxs):
                if p < 0:
                    continue
                owner = self._passage_owner[p]
                if s > best.get(owner, -1.0):
                    best[owner] = float(s)
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [RetrievedChunk(chunk=self.chunks[i], score=s) for i, s in ranked]
