"""Sentence-Transformers embedding wrapper (all-MiniLM-L6-v2 by default)."""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    dimension: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalised float32 vectors, shape (len(texts), dimension)."""


class EmbeddingError(Exception):
    pass


class SentenceTransformerEmbedder:
    """Wraps a SentenceTransformer model and returns normalised embeddings.

    all-MiniLM-L6-v2 truncates input at 256 word-pieces, so the retrieval layer
    embeds each 400–600 token chunk as several shorter passages (see
    core/retrieval.py) instead of silently losing the end of each chunk.
    """

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name, device="cpu")
        except Exception as exc:  # noqa: BLE001 - network/model errors vary
            raise EmbeddingError(
                f"Could not load the embedding model '{model_name}'. The first run downloads it from "
                "Hugging Face, so an internet connection is required."
            ) from exc
        self.model_name = model_name
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dimension = int(get_dim())
        self.max_seq_length = int(getattr(self.model, "max_seq_length", 256) or 256)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        vectors = self.model.encode(
            list(texts), batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype="float32")


@lru_cache(maxsize=2)
def load_embedder(model_name: str) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(model_name)
