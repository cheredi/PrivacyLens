"""Central configuration for PrivacyLens.

All secrets are read from environment variables (or Streamlit secrets when
deployed to Streamlit Community Cloud). Nothing sensitive lives in source code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # Optional: load a local .env file during development.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - python-dotenv is in requirements
    pass

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_POLICY_PATH = DATA_DIR / "sample_policy.txt"


def _read_setting(name: str, default: str | None = None) -> str | None:
    """Read a setting from the environment, falling back to Streamlit secrets."""
    value = os.environ.get(name)
    if value:
        return value
    try:  # Streamlit secrets are only available inside a Streamlit runtime.
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # noqa: BLE001 - secrets file missing or not in Streamlit
        pass
    return default


@dataclass(frozen=True)
class Settings:
    # LLM
    openai_api_key: str | None = field(default_factory=lambda: _read_setting("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: _read_setting("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini")
    openai_base_url: str | None = field(default_factory=lambda: _read_setting("OPENAI_BASE_URL"))
    # Empty string disables sending temperature (some reasoning models reject it).
    openai_temperature: str = field(default_factory=lambda: _read_setting("OPENAI_TEMPERATURE", "0") or "")
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2  # extra attempts after the first when JSON is invalid

    # Embeddings / retrieval
    embedding_model: str = field(
        default_factory=lambda: _read_setting("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        or "sentence-transformers/all-MiniLM-L6-v2"
    )
    chunk_target_tokens: int = 480  # within the 400–600 token target
    chunk_overlap_tokens: int = 80
    min_section_tokens: int = 70  # tiny sections are merged with their neighbour
    qa_top_k: int = 5
    category_top_k: int = 6

    # Fetching
    fetch_timeout_s: float = 12.0
    fetch_max_bytes: int = 3_000_000
    fetch_max_redirects: int = 5

    # Input validation
    min_policy_words: int = 120
    max_policy_chars: int = 400_000

    @property
    def llm_enabled(self) -> bool:
        return bool(self.openai_api_key)


def get_settings() -> Settings:
    return Settings()
