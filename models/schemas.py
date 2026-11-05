"""Pydantic schemas for every structured LLM output.

Model output is parsed into these classes before anything is displayed. The
`*Draft` models describe what the LLM must return; evidence validation then
produces the final display models with verified citations.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EvidenceStatus(str, Enum):
    CLEARLY_STATED = "clearly_stated"
    UNCLEAR = "unclear"
    NOT_FOUND = "not_found"

    @property
    def label(self) -> str:
        return {"clearly_stated": "Clearly stated", "unclear": "Unclear", "not_found": "Not found"}[self.value]


class Category(str, Enum):
    COLLECTION = "Data Collection"
    USAGE = "Data Usage"
    SHARING = "Data Sharing"
    RETENTION = "Data Retention"
    RIGHTS = "User Rights and Choices"


def _clean_ids(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    out: list[str] = []
    for v in value:  # type: ignore[union-attr]
        s = str(v).strip()
        if s and s not in out:
            out.append(s)
    return out


class Finding(BaseModel):
    model_config = ConfigDict(extra="ignore")

    category: Category
    finding: str = Field(min_length=3, max_length=600)
    plain_language_explanation: str = Field(min_length=3, max_length=800)
    evidence_status: EvidenceStatus
    source_ids: list[str] = Field(default_factory=list)
    verification_note: str | None = None  # added by evidence validation, never by the LLM

    @field_validator("source_ids", mode="before")
    @classmethod
    def normalise_ids(cls, value: object) -> list[str]:
        return _clean_ids(value)


class CategoryDraft(BaseModel):
    """What the LLM returns for one category."""

    model_config = ConfigDict(extra="ignore")

    category: Category
    summary: str = Field(min_length=3, max_length=700)
    evidence_status: EvidenceStatus
    findings: list[Finding] = Field(default_factory=list, max_length=3)
    limitations: str | None = None

    @field_validator("findings", mode="before")
    @classmethod
    def _cap_findings(cls, value: object) -> object:
        return value[:3] if isinstance(value, list) else value


class CategorySummary(CategoryDraft):
    """Validated category ready for display."""

    retrieved_ids: list[str] = Field(default_factory=list)
    error: str | None = None


class PolicyOverview(BaseModel):
    categories: list[CategorySummary]


class QADraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    answer: str = Field(min_length=1, max_length=2500)
    evidence_status: EvidenceStatus
    source_ids: list[str] = Field(default_factory=list)
    limitations: str | None = None

    @field_validator("source_ids", mode="before")
    @classmethod
    def normalise_ids(cls, value: object) -> list[str]:
        return _clean_ids(value)


class QAAnswer(QADraft):
    retrieved_ids: list[str] = Field(default_factory=list)
    rejected_ids: list[str] = Field(default_factory=list)
    verification_note: str | None = None
