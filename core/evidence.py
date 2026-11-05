"""Citation validation and evidence-status enforcement.

The LLM's own evidence labels are never trusted blindly:

1. `validate_source_ids` rejects any ID that is not in the current policy
   index (and, for answers, not among the chunks actually retrieved and shown
   to the model).
2. A claim labelled clearly_stated / unclear with no valid citation is treated
   as unsupported: overview findings are dropped, answers are marked unverified.
3. `support_score` performs a lightweight lexical check that the claim shares
   meaningful vocabulary with the cited excerpts. Weak overlap downgrades
   "clearly_stated" to "unclear" with a note. This is a heuristic guard against
   mis-citation, not a proof of correctness.
"""

from __future__ import annotations

import re
from typing import Iterable

from core.preprocessor import Chunk
from models.schemas import CategoryDraft, CategorySummary, EvidenceStatus, Finding, QAAnswer, QADraft

SUPPORT_THRESHOLD = 0.25

_STOPWORDS = set(
    """a about above after again against all also am an and any are as at be because been before being below between
    both but by can could did do does doing down during each either few for from further had has have having he her
    here hers him his how i if in into is it its itself just may me might more most must my no nor not now of off on
    once only or other our ours out over own policy same says she should so some state stated states such than that
    the their them then there these they this those through to too under until up very was we were what when where
    which while who whom why will with would you your yours information data personal service services company
    describe describes described mention mentions mentioned""".split()
)


def _terms(text: str) -> set[str]:
    words = re.findall(r"[a-z][a-z0-9\-']+", (text or "").lower())
    out = set()
    for w in words:
        w = w.strip("'-")
        if len(w) < 3 or w in _STOPWORDS:
            continue
        for suffix in ("ing", "ies", "ed", "es", "s"):
            if w.endswith(suffix) and len(w) - len(suffix) >= 4:
                w = w[: -len(suffix)]
                break
        out.add(w)
    return out


def support_score(claim: str, excerpts: Iterable[str]) -> float:
    """Fraction of the claim's content terms that appear in the cited excerpts."""
    claim_terms = _terms(claim)
    if not claim_terms:
        return 1.0
    evidence_terms: set[str] = set()
    for e in excerpts:
        evidence_terms |= _terms(e)
    return len(claim_terms & evidence_terms) / len(claim_terms)


def validate_source_ids(source_ids: Iterable[str], allowed_ids: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split IDs into (valid, rejected) against the allowed set, preserving order."""
    allowed = set(allowed_ids)
    valid: list[str] = []
    rejected: list[str] = []
    for sid in source_ids:
        sid = str(sid).strip()
        if not sid:
            continue
        if sid in allowed:
            if sid not in valid:
                valid.append(sid)
        elif sid not in rejected:
            rejected.append(sid)
    return valid, rejected


def _texts(ids: list[str], lookup: dict[str, Chunk]) -> list[str]:
    return [f"{lookup[i].heading or ''} {lookup[i].text}" for i in ids if i in lookup]


def verify_finding(finding: Finding, allowed_ids: set[str], lookup: dict[str, Chunk]) -> Finding | None:
    """Return a verified copy of the finding, or None if it is unsupported."""
    valid, rejected = validate_source_ids(finding.source_ids, allowed_ids)
    notes: list[str] = []
    status = finding.evidence_status

    if status != EvidenceStatus.NOT_FOUND and not valid:
        return None  # a positive claim with no verifiable citation is not shown
    if rejected:
        notes.append("Some citations could not be matched to the policy and were removed.")
    if status == EvidenceStatus.CLEARLY_STATED:
        score = support_score(finding.finding, _texts(valid, lookup))
        if score < SUPPORT_THRESHOLD:
            status = EvidenceStatus.UNCLEAR
            notes.append("Automated check found weak overlap between this finding and its cited text; verify the source.")
    return finding.model_copy(
        update={"source_ids": valid, "evidence_status": status, "verification_note": " ".join(notes) or None}
    )


def verify_category(draft: CategoryDraft, expected_category: str, retrieved: list[Chunk]) -> CategorySummary:
    lookup = {c.id: c for c in retrieved}
    allowed = set(lookup)
    findings: list[Finding] = []
    for f in draft.findings:
        if f.category.value != expected_category:
            f = f.model_copy(update={"category": expected_category})
        checked = verify_finding(f, allowed, lookup)
        if checked is not None and checked.evidence_status != EvidenceStatus.NOT_FOUND:
            findings.append(checked)

    status = draft.evidence_status
    limitations = draft.limitations
    if not findings:
        status = EvidenceStatus.NOT_FOUND
        if draft.findings and draft.evidence_status != EvidenceStatus.NOT_FOUND:
            extra = "Findings proposed for this category could not be verified against the policy text, so they are not shown."
            limitations = f"{limitations} {extra}".strip() if limitations else extra
    else:
        # The category status is derived from its verified findings.
        any_clear = any(f.evidence_status == EvidenceStatus.CLEARLY_STATED for f in findings)
        status = EvidenceStatus.CLEARLY_STATED if any_clear else EvidenceStatus.UNCLEAR

    return CategorySummary(
        category=expected_category,
        summary=draft.summary,
        evidence_status=status,
        findings=findings,
        limitations=limitations,
        retrieved_ids=[c.id for c in retrieved],
    )


def verify_answer(draft: QADraft, retrieved: list[Chunk]) -> QAAnswer:
    lookup = {c.id: c for c in retrieved}
    valid, rejected = validate_source_ids(draft.source_ids, lookup.keys())
    status = draft.evidence_status
    notes: list[str] = []

    if rejected:
        notes.append(f"{len(rejected)} citation(s) did not match the retrieved policy text and were removed.")
    if status != EvidenceStatus.NOT_FOUND and not valid:
        status = EvidenceStatus.UNCLEAR
        notes.append("This answer has no verifiable citation in the policy, so treat it as unverified.")
    elif status == EvidenceStatus.CLEARLY_STATED:
        if support_score(draft.answer, _texts(valid, lookup)) < SUPPORT_THRESHOLD:
            status = EvidenceStatus.UNCLEAR
            notes.append("Automated check found weak overlap between this answer and its cited text; compare with the source.")

    return QAAnswer(
        answer=draft.answer,
        evidence_status=status,
        source_ids=valid,
        limitations=draft.limitations,
        retrieved_ids=[c.id for c in retrieved],
        rejected_ids=rejected,
        verification_note=" ".join(notes) or None,
    )
