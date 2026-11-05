from core.evidence import support_score, validate_source_ids, verify_answer, verify_category
from core.preprocessor import Chunk
from models.schemas import CategoryDraft, EvidenceStatus, QADraft

RET = Chunk(id="chunk_4", index=3, heading="Data Retention",
            text="If you delete your account, we will delete your User Content within 30 days. "
                 "We retain usage information for as long as necessary.")
SHARE = Chunk(id="chunk_2", index=1, heading="How We Share",
              text="We may share device identifiers with advertising partners in the free version.")


def test_validate_source_ids_rejects_unknown_and_dedupes():
    valid, rejected = validate_source_ids(["chunk_4", "chunk_99", "chunk_4", "", "Source 1"], {"chunk_4", "chunk_2"})
    assert valid == ["chunk_4"]
    assert rejected == ["chunk_99", "Source 1"]


def test_support_score():
    assert support_score("User content is deleted within 30 days of account deletion", [RET.text]) > 0.6
    assert support_score("Biometric templates are encrypted with quantum keys", [RET.text]) < 0.25


def _finding(text, status="clearly_stated", ids=("chunk_4",), category="Data Retention"):
    return {"category": category, "finding": text, "plain_language_explanation": "Plain words.",
            "evidence_status": status, "source_ids": list(ids)}


def test_category_drops_fabricated_citations_and_unsupported_findings():
    draft = CategoryDraft.model_validate({
        "category": "Data Retention",
        "summary": "Summary.",
        "evidence_status": "clearly_stated",
        "findings": [
            _finding("User Content is deleted within 30 days after account deletion."),
            _finding("Data is kept forever.", ids=["chunk_77"]),  # fabricated id → dropped
        ],
    })
    result = verify_category(draft, "Data Retention", [RET, SHARE])
    assert len(result.findings) == 1
    assert result.findings[0].source_ids == ["chunk_4"]
    assert result.evidence_status == EvidenceStatus.CLEARLY_STATED


def test_category_with_no_verifiable_findings_becomes_not_found():
    draft = CategoryDraft.model_validate({
        "category": "Data Retention", "summary": "Summary.", "evidence_status": "clearly_stated",
        "findings": [_finding("Kept for 7 years.", ids=[])],
    })
    result = verify_category(draft, "Data Retention", [RET])
    assert result.findings == []
    assert result.evidence_status == EvidenceStatus.NOT_FOUND
    assert "could not be verified" in result.limitations


def test_weakly_supported_claim_is_downgraded():
    draft = CategoryDraft.model_validate({
        "category": "Data Retention", "summary": "Summary.", "evidence_status": "clearly_stated",
        "findings": [_finding("Biometric templates are encrypted with quantum keys.")],
    })
    result = verify_category(draft, "Data Retention", [RET])
    assert result.findings[0].evidence_status == EvidenceStatus.UNCLEAR
    assert result.findings[0].verification_note
    assert result.evidence_status == EvidenceStatus.UNCLEAR


def test_findings_capped_at_three():
    draft = CategoryDraft.model_validate({
        "category": "Data Retention", "summary": "Summary.", "evidence_status": "clearly_stated",
        "findings": [_finding(f"User Content deleted within 30 days ({i}).") for i in range(5)],
    })
    assert len(draft.findings) == 3


def test_answer_only_accepts_retrieved_ids():
    draft = QADraft(answer="Device identifiers may be shared with advertising partners.",
                    evidence_status="clearly_stated", source_ids=["chunk_2", "chunk_4"])
    # chunk_4 exists in the policy but was not retrieved for this answer.
    ans = verify_answer(draft, [SHARE])
    assert ans.source_ids == ["chunk_2"]
    assert ans.rejected_ids == ["chunk_4"]
    assert ans.evidence_status == EvidenceStatus.CLEARLY_STATED


def test_answer_without_valid_citation_is_marked_unverified():
    draft = QADraft(answer="Yes, they sell your data.", evidence_status="clearly_stated", source_ids=["chunk_42"])
    ans = verify_answer(draft, [SHARE])
    assert ans.source_ids == []
    assert ans.evidence_status == EvidenceStatus.UNCLEAR
    assert "unverified" in ans.verification_note


def test_not_found_answer_passes_through():
    draft = QADraft(answer="The processed policy does not say.", evidence_status="not_found", source_ids=[])
    ans = verify_answer(draft, [SHARE])
    assert ans.evidence_status == EvidenceStatus.NOT_FOUND
    assert ans.verification_note is None
