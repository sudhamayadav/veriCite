"""Stage 5: judge claim–paper pairs and validate cited evidence.

Public API:
-----------
assess_claim_reference_pair(claim, source, passages, is_retracted) -> Assessment
"""
from __future__ import annotations

import logging
import uuid
import re

from app.models import (
    Assessment,
    AssessmentStatus,
    Claim,
    CoverageStatus,
    HumanReviewStatus,
    IdentityStatus,
    Passage,
    ResolvedSource,
    SemanticVerdict,
)
from app.providers import classify_claim_gemini

logger = logging.getLogger(__name__)

# Heuristic patterns for claims that likely require non-abstract details
_DETAILED_DATA_PATTERNS = re.compile(
    r"\b(table\s+\d+|supplementary|subgroup\s+analysis|column\s+\d+|row\s+\d+|appendix|raw\s+data)\b",
    re.IGNORECASE,
)


async def assess_claim_reference_pair(
    claim: Claim,
    source: ResolvedSource,
    passages: list[Passage],
    is_retracted: bool = False,
) -> Assessment:
    """Assess whether a cited source supports an atomic claim.

    Follows Section 7 verdict decision rules:
    - If source identity is not matched or coverage is unavailable, abstain (unable_to_assess).
    - If coverage is abstract_only and claim requires data/table unavailable in abstract, abstain.
    - If no passages are available, abstain (unable_to_assess).
    - Validate all evidence passage IDs returned by LLM against the actual passages for this source.
    - Retain retraction flag and age flag as editorial issue flags.
    """
    assessment_id = f"asmt_{uuid.uuid4().hex[:12]}"
    issue_flags: list[str] = []

    if is_retracted or source.is_retracted:
        issue_flags.append("retraction_warning")
    if source.age_flag:
        issue_flags.append("age_review")

    # 1. Gate: Check identity status
    if source.identity_status != IdentityStatus.matched:
        reason = (
            f"Source resolution status is '{source.identity_status}'. "
            "Cannot assess evidence against an unresolved or mismatched paper."
        )
        if source.identity_status == IdentityStatus.metadata_mismatch:
            issue_flags.append("metadata_mismatch")
        elif source.identity_status == IdentityStatus.lookup_failed:
            issue_flags.append("lookup_failed")
        else:
            issue_flags.append("unresolved_source")

        return Assessment(
            assessment_id=assessment_id,
            claim_id=claim.claim_id,
            ref_id=source.ref_id,
            assessment_status=AssessmentStatus.unable_to_assess,
            semantic_verdict=None,
            evidence_passage_ids=[],
            explanation=reason,
            issue_flags=issue_flags,
            human_review_status=HumanReviewStatus.pending,
            confidence_score=None,
            confidence_basis=None,
        )

    # 2. Gate: Check coverage status
    if source.coverage == CoverageStatus.unavailable:
        return Assessment(
            assessment_id=assessment_id,
            claim_id=claim.claim_id,
            ref_id=source.ref_id,
            assessment_status=AssessmentStatus.unable_to_assess,
            semantic_verdict=None,
            evidence_passage_ids=[],
            explanation="Source document or abstract was unavailable for inspection.",
            issue_flags=issue_flags + ["source_unavailable"],
            human_review_status=HumanReviewStatus.pending,
            confidence_score=None,
            confidence_basis=None,
        )

    # 3. Gate: Check passages
    if not passages:
        return Assessment(
            assessment_id=assessment_id,
            claim_id=claim.claim_id,
            ref_id=source.ref_id,
            assessment_status=AssessmentStatus.unable_to_assess,
            semantic_verdict=None,
            evidence_passage_ids=[],
            explanation="evidence_not_found: No relevant evidence passages could be retrieved from the source.",
            issue_flags=issue_flags + ["evidence_not_found"],
            human_review_status=HumanReviewStatus.pending,
            confidence_score=None,
            confidence_basis=None,
        )

    # 4. Gate: Abstract-only coverage with claim depending on detailed data/table
    if source.coverage == CoverageStatus.abstract_only:
        issue_flags.append("abstract_only_evidence")
        if _DETAILED_DATA_PATTERNS.search(claim.normalized_text) or _DETAILED_DATA_PATTERNS.search(claim.original_span):
            return Assessment(
                assessment_id=assessment_id,
                claim_id=claim.claim_id,
                ref_id=source.ref_id,
                assessment_status=AssessmentStatus.unable_to_assess,
                semantic_verdict=None,
                evidence_passage_ids=[],
                explanation="Claim depends on detailed data, table, or subgroup analysis not accessible in abstract-only coverage.",
                issue_flags=issue_flags + ["insufficient_abstract_coverage"],
                human_review_status=HumanReviewStatus.pending,
                confidence_score=None,
                confidence_basis=None,
            )

    # 5. Build passage payload for Gemini
    passage_payload = [
        {
            "passage_id": p.passage_id,
            "text": p.verbatim_text,
            "section": p.section,
            "page": p.page,
        }
        for p in passages
    ]
    valid_passage_ids = {p.passage_id for p in passages}

    try:
        raw_result = await classify_claim_gemini(
            claim_text=claim.normalized_text,
            passages=passage_payload,
            coverage=str(source.coverage.value if hasattr(source.coverage, "value") else source.coverage),
            source_title=source.matched_title or "Unknown Source",
        )
    except Exception as exc:
        logger.exception("classify_claim_gemini threw unexpected exception: %s", exc)
        return Assessment(
            assessment_id=assessment_id,
            claim_id=claim.claim_id,
            ref_id=source.ref_id,
            assessment_status=AssessmentStatus.failed,
            semantic_verdict=None,
            evidence_passage_ids=[],
            explanation=f"Provider error during classification: {exc}",
            issue_flags=issue_flags + ["provider_error"],
            human_review_status=HumanReviewStatus.pending,
            confidence_score=None,
            confidence_basis=None,
        )

    # 6. Parse and validate verdict
    raw_verdict = raw_result.get("verdict")
    verdict_enum: SemanticVerdict | None = None
    if raw_verdict:
        verdict_str = str(raw_verdict).strip().upper()
        if verdict_str in SemanticVerdict.__members__:
            verdict_enum = SemanticVerdict[verdict_str]

    # Validate evidence passage IDs: ensure they belong strictly to this source's valid passages
    returned_passage_ids = raw_result.get("evidence_passage_ids", [])
    validated_passage_ids = [pid for pid in returned_passage_ids if pid in valid_passage_ids]

    combined_issue_flags = list(set(issue_flags + raw_result.get("issue_flags", [])))

    status = AssessmentStatus.assessed
    explanation = raw_result.get("explanation", "").strip()

    if verdict_enum is None:
        status = AssessmentStatus.unable_to_assess
        if not explanation:
            explanation = "Evidence was inconclusive or model abstained from assigning a verdict."

    return Assessment(
        assessment_id=assessment_id,
        claim_id=claim.claim_id,
        ref_id=source.ref_id,
        assessment_status=status,
        semantic_verdict=verdict_enum,
        evidence_passage_ids=validated_passage_ids,
        explanation=explanation,
        issue_flags=combined_issue_flags,
        human_review_status=HumanReviewStatus.pending,
        confidence_score=None,
        confidence_basis=None,
    )


__all__ = ["assess_claim_reference_pair"]
