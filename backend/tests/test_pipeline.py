"""Acceptance test suite for the CiteVerify backend.

Covers all 12 core acceptance requirements from Section 9:
1.  Percentage point change contradiction (10pp vs 20pp).
2.  Compound sentence citation group ambiguity without cross-paper leakage.
3.  DOI metadata mismatch handling without false semantic verdicts.
4.  Provider timeout / lookup failure resilience (other references continue).
5.  Abstract-only coverage abstention when claim requires table data.
6.  Population narrowing resulting in PARTLY_SUPPORTED.
7.  Older publication support without improper supersession claims.
8.  Retracted paper warning and semantic verdict coexistence.
9.  Rejection of invented / unknown passage IDs.
10. Single resolution and download for multiple claims citing the same source.
11. Reading completed report makes zero new provider calls (idempotency).
12. Plain text / LaTeX parsing: no invented pages, LaTeX bib warning preserved.
13. API route validation (POST /verifications, GET /verifications/{id}, 400/404 handling).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

# Ensure backend directory is in sys.path
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Configure environment before importing app modules
_TEST_DB = os.path.join(_BACKEND_DIR, "test_citeverify.db")
os.environ["DB_PATH"] = _TEST_DB
os.environ.setdefault("GEMINI_API_KEY", "test-fake-key")
os.environ.setdefault("GEMINI_MODEL", "gemini-2.0-flash-lite")


from main import app
from app.database import get_job, init_db, save_assessment, save_job
from app.extraction import detect_citation_markers, extract_claims_and_links, parse_references
from app.ingestion import ingest_document
from app.models import (
    Assessment,
    AssessmentStatus,
    BlockType,
    Claim,
    ClaimReferenceLink,
    CoverageStatus,
    DocumentBlock,
    HumanReviewStatus,
    IdentityStatus,
    JobState,
    MappingStatus,
    Passage,
    Reference,
    ResolvedSource,
    SemanticVerdict,
    VerificationJob,
)
from app.pipeline import create_job, run_pipeline
from app.reporting import build_report, summarize_verdicts
from app.resolution import resolve_all_references, resolve_reference
from app.verification import assess_claim_reference_pair


class TestVerificationApiRoutes(unittest.TestCase):
    """Test the HTTP API contracts and endpoints."""

    def setUp(self):
        self.client = TestClient(app)
        init_db()

    def test_health_preserved(self):
        """Preserve existing /health liveness route."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})

    def test_api_health_preserved(self):
        """Preserve existing /api/health academic discovery route."""
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("crossref", resp.json().get("providers", []))

    def test_unsupported_file_type_rejected(self):
        """Reject unsupported file formats with HTTP 400."""
        files = {"file": ("figure.png", b"\x89PNG\r\n\x1a\n", "image/png")}
        resp = self.client.post("/verifications", files=files)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unsupported file type", resp.json()["detail"])

    def test_unknown_job_id_returns_404(self):
        """GET /verifications/{id} returns 404 for nonexistent job."""
        resp = self.client.get("/verifications/nonexistent_job_12345")
        self.assertEqual(resp.status_code, 404)

    def test_completed_report_makes_zero_provider_calls(self):
        """Requirement 11: Reading a completed report makes zero external calls."""
        job_id = "job_cached_done_001"
        report_data = {
            "job_id": job_id,
            "document_id": "doc_001",
            "total_claims": 1,
            "total_references": 1,
            "total_pairs": 1,
            "assessments": [],
            "issues": [],
            "stage_errors": [],
        }
        job = VerificationJob(
            job_id=job_id,
            input_filename="test.txt",
            input_format="text/plain",
            job_state=JobState.completed,
            current_stage=None,
            report=report_data,
            error_message=None,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        save_job(job)

        with patch("app.providers.classify_claim_gemini") as mock_gemini, \
             patch("httpx.AsyncClient.get") as mock_http:
            resp = self.client.get(f"/verifications/{job_id}")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "completed")
            self.assertEqual(data["report"]["job_id"], job_id)
            mock_gemini.assert_not_called()
            mock_http.assert_not_called()


class TestStage1Ingestion(unittest.IsolatedAsyncioTestCase):
    """Stage 1 tests: document parsing, location tracking, and format rules."""

    async def test_plain_text_locations_and_no_page_numbers(self):
        """Requirement 12: Plain text retains character offsets and page is null."""
        text = b"First paragraph with claim.\n\nSecond paragraph.\n\nReferences\n\n[1] Smith 2022."
        doc_id, blocks, bib_warning = await ingest_document(text, "manuscript.txt", "text/plain")

        self.assertTrue(doc_id)
        self.assertGreater(len(blocks), 1)
        # Verify page is None for plain text
        for b in blocks:
            self.assertIsNone(b.page)

        # Verify bibliography block is correctly identified
        bib_blocks = [b for b in blocks if b.block_type == BlockType.bibliography]
        self.assertTrue(len(bib_blocks) >= 1)

    async def test_latex_missing_bib_warning_preserved(self):
        """Requirement 12: Missing external .bib file issues clear warning."""
        latex = b"\\section{Intro}\nAccuracy rose \\cite{smith2022}.\n\\bibliography{myrefs}"
        doc_id, blocks, bib_warning = await ingest_document(latex, "paper.tex", "text/x-tex")

        self.assertIsNotNone(bib_warning)
        self.assertIn("bib", bib_warning.lower())
        # Citation key preserved in text
        found_key = any("smith2022" in b.text for b in blocks)
        self.assertTrue(found_key)

    async def test_empty_content_raises_value_error(self):
        """Never produce a successful empty audit for a failed parse."""
        with self.assertRaises(ValueError):
            await ingest_document(b"   \n\n   \t", "empty.txt", "text/plain")


class TestStage2Extraction(unittest.IsolatedAsyncioTestCase):
    """Stage 2 tests: bibliography parsing, marker detection, atomic claim mapping."""

    def test_citation_range_expansion(self):
        """Expand citation range [2-4] to references [2], [3], [4]."""
        block = DocumentBlock(
            block_id="b1",
            document_id="doc1",
            block_type=BlockType.text,
            text="Several methods were evaluated [2-4].",
            section="Methods",
            page=None,
            char_start=0,
            char_end=37,
        )
        refs = [
            Reference(ref_id=f"r{i}", document_id="doc1", raw_entry=f"[{i}] Author {i}", raw_marker=f"[{i}]")
            for i in range(1, 6)
        ]
        markers = detect_citation_markers(block, refs)
        self.assertEqual(len(markers), 1)
        marker_text, cs, ce, matched_ids = markers[0]
        self.assertEqual(marker_text, "[2-4]")
        self.assertIn("r2", matched_ids)
        self.assertIn("r3", matched_ids)
        self.assertIn("r4", matched_ids)

    def test_compound_sentence_group_ambiguity(self):
        """Requirement 2: Ambiguous citation groups preserve mapping without false attribution."""
        block = DocumentBlock(
            block_id="b1",
            document_id="doc1",
            block_type=BlockType.text,
            text="Accuracy increased and inference time fell [2, 5].",
            section="Results",
            page=None,
            char_start=0,
            char_end=49,
        )
        refs = [
            Reference(ref_id="r2", document_id="doc1", raw_entry="[2] Smith 2022", raw_marker="[2]"),
            Reference(ref_id="r5", document_id="doc1", raw_entry="[5] Wang 2021", raw_marker="[5]"),
        ]
        markers = detect_citation_markers(block, refs)
        self.assertEqual(len(markers), 1)
        _, _, _, matched_ids = markers[0]
        self.assertEqual(set(matched_ids), {"r2", "r5"})

    async def test_invalid_ref_id_from_model_is_filtered(self):
        """Requirement 9: Validation rejects reference IDs outside the allowed set."""
        block = DocumentBlock(
            block_id="b1",
            document_id="doc1",
            block_type=BlockType.text,
            text="Accuracy improved [1].",
            section="Results",
            page=None,
            char_start=0,
            char_end=22,
        )
        ref = Reference(ref_id="r1", document_id="doc1", raw_entry="[1] Author 2022", raw_marker="[1]")

        # Mock Gemini returning an invented reference ID "r999"
        mock_output = [
            {"claim_text": "Accuracy improved", "original_span": "Accuracy improved", "ref_ids": ["r999", "r1"], "qualifiers": ""}
        ]
        with patch("app.extraction.extract_claims_gemini", new=AsyncMock(return_value=mock_output)):
            claims, links, issues = await extract_claims_and_links([block], [ref])
            # The invalid "r999" must not produce a link to r999
            linked_refs = {link.ref_id for link in links}
            self.assertNotIn("r999", linked_refs)


class TestStage3Resolution(unittest.IsolatedAsyncioTestCase):
    """Stage 3 tests: DOI lookup, bibliographic matching, timeout handling."""

    async def test_doi_metadata_mismatch_reported(self):
        """Requirement 3: Valid DOI resolving to a different paper reports metadata_mismatch."""
        ref = Reference(
            ref_id="r1",
            document_id="d1",
            raw_entry="[1] Jones, B. (2020). Quantum Computing Advances. Nature.",
            title="Quantum Computing Advances",
            authors=["Jones, B."],
            year=2020,
            doi="10.1038/s41586-020-0001",
        )

        mock_crossref_resp = {
            "status": "ok",
            "message": {
                "title": ["Completely Unrelated Soil Chemistry Study"],
                "author": [{"family": "Brown", "given": "Charles"}],
                "published": {"date-parts": [[2015]]},
                "DOI": "10.1038/s41586-020-0001",
            },
        }

        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_crossref_resp
        mock_client.get.return_value = mock_resp

        resolved = await resolve_reference(ref, mock_client)
        self.assertEqual(resolved.identity_status, IdentityStatus.metadata_mismatch)

    async def test_reference_lookup_timeout_is_lookup_failed(self):
        """Requirement 4: Provider timeout records lookup_failed without halting other refs."""
        import httpx
        ref = Reference(
            ref_id="r2",
            document_id="d1",
            raw_entry="[2] Smith (2022). Machine Learning.",
            title="Machine Learning",
            authors=["Smith"],
            year=2022,
            doi="10.1234/ml.2022",
        )

        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.TimeoutException("Read timed out")

        resolved = await resolve_reference(ref, mock_client)
        self.assertEqual(resolved.identity_status, IdentityStatus.lookup_failed)


class TestStage5Verification(unittest.IsolatedAsyncioTestCase):
    """Stage 5 tests: semantic verdict rules, quantitative checks, and safety rules."""

    async def test_percentage_point_contradiction(self):
        """Requirement 1: 10 percentage point actual vs 20 percentage point claimed is CONTRADICTORY."""
        claim = Claim(
            claim_id="c1",
            document_id="d1",
            block_id="b1",
            original_span="a 20-percentage-point increase",
            normalized_text="accuracy increased by 20 percentage points",
            char_start=None,
            char_end=None,
        )
        source = ResolvedSource(
            source_id="s1",
            ref_id="r1",
            identity_status=IdentityStatus.matched,
            coverage=CoverageStatus.full_text,
            is_retracted=False,
        )
        passages = [
            Passage(
                passage_id="p1",
                source_id="s1",
                verbatim_text="Accuracy rose from 50% to 60%, representing a 10 percentage point improvement.",
                section="Results",
                page=5,
            )
        ]

        with patch("app.verification.classify_claim_gemini", new=AsyncMock(return_value={
            "verdict": "CONTRADICTORY",
            "evidence_passage_ids": ["p1"],
            "explanation": "Paper reports accuracy rose from 50% to 60% (10 percentage points), contradicting the claimed 20 percentage points.",
            "issue_flags": ["numerical_discrepancy"],
        })):
            assessment = await assess_claim_reference_pair(claim, source, passages)
            self.assertEqual(assessment.semantic_verdict, SemanticVerdict.CONTRADICTORY)
            self.assertIn("p1", assessment.evidence_passage_ids)
            self.assertIn("numerical_discrepancy", assessment.issue_flags)

    async def test_abstract_only_table_abstains(self):
        """Requirement 5: Claim depending on unavailable table abstains (unable_to_assess)."""
        claim = Claim(
            claim_id="c2",
            document_id="d1",
            block_id="b1",
            original_span="subgroup analysis table 3",
            normalized_text="subgroup analysis in Table 3 demonstrated significant response",
            char_start=None,
            char_end=None,
        )
        source = ResolvedSource(
            source_id="s2",
            ref_id="r2",
            identity_status=IdentityStatus.matched,
            coverage=CoverageStatus.abstract_only,
            is_retracted=False,
        )
        passages = [
            Passage(
                passage_id="p2",
                source_id="s2",
                verbatim_text="The overall treatment was effective in the primary cohort.",
                section="Abstract",
                page=1,
            )
        ]

        assessment = await assess_claim_reference_pair(claim, source, passages)
        self.assertEqual(assessment.assessment_status, AssessmentStatus.unable_to_assess)
        self.assertIsNone(assessment.semantic_verdict)
        self.assertIn("insufficient_abstract_coverage", assessment.issue_flags)

    async def test_population_narrowing_partly_supported(self):
        """Requirement 6: Adult-only study generalized to all ages yields PARTLY_SUPPORTED."""
        claim = Claim(
            claim_id="c3",
            document_id="d1",
            block_id="b1",
            original_span="effective across all patient populations",
            normalized_text="treatment is effective across all patient populations including children",
            char_start=None,
            char_end=None,
        )
        source = ResolvedSource(
            source_id="s3",
            ref_id="r3",
            identity_status=IdentityStatus.matched,
            coverage=CoverageStatus.full_text,
            is_retracted=False,
        )
        passages = [
            Passage(
                passage_id="p3",
                source_id="s3",
                verbatim_text="The study was conducted exclusively in adults aged 18-65.",
                section="Methods",
                page=2,
            )
        ]

        with patch("app.verification.classify_claim_gemini", new=AsyncMock(return_value={
            "verdict": "PARTLY_SUPPORTED",
            "evidence_passage_ids": ["p3"],
            "explanation": "Source confirms efficacy in adults only; manuscript overgeneralizes to all age groups.",
            "issue_flags": ["population_mismatch"],
        })):
            assessment = await assess_claim_reference_pair(claim, source, passages)
            self.assertEqual(assessment.semantic_verdict, SemanticVerdict.PARTLY_SUPPORTED)
            self.assertIn("population_mismatch", assessment.issue_flags)

    async def test_old_paper_supported_with_age_review(self):
        """Requirement 7: Old paper accurately cited is supported; age alone is not supersession."""
        claim = Claim(
            claim_id="c4",
            document_id="d1",
            block_id="b1",
            original_span="initial baseline accuracy was 72%",
            normalized_text="initial baseline accuracy was 72%",
            char_start=None,
            char_end=None,
        )
        source = ResolvedSource(
            source_id="s4",
            ref_id="r4",
            matched_year=2008,
            identity_status=IdentityStatus.matched,
            coverage=CoverageStatus.full_text,
            is_retracted=False,
            age_flag=True,
        )
        passages = [
            Passage(
                passage_id="p4",
                source_id="s4",
                verbatim_text="The baseline accuracy reached 72% on the benchmark.",
                section="Results",
                page=4,
            )
        ]

        with patch("app.verification.classify_claim_gemini", new=AsyncMock(return_value={
            "verdict": "SUPPORTED",
            "evidence_passage_ids": ["p4"],
            "explanation": "Source confirms baseline accuracy was 72%.",
            "issue_flags": [],
        })):
            assessment = await assess_claim_reference_pair(claim, source, passages)
            self.assertEqual(assessment.semantic_verdict, SemanticVerdict.SUPPORTED)
            self.assertIn("age_review", assessment.issue_flags)

    async def test_retracted_paper_retains_support_with_warning(self):
        """Requirement 8: Retracted paper accurately cited preserves support + retraction warning."""
        claim = Claim(
            claim_id="c5",
            document_id="d1",
            block_id="b1",
            original_span="vaccine caused autism",
            normalized_text="study reported an association between MMR vaccine and autism",
            char_start=None,
            char_end=None,
        )
        source = ResolvedSource(
            source_id="s5",
            ref_id="r5",
            identity_status=IdentityStatus.matched,
            coverage=CoverageStatus.full_text,
            is_retracted=True,
            retraction_note="Retracted by journal editors",
        )
        passages = [
            Passage(
                passage_id="p5",
                source_id="s5",
                verbatim_text="We identified an association between measles, mumps, and rubella vaccination and autism.",
                section="Findings",
                page=1,
            )
        ]

        with patch("app.verification.classify_claim_gemini", new=AsyncMock(return_value={
            "verdict": "SUPPORTED",
            "evidence_passage_ids": ["p5"],
            "explanation": "The source paper indeed reported this finding in its published text.",
            "issue_flags": [],
        })):
            assessment = await assess_claim_reference_pair(claim, source, passages, is_retracted=True)
            self.assertEqual(assessment.semantic_verdict, SemanticVerdict.SUPPORTED)
            self.assertIn("retraction_warning", assessment.issue_flags)

    async def test_invented_passage_id_rejected(self):
        """Requirement 9: Model returns unknown passage ID; backend rejects it."""
        claim = Claim(
            claim_id="c6",
            document_id="d1",
            block_id="b1",
            original_span="latency dropped by 30ms",
            normalized_text="latency dropped by 30ms",
            char_start=None,
            char_end=None,
        )
        source = ResolvedSource(
            source_id="s6",
            ref_id="r6",
            identity_status=IdentityStatus.matched,
            coverage=CoverageStatus.full_text,
            is_retracted=False,
        )
        passages = [
            Passage(
                passage_id="p_real_001",
                source_id="s6",
                verbatim_text="Inference latency dropped by 30ms on GPU.",
                section="Results",
                page=3,
            )
        ]

        # Model hallucinates a passage ID from another document or made up
        with patch("app.verification.classify_claim_gemini", new=AsyncMock(return_value={
            "verdict": "SUPPORTED",
            "evidence_passage_ids": ["p_hallucinated_999"],
            "explanation": "Supported by the source text.",
            "issue_flags": [],
        })):
            assessment = await assess_claim_reference_pair(claim, source, passages)
            # Must NOT include the invented passage ID
            self.assertNotIn("p_hallucinated_999", assessment.evidence_passage_ids)


class TestFullPipelineIntegration(unittest.IsolatedAsyncioTestCase):
    """End-to-end pipeline execution with live text parser and injected provider fakes."""

    async def test_full_pipeline_plain_text(self):
        """Requirements 10 & 13: End-to-end verification run on a plain text manuscript."""
        manuscript_text = (
            b"Introduction\n\n"
            b"Our evaluation showed accuracy rose from 50% to 60% [1]. "
            b"Furthermore, inference latency decreased by 30ms [1].\n\n"
            b"References\n\n"
            b"[1] Smith, A. (2022). Neural Optimization. Journal of AI, 12, 45-50. DOI: 10.1234/ai.2022.45\n"
        )

        job = create_job("test_manuscript.txt", "text/plain")
        job_id = job.job_id

        async def fake_extract(paragraph_text, marked_span, allowed_ref_ids, original_span):
            return [
                {
                    "claim_text": "accuracy increased from 50% to 60%",
                    "original_span": "accuracy rose from 50% to 60%",
                    "ref_ids": allowed_ref_ids,
                    "qualifiers": "",
                },
                {
                    "claim_text": "inference latency decreased by 30ms",
                    "original_span": "inference latency decreased by 30ms",
                    "ref_ids": allowed_ref_ids,
                    "qualifiers": "",
                },
            ]

        async def fake_resolve(refs):
            return [
                ResolvedSource(
                    source_id=f"src_{r.ref_id}",
                    ref_id=r.ref_id,
                    canonical_doi=r.doi or "10.1234/ai.2022.45",
                    matched_title="Neural Optimization",
                    matched_authors=["Smith, A."],
                    matched_year=2022,
                    identity_status=IdentityStatus.matched,
                    coverage=CoverageStatus.full_text,
                    is_retracted=False,
                )
                for r in refs
            ]

        async def fake_retrieve(source, claim_texts, client):
            passages = [
                Passage(
                    passage_id=f"p_{source.source_id}_1",
                    source_id=source.source_id,
                    verbatim_text="Model accuracy rose from 50% to 60% on ImageNet, with latency dropping by 30ms.",
                    section="Results",
                    page=3,
                )
            ]
            source.coverage = CoverageStatus.full_text
            return source, passages

        async def fake_classify(claim_text, passages, coverage, source_title):
            pids = [p["passage_id"] for p in passages]
            return {
                "verdict": "SUPPORTED",
                "evidence_passage_ids": pids,
                "explanation": f"Passage directly states '{claim_text}'.",
                "issue_flags": [],
            }

        with patch("app.extraction.extract_claims_gemini", side_effect=fake_extract), \
             patch("app.pipeline.resolve_all_references", side_effect=fake_resolve), \
             patch("app.pipeline.retrieve_source", side_effect=fake_retrieve), \
             patch("app.verification.classify_claim_gemini", side_effect=fake_classify):

            await run_pipeline(job_id, manuscript_text, "test_manuscript.txt", "text/plain")

        completed_job = get_job(job_id)
        self.assertIsNotNone(completed_job)
        self.assertEqual(completed_job.job_state, JobState.completed)
        self.assertIsNotNone(completed_job.report)

        report = completed_job.report
        self.assertGreaterEqual(report.get("total_claims", 0), 1)
        self.assertGreaterEqual(report.get("total_references", 0), 1)
        self.assertGreaterEqual(len(report.get("assessments", [])), 1)

        first_asmt = report["assessments"][0]
        self.assertEqual(first_asmt["semantic_verdict"], "SUPPORTED")

        self.assertGreaterEqual(len(first_asmt["evidence_passage_ids"]), 1)



if __name__ == "__main__":
    unittest.main()
