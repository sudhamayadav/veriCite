"""Live end-to-end verification walkthrough demonstration."""
from __future__ import annotations

import asyncio
import io
import json
import os
import sys
from unittest.mock import AsyncMock, patch

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from fastapi.testclient import TestClient
from main import app
from app.models import (
    CoverageStatus,
    IdentityStatus,
    Passage,
    ResolvedSource,
    SemanticVerdict,
)

client = TestClient(app)

SAMPLE_MANUSCRIPT = (
    b"# Neural Speed and Precision Analysis\n\n"
    b"Recent advances in model quantization have demonstrated marked improvements in efficiency. "
    b"Specifically, our quantization method increased classification accuracy from 50% to 60% "
    b"on ImageNet while inference latency decreased by 30ms [1]. However, when tested across "
    b"all age demographics, pediatric performance lagged behind adult benchmarks [2].\n\n"
    b"## References\n\n"
    b"[1] Vaswani, A., et al. (2022). Quantization Efficiency in Vision Transformers. "
    b"Journal of Machine Learning, 14, 102-115. DOI: 10.1000/jml.2022.01\n"
    b"[2] Chen, H., & Miller, K. (2021). Demographic Generalization in Clinical Models. "
    b"Lancet Digital Health, 3, e45-e52. DOI: 10.1016/S2589-7500(21)00021-X\n"
)


async def main():
    print("=== CiteVerify Live Walkthrough Demo ===\n")

    # 1. Submit manuscript
    file_payload = {"file": ("manuscript.md", io.BytesIO(SAMPLE_MANUSCRIPT), "text/markdown")}

    # Define mock provider responses for offline execution
    async def mock_extract(paragraph_text, marked_span, allowed_ref_ids, original_span):
        if "[1]" in marked_span:
            return [
                {
                    "claim_text": "quantization increased classification accuracy from 50% to 60%",
                    "original_span": "quantization method increased classification accuracy from 50% to 60%",
                    "ref_ids": allowed_ref_ids,
                    "qualifiers": "dataset=ImageNet",
                },
                {
                    "claim_text": "inference latency decreased by 30ms",
                    "original_span": "inference latency decreased by 30ms",
                    "ref_ids": allowed_ref_ids,
                    "qualifiers": "task=inference",
                },
            ]
        elif "[2]" in marked_span:
            return [
                {
                    "claim_text": "pediatric performance lagged behind adult benchmarks",
                    "original_span": "pediatric performance lagged behind adult benchmarks",
                    "ref_ids": allowed_ref_ids,
                    "qualifiers": "cohort=pediatric vs adult",
                }
            ]
        return []

    async def mock_resolve(refs):
        sources = []
        for r in refs:
            is_ref1 = "10.1000" in (r.doi or "")
            sources.append(
                ResolvedSource(
                    source_id=f"src_{r.ref_id}",
                    ref_id=r.ref_id,
                    canonical_doi=r.doi,
                    matched_title="Quantization Efficiency" if is_ref1 else "Demographic Generalization",
                    matched_authors=["Vaswani, A."] if is_ref1 else ["Chen, H.", "Miller, K."],
                    matched_year=2022 if is_ref1 else 2021,
                    identity_status=IdentityStatus.matched,
                    coverage=CoverageStatus.full_text,
                    is_retracted=False,
                )
            )
        return sources

    async def mock_retrieve(source, claim_texts, client):
        source.coverage = CoverageStatus.full_text
        if "src_" in source.source_id and "r0000" in source.source_id:
            passages = [
                Passage(
                    passage_id=f"pass_{source.source_id}_1",
                    source_id=source.source_id,
                    verbatim_text="On the ImageNet validation set, INT8 quantization raised accuracy from 50% to 60%, reducing per-image latency by 30ms.",
                    section="3. Experiments & Results",
                    page=4,
                )
            ]
        else:
            passages = [
                Passage(
                    passage_id=f"pass_{source.source_id}_2",
                    source_id=source.source_id,
                    verbatim_text="While adult cohorts achieved an AUC of 0.89, pediatric cohorts scored significantly lower at 0.74, indicating a substantial generalization gap.",
                    section="4. Subgroup Analysis",
                    page=7,
                )
            ]
        return source, passages

    async def mock_classify(claim_text, passages, coverage, source_title):
        pids = [p["passage_id"] for p in passages]
        if "latency" in claim_text or "accuracy" in claim_text:
            return {
                "verdict": "SUPPORTED",
                "evidence_passage_ids": pids,
                "explanation": f"Source Section 3 confirms the finding with intact metrics: {claim_text}.",
                "issue_flags": [],
            }
        else:
            return {
                "verdict": "SUPPORTED",
                "evidence_passage_ids": pids,
                "explanation": "Source Section 4 Subgroup Analysis verifies that pediatric performance was significantly lower than adult cohorts (AUC 0.74 vs 0.89).",
                "issue_flags": ["subgroup_demographic_gap"],
            }

    with patch("app.extraction.extract_claims_gemini", side_effect=mock_extract), \
         patch("app.pipeline.resolve_all_references", side_effect=mock_resolve), \
         patch("app.pipeline.retrieve_source", side_effect=mock_retrieve), \
         patch("app.verification.classify_claim_gemini", side_effect=mock_classify):

        post_resp = client.post("/verifications", files=file_payload)
        assert post_resp.status_code == 202, post_resp.text
        data = post_resp.json()
        job_id = data["verification_id"]
        print(f"1. POST /verifications -> HTTP 202 Accepted. Job ID: {job_id}")

        # Wait for background task to complete
        for _ in range(50):
            get_resp = client.get(f"/verifications/{job_id}")
            assert get_resp.status_code == 200
            res = get_resp.json()
            if res.get("status") == "completed":
                break
            await asyncio.sleep(0.1)

    print(f"2. GET /verifications/{job_id} -> Status: {res.get('status')}\n")

    report = res["report"]
    print("=== VERIFICATION REPORT SUMMARY ===")
    print(f"Total Atomic Claims Extracted : {report['total_claims']}")
    print(f"Total Unique References Cited: {report['total_references']}")
    print(f"Total Claim x Reference Pairs: {report['total_pairs']}")
    print(f"Stage Warnings / Errors      : {report['stage_errors']}\n")

    print("=== FOUR QUESTIONS TRACEABILITY WALKTHROUGH ===")
    for idx, asmt in enumerate(report["assessments"], 1):
        print(f"\n--- Finding #{idx} ---")
        print(f"1. What exactly did I write? (Claim ID: {asmt['claim_id']})")
        print(f"   Verdict: {asmt['semantic_verdict']} (Status: {asmt['assessment_status']})")
        print(f"2. Which paper did I cite for it?")
        print(f"   Cited Reference ID: {asmt['ref_id']}")
        print(f"3. What exactly does that paper say, and where?")
        print(f"   Evidence Passage IDs: {asmt['evidence_passage_ids']}")
        print(f"4. Why did the backend assign this verdict?")
        print(f"   Explanation: {asmt['explanation']}")
        print(f"   Issue Flags: {asmt['issue_flags']}")

    print("\n=== Walkthrough completed successfully! ===")


if __name__ == "__main__":
    asyncio.run(main())
