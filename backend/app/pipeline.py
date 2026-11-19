"""CiteVerify pipeline orchestrator.

Runs all 5 verification stages sequentially for a single job and persists
results at each stage so the API can surface progress in real time.
"""
from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from datetime import datetime, timezone

import httpx

from app.models import (
    VerificationJob,
    JobState,
    DocumentBlock,
    Reference,
    Claim,
    ClaimReferenceLink,
    ResolvedSource,
    Passage,
    Assessment,
    VerificationReport,
)
from app.database import (
    init_db,
    save_job,
    get_job,
    save_blocks,
    get_blocks,
    save_references,
    get_references,
    save_claims,
    get_claims,
    save_links,
    get_links,
    save_resolved_source,
    get_resolved_source,
    save_passages,
    get_passages,
    save_assessment,
    get_all_assessments,
)
from app.ingestion import ingest_document
from app.extraction import parse_references, extract_claims_and_links
from app.resolution import resolve_all_references
from app.retrieval import retrieve_source
from app.verification import assess_claim_reference_pair
from app.reporting import build_report
from app.config import PROVIDER_TIMEOUT


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def create_job(filename: str | None, content_type: str | None) -> VerificationJob:
    """Create and persist a new VerificationJob in the *queued* state.

    Args:
        filename:     Original upload filename (may be None).
        content_type: MIME type of the uploaded document (may be None).

    Returns:
        The newly-created VerificationJob (already saved to the database).
    """
    job = VerificationJob(
        job_id=f"job_{uuid.uuid4().hex[:16]}",
        input_filename=filename,
        input_format=content_type,
        job_state=JobState.queued,
        current_stage=None,
        report=None,
        error_message=None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    save_job(job)
    return job


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


async def run_pipeline(
    job_id: str,
    content: bytes,
    filename: str,
    content_type: str,
) -> None:
    """Run the full 5-stage CiteVerify pipeline for *job_id*.

    This coroutine is designed to be launched via ``asyncio.create_task`` and
    runs entirely in the background.  Job state and all intermediate artefacts
    are written to the database after every stage so callers can poll progress
    via the API.

    Stages:
        1. Ingest — parse the document into DocumentBlocks.
        2. Extract — parse bibliography references and extract claim–reference links.
        3. Resolve — resolve each reference to a canonical identity.
        4. Retrieve — fetch passages relevant to each claim from every source.
        5. Assess — produce a semantic verdict for every (claim, reference) pair.

    A VerificationReport is built from all stage outputs and persisted with the
    completed job record.  Any non-fatal errors accumulated during processing
    are included in the report's *stage_errors* list.

    Args:
        job_id:       Identifier of the VerificationJob to run (must exist in DB).
        content:      Raw bytes of the uploaded document.
        filename:     Original filename, used for format detection.
        content_type: MIME type, used as a secondary format hint.
    """
    stage_errors: list[str] = []

    # ------------------------------------------------------------------
    # Helper: persist job state changes
    # ------------------------------------------------------------------
    def update_job(stage: int | None, state: JobState, error: str | None = None) -> None:
        job = get_job(job_id)
        if job:
            job.current_stage = stage
            job.job_state = state
            job.error_message = error
            job.updated_at = datetime.now(timezone.utc)
            save_job(job)

    try:
        # ------------------------------------------------------------------
        # STAGE 1: Ingest
        # ------------------------------------------------------------------
        update_job(1, JobState.running)
        try:
            document_id, blocks, bib_warning = await ingest_document(
                content, filename, content_type
            )
            if bib_warning:
                stage_errors.append(f"Stage 1: {bib_warning}")
            save_blocks(blocks)
        except ValueError as e:
            update_job(1, JobState.failed, str(e))
            return

        # ------------------------------------------------------------------
        # STAGE 2: Extract claims and references
        # ------------------------------------------------------------------
        update_job(2, JobState.running)
        refs: list[Reference] = []
        claims: list[Claim] = []
        links: list[ClaimReferenceLink] = []
        try:
            refs = parse_references(blocks)
            save_references(refs)
            claims, links, issues = await extract_claims_and_links(blocks, refs)
            stage_errors.extend([f"Stage 2: {i}" for i in issues])
            save_claims(claims)
            save_links(links)
        except Exception as e:
            stage_errors.append(f"Stage 2 error: {e}")

        # ------------------------------------------------------------------
        # STAGE 3: Resolve references
        # ------------------------------------------------------------------
        update_job(3, JobState.running)
        resolved_sources: list[ResolvedSource] = []
        try:
            resolved_sources = await resolve_all_references(refs)
            for src in resolved_sources:
                save_resolved_source(src)
        except Exception as e:
            stage_errors.append(f"Stage 3 error: {e}")

        # ------------------------------------------------------------------
        # STAGE 4: Retrieve passages per unique source
        # ------------------------------------------------------------------
        update_job(4, JobState.running)
        all_passages: dict[str, list[Passage]] = {}  # source_id → passages
        updated_sources: dict[str, ResolvedSource] = {}
        try:
            # Group claim texts by ref_id so the retriever can use them as
            # search queries when fetching relevant passages.
            claims_by_ref: dict[str, list[str]] = defaultdict(list)
            for link in links:
                for claim in claims:
                    if claim.claim_id == link.claim_id:
                        claims_by_ref[link.ref_id].append(claim.normalized_text)

            async with httpx.AsyncClient(
                timeout=PROVIDER_TIMEOUT,
                follow_redirects=True,
                max_redirects=3,
            ) as client:

                async def retrieve_one(src: ResolvedSource) -> None:
                    claim_texts = claims_by_ref.get(src.ref_id, [])
                    try:
                        updated_src, passages = await retrieve_source(
                            src, claim_texts, client
                        )
                        save_resolved_source(updated_src)
                        save_passages(passages)
                        all_passages[src.source_id] = passages
                        updated_sources[src.ref_id] = updated_src
                    except Exception as exc:
                        stage_errors.append(
                            f"Stage 4 error for ref {src.ref_id}: {exc}"
                        )
                        all_passages[src.source_id] = []
                        updated_sources[src.ref_id] = src

                await asyncio.gather(*[retrieve_one(src) for src in resolved_sources])
        except Exception as e:
            stage_errors.append(f"Stage 4 error: {e}")

        # ------------------------------------------------------------------
        # STAGE 5: Assess each claim–reference pair
        # ------------------------------------------------------------------
        update_job(5, JobState.running)
        all_assessments: list[Assessment] = []
        final_sources = [updated_sources.get(src.ref_id, src) for src in resolved_sources]
        try:
            source_by_ref: dict[str, ResolvedSource] = {
                src.ref_id: src for src in final_sources
            }

            for link in links:
                claim = next(
                    (c for c in claims if c.claim_id == link.claim_id), None
                )
                source = source_by_ref.get(link.ref_id)
                if claim is None or source is None:
                    continue
                passages = all_passages.get(source.source_id, [])
                try:
                    assessment = await assess_claim_reference_pair(
                        claim, source, passages, source.is_retracted
                    )
                    save_assessment(assessment)
                    all_assessments.append(assessment)
                except Exception as e:
                    stage_errors.append(
                        f"Stage 5 error for claim {link.claim_id} \u00d7 ref {link.ref_id}: {e}"
                    )
        except Exception as e:
            stage_errors.append(f"Stage 5 error: {e}")

        # ------------------------------------------------------------------
        # Build final report
        # ------------------------------------------------------------------
        report = build_report(
            job_id=job_id,
            document_id=document_id if "document_id" in dir() else None,
            claims=claims,
            refs=refs,
            links=links,
            sources=final_sources,
            assessments=all_assessments,
            stage_errors=stage_errors,
        )

        # Persist completed job
        job = get_job(job_id)
        if job:
            job.job_state = JobState.completed
            job.current_stage = None
            job.report = report.model_dump(mode="json")
            job.updated_at = datetime.now(timezone.utc)
            save_job(job)

    except Exception as e:
        update_job(None, JobState.failed, f"Pipeline error: {e}")
