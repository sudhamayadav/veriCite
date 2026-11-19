"""FastAPI routes for CiteVerify manuscript verification."""
from __future__ import annotations

import asyncio
import os
import threading
from typing import Any
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status

from app.config import MAX_FILE_SIZE_MB
from app.database import get_job, init_db
from app.models import JobState, VerificationJob
from app.pipeline import create_job, run_pipeline

router = APIRouter(prefix="/verifications", tags=["verification"])

# Ensure tables exist at startup
init_db()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".tex"}
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/markdown",
    "text/x-tex",
    "application/octet-stream",  # often sent by generic clients
}


def _run_in_thread(coro: Any) -> None:
    """Run an async coroutine in a dedicated thread event loop."""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(coro)
    finally:
        loop.close()


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def submit_verification(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Accept a scientific manuscript, persist the input job, and trigger verification."""
    filename = file.filename or "manuscript"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{ext or file.content_type}'. "
                "Accepted formats are: PDF, DOCX, LaTeX (.tex), TXT, and Markdown (.md)."
            ),
        )

    # Read content with size check
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_MB}MB.",
        )

    init_db()
    job = create_job(filename, file.content_type)

    # Run the 5-stage pipeline asynchronously in background
    background_tasks.add_task(
        _run_in_thread,
        run_pipeline(
            job_id=job.job_id,
            content=content,
            filename=filename,
            content_type=file.content_type or "application/octet-stream",
        ),
    )

    return {
        "verification_id": job.job_id,
        "job_id": job.job_id,
        "status": job.job_state.value if hasattr(job.job_state, "value") else str(job.job_state),
        "message": "Manuscript accepted for verification.",
    }


@router.get("/{verification_id}")
async def get_verification(verification_id: str) -> dict[str, Any]:
    """Return job progress, status, or completed verification report.

    Reading a completed report makes zero external provider or model calls.
    """
    job = get_job(verification_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Verification job '{verification_id}' not found.",
        )

    state_val = job.job_state.value if hasattr(job.job_state, "value") else str(job.job_state)

    if state_val == JobState.completed.value:
        return {
            "verification_id": job.job_id,
            "job_id": job.job_id,
            "status": "completed",
            "report": job.report,
            "created_at": job.created_at.isoformat() if hasattr(job.created_at, "isoformat") else str(job.created_at),
            "updated_at": job.updated_at.isoformat() if hasattr(job.updated_at, "isoformat") else str(job.updated_at),
        }

    if state_val == JobState.failed.value:
        return {
            "verification_id": job.job_id,
            "job_id": job.job_id,
            "status": "failed",
            "error": job.error_message or "Pipeline execution failed.",
            "stage": job.current_stage,
            "created_at": job.created_at.isoformat() if hasattr(job.created_at, "isoformat") else str(job.created_at),
            "updated_at": job.updated_at.isoformat() if hasattr(job.updated_at, "isoformat") else str(job.updated_at),
        }

    return {
        "verification_id": job.job_id,
        "job_id": job.job_id,
        "status": state_val,
        "current_stage": job.current_stage,
        "message": f"Processing Stage {job.current_stage or 1} in progress.",
        "created_at": job.created_at.isoformat() if hasattr(job.created_at, "isoformat") else str(job.created_at),
        "updated_at": job.updated_at.isoformat() if hasattr(job.updated_at, "isoformat") else str(job.updated_at),
    }
