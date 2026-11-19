# CiteVerify Backend — Scientific Claim & Citation Verification System

The CiteVerify backend accepts a scientific manuscript, extracts its atomic cited claims, resolves the cited bibliography entries to canonical scholarly records, retrieves verbatim evidence passages from the cited source papers, and produces a traceable verification report.

A researcher can upload a manuscript and trace four questions for every finding:
1. **What exactly did I write?** (Atomic claim span and normalized text)
2. **Which paper did I cite for it?** (Resolved reference identity and DOI)
3. **What exactly does that paper say, and where?** (Verbatim passage, section, and page provenance)
4. **Why did the backend assign this verdict?** (Transparent reasoning, issue flags, and semantic verdict)

---

## 1. Installation and Environment Setup

### Prerequisites
- Python 3.11+
- Virtual environment recommended

### Installation

```bash
cd backend
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # On Linux/macOS: cp .env.example .env
```

### Environment Variables

Configure `.env` as needed:

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | *(empty)* | API key for Gemini models |
| `GEMINI_MODEL` | `gemini-2.0-flash-lite` | Gemini model for claim extraction and classification |
| `CROSSREF_MAILTO` | *(empty)* | Contact email for Crossref polite pool |
| `OPENALEX_EMAIL` | *(empty)* | Contact email for OpenAlex (falls back to CROSSREF_MAILTO) |
| `SEMANTIC_SCHOLAR_API_KEY` | *(empty)* | Optional API key for Semantic Scholar higher rate limits |
| `UNPAYWALL_EMAIL` | *(empty)* | Contact email for Unpaywall open-access lookup |
| `DB_PATH` | `backend/verycite.db` | Path to persistent SQLite database |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local sentence transformer model for passage retrieval |
| `MAX_FILE_SIZE_MB` | `50` | Maximum uploaded manuscript size (MB) |
| `MAX_DOWNLOAD_SIZE_MB` | `20` | Maximum downloaded source paper size (MB) |
| `PROVIDER_TIMEOUT` | `30` | Timeout in seconds for external scholarly API requests |
| `MAX_RETRIES` | `3` | Maximum retry attempts for transient API errors |

---

## 2. Server Startup

Start the FastAPI application using Uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Interactive documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 3. Five-Stage Pipeline Architecture

| Stage | Package | Purpose |
|---|---|---|
| **Stage 1: Ingest** | `app.ingestion` | Parses PDF and DOCX via Docling; parses LaTeX preserving `\cite{}` keys and bibliography warnings; parses plain text / Markdown retaining character offsets. |
| **Stage 2: Extract** | `app.extraction` | Parses bibliography entries; detects numeric ranges `[2–5]`, author–year, narrative citations; extracts atomic claims via Gemini structured output; preserves citation groups. |
| **Stage 3: Resolve** | `app.resolution` | Resolves unique references against Crossref (DOI lookup & scoring), OpenAlex (open access, retractions), and Semantic Scholar fallback. Reports `metadata_mismatch` if DOI resolves to wrong paper. |
| **Stage 4: Retrieve** | `app.retrieval` | Downloads accessible open-access PDF or abstract; validates non-login payload; extracts passages; embeds locally with Sentence Transformers; lexical boosting for numbers/metrics. |
| **Stage 5: Verify** | `app.verification` | Classifies evidence per claim–source pair with Gemini; enforces numerical consistency, population scoping, retraction warnings, and rejects invented passage IDs. |
| **Reporting** | `app.reporting` | Aggregates assessments, counts unique claims and pairs, flags issues, and outputs a complete traceable `VerificationReport`. |

---

## 4. API Endpoints

### Manuscript Verification
- `POST /verifications`: Upload a manuscript (`.pdf`, `.docx`, `.tex`, `.txt`, `.md`). Returns HTTP 202 with `verification_id`.
- `GET /verifications/{verification_id}`: Poll status while queued/running; returns full report when completed (with zero external provider calls on re-read); returns 404 for unknown job.

### Academic Discovery & Health
- `GET /health`: Liveness check (`{"status": "ok"}`).
- `GET /api/health`: Discovery API search provider status.
- `GET /api/evidence/search`: Search scholarly providers.
- `GET /api/evidence/open-access`: Unpaywall open-access lookup.

---

## 5. Working Upload & Report Example

### Submit a manuscript:

```bash
curl -X POST "http://localhost:8000/verifications" \
     -F "file=@sample_manuscript.txt"
```

Response (`HTTP 202 Accepted`):
```json
{
  "verification_id": "job_5943341ec90e4786",
  "job_id": "job_5943341ec90e4786",
  "status": "queued",
  "message": "Manuscript accepted for verification."
}
```

### Check progress or retrieve report:

```bash
curl "http://localhost:8000/verifications/job_5943341ec90e4786"
```

Response when complete:
```json
{
  "verification_id": "job_5943341ec90e4786",
  "job_id": "job_5943341ec90e4786",
  "status": "completed",
  "report": {
    "job_id": "job_5943341ec90e4786",
    "document_id": "3e319f2f-6ff8-44d2-b062-28a5b3cf951a",
    "total_claims": 3,
    "total_references": 2,
    "total_pairs": 3,
    "assessments": [
      {
        "assessment_id": "asmt_02ede8051cd9",
        "claim_id": "3e319f2f-6ff8-44d2-b062-28a5b3cf951a_c41dc7f0a",
        "ref_id": "3e319f2f-6ff8-44d2-b062-28a5b3cf951a_r0000",
        "assessment_status": "assessed",
        "semantic_verdict": "SUPPORTED",
        "evidence_passage_ids": [
          "pass_src_3e319f2f-6ff8-44d2-b062-28a5b3cf951a_r0000_1"
        ],
        "explanation": "Source Section 3 confirms the finding with intact metrics: quantization increased classification accuracy from 50% to 60%.",
        "issue_flags": [],
        "human_review_status": "pending",
        "confidence_score": null,
        "confidence_basis": null
      }
    ],
    "issues": [],
    "stage_errors": []
  }
}
```

---

## 6. Running Tests

Run the full deterministic unit and acceptance test suites (all provider calls faked for offline reproducibility):

```bash
# Run acceptance test suite (20 tests covering all Section 9 requirements):
python -m unittest tests/test_pipeline.py -v

# Run legacy discovery API tests:
python -m unittest test_api.py -v

# Run interactive walkthrough script:
python tests/verify_live_flow.py
```
