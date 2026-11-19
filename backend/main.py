"""veriCite academic research discovery API."""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import httpx
from integrations.providers import SEARCH_PROVIDERS, search_all, unpaywall

load_dotenv()
app = FastAPI(title="veriCite Evidence API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://localhost:5173"],
                   allow_credentials=False, allow_methods=["GET"], allow_headers=["*"])

@app.get("/api/health")
def health():
    return {"status": "ok", "providers": list(SEARCH_PROVIDERS)}

@app.get("/api/evidence/search")
async def evidence_search(q: str = Query(min_length=3, max_length=500),
                          limit: int = Query(5, ge=1, le=20),
                          providers: str | None = None):
    selected = [s.strip() for s in providers.split(",") if s.strip()] if providers else None
    try:
        return await search_all(q, limit, selected)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.get("/api/evidence/open-access")
async def open_access(doi: str = Query(min_length=3, max_length=250)):
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            return await unpaywall(client, doi.removeprefix("https://doi.org/"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Unpaywall returned HTTP {exc.response.status_code}") from exc

    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail="Unpaywall is temporarily unreachable; please retry.") from exc


@app.get("/health")
def liveness() -> dict[str, str]:
    return {"status": "ok"}

from app.verification_routes import router as verification_router
app.include_router(verification_router)

