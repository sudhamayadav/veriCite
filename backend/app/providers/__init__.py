"""Gemini LLM provider functions for CiteVerify.

Exposes two async helpers used by the extraction and verification stages.
No new packages are installed — google-genai is already in requirements.txt.
"""
from __future__ import annotations

import json
import logging

from app.config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _make_client():
    try:
        import google.genai as genai
    except ImportError:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(f"google-genai package is not installed: {exc}") from exc
    return genai.Client(api_key=GEMINI_API_KEY)



# ---------------------------------------------------------------------------
# Stage 2 – Claim extraction
# ---------------------------------------------------------------------------


async def extract_claims_gemini(
    paragraph_text: str,
    marked_span: str,
    allowed_ref_ids: list[str],
    original_span: str,
) -> list[dict]:
    """Call Gemini to extract atomic claims from *paragraph_text*.

    Returns a list of dicts, each with keys:
        claim_text, original_span, ref_ids, qualifiers

    Any item whose ref_ids contain values not in *allowed_ref_ids* is filtered
    out.  Returns [] on empty or parse-failure.  Raises RuntimeError on API
    error.
    """
    allowed_set = set(allowed_ref_ids)

    prompt = (
        "You are a scientific claim extractor. Given a paragraph from a manuscript "
        "and a marked citation span, extract all ATOMIC claims being made that are "
        "linked to the citations. Each atomic claim must be a single verifiable "
        "assertion with its qualifiers, units, comparators, population, and time point "
        "preserved. Do not merge multiple assertions into one claim.\n\n"
        "Return a JSON array. Each element: "
        '{"claim_text": str, "original_span": str, "ref_ids": [str], "qualifiers": str}\n'
        "- claim_text: normalized atomic assertion\n"
        "- original_span: exact substring of the paragraph that contains this claim\n"
        "- ref_ids: subset of the allowed_ref_ids that this claim specifically applies "
        "to (may be all of them if ambiguous)\n"
        "- qualifiers: any important qualifiers (population, time, units, negation)\n\n"
        "If no verifiable claims are found, return [].\n"
        f"Important: ref_ids must only contain values from: {allowed_ref_ids}\n\n"
        f"Paragraph:\n{paragraph_text}\n\n"
        f"Marked citation span: {marked_span}\n"
        f"Representative clause: {original_span}"
    )

    try:
        client = _make_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        raw = response.text or "[]"
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc

    try:
        items: list[dict] = json.loads(raw)
        if not isinstance(items, list):
            return []
    except json.JSONDecodeError:
        logger.warning("extract_claims_gemini: JSON parse failure; raw=%r", raw[:200])
        return []

    # Validate: drop any ref_ids outside allowed_ref_ids (keep item, prune ids)
    validated: list[dict] = []
    for item in items:
        ref_ids = item.get("ref_ids", [])
        if isinstance(ref_ids, list):
            item["ref_ids"] = [r for r in ref_ids if r in allowed_set]
        validated.append(item)

    return validated


# ---------------------------------------------------------------------------
# Stage 5 – Evidence classification
# ---------------------------------------------------------------------------


async def classify_claim_gemini(
    claim_text: str,
    passages: list[dict],
    coverage: str,
    source_title: str,
) -> dict:
    """Call Gemini to classify evidence for *claim_text* against *passages*.

    Each passage dict: {"passage_id": str, "text": str, "section": str|None,
                        "page": int|None}

    Returns a dict with keys: verdict, evidence_passage_ids, explanation,
    issue_flags.  On any error, returns a safe fallback dict.
    """
    valid_passage_ids = {p["passage_id"] for p in passages}

    # Build numbered passage block
    passage_lines: list[str] = []
    for idx, p in enumerate(passages, 1):
        section_info = f" [{p.get('section', '')}]" if p.get("section") else ""
        page_info = f" (p.{p['page']})" if p.get("page") is not None else ""
        passage_lines.append(
            f"[{idx}] passage_id={p['passage_id']}{section_info}{page_info}\n{p['text']}"
        )
    passages_block = "\n\n".join(passage_lines)

    prompt = (
        "You are a scientific evidence classifier. Given an atomic claim from a "
        "manuscript and numbered passages from the cited source paper, determine "
        "whether the source supports the claim.\n\n"
        "Verdict definitions:\n"
        "- SUPPORTED: source reports the same finding with material qualifiers intact\n"
        "- PARTLY_SUPPORTED: source establishes a narrower version (e.g. one subgroup "
        "when manuscript generalizes to all)\n"
        "- CONTRADICTORY: source gives positive contrary evidence for the same assertion, "
        "metric, population, and period\n"
        "- UNSUPPORTED: source was adequately inspected and does not substantiate the "
        "attribution\n"
        "- null verdict with status=unable_to_assess: source is unresolved, evidence "
        "unavailable, or retrieval inconclusive\n\n"
        f"Coverage: {coverage}\n"
        f"Source: {source_title}\n\n"
        "Rules:\n"
        "- Compare like quantities: check metric, unit, denominator, population, "
        "comparator, time window\n"
        "- If the manuscript gives a definite number and the source gives a different "
        "comparable number, the numerical assertion is CONTRADICTORY\n"
        "- A source that only discusses the same topic (without the specific finding) "
        "does NOT support the claim\n"
        "- Abstract-only evidence can support a claim explicitly stated in the abstract; "
        "abstain if claim depends on unavailable data\n\n"
        'Return JSON: {"verdict": "SUPPORTED"|"PARTLY_SUPPORTED"|"CONTRADICTORY"'
        '|"UNSUPPORTED"|null, "evidence_passage_ids": [str], "explanation": str, '
        '"issue_flags": [str]}\n'
        "- evidence_passage_ids: list of passage_ids you used (must be from the "
        "provided list)\n"
        "- explanation: 2-3 sentence reasoning\n"
        '- issue_flags: list of strings like "population_mismatch", '
        '"numerical_discrepancy", "unit_mismatch", "negation_present", '
        '"abstract_only_evidence"\n\n'
        f"Claim: {claim_text}\n\n"
        f"Passages:\n{passages_block}"
    )

    try:
        client = _make_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        raw = response.text or "{}"
    except Exception as exc:
        return {
            "verdict": None,
            "evidence_passage_ids": [],
            "explanation": str(exc),
            "issue_flags": ["provider_error"],
        }

    try:
        result: dict = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("classify_claim_gemini: JSON parse failure; raw=%r", raw[:200])
        return {
            "verdict": None,
            "evidence_passage_ids": [],
            "explanation": "JSON parse failure from Gemini.",
            "issue_flags": ["provider_error"],
        }

    # Validate passage_ids — only keep ones actually in provided list
    raw_ids = result.get("evidence_passage_ids", [])
    result["evidence_passage_ids"] = (
        [pid for pid in raw_ids if pid in valid_passage_ids]
        if isinstance(raw_ids, list)
        else []
    )

    # Ensure required keys exist with sane defaults
    result.setdefault("verdict", None)
    result.setdefault("explanation", "")
    result.setdefault("issue_flags", [])

    return result


__all__ = [
    "extract_claims_gemini",
    "classify_claim_gemini",
]
