"""
frontend/api_client.py
========================
Thin HTTP client for the FastAPI backend.

API contract:
    POST {API_BASE_URL}/query
    body: {"question": str}
    response: {"answer": str, "sources": list}

`sources` may come back as a list of plain strings, or as a list of metadata
objects (source/page/text/distance, e.g. from a ChromaDB-backed service).
This client normalizes either shape into a consistent structure so the UI
can render rich citation cards when metadata is available, and a plain
label when it isn't — without crashing either way.
"""

import os
from typing import Any, List, TypedDict, Union

import requests
from dotenv import load_dotenv

load_dotenv()  # reads .env if present; safe no-op if it doesn't exist

# Reads API_BASE_URL from the environment / .env, with a safe fallback.
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
QUERY_ENDPOINT = f"{API_BASE_URL}/query"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

REQUEST_TIMEOUT_SECONDS = 60


class NormalizedSource(TypedDict):
    label: str                    # e.g. "policy.pdf — page 3"
    file_name: str                # bare file name, for badges/icons
    page: Union[int, str, None]
    detail: Union[str, None]      # retrieved chunk snippet, if provided
    score: Union[float, None]     # similarity score / distance, if provided
    raw: Any                      # original item, untouched


class QueryResult(TypedDict):
    question: str
    answer: str
    sources: List[NormalizedSource]


class BackendUnavailableError(Exception):
    """The backend could not be reached at all: connection refused, DNS failure, timeout."""


class BackendRequestError(Exception):
    """The backend responded, but with an error status or an unparsable/unexpected body."""


def ask_question(question: str, top_k: int = 4) -> QueryResult:
    """
    Send a question to POST /query and return the parsed, normalized answer
    + sources. The core required payload is exactly {"question": question};
    top_k is included as an additive field so the sidebar's retrieval slider
    has an effect on backends that support it — FastAPI/Pydantic backends
    ignore unrecognized extra fields by default, so this is safe even
    against a backend that only reads "question".

    Raises BackendUnavailableError (network-level failure) or
    BackendRequestError (backend reachable but responded with an error / bad
    body) — callers should catch these separately to show the right message.
    """
    payload = {"question": question, "top_k": top_k}

    try:
        response = requests.post(QUERY_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.ConnectionError as e:
        raise BackendUnavailableError(
            f"Could not connect to the backend at {API_BASE_URL}. "
            "Make sure the FastAPI server is running "
            "(uvicorn app.main:app --reload --port 8000 from the backend/ directory)."
        ) from e
    except requests.exceptions.Timeout as e:
        raise BackendUnavailableError(f"The backend at {API_BASE_URL} took too long to respond.") from e
    except requests.exceptions.RequestException as e:
        raise BackendUnavailableError(f"Unexpected network error contacting backend: {e}") from e

    if response.status_code != 200:
        raise BackendRequestError(f"Backend returned {response.status_code}: {_extract_error_detail(response)}")

    try:
        data = response.json()
    except ValueError as e:
        raise BackendRequestError(f"Backend response was not valid JSON: {e}") from e

    if not isinstance(data, dict) or "answer" not in data:
        raise BackendRequestError(f"Backend response is missing an 'answer' field: {data}")

    raw_sources = data.get("sources", []) or []
    if not isinstance(raw_sources, list):
        raw_sources = [raw_sources]

    return QueryResult(
        question=data.get("question", question),
        answer=data["answer"],
        sources=[_normalize_source(s, i) for i, s in enumerate(raw_sources)],
    )


def check_backend_health() -> bool:
    """Lightweight check used to drive the live connection-status indicator."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def _normalize_source(item: Any, index: int) -> NormalizedSource:
    """Accepts a plain string or a metadata dict; returns a common shape either way."""
    if isinstance(item, str):
        return NormalizedSource(
            label=item, file_name=item, page=None, detail=None, score=None, raw=item
        )

    if isinstance(item, dict):
        file_name = str(item.get("source") or item.get("file") or item.get("filename") or f"Source {index + 1}")
        page = item.get("page")
        label = f"{file_name} — page {page}" if page not in (None, "") else file_name
        detail = item.get("text") or item.get("chunk") or item.get("snippet")
        score = item.get("distance")
        if score is None:
            score = item.get("score")
        return NormalizedSource(
            label=label, file_name=file_name, page=page, detail=detail, score=score, raw=item
        )

    # Anything else (number, None, nested list, etc.) — stringify rather than crash.
    return NormalizedSource(
        label=str(item), file_name=str(item), page=None, detail=None, score=None, raw=item
    )


def _extract_error_detail(response: requests.Response) -> str:
    try:
        body = response.json()
        if isinstance(body, dict):
            return str(body.get("detail", body))
        return str(body)
    except ValueError:
        return response.text
