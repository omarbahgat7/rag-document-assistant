"""
app/api/routes/query.py
=========================
GET  /health   - readiness check
POST /query    - retrieve relevant chunks and return a grounded answer
"""

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.schemas.query import HealthResponse, QueryRequest, QueryResponse
from app.services import generation, retrieval

router = APIRouter(tags=["query"])


@router.get("/health", response_model=HealthResponse)
def health():
    if not retrieval.is_ready():
        raise HTTPException(status_code=503, detail="Retrieval service not initialized yet.")

    return HealthResponse(
        status="ok",
        collection=settings.collection_name,
        vector_store_path=settings.vector_store_path,
        chunks_loaded=retrieval.get_chunk_count(),
        embedding_model=settings.embedding_model_name,
    )


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not retrieval.is_ready():
        raise HTTPException(status_code=503, detail="Retrieval service not initialized yet.")

    chunks = retrieval.retrieve_chunks(request.question, request.top_k)
    answer = generation.generate_answer(request.question, chunks)

    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=chunks,
    )
