"""
app/schemas/query.py
=====================
Request/response models for the /query (and /health) endpoints.
"""

from typing import List

from pydantic import BaseModel, Field

from app.core.config import settings


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's natural-language question.")
    top_k: int = Field(
        default=settings.default_top_k,
        ge=1,
        le=settings.max_top_k,
        description="Number of chunks to retrieve from the vector store.",
    )


class SourceChunk(BaseModel):
    source: str = Field(..., description="Originating file name.")
    page: int = Field(..., description="Page number within the source file.")
    chunk_index: int = Field(..., description="Index of this chunk within its page.")
    text: str = Field(..., description="The chunk's raw text.")
    distance: float = Field(..., description="Similarity distance from the query (lower = more similar).")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceChunk]


class HealthResponse(BaseModel):
    status: str
    collection: str
    vector_store_path: str
    chunks_loaded: int
    embedding_model: str
