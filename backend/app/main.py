"""
app/main.py
=============
FastAPI application entrypoint.

Run from the `backend/` directory with:
    uvicorn app.main:app --reload --port 8000
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import query as query_routes
from app.core.config import settings
from app.services import retrieval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-backend")

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="RAG-powered document assistant API — retrieval over a persisted ChromaDB store, with optional LLM generation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    logger.info("Starting up — loading embedding model and vector store...")
    retrieval.initialize()
    logger.info("Startup complete.")


app.include_router(query_routes.router)


@app.get("/")
def root():
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "endpoints": ["/health", "/query"],
    }
