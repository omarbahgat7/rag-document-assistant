"""
app/services/retrieval.py
===========================
Owns the connection to the persisted ChromaDB collection and the embedding
model. Both are expensive to load, so they're initialized once (via
`initialize()`, called from the FastAPI startup event in app/main.py) and
reused across requests through module-level singletons.
"""

import logging
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.schemas.query import SourceChunk

logger = logging.getLogger("rag-backend.retrieval")

# Module-level singletons, populated by initialize().
_embedding_model: Optional[SentenceTransformer] = None
_collection = None


def initialize() -> None:
    """
    Load the embedding model and open the persisted ChromaDB collection.
    Must be called once before any retrieval happens (e.g. from a FastAPI
    startup event). Raises RuntimeError with a clear message if the vector
    store or collection can't be found, rather than failing silently.
    """
    global _embedding_model, _collection

    store_path = Path(settings.vector_store_path).resolve()
    if not store_path.exists():
        raise RuntimeError(
            f"Vector store path '{store_path}' does not exist. "
            "Run the Phase 2 ingestion notebook first to create and populate it."
        )

    logger.info(f"Loading embedding model '{settings.embedding_model_name}'...")
    _embedding_model = SentenceTransformer(settings.embedding_model_name)

    logger.info(f"Opening persistent ChromaDB client at '{store_path}'...")
    client = chromadb.PersistentClient(
        path=str(store_path),
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    try:
        _collection = client.get_collection(settings.collection_name)
    except Exception as e:
        raise RuntimeError(
            f"Could not open collection '{settings.collection_name}' at '{store_path}'. "
            f"Make sure Phase 2 has been run and the path/name in .env match exactly. "
            f"Original error: {e}"
        )

    count = _collection.count()
    logger.info(f"Loaded collection '{settings.collection_name}' with {count} chunks.")
    if count == 0:
        logger.warning("Collection is empty — retrieval will return no results until it's populated.")


def is_ready() -> bool:
    return _embedding_model is not None and _collection is not None


def get_chunk_count() -> int:
    return _collection.count() if _collection is not None else 0


def retrieve_chunks(question: str, top_k: int) -> List[SourceChunk]:
    """Embed the question and return the top_k most similar chunks from Chroma."""
    if not is_ready():
        raise RuntimeError("Retrieval service not initialized. Call initialize() at startup.")

    query_vector = _embedding_model.encode([question]).tolist()
    results = _collection.query(query_embeddings=query_vector, n_results=top_k)

    if not results["ids"] or not results["ids"][0]:
        return []

    chunks: List[SourceChunk] = []
    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        chunks.append(
            SourceChunk(
                source=metadata.get("source", "unknown"),
                page=metadata.get("page", -1),
                chunk_index=metadata.get("chunk_index", -1),
                text=results["documents"][0][i],
                distance=results["distances"][0][i],
            )
        )
    return chunks
