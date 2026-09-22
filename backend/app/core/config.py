"""
app/core/config.py
===================
Centralized app configuration, loaded from environment variables / a `.env`
file via pydantic-settings. Every other module should read config through
`settings` (imported from here) rather than reading os.environ directly, so
there is exactly one source of truth.
"""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- App metadata ---
    app_name: str = "RAG Document Assistant API"
    app_version: str = "1.0.0"

    # --- Vector store (MUST match Phase 2 ingestion exactly) ---
    vector_store_path: str = "F:/rag-assistant-project/backend/data/vector_store"

    collection_name: str = "document_assistant"
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # --- Retrieval defaults ---
    default_top_k: int = 4
    max_top_k: int = 20

    # --- Generation (optional LLM call) ---
    anthropic_api_key: str | None = None
    generation_model: str = "claude-sonnet-4-6"
    generation_max_tokens: int = 500

    # --- CORS ---
    # Comma-separated in .env, e.g. CORS_ORIGINS=http://localhost:3000,http://localhost:5173
    cors_origins: str = "*"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> List[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings instance. Using a function (rather than a bare module-level
    Settings() call) makes it easy to override in tests via dependency
    injection / cache clearing if needed later.
    """
    return Settings()


# Convenience singleton for straightforward `from app.core.config import settings` imports.
settings = get_settings()
