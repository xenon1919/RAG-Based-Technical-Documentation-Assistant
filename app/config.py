"""
Configuration management using Pydantic Settings.

All settings are loaded from environment variables or the .env file.
Using @lru_cache ensures the Settings object is only created once
and reused across the entire application (singleton pattern).
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide settings.

    Pydantic automatically reads these from environment variables or .env.
    Variable names are case-sensitive by default.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # silently ignore unknown env vars
    )

    # ── Google Gemini LLM ────────────────────────────────────────────────────
    GOOGLE_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.0-flash"
    GEMINI_TEMPERATURE: float = 0.1
    GEMINI_MAX_OUTPUT_TOKENS: int = 2048

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    COLLECTION_NAME: str = "technical_docs"

    # ── Embedding Model ───────────────────────────────────────────────────────
    # sentence-transformers/all-MiniLM-L6-v2 produces 384-dim embeddings
    # Excellent speed/quality tradeoff, ~90MB download on first use
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"  # use "cuda" if GPU available

    # ── Document Chunking ─────────────────────────────────────────────────────
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    # ── Retrieval ─────────────────────────────────────────────────────────────
    TOP_K_RETRIEVAL: int = 5  # number of chunks retrieved per query

    # ── LangGraph Workflow ────────────────────────────────────────────────────
    MAX_RETRIES: int = 3  # max query rewrite attempts before fallback

    # ── FastAPI Server ────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: list[str] = ["*"]

    # ── File Paths ────────────────────────────────────────────────────────────
    DOCS_DIR: str = "./docs"


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings instance (created only once at startup)."""
    return Settings()
