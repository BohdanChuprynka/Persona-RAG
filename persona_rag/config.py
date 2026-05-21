from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Persona
    PERSONA_NAME: str
    PERSONA_LANGUAGE: str = "en"
    PERSONA_DESCRIPTION: str = ""

    # Telegram
    TELEGRAM_BOT_TOKEN: str
    ADMIN_TELEGRAM_ID: int

    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "persona_turns"

    # LangSmith
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_API_KEY: str | None = None
    LANGCHAIN_PROJECT: str = "persona-rag"

    # MLflow
    MLFLOW_TRACKING_URI: str = "file:./mlruns"
    MLFLOW_EXPERIMENT: str = "persona-rag-eval"

    # Storage
    USER_DB_PATH: Path = Path("data/persona.db")
    SHADOW_LOG_PATH: Path = Path("data/shadow_log.jsonl")

    # Ingest tuning
    MESSAGE_BURST_SECONDS: int = 300
    SESSION_BREAK_HOURS: int = 6
    MIN_SESSION_TURNS: int = 4
    INCLUDE_GROUP_CHATS: bool = False
    CONTEXT_TURNS: int = 10

    # PII
    PII_PATTERNS: str = "phone,email,address,iban,credit_card"
    PII_NAMES: str = ""
    PII_REPLACE_TOKEN: str = "<REDACTED>"
    STRIP_URLS: bool = False

    # Retrieval
    TOP_K: int = 8
    RECENCY_HALF_LIFE_DAYS: int = 180
    HYBRID_DENSE_ALPHA: float = Field(default=0.7, ge=0.0, le=1.0)

    # Generation
    MAX_REPLY_TOKENS: int = 300
    TEMPERATURE: float = 0.8
    ENABLE_PROMPT_CACHING: bool = True

    # Conversation state
    CURRENT_SESSION_WINDOW: int = 10
    SESSION_TIMEOUT_MINUTES: int = 30
    # LLM-distill user memory every N completed turns (user+assistant pair).
    # Set to 0 to disable (memory stays whatever it was). 4 is a sensible
    # default — extra LLM call once every 4 messages, not every turn.
    MEMORY_UPDATE_INTERVAL_TURNS: int = 4

    # Shadow
    SHADOW_MODE: bool = False

    # Rate limits
    MAX_MESSAGES_PER_MINUTE: int = 6
    MAX_OPENAI_RPS: int = 2
    PENDING_BUFFER_SIZE: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
