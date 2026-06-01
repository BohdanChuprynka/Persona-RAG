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
    TOP_K: int = 4
    # Drop retrieved turns whose final hybrid score is below this floor.
    # Stops weak past-turn matches from feeding vocabulary into the prompt
    # that the model then parrots out of context. 0.0 disables.
    HYBRID_SCORE_FLOOR: float = 0.15
    RECENCY_HALF_LIFE_DAYS: int = 180
    HYBRID_DENSE_ALPHA: float = Field(default=0.7, ge=0.0, le=1.0)
    # MMR (Maximal Marginal Relevance) reranking — see
    # docs/superpowers/specs/2026-05-31-mmr-retrieval-design.md
    MMR_ENABLED: bool = True
    MMR_POOL_SIZE: int = 30
    MMR_LAMBDA: float = Field(default=0.6, ge=0.0, le=1.0)

    # Generation
    MAX_REPLY_TOKENS: int = 500
    TEMPERATURE: float = 0.8
    ENABLE_PROMPT_CACHING: bool = True
    # Shape-conditioning: read the typical message-count of the moment off the
    # retrieved example replies and INSTRUCT the model to match it. The model
    # ignores soft "be short" rules but obeys an enforced per-reply directive.
    SHAPE_HINT_ENABLED: bool = True
    # Register-aware generation: classify the incoming as heated / serious /
    # casual and adapt. serious (someone opening up / asking for real help)
    # drops the brevity cap and injects an engagement directive so the bot
    # stops brushing off vulnerable messages with a flippant one-liner.
    REGISTER_AWARE_ENABLED: bool = True
    # Decoding-side voice levers (research item 3). Both default-off so the live
    # bot is unchanged until measured.
    #   PAREN_LOGIT_BIAS: positive OpenAI logit bias (1..5 sane) on the ")" /
    #     "))" tokens to nudge Bohdan's paren-smiley tic. 0 = off.
    #   BEST_OF_N: sample N candidates and keep the one closest to Bohdan's
    #     style centroid (needs the authorship scorer). 1 = off. Multiplies
    #     generation token cost by N — keep small.
    PAREN_LOGIT_BIAS: int = 0
    BEST_OF_N: int = 1
    BEST_OF_N_TEMPERATURE: float = 1.0
    # Real chat behavior: split replies on \n and send each fragment as
    # its own Telegram message, with a small human-like typing delay
    # between them. Disable to keep everything in one bubble.
    REPLY_SPLIT_NEWLINES: bool = True
    REPLY_CHUNK_DELAY_BASE_MS: int = 300
    REPLY_CHUNK_DELAY_PER_CHAR_MS: int = 20
    REPLY_CHUNK_DELAY_MAX_MS: int = 1800
    # Random pause variance between messages: 0.0 = deterministic,
    # 0.5 = +/-50% (e.g. 800ms base becomes a uniform draw in 400..1200ms).
    # Caps at 0.95 internally to avoid 0ms delays.
    REPLY_CHUNK_DELAY_JITTER_PCT: float = 0.5

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

    # Insights pipeline
    INSIGHTS_ENABLED: bool = True
    INSIGHTS_EXTRACT_MODEL: str = "gpt-4o"
    INSIGHTS_CONSOLIDATE_MODEL: str = "gpt-4o-mini"
    INSIGHTS_HISTORY_YEARS: float = 2.5
    INSIGHTS_MIN_SESSION_TURNS: int = 10
    INSIGHTS_MIN_SESSION_CHARS: int = 300
    INSIGHTS_MAX_SESSIONS: int = 600
    INSIGHTS_TOP_K_SEMANTIC: int = 6
    # Minimum recency-aware score for a retrieved insight to be rendered into
    # the prompt. Drops weak matches that would otherwise leak in when K is
    # bumped. 0.0 disables.
    INSIGHTS_MIN_SCORE_FLOOR: float = 0.2
    INSIGHTS_TOP_N_STATIC: int = 5
    INSIGHTS_CONFIDENCE_THRESHOLD: float = 0.7
    INSIGHTS_MIN_EVIDENCE: int = 3
    # Stage E: require evidence from at least N distinct chat partners
    # (counts unique recipient_id_hash across source_session_ids). Stops
    # 3 misattributions from the same chat thread looking like 3 confirmations.
    INSIGHTS_MIN_DISTINCT_PARTNERS: int = 2
    # Stage C->D verification gate — see specs/2026-05-31-insights-extraction-accuracy-design.md
    INSIGHTS_VERIFY_MODEL: str = "gpt-4o-mini"
    INSIGHTS_VERIFY_CONCURRENCY: int = 10
    INSIGHTS_VERIFY_ENABLED: bool = True
    # AMBIGUOUS verifier verdicts count as N units of evidence in Stage D.
    INSIGHTS_AMBIGUOUS_EVIDENCE_WEIGHT: float = 0.5
    INSIGHTS_RECENCY_HALF_LIFE_DAYS: int = 365
    INSIGHTS_STALE_DEMOTE_YEARS: float = 2.0
    INSIGHTS_STALE_DEMOTE_MIN_EVIDENCE: int = 5
    INSIGHTS_BUDGET_HARD_CAP_USD: float = 5.0
    INSIGHTS_SYNONYMS_PATH: Path | None = None
    INSIGHTS_ONBOARDING_PATH: Path | None = None
    INSIGHTS_STATIC_PATTERNS_ENABLED: bool = True
    INSIGHTS_PROMPT_TOP_ENTITIES: int = 3
    INSIGHTS_USE_GENERATED_PERSONA_DESCRIPTION: bool = True
    QDRANT_INSIGHTS_COLLECTION: str = "self_insights"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
