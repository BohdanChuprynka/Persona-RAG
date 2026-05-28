from __future__ import annotations

import uuid
from datetime import datetime

from sqlmodel import Field, SQLModel


class Conversation(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    chat_id_hash: str
    channel: str
    started_at: datetime
    ended_at: datetime
    message_count: int


class Message(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    conversation_id: int = Field(foreign_key="conversation.id")
    sender_id_hash: str
    is_persona: bool
    text: str
    timestamp: datetime
    language: str | None = None


class PersonaTurnRow(SQLModel, table=True):
    id: str = Field(primary_key=True)
    your_reply: str
    incoming_context_json: str
    channel: str
    chat_id_hash: str
    recipient_id_hash: str
    timestamp: datetime
    language: str
    your_reply_len_chars: int
    your_reply_emoji_count: int
    eval_split: bool = False


class User(SQLModel, table=True):
    telegram_id: int = Field(primary_key=True)
    username: str | None = None
    first_name: str | None = None
    state: str
    first_seen: datetime
    last_interaction: datetime | None = None
    approved_by: int | None = None
    approved_at: datetime | None = None
    notes: str | None = None


class ContactMemory(SQLModel, table=True):
    __tablename__ = "contact_memory"

    user_id: int = Field(primary_key=True)
    summary: str
    last_interaction: datetime | None = None
    updated_at: datetime


class PendingMessage(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.telegram_id")
    text: str
    timestamp: datetime


class AuditLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime
    actor_id: int
    action: str
    target_id: int | None = None
    details: str | None = None


class AlgoSignal(SQLModel, table=True):
    __tablename__ = "algo_signal"

    id: int | None = Field(default=None, primary_key=True)
    kind: str = Field(index=True)  # entity|rhythm|language|phase|style
    subject: str = Field(index=True)
    value_json: str
    first_seen: datetime
    last_seen: datetime
    evidence_count: int
    updated_at: datetime


class InsightRow(SQLModel, table=True):
    __tablename__ = "insight_row"

    id: str = Field(primary_key=True)
    category: str = Field(index=True)  # bio|opinion|interest|behavior
    subject: str = Field(index=True)
    text: str
    confidence: float
    evidence_count: int = 1
    earliest_date: datetime
    latest_date: datetime
    trajectory: str | None = None
    source_session_ids: str  # JSON list[str], empty list for onboarding
    source: str = Field(index=True)  # chat|user_verified|onboarding
    review_status: str = Field(index=True)  # auto|pending|approved|rejected
    edited_text: str | None = None
    created_at: datetime
    updated_at: datetime


class InsightRunState(SQLModel, table=True):
    __tablename__ = "insight_run_state"

    session_id: str = Field(primary_key=True)
    last_extracted_at: datetime
    insights_count: int
    failed: bool = False
    error_message: str | None = None


class RawInsightRow(SQLModel, table=True):
    """Stage C checkpoint table — one row per LLM-extracted raw insight.

    Persisted atomically alongside InsightRunState in each Stage C iteration so
    that a crash anywhere downstream (Stage D, E, F) leaves the extracted raws
    recoverable on the next incremental run. See
    docs/superpowers/specs/2026-05-26-stage-c-checkpointing-design.md.
    """

    __tablename__ = "raw_insight"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    session_id: str = Field(index=True)
    category: str  # bio | opinion | interest | behavior
    subject: str  # raw, NOT normalized
    text: str
    confidence: float
    source_quote: str
    extracted_at: datetime


class VerificationSession(SQLModel, table=True):
    __tablename__ = "verification_session"

    user_id: int = Field(primary_key=True)
    phase: str  # idle|phase1_in_progress|phase1_done|phase2_in_progress|phase2_done
    current_insight_id: str | None = None
    current_question_id: str | None = None
    started_at: datetime
    updated_at: datetime
