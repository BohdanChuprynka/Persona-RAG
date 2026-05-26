from __future__ import annotations

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
