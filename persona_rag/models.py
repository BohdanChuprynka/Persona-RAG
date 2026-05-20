from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, TypedDict

from pydantic import BaseModel

Channel = Literal["telegram", "instagram"]


class UserState(str, Enum):
    UNKNOWN = "unknown"
    PENDING = "pending"
    WHITELISTED = "whitelisted"
    BLOCKED = "blocked"


class RawMessage(BaseModel):
    channel: Channel
    chat_id: str
    sender_id: str
    sender_name: str
    text: str
    timestamp: datetime
    is_group: bool


class PersonaTurn(BaseModel):
    id: str
    your_reply: str
    incoming_context: list[str]
    channel: Channel
    chat_id_hash: str
    recipient_id_hash: str
    timestamp: datetime
    language: str
    your_reply_len_chars: int
    your_reply_emoji_count: int
    eval_split: bool = False


class StyleAnchors(BaseModel):
    avg_len_chars: float
    median_len_chars: float
    emoji_rate_per_char: float
    lang_distribution: dict[str, float]
    top_bigrams: list[str]
    n_turns: int
    primary_language: str


class RetrievedTurn(BaseModel):
    turn: PersonaTurn
    score: float
    score_dense: float = 0.0
    score_bm25: float = 0.0


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class GraphState(TypedDict, total=False):
    incoming: str
    user_id: int
    chat_id: int
    auth_state: str
    retrieved: list[RetrievedTurn]
    memory: str
    session: list[ChatMessage]
    prompt: list[dict[str, str]]
    reply: str
    shadow: bool
    session_id: str
