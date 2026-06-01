from __future__ import annotations

from datetime import UTC, datetime

from sqlmodel import Session

from persona_rag.db.engine import make_engine
from persona_rag.db.models import ContactMemory


def load_contact_memory(user_id: int) -> str:
    with Session(make_engine()) as s:
        row = s.get(ContactMemory, user_id)
        return row.summary if row else ""


def save_contact_memory(user_id: int, summary: str) -> None:
    now = datetime.now(UTC)
    with Session(make_engine()) as s:
        row = s.get(ContactMemory, user_id)
        if row is None:
            s.add(
                ContactMemory(
                    user_id=user_id,
                    summary=summary,
                    last_interaction=now,
                    updated_at=now,
                )
            )
        else:
            row.summary = summary
            row.last_interaction = now
            row.updated_at = now
            s.add(row)
        s.commit()
