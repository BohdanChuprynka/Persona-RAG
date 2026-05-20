from __future__ import annotations

from pathlib import Path

from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from persona_rag.config import get_settings


def make_engine(path: str | Path | None = None) -> Engine:
    target = path or get_settings().USER_DB_PATH
    url = f"sqlite:///{target}"
    engine = create_engine(url, echo=False)
    SQLModel.metadata.create_all(engine)
    return engine


def session() -> Session:
    return Session(make_engine())
