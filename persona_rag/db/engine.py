from __future__ import annotations

from pathlib import Path

from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from persona_rag.config import get_settings


def make_engine(path: str | Path | None = None) -> Engine:
    target = Path(path) if path else get_settings().USER_DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite:///{target}"
    engine = create_engine(url, echo=False)
    SQLModel.metadata.create_all(engine)

    # One-time rename: usermemory → contact_memory. Idempotent.
    with engine.begin() as conn:
        _sql = (
            "SELECT name FROM sqlite_master"
            " WHERE type='table' AND name IN ('usermemory', 'contact_memory')"
        )
        rows = conn.exec_driver_sql(_sql).fetchall()
        names = {r[0] for r in rows}
        if "usermemory" in names and "contact_memory" not in names:
            conn.exec_driver_sql("ALTER TABLE usermemory RENAME TO contact_memory")
        elif "usermemory" in names and "contact_memory" in names:
            # Both exist — assume a previous half-migration. Drop the empty one.
            old_count = conn.exec_driver_sql("SELECT COUNT(*) FROM usermemory").scalar() or 0
            new_count = conn.exec_driver_sql("SELECT COUNT(*) FROM contact_memory").scalar() or 0
            if old_count > 0 and new_count == 0:
                conn.exec_driver_sql("DROP TABLE contact_memory")
                conn.exec_driver_sql("ALTER TABLE usermemory RENAME TO contact_memory")
            else:
                conn.exec_driver_sql("DROP TABLE usermemory")

        # Additive migration (spec 2026-06-03): InsightRow.text_en for the
        # query-language fact card. Idempotent; no-op on fresh DBs (create_all
        # already added the column from the model).
        ins_cols = {r[1] for r in conn.exec_driver_sql("PRAGMA table_info(insight_row)").fetchall()}
        if ins_cols and "text_en" not in ins_cols:
            conn.exec_driver_sql("ALTER TABLE insight_row ADD COLUMN text_en VARCHAR")

    return engine


def session() -> Session:
    return Session(make_engine())
