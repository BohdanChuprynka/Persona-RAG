from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_pipeline_writes_db_dryrun(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_PATH", str(tmp_path / "p.db"))
    monkeypatch.setenv("ADMIN_TELEGRAM_ID", "222")
    monkeypatch.setenv("MIN_SESSION_TURNS", "1")

    # Clear lru_cache so Settings re-reads env vars
    import persona_rag.config as cfg

    cfg.get_settings.cache_clear()

    # Redirect data/ writes to tmp_path
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()

    tg_fixture = Path(__file__).parent / "fixtures" / "tg_export_small.json"
    from persona_rag.ingest.pipeline import run_ingest

    summary = await run_ingest(telegram_path=tg_fixture, ig_root=None, dry_run_embeddings=True)
    assert summary["turns_written"] >= 1
    assert (tmp_path / "p.db").exists()

    # Restore cache so other tests aren't affected
    cfg.get_settings.cache_clear()
