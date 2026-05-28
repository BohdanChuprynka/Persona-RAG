import os

from persona_rag.config import Settings


def test_settings_load_from_env(monkeypatch, tmp_path):
    for key in [
        "PERSONA_NAME",
        "PERSONA_LANGUAGE",
        "PERSONA_DESCRIPTION",
        "TELEGRAM_BOT_TOKEN",
        "ADMIN_TELEGRAM_ID",
        "OPENAI_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "PERSONA_NAME=TestPerson\n"
        "PERSONA_LANGUAGE=en\n"
        "PERSONA_DESCRIPTION=Test desc\n"
        "TELEGRAM_BOT_TOKEN=test-token\n"
        "ADMIN_TELEGRAM_ID=12345\n"
        "OPENAI_API_KEY=sk-test\n"
    )
    s = Settings(_env_file=str(env))
    assert s.PERSONA_NAME == "TestPerson"
    assert s.ADMIN_TELEGRAM_ID == 12345
    assert s.TOP_K == 4  # default
    assert s.HYBRID_SCORE_FLOOR == 0.15  # default
    assert s.HYBRID_DENSE_ALPHA == 0.7  # default


def test_insights_settings_defaults(monkeypatch, tmp_path):
    # purge process env so the test relies on defaults
    for k in list(os.environ):
        if k.startswith("INSIGHTS_") or k == "QDRANT_INSIGHTS_COLLECTION":
            monkeypatch.delenv(k, raising=False)
    # need minimum required vars; reuse the existing conftest seed
    from persona_rag.config import Settings, get_settings

    get_settings.cache_clear()
    s = Settings()
    assert s.INSIGHTS_ENABLED is True
    assert s.INSIGHTS_EXTRACT_MODEL == "gpt-4o"
    assert s.INSIGHTS_CONSOLIDATE_MODEL == "gpt-4o-mini"
    assert s.INSIGHTS_HISTORY_YEARS == 2.5
    assert s.INSIGHTS_MIN_SESSION_TURNS == 10
    assert s.INSIGHTS_MIN_SESSION_CHARS == 300
    assert s.INSIGHTS_MAX_SESSIONS == 600
    assert s.INSIGHTS_TOP_K_SEMANTIC == 6
    assert s.INSIGHTS_MIN_SCORE_FLOOR == 0.2
    assert s.INSIGHTS_TOP_N_STATIC == 5
    assert s.INSIGHTS_CONFIDENCE_THRESHOLD == 0.7
    assert s.INSIGHTS_MIN_EVIDENCE == 2
    assert s.INSIGHTS_RECENCY_HALF_LIFE_DAYS == 365
    assert s.INSIGHTS_STALE_DEMOTE_YEARS == 2.0
    assert s.INSIGHTS_STALE_DEMOTE_MIN_EVIDENCE == 5
    assert s.INSIGHTS_BUDGET_HARD_CAP_USD == 5.0
    assert s.INSIGHTS_SYNONYMS_PATH is None
    assert s.INSIGHTS_ONBOARDING_PATH is None
    assert s.INSIGHTS_STATIC_PATTERNS_ENABLED is True
    assert s.INSIGHTS_PROMPT_TOP_ENTITIES == 3
    assert s.INSIGHTS_USE_GENERATED_PERSONA_DESCRIPTION is True
    assert s.QDRANT_INSIGHTS_COLLECTION == "self_insights"
