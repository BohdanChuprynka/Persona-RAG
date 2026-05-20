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
    assert s.TOP_K == 8  # default
    assert s.HYBRID_DENSE_ALPHA == 0.7  # default
