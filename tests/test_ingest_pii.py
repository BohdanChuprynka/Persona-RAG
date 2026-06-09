from persona_rag.ingest.pii import redact


def test_redact_phone():
    out = redact("call me at +15555550100 please")
    assert "+15555550100" not in out
    assert "<REDACTED>" in out


def test_redact_email():
    assert "bob@example.com" not in redact("mail bob@example.com")


def test_redact_custom_names():
    out = redact("hey Alice how are you", names=["alice"])
    assert "Alice" not in out
    assert "alice" not in out.lower().replace("<redacted>", "")


def test_preserves_emojis_and_case():
    text = "OMG YES 🎉 finally!!!"
    assert redact(text) == text
