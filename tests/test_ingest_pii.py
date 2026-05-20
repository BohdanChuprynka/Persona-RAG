from persona_rag.ingest.pii import redact


def test_redact_phone():
    out = redact("call me at +12163761384 please")
    assert "+12163761384" not in out
    assert "<REDACTED>" in out


def test_redact_email():
    assert "bob@example.com" not in redact("mail bob@example.com")


def test_redact_custom_names():
    out = redact("hey Oksana how are you", names=["oksana"])
    assert "Oksana" not in out
    assert "oksana" not in out.lower().replace("<redacted>", "")


def test_preserves_emojis_and_case():
    text = "OMG YES 🎉 finally!!!"
    assert redact(text) == text
