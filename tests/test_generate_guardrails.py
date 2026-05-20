from persona_rag.generate.guardrails import apply_guardrails


def test_redacted_token_triggers_block():
    out, ok = apply_guardrails("Sure, the address is <REDACTED> on Main")
    assert not ok


def test_plain_reply_passes():
    out, ok = apply_guardrails("yeah sure")
    assert ok
    assert out == "yeah sure"


def test_empty_reply_falls_back():
    out, ok = apply_guardrails("   ")
    assert ok
    assert out == "..."


def test_overlong_truncates():
    long = "a sentence. " * 200
    out, ok = apply_guardrails(long, max_chars=200)
    assert ok
    assert len(out) <= 200
