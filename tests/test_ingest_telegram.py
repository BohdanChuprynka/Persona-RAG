from pathlib import Path

from persona_rag.ingest.telegram_parser import parse_telegram_export


def test_parse_personal_chat_messages():
    fixture = Path("tests/fixtures/tg_export_small.json")
    msgs = list(parse_telegram_export(fixture))
    texts = [m.text for m in msgs]
    assert "hey" in texts
    assert "yo" in texts
    assert "g" not in texts  # group filtered (INCLUDE_GROUP_CHATS=False default)
    for m in msgs:
        assert m.channel == "telegram"
        assert not m.is_group
