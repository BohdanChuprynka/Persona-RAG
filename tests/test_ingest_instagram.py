from pathlib import Path

from persona_rag.ingest.instagram_parser import parse_instagram_export


def test_parse_messages():
    msgs = list(parse_instagram_export(Path("tests/fixtures/ig_export_small.json")))
    assert len(msgs) == 2
    assert msgs[0].sender_name == "Friend B"
    assert msgs[0].text == "sup"
    # mojibake decode: "Ã©" (latin-1 bytes C3 A9) → "é"
    assert "é" in msgs[1].text
    for m in msgs:
        assert m.channel == "instagram"
        assert not m.is_group
