import json

from persona_rag.config import get_settings


def test_appends_jsonl(tmp_path, monkeypatch):
    log_path = tmp_path / "shadow.jsonl"
    monkeypatch.setenv("SHADOW_LOG_PATH", str(log_path))
    get_settings.cache_clear()

    from persona_rag.shadow.logger import write_shadow_entry

    write_shadow_entry(
        user_id_hash="x",
        incoming="hi",
        context=[],
        retrieved_ids=["a"],
        memory="",
        generated_reply="hey",
        params={"top_k": 8},
    )
    line = log_path.read_text().strip()
    entry = json.loads(line)
    assert entry["generated_reply"] == "hey"
    assert entry["your_actual_reply"] is None

    get_settings.cache_clear()
