import json

import structlog

from persona_rag._logging import configure_logging, get_logger


def test_log_emits_json(capsys):
    configure_logging()
    log = get_logger()
    log.info("test_event", foo="bar")
    out = capsys.readouterr().out.strip()
    parsed = json.loads(out)
    assert parsed["event"] == "test_event"
    assert parsed["foo"] == "bar"
    assert parsed["level"] == "info"


def test_contextvars_bind(capsys):
    configure_logging()
    log = get_logger()
    with structlog.contextvars.bound_contextvars(user_id=42):
        log.info("bound_event")
    parsed = json.loads(capsys.readouterr().out.strip())
    assert parsed["user_id"] == 42
