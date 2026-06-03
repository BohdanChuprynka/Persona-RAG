# ruff: noqa: RUF001
from __future__ import annotations

from persona_rag.generate.lang_detect import detect_language


def test_english():
    assert detect_language("tell me about yourself") == "en"
    assert detect_language("where do you study?") == "en"


def test_ukrainian():
    assert detect_language("розкажи про себе") == "uk"
    assert detect_language("де ти живеш?") == "uk"


def test_russian_distinctive_chars():
    assert detect_language("расскажи о себе, кто ты") == "ru"


def test_mixed_defaults_uk():
    assert detect_language("ok розкажи") == "uk"
    assert detect_language("") == "uk"
