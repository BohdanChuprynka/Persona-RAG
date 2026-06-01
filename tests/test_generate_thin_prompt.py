# ruff: noqa: RUF001
# Reason: Cyrillic persona system string + test messages.
"""Backend-aware serving prompt.

The fine-tuned LoRA is trained on the THIN shape
``[system: THIN_SYSTEM][user: joined incoming context][assistant: reply]`` with
``train_on_responses_only``. So when ``GENERATION_BACKEND == "ollama"`` the
serving prompt MUST match that shape exactly — not the 1600-token English
SYSTEM_TEMPLATE + retrieved few-shot the gpt-4o-mini path uses. Feeding the LoRA
the heavy prompt drags the small model back to its generic instruct register and
undoes the fine-tune (the audit's dominant finding, TS1/Q1/D1-trainserve).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from persona_rag.config import get_settings
from persona_rag.generate.persona import THIN_SYSTEM
from persona_rag.generate.prompt import build_messages, build_thin_messages
from persona_rag.models import ChatMessage, PersonaTurn, RetrievedTurn, StyleAnchors

_ANCHORS = StyleAnchors(
    avg_len_chars=20,
    median_len_chars=18,
    emoji_rate_per_char=0.01,
    lang_distribution={"uk": 1.0},
    top_bigrams=["ну да"],
    n_turns=10,
    primary_language="uk",
)


def _r(reply: str, ctx: str) -> RetrievedTurn:
    return RetrievedTurn(
        turn=PersonaTurn(
            id=reply,
            your_reply=reply,
            incoming_context=[ctx],
            channel="telegram",
            chat_id_hash="x",
            recipient_id_hash="y",
            timestamp=datetime.now(UTC),
            language="uk",
            your_reply_len_chars=len(reply),
            your_reply_emoji_count=0,
        ),
        score=1.0,
    )


class TestBuildThinMessages:
    def test_minimal_is_system_then_user(self) -> None:
        msgs = build_thin_messages(incoming="шо там", session=[])
        assert msgs == [
            {"role": "system", "content": THIN_SYSTEM},
            {"role": "user", "content": "шо там"},
        ]

    def test_session_and_incoming_join_into_one_user_turn(self) -> None:
        session = [
            ChatMessage(role="user", content="ти де"),
            ChatMessage(role="assistant", content="вже виходжу"),
        ]
        msgs = build_thin_messages(incoming="ну давай швидше", session=session)
        # exactly two turns: the trained shape has a SINGLE user turn, never
        # interleaved assistant turns (those break the response-only mask).
        assert [m["role"] for m in msgs] == ["system", "user"]
        assert msgs[1]["content"] == "ти де\nвже виходжу\nну давай швидше"

    def test_blank_context_lines_dropped(self) -> None:
        session = [ChatMessage(role="user", content="  "), ChatMessage(role="user", content="йо")]
        msgs = build_thin_messages(incoming="шо", session=session)
        assert msgs[1]["content"] == "йо\nшо"

    def test_context_tail_truncated(self) -> None:
        session = [ChatMessage(role="user", content="x" * 5000)]
        msgs = build_thin_messages(incoming="y", session=session, max_ctx_chars=100)
        assert len(msgs[1]["content"]) == 100
        # tail-truncation keeps the most recent chars (the incoming end)
        assert msgs[1]["content"].endswith("y")

    def test_facts_appended_to_system_turn(self) -> None:
        msgs = build_thin_messages(
            incoming="ти де вчишся", session=[], facts="ходиш в Lincoln High"
        )
        assert msgs[0]["role"] == "system"
        assert THIN_SYSTEM in msgs[0]["content"]
        assert "Lincoln High" in msgs[0]["content"]
        # facts never create extra turns
        assert [m["role"] for m in msgs] == ["system", "user"]

    def test_never_emits_assistant_or_fewshot_turns(self) -> None:
        session = [ChatMessage(role="assistant", content="попередня відповідь")]
        msgs = build_thin_messages(incoming="ок", session=session)
        assert all(m["role"] != "assistant" for m in msgs)


class TestBuildMessagesBackendBranch:
    def test_ollama_backend_returns_thin_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GENERATION_BACKEND", "ollama")
        get_settings.cache_clear()
        try:
            msgs = build_messages(
                persona_name="Bohdan",
                persona_description="t",
                style_anchors=_ANCHORS,
                user_memory="",
                retrieved=[_r("ок", "привіт"), _r("ага", "як ти")],
                session=[ChatMessage(role="user", content="привіт")],
                incoming="шо там",
            )
        finally:
            get_settings.cache_clear()
        # thin: system==THIN_SYSTEM, no heavy template, no retrieved few-shot
        assert [m["role"] for m in msgs] == ["system", "user"]
        assert msgs[0]["content"].startswith(THIN_SYSTEM)
        assert "You are" not in msgs[0]["content"]
        assert "comeback" not in msgs[0]["content"].lower()
        assert msgs[-1]["content"].endswith("шо там")

    def test_openai_backend_still_returns_heavy_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GENERATION_BACKEND", "openai")
        get_settings.cache_clear()
        try:
            msgs = build_messages(
                persona_name="Bohdan",
                persona_description="t",
                style_anchors=_ANCHORS,
                user_memory="",
                retrieved=[_r("ок", "привіт"), _r("ага", "як ти")],
                session=[],
                incoming="шо там",
            )
        finally:
            get_settings.cache_clear()
        # heavy: the big English template + retrieved few-shot assistant turns
        assert "You are Bohdan" in msgs[0]["content"]
        assert any(m["role"] == "assistant" for m in msgs)
