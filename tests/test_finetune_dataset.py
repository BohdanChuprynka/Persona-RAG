# ruff: noqa: RUF001
"""Tests for the ShareGPT export used by the Colab fine-tune kit."""

from __future__ import annotations

from persona_rag.finetune.dataset import to_sharegpt


def test_basic_pair_maps_to_human_gpt_turns():
    rec = to_sharegpt(["hi", "how r u"], "норм)")
    assert rec == {
        "conversations": [
            {"from": "human", "value": "hi\nhow r u"},
            {"from": "gpt", "value": "норм)"},
        ]
    }


def test_reply_newlines_are_preserved_so_model_learns_bursts():
    rec = to_sharegpt(["ти де"], "ще не\nзара буду збиратись")
    assert rec["conversations"][-1]["value"] == "ще не\nзара буду збиратись"


def test_system_turn_is_prepended_when_given():
    rec = to_sharegpt(["yo"], "хай", system="Ти Богдан.")
    assert rec["conversations"][0] == {"from": "system", "value": "Ти Богдан."}
    assert [t["from"] for t in rec["conversations"]] == ["system", "human", "gpt"]


def test_blank_context_lines_are_dropped():
    rec = to_sharegpt(["", "  ", "real line"], "ok")
    assert rec["conversations"][0]["from"] in ("system", "human")
    human = next(t for t in rec["conversations"] if t["from"] == "human")
    assert human["value"] == "real line"


def test_context_is_tail_truncated():
    long_ctx = ["x" * 5000]
    rec = to_sharegpt(long_ctx, "ok", max_ctx_chars=100)
    human = next(t for t in rec["conversations"] if t["from"] == "human")
    assert len(human["value"]) == 100
