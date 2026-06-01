# ruff: noqa: RUF001
"""Tests for the ShareGPT export used by the Colab fine-tune kit."""

from __future__ import annotations

from persona_rag.finetune.dataset import clean_reply, eval_split_for, to_sharegpt


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


class TestCleanReply:
    """Don't teach the LoRA to emit scrubber scars or memorized URLs."""

    def test_clean_reply_unchanged(self):
        assert clean_reply("норм) побачимось") == "норм) побачимось"

    def test_redacted_token_drops_the_row(self):
        # a <REDACTED> scar in the reply would be reproduced verbatim — drop it
        assert clean_reply("мій номер <REDACTED>") is None
        assert clean_reply("<REDACTED>") is None

    def test_bare_url_reply_dropped(self):
        assert clean_reply("https://example.com/x") is None
        assert clean_reply("www.example.com") is None

    def test_url_stripped_but_surrounding_burst_kept(self):
        # the link goes, the voice around it stays — multi-bubble preserved
        out = clean_reply("дивись\nhttps://example.com/x\nкорисна штука")
        assert out == "дивись\nкорисна штука"

    def test_url_inline_stripped_keeps_text(self):
        out = clean_reply("глянь тут t.me/somechannel ок")
        assert "t.me" not in out
        assert "глянь тут" in out and "ок" in out


class TestEvalSplitFor:
    """Recipient-stratified hold-out via a deterministic per-turn hash — fixes
    the temporal-tail split that made the latin target unreachable."""

    def test_deterministic(self):
        assert eval_split_for("turn-abc") == eval_split_for("turn-abc")

    def test_not_all_same(self):
        results = {eval_split_for(f"turn-{i}") for i in range(200)}
        assert results == {True, False}

    def test_roughly_ten_percent(self):
        held = sum(1 for i in range(10000) if eval_split_for(f"turn-{i}"))
        assert 0.07 <= held / 10000 <= 0.13
