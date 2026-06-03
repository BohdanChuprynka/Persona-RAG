# ruff: noqa: RUF001
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from persona_rag.graph.nodes.retrieve_insights import retrieve_insights


@pytest.mark.asyncio
async def test_retrieve_insights_no_collection_returns_empty(monkeypatch):
    monkeypatch.setattr(
        "persona_rag.graph.nodes.retrieve_insights.make_client",
        lambda: (_ for _ in ()).throw(RuntimeError("no qdrant")),
    )
    state = {"user_id": 1, "chat_id": 1, "incoming": "hi"}
    out = await retrieve_insights(state)
    assert out["insights"]["semantic"] == []
    assert out["insights"]["static"] == {}


@pytest.mark.asyncio
async def test_retrieve_insights_pulls_pool_then_reranks(monkeypatch):
    fake_client = MagicMock()
    fake_resp = MagicMock()
    fake_resp.points = []
    fake_client.query_points = MagicMock(return_value=fake_resp)
    monkeypatch.setattr(
        "persona_rag.graph.nodes.retrieve_insights.make_client",
        lambda: fake_client,
    )
    monkeypatch.setattr(
        "persona_rag.graph.nodes.retrieve_insights.anchor_vecs",
        AsyncMock(return_value=[[0.0] * 1536]),
    )
    with patch(
        "persona_rag.graph.nodes.retrieve_insights.embed_batch",
        AsyncMock(return_value=[[0.0] * 1536]),
    ):
        state = {"user_id": 1, "chat_id": 1, "incoming": "hi"}
        out = await retrieve_insights(state)
    assert out["insights"]["semantic"] == []

    # Verify the Qdrant filter excludes rejected/pending insights
    call_kwargs = fake_client.query_points.call_args.kwargs
    assert "query_filter" in call_kwargs
    qfilter = call_kwargs["query_filter"]
    # Must contain a review_status condition matching auto/approved
    from qdrant_client.models import Filter as QFilter

    assert isinstance(qfilter, QFilter)
    assert qfilter.must is not None
    cond = qfilter.must[0]
    assert cond.key == "review_status"
    assert set(cond.match.any) == {"auto", "approved"}


@pytest.mark.asyncio
async def test_retrieve_insights_drops_below_score_floor(monkeypatch):
    """Insights with final_score below INSIGHTS_MIN_SCORE_FLOOR are filtered out."""
    from datetime import UTC, datetime

    from persona_rag.insights.recency import RankedInsight

    now = datetime.now(UTC)
    high = RankedInsight(
        id="aaa",
        category="bio",
        subject="school",
        text="goes to school",
        confidence=1.0,
        evidence_count=5,
        earliest_date=now,
        latest_date=now,
        trajectory=None,
        source="chat",
        semantic_score=0.5,
        final_score=0.5,
    )
    weak = RankedInsight(
        id="bbb",
        category="opinion",
        subject="snow",
        text="dislikes snow",
        confidence=0.5,
        evidence_count=1,
        earliest_date=now,
        latest_date=now,
        trajectory=None,
        source="chat",
        semantic_score=0.1,
        final_score=0.1,
    )

    fake_client = MagicMock()
    fake_resp = MagicMock()
    fake_resp.points = ["high_point", "weak_point"]  # opaque — from_qdrant_point is mocked
    fake_client.query_points = MagicMock(return_value=fake_resp)

    monkeypatch.setattr(
        "persona_rag.graph.nodes.retrieve_insights.make_client",
        lambda: fake_client,
    )
    monkeypatch.setattr(
        "persona_rag.graph.nodes.retrieve_insights.from_qdrant_point",
        lambda p: high if p == "high_point" else weak,
    )
    # Bypass the recency decay so final_score stays = semantic_score as seeded above.
    monkeypatch.setattr(
        "persona_rag.graph.nodes.retrieve_insights.rerank_with_recency",
        lambda items, half_life_days: sorted(items, key=lambda x: x.final_score, reverse=True),
    )
    monkeypatch.setattr(
        "persona_rag.graph.nodes.retrieve_insights.anchor_vecs",
        AsyncMock(return_value=[[0.0] * 1536]),
    )

    with patch(
        "persona_rag.graph.nodes.retrieve_insights.embed_batch",
        AsyncMock(return_value=[[0.0] * 1536]),
    ):
        out = await retrieve_insights({"user_id": 1, "chat_id": 1, "incoming": "hi"})

    subjects = [r.subject for r in out["insights"]["semantic"]]
    assert "school" in subjects  # final_score=0.5 >= floor 0.2 -> keep
    assert "snow" not in subjects  # final_score=0.1 < floor 0.2 -> drop


@pytest.mark.asyncio
async def test_self_desc_query_sets_core_lane(monkeypatch):
    """A self-description query routes to the CORE lane (by route, not similarity)."""
    from datetime import UTC, datetime

    from persona_rag.insights.recency import RankedInsight

    fake_client = MagicMock()
    fake_resp = MagicMock()
    fake_resp.points = []
    fake_client.query_points = MagicMock(return_value=fake_resp)
    monkeypatch.setattr(
        "persona_rag.graph.nodes.retrieve_insights.make_client", lambda: fake_client
    )
    now = datetime.now(UTC)
    core = [
        RankedInsight(
            id="b",
            text="navch",
            text_en="studies",
            category="bio",
            subject="school",
            confidence=1.0,
            evidence_count=1,
            earliest_date=now,
            latest_date=now,
            trajectory=None,
            source="vault",
            semantic_score=1.0,
            final_score=1.0,
        )
    ]
    monkeypatch.setattr(
        "persona_rag.graph.nodes.retrieve_insights.classify_self_description",
        lambda vec, anchors, threshold: True,
    )
    monkeypatch.setattr(
        "persona_rag.graph.nodes.retrieve_insights.load_core_facts",
        lambda *, limit, query_lang: core,
    )
    with (
        patch(
            "persona_rag.graph.nodes.retrieve_insights.embed_batch",
            AsyncMock(return_value=[[0.0] * 1536]),
        ),
        patch(
            "persona_rag.graph.nodes.retrieve_insights.anchor_vecs",
            AsyncMock(return_value=[[0.0] * 1536]),
        ),
    ):
        out = await retrieve_insights({"user_id": 1, "chat_id": 1, "incoming": "розкажи про себе"})
    ins = out["insights"]
    assert ins["lane"] == "self_desc"
    assert ins["query_lang"] == "uk"
    assert [c.subject for c in ins["core"]] == ["school"]
