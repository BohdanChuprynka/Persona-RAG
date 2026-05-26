"""Self-insights distillation pipeline.

See docs/superpowers/specs/2026-05-22-persona-insights-design.md.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path

from sqlmodel import Session, select

from persona_rag._logging import configure_logging, get_logger
from persona_rag.config import get_settings
from persona_rag.db.engine import make_engine
from persona_rag.db.models import (
    AlgoSignal,
    InsightRow,
    InsightRunState,
    PersonaTurnRow,
)
from persona_rag.index.qdrant_store import ensure_insights_collection, make_client
from persona_rag.insights.algo import run_stage_a
from persona_rag.insights.consolidator import consolidate, load_synonyms
from persona_rag.insights.extractor import extract_from_session
from persona_rag.insights.persistence import (
    persist_algo_signals,
    persist_insights,
)
from persona_rag.insights.router import route_insight
from persona_rag.insights.sessions import build_sessions, filter_high_signal

log = get_logger()


def _default_synonyms_path() -> Path:
    return Path(__file__).parent.parent / "persona_rag" / "insights" / "synonyms.yaml"


def _full_truncate() -> None:
    """Wipe insight tables for a clean rebuild."""
    with Session(make_engine()) as s:
        for tbl in (InsightRow, InsightRunState, AlgoSignal):
            for row in s.exec(select(tbl)).all():
                s.delete(row)
        s.commit()


async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    log.info("insights_run_start", mode=args.mode, max_sessions=args.max_sessions)

    if args.mode == "full":
        _full_truncate()

    # Load all persona turns (full corpus — Stage B applies the temporal cutoff)
    with Session(make_engine()) as s:
        all_turns = list(
            s.exec(select(PersonaTurnRow).where(PersonaTurnRow.eval_split == False)).all()  # noqa: E712
        )
    log.info("insights_corpus_loaded", n_turns=len(all_turns))

    # Stage A — always re-runs (cheap)
    stage_a = run_stage_a(all_turns)
    persist_algo_signals(stage_a)
    log.info("insights_stage_a_done", **{k: len(v) for k, v in stage_a.items()})
    entity_hints = [e["subject"] for e in stage_a["entity"][:10]]

    # Stage B
    sessions = build_sessions(all_turns, gap_hours=6)
    max_sessions = args.max_sessions or settings.INSIGHTS_MAX_SESSIONS
    high = filter_high_signal(
        sessions,
        history_years=settings.INSIGHTS_HISTORY_YEARS,
        min_turns=settings.INSIGHTS_MIN_SESSION_TURNS,
        min_chars=settings.INSIGHTS_MIN_SESSION_CHARS,
        max_sessions=max_sessions,
    )
    log.info("insights_stage_b_done", n_sessions=len(high))

    if args.dry_run:
        # Approximate cost: 1 LLM call per session, ~1K tokens in + 200 out at gpt-4o
        est_input = len(high) * 1000
        est_output = len(high) * 200
        est_usd = est_input / 1_000_000 * 2.50 + est_output / 1_000_000 * 10.0
        log.info("insights_dry_run_estimate", est_usd=round(est_usd, 2))
        return 0

    if args.mode == "reembed":
        return await _reembed_only()

    if args.mode == "reconsolidate":
        return await _reconsolidate_only(settings)

    # Stage C — extract (skip already-done sessions in incremental mode)
    skip_ids: set[str] = set()
    if args.mode == "incremental":
        with Session(make_engine()) as s:
            for row in s.exec(select(InsightRunState).where(InsightRunState.failed == False)).all():  # noqa: E712
                skip_ids.add(row.session_id)

    to_process = [
        s for s in high if s.session_id not in skip_ids or args.force_session == s.session_id
    ]
    log.info(
        "insights_stage_c_start",
        sessions_to_process=len(to_process),
        sessions_skipped=len(high) - len(to_process),
    )
    raws: list = []
    successes = 0
    failures = 0
    run_t0 = time.monotonic()
    for session_num, session in enumerate(to_process, start=1):
        session_t0 = time.monotonic()
        try:
            extracted = await extract_from_session(
                session, persona_name=settings.PERSONA_NAME, entity_hints=entity_hints
            )
            raws.extend(extracted)
            _mark_session_extracted(session.session_id, len(extracted))
            successes += 1
            log.info(
                "insights_session_done",
                session_num=session_num,
                total_sessions=len(to_process),
                session_id=session.session_id,
                insights_extracted=len(extracted),
                raws_total=len(raws),
                successes=successes,
                failures=failures,
                session_elapsed_s=round(time.monotonic() - session_t0, 2),
                run_elapsed_s=round(time.monotonic() - run_t0, 1),
            )
        except Exception as e:
            failures += 1
            log.warning(
                "insights_session_failed",
                session_num=session_num,
                total_sessions=len(to_process),
                session_id=session.session_id,
                error=str(e)[:300],
                successes=successes,
                failures=failures,
            )
            _mark_session_failed(session.session_id, str(e))
    log.info(
        "insights_stage_c_done",
        raws_total=len(raws),
        successes=successes,
        failures=failures,
        run_elapsed_s=round(time.monotonic() - run_t0, 1),
    )

    # Stage D — consolidate
    synonyms_path = settings.INSIGHTS_SYNONYMS_PATH or _default_synonyms_path()
    synonyms = load_synonyms(synonyms_path) if synonyms_path.exists() else {}
    consolidated = await consolidate(raws, synonyms=synonyms)
    log.info("insights_stage_d_done", n_consolidated=len(consolidated))

    # Stage E — route
    now = datetime.now(UTC)
    statuses = {
        ci.id: route_insight(
            ci,
            confidence_threshold=settings.INSIGHTS_CONFIDENCE_THRESHOLD,
            min_evidence=settings.INSIGHTS_MIN_EVIDENCE,
            stale_years=settings.INSIGHTS_STALE_DEMOTE_YEARS,
            stale_min_evidence=settings.INSIGHTS_STALE_DEMOTE_MIN_EVIDENCE,
            now=now,
        )
        for ci in consolidated
    }

    # Stage F — persist
    client = make_client()
    ensure_insights_collection(client, settings.QDRANT_INSIGHTS_COLLECTION)
    await persist_insights(
        consolidated,
        statuses=statuses,
        qdrant_client=client,
        collection=settings.QDRANT_INSIGHTS_COLLECTION,
    )
    log.info(
        "insights_run_finished",
        n_active=sum(1 for v in statuses.values() if v == "auto"),
        n_pending=sum(1 for v in statuses.values() if v == "pending"),
    )
    return 0


def _mark_session_extracted(session_id: str, n_insights: int) -> None:
    with Session(make_engine()) as s:
        row = s.get(InsightRunState, session_id)
        now = datetime.now(UTC)
        if row is None:
            s.add(
                InsightRunState(
                    session_id=session_id,
                    last_extracted_at=now,
                    insights_count=n_insights,
                    failed=False,
                )
            )
        else:
            row.last_extracted_at = now
            row.insights_count = n_insights
            row.failed = False
            row.error_message = None
            s.add(row)
        s.commit()


def _mark_session_failed(session_id: str, msg: str) -> None:
    with Session(make_engine()) as s:
        row = s.get(InsightRunState, session_id)
        now = datetime.now(UTC)
        if row is None:
            s.add(
                InsightRunState(
                    session_id=session_id,
                    last_extracted_at=now,
                    insights_count=0,
                    failed=True,
                    error_message=msg[:500],
                )
            )
        else:
            row.failed = True
            row.error_message = msg[:500]
            s.add(row)
        s.commit()


async def _reembed_only() -> int:
    """Re-upsert all active SQLite insights to Qdrant. No LLM calls."""
    s = get_settings()
    client = make_client()
    ensure_insights_collection(client, s.QDRANT_INSIGHTS_COLLECTION)

    with Session(make_engine()) as sess:
        rows = list(
            sess.exec(
                select(InsightRow).where(InsightRow.review_status.in_(("auto", "approved")))  # type: ignore[attr-defined]
            ).all()
        )
    if not rows:
        log.info("insights_reembed_empty")
        return 0

    from qdrant_client.models import PointStruct

    from persona_rag.index.embedder import embed_batch

    vectors = await embed_batch([r.text for r in rows])
    points = [
        PointStruct(
            id=r.id,
            vector=vec,
            payload={
                "category": r.category,
                "subject": r.subject,
                "text": r.text,
                "confidence": r.confidence,
                "evidence_count": r.evidence_count,
                "earliest_date": r.earliest_date.isoformat(),
                "latest_date": r.latest_date.isoformat(),
                "trajectory": r.trajectory,
                "source": r.source,
                "review_status": r.review_status,
            },
        )
        for r, vec in zip(rows, vectors, strict=True)
    ]
    client.upsert(collection_name=s.QDRANT_INSIGHTS_COLLECTION, points=points)
    log.info("insights_reembed_done", n=len(points))
    return 0


async def _reconsolidate_only(settings: object) -> int:
    """Re-run Stage D on existing InsightRow data using current synonyms."""
    log.warning("insights_reconsolidate_not_yet_implemented")
    return 0


def main() -> None:
    configure_logging()
    p = argparse.ArgumentParser(description="Self-insights distillation pipeline.")
    p.add_argument(
        "--mode",
        choices=["incremental", "full", "reconsolidate", "reembed"],
        default="incremental",
    )
    p.add_argument("--force-session", default=None, help="Force re-extract a specific session_id.")
    p.add_argument("--max-sessions", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="Estimate cost only; no LLM calls.")
    args = p.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
