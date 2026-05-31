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
    RawInsightRow,
)
from persona_rag.index.qdrant_store import ensure_insights_collection, make_client
from persona_rag.insights.algo import run_stage_a
from persona_rag.insights.consolidator import consolidate, load_synonyms
from persona_rag.insights.extractor import RawInsight, extract_from_session
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
    """Wipe insight tables for a clean rebuild.

    Includes raw_insight so that the Stage C checkpoint table starts empty
    too — otherwise a prior partial run's raws would leak into Stage D.
    """
    with Session(make_engine()) as s:
        for tbl in (InsightRow, InsightRunState, AlgoSignal, RawInsightRow):
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

    # Stage A — always re-runs (cheap). Build whitelist from synonyms so
    # high-frequency user-blessed tokens (e.g. "мама" / "школа") never get
    # filtered by the narrow function-word blocklist added in spec §5.1.b.
    synonyms_path = settings.INSIGHTS_SYNONYMS_PATH or _default_synonyms_path()
    synonyms_for_whitelist = load_synonyms(synonyms_path) if synonyms_path.exists() else {}
    entity_whitelist: set[str] = set()
    for canonical, variants in synonyms_for_whitelist.items():
        entity_whitelist.add(canonical.lower())
        for v in variants:
            entity_whitelist.add(v.lower())
    stage_a = run_stage_a(all_turns, entity_whitelist=entity_whitelist)
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

    # Stage C — extract. In incremental mode, hydrate raws from DB for sessions
    # that completed on a prior run so a downstream crash never costs us those
    # extractions a second time.
    skip_ids, resumed_raws = _load_resume_state(mode=args.mode)
    log.info(
        "insights_stage_c_resumed",
        sessions_resumed=len(skip_ids),
        raws_loaded=len(resumed_raws),
    )

    to_process = [
        s for s in high if s.session_id not in skip_ids or args.force_session == s.session_id
    ]
    log.info(
        "insights_stage_c_start",
        sessions_to_process=len(to_process),
        sessions_skipped=len(high) - len(to_process),
    )
    raws: list[RawInsight] = list(resumed_raws)
    successes = 0
    failures = 0
    run_t0 = time.monotonic()
    for session_num, session in enumerate(to_process, start=1):
        session_t0 = time.monotonic()
        try:
            extracted = await extract_from_session(
                session, persona_name=settings.PERSONA_NAME, entity_hints=entity_hints
            )
            _persist_raws_and_mark(session.session_id, extracted)
            raws.extend(extracted)
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
            min_distinct_partners=settings.INSIGHTS_MIN_DISTINCT_PARTNERS,
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


def _load_resume_state(*, mode: str) -> tuple[set[str], list[RawInsight]]:
    """Return (skip_ids, resumed_raws) for the current run mode.

    Incremental: skip sessions whose InsightRunState.failed=False and load their
    raws from raw_insight back into memory. Stage D then sees a union of
    (resumed) + (newly extracted) raws — identical to the no-crash baseline.

    Any other mode (full, dry-run, reembed, reconsolidate): return empties. Full
    has already truncated; the rest don't go through Stage C.
    """
    if mode != "incremental":
        return set(), []
    with Session(make_engine()) as s:
        done = list(
            s.exec(
                select(InsightRunState).where(InsightRunState.failed == False)  # noqa: E712
            ).all()
        )
        skip_ids = {r.session_id for r in done}
        if not skip_ids:
            return set(), []
        persisted = list(
            s.exec(
                select(RawInsightRow).where(RawInsightRow.session_id.in_(skip_ids))  # type: ignore[attr-defined]
            ).all()
        )
    return skip_ids, [_raw_from_row(r) for r in persisted]


def _row_from_raw(raw: RawInsight) -> RawInsightRow:
    return RawInsightRow(
        session_id=raw.session_id,
        category=raw.category,
        subject=raw.subject,
        text=raw.text,
        confidence=raw.confidence,
        source_quote=raw.source_quote,
        extracted_at=raw.extracted_at,
    )


def _raw_from_row(row: RawInsightRow) -> RawInsight:
    return RawInsight(
        session_id=row.session_id,
        category=row.category,
        subject=row.subject,
        text=row.text,
        confidence=row.confidence,
        source_quote=row.source_quote,
        extracted_at=row.extracted_at,
    )


def _persist_raws_and_mark(session_id: str, raws: list[RawInsight]) -> None:
    """Atomically persist raws AND mark the session done in one transaction.

    Stage C checkpoint: if the process dies between the LLM call and this commit,
    nothing lands — the session is re-extracted on the next run. If commit
    succeeds, both raws and InsightRunState are durable, so resume can load the
    raws back into memory without recharging the LLM.

    Replaces the old _mark_session_extracted helper.
    """
    with Session(make_engine()) as s:
        # Defense-in-depth: drop any prior raws for this session. Handles
        # --force-session and any unexpected re-entry. New runs almost always
        # see an empty result here, so this is a no-op on the happy path.
        for old in s.exec(
            select(RawInsightRow).where(RawInsightRow.session_id == session_id)
        ).all():
            s.delete(old)
        for raw in raws:
            s.add(_row_from_raw(raw))
        row = s.get(InsightRunState, session_id)
        now = datetime.now(UTC)
        if row is None:
            s.add(
                InsightRunState(
                    session_id=session_id,
                    last_extracted_at=now,
                    insights_count=len(raws),
                    failed=False,
                )
            )
        else:
            row.last_extracted_at = now
            row.insights_count = len(raws)
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
    from persona_rag.index.qdrant_store import to_qdrant_point_id

    vectors = await embed_batch([r.text for r in rows])
    points = [
        PointStruct(
            id=to_qdrant_point_id(r.id),
            vector=vec,
            payload={
                "sqlite_id": r.id,
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
