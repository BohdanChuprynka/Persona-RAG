"""Eval CLI for Persona-RAG.

Loads held-out persona turns from SQLite, generates replies via the LangGraph,
computes stylometric MAD vs real replies, logs run to MLflow.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from sqlmodel import Session, select

from persona_rag._logging import configure_logging, get_logger
from persona_rag.config import get_settings
from persona_rag.db.engine import make_engine
from persona_rag.db.models import PersonaTurnRow
from persona_rag.eval.mlflow_wrap import log_eval_run
from persona_rag.eval.stylometry import mean_abs_deviation
from persona_rag.graph.compile import build_graph

log = get_logger()


async def _run_stylometry(run_name: str) -> None:
    s = get_settings()
    with Session(make_engine()) as sess:
        held_out = list(
            sess.exec(
                select(PersonaTurnRow).where(PersonaTurnRow.eval_split == True)  # noqa: E712
            ).all()
        )
    if not held_out:
        log.warning("no_eval_turns", message="Run ingest first")
        return

    graph = build_graph()
    generated: list[str] = []
    real: list[str] = []
    for row in held_out[:50]:  # cap for speed
        ctx = json.loads(row.incoming_context_json)
        incoming = ctx[-1] if ctx else ""
        final = await graph.ainvoke(
            {"user_id": s.ADMIN_TELEGRAM_ID, "chat_id": 0, "incoming": incoming},
        )
        gen = final.get("reply", "")
        if gen:
            generated.append(gen)
            real.append(row.your_reply)

    if not generated:
        log.warning("no_replies_generated")
        return

    mad = mean_abs_deviation(generated, real)
    composite = sum(mad.values())

    metrics = {f"stylometry_{k}_mad": v for k, v in mad.items()}
    metrics["stylometry_composite"] = composite
    params = {
        "top_k": s.TOP_K,
        "alpha": s.HYBRID_DENSE_ALPHA,
        "model": s.OPENAI_CHAT_MODEL,
        "temperature": s.TEMPERATURE,
        "n_eval_turns": len(generated),
    }
    tags = {"persona_name": s.PERSONA_NAME, "prompt_version": "v1"}

    report_dir = Path("data/eval")
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / f"{run_name}-pairs.csv"
    with csv_path.open("w") as f:
        f.write("real,generated\n")
        for r, g in zip(real, generated, strict=True):
            f.write(f"{json.dumps(r)},{json.dumps(g)}\n")

    run_id = log_eval_run(
        run_name=run_name,
        params=params,
        metrics=metrics,
        tags=tags,
        artifacts=[csv_path],
    )
    log.info("eval_logged", run_id=run_id, composite=composite)


def main() -> None:
    configure_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--metric", choices=["stylometry"], default="stylometry")
    p.add_argument(
        "--name",
        default=f"{datetime.now().strftime('%Y-%m-%d-%H%M')}-baseline",
    )
    args = p.parse_args()
    if args.metric == "stylometry":
        asyncio.run(_run_stylometry(args.name))


if __name__ == "__main__":
    main()
