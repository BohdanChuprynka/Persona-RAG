# ruff: noqa: RUF001
# Reason: intentional Cyrillic content in prompt strings used to exercise the
# Ukrainian-language persona pipeline.
"""Persona-bot stress test.

Drives the full LangGraph pipeline with a probing prompt set in SHADOW_MODE,
captures retrieval evidence and final reply for each, writes a markdown
report. Used to find hallucinations and accuracy issues without needing the
Telegram round-trip.

Usage:
    uv run python scripts/stress_test_persona.py
    uv run python scripts/stress_test_persona.py --out /tmp/stress.md
    uv run python scripts/stress_test_persona.py --limit 5

Requires Qdrant + .env loaded. Writes scratch conversation rows under
chat_id range 990000000+ (cleanup SQL at end of report).
"""

from __future__ import annotations

# Force shadow mode BEFORE importing anything that reads settings, so the
# graph short-circuits past send_reply (no bot attached in stress mode).
import os

os.environ["SHADOW_MODE"] = "true"

import argparse
import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path

from persona_rag._logging import configure_logging, get_logger
from persona_rag.graph.compile import build_graph
from persona_rag.graph.state import GraphState

log = get_logger()


# 20 prompts grouped by known failure categories. Each label is a short
# slug so the report and any future regression set can reference them.
PROMPTS: list[tuple[str, str]] = [
    # --- sport / activity (probes the football/basketball misattribution) ---
    ("sport.what", "ти займаєшся спортом?"),
    ("sport.football", "любиш футбол?"),
    ("sport.basket", "в баскет граєш?"),
    # --- location / identity (probes past-turn vocabulary parroting) ---
    ("loc.now", "ти зараз де?"),
    ("loc.school", "куди в школу ходиш?"),
    ("loc.city", "в якому місті живеш?"),
    # --- work / job (probes McDonald's/office conflation) ---
    ("work.now", "ти зара працюєш?"),
    ("work.where", "де працюєш?"),
    # --- family / relationships (probes refusal + invention) ---
    ("fam.parents", "як батьків звати?"),
    ("rel.gf", "є дівчина?"),
    # --- opinions / preferences (probes opinion-style generation) ---
    ("op.coffee", "любиш каву?"),
    ("op.music", "яка музика подобається?"),
    ("op.crypto", "що думаєш про крипту?"),
    # --- past / recall (probes grounding vs invention) ---
    ("past.yest", "що вчора робив?"),
    ("past.weekend", "як минулі вихідні пройшли?"),
    # --- future / plans (probes speculation grounding) ---
    ("future.tom", "що плануєш на завтра?"),
    ("future.summer", "куди їдеш в літі?"),
    # --- casual / style (probes register match) ---
    ("style.mood", "як настрій?"),
    ("style.what", "шо там?"),
    # --- explicit length request (probes paragraph-on-request rule) ---
    ("style.para", "розкажи про себе параграф"),
]


# Bohdan's own Telegram ID (already whitelisted in the auth table).
WHITELISTED_USER_ID = 1037155651
SCRATCH_CHAT_ID_BASE = 990_000_000


async def run_one(graph, prompt_label: str, prompt_text: str, chat_id: int) -> dict:
    """Invoke the full graph once and distill the final state into a row."""
    state: GraphState = {
        "user_id": WHITELISTED_USER_ID,
        "chat_id": chat_id,
        "incoming": prompt_text,
    }
    t0 = time.monotonic()
    out = await graph.ainvoke(state)
    elapsed = time.monotonic() - t0

    insights = out.get("insights") or {}
    semantic_raw = insights.get("semantic") or []
    semantic = []
    for it in semantic_raw[:6]:
        d = it.model_dump() if hasattr(it, "model_dump") else dict(it)
        semantic.append(
            {
                "category": d.get("category"),
                "subject": d.get("subject"),
                "score": round(float(d.get("final_score") or 0), 3),
                "ev": d.get("evidence_count"),
                "text": (d.get("text") or "")[:160],
            }
        )

    static = insights.get("static") or {}

    retrieved = []
    for r in (out.get("retrieved") or [])[:4]:
        t = r.turn
        retrieved.append(
            {
                "score": round(float(r.score or 0), 3),
                "score_dense": round(float(r.score_dense or 0), 3),
                "score_bm25": round(float(r.score_bm25 or 0), 3),
                "lang": t.language,
                "reply": (t.your_reply or "")[:140],
            }
        )

    sys_msg_full = None
    for m in out.get("prompt") or []:
        if m.get("role") == "system":
            sys_msg_full = m.get("content")
            break

    return {
        "label": prompt_label,
        "incoming": prompt_text,
        "elapsed_s": round(elapsed, 2),
        "semantic": semantic,
        "static_entities": [e.get("subject") for e in (static.get("entities") or [])[:5]],
        "static_languages": [
            (lang.get("subject"), round(lang.get("percentage") or 0, 3))
            for lang in (static.get("languages") or [])[:3]
        ],
        "retrieved": retrieved,
        "reply": out.get("reply") or "",
        # Keep just the insights-block region of the system prompt to ease
        # eyeballing; full prompt is in LangSmith if needed.
        "system_prompt_excerpt": (sys_msg_full or "")[:2500] if sys_msg_full else None,
    }


def render_markdown(results: list[dict], out_path: str) -> None:
    lines: list[str] = []
    ts = datetime.now(UTC).isoformat(timespec="seconds")
    lines.append(f"# Persona-bot stress test — {ts}\n")
    lines.append(
        f"{len(results)} prompts driven through the full LangGraph pipeline "
        f"in SHADOW_MODE. Captures: top-6 semantic insights, top-4 hybrid "
        f"turns, final reply. Use this to spot hallucinations and "
        f"misattribution.\n"
    )

    lines.append("## Index\n")
    for r in results:
        lines.append(f"- `{r['label']}` — {r['incoming']!r}")
    lines.append("")

    for r in results:
        lines.append(f"## `{r['label']}` — {r['elapsed_s']}s\n")
        lines.append(f"**Incoming:** {r['incoming']!r}\n")

        if r["semantic"]:
            lines.append("**Retrieved insights (semantic, top-6):**\n")
            lines.append("| # | cat | subject | ev | score | text |")
            lines.append("|---|---|---|---|---|---|")
            for i, ins in enumerate(r["semantic"], 1):
                txt = ins["text"].replace("|", "\\|").replace("\n", " ")
                lines.append(
                    f"| {i} | {ins['category']} | {ins['subject']} "
                    f"| {ins['ev']} | {ins['score']} | {txt} |"
                )
            lines.append("")
        else:
            lines.append("**Retrieved insights:** (none above floor)\n")

        if r["retrieved"]:
            lines.append("**Retrieved hybrid turns (top-4):**\n")
            for i, t in enumerate(r["retrieved"], 1):
                reply = t["reply"].replace("\n", " ")
                lines.append(
                    f"  {i}. score={t['score']} "
                    f"(dense={t['score_dense']}, bm25={t['score_bm25']}) "
                    f"lang={t['lang']}\n"
                    f"     _you said:_ `{reply!r}`"
                )
            lines.append("")
        else:
            lines.append("**Retrieved hybrid turns:** (none above floor)\n")

        if r["static_entities"] or r["static_languages"]:
            lines.append(
                "**Static patterns:** "
                f"langs={r['static_languages']} "
                f"entities={r['static_entities']}\n"
            )

        lines.append("**Reply:**\n")
        lines.append("```\n" + (r["reply"] or "<empty>") + "\n```\n")
        lines.append("---\n")

    lines.append("## Cleanup\n")
    lines.append(
        "Scratch chat rows live in the conversation/message tables "
        f"under chat_id_hash derived from {SCRATCH_CHAT_ID_BASE} + i. "
        "Safe to leave; or to wipe:\n"
    )
    lines.append("```sql")
    lines.append("-- DB Browser / sqlite3 data/persona.db")
    lines.append(
        f"delete from message where chat_id_hash in "
        f"(select chat_id_hash from conversation where chat_id "
        f"between {SCRATCH_CHAT_ID_BASE} and "
        f"{SCRATCH_CHAT_ID_BASE + 99});"
    )
    lines.append(
        f"delete from conversation where chat_id between "
        f"{SCRATCH_CHAT_ID_BASE} and {SCRATCH_CHAT_ID_BASE + 99};"
    )
    lines.append("```")

    Path(out_path).write_text("\n".join(lines))


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/persona-stress-test.md")
    ap.add_argument(
        "--limit", type=int, default=None, help="Only run first N prompts (for quick iteration)"
    )
    args = ap.parse_args()

    configure_logging()
    log.info("stress_test_start", n_prompts=len(PROMPTS), shadow_mode=True)
    graph = build_graph()

    prompts = PROMPTS[: args.limit] if args.limit else PROMPTS
    results: list[dict] = []
    for i, (label, text) in enumerate(prompts, 1):
        chat_id = SCRATCH_CHAT_ID_BASE + i
        try:
            r = await run_one(graph, label, text, chat_id)
            results.append(r)
            log.info(
                "stress_prompt_done",
                i=i,
                n=len(prompts),
                label=label,
                n_insights=len(r["semantic"]),
                n_turns=len(r["retrieved"]),
                elapsed_s=r["elapsed_s"],
                reply_preview=(r["reply"] or "")[:80],
            )
        except Exception as e:
            log.warning("stress_prompt_failed", label=label, error=str(e)[:300])
            results.append(
                {
                    "label": label,
                    "incoming": text,
                    "elapsed_s": 0,
                    "semantic": [],
                    "static_entities": [],
                    "static_languages": [],
                    "retrieved": [],
                    "reply": f"<ERROR: {e}>",
                    "system_prompt_excerpt": None,
                }
            )

    render_markdown(results, args.out)
    print(f"\nWrote {len(results)} results to {args.out}")
    print(f"  errors:       {sum(1 for r in results if r['reply'].startswith('<ERROR'))}")
    print(f"  zero-insight: {sum(1 for r in results if not r['semantic'])}")
    print(f"  zero-turn:    {sum(1 for r in results if not r['retrieved'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
