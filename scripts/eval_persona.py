"""Persona-accuracy eval for Persona-RAG.

Samples held-out persona turns (eval_split=1), replays each through the full
LangGraph with its REAL prior context injected as session, and scores the
generated replies against the real ones by *distribution* (message-shape,
per-bubble length, punctuation, code-switch, opener monotony) — not
mean-of-means, which is blind to the shape-uniformity failure. Optionally adds
a style-embedding "is this me?" cosine (style_self_sim) when the authorship
scorer + its model are available.

Outputs under data/eval/<name>/:
  - pairs.csv      (incoming, real, generated) for blind real-vs-generated A/B
  - scorecard.json (distributional distances + params)
and prints a human scorecard.

    uv run python scripts/eval_persona.py --n 80 --seed 0 --name baseline
    uv run python scripts/eval_persona.py --n 80 --name reg-off --register off
"""

from __future__ import annotations

# Set env BEFORE importing settings-readers (get_settings is cached on first read):
# - SHADOW_MODE: the graph skips the Telegram round-trip; we read state["reply"].
# - MEMORY_UPDATE_INTERVAL_TURNS=0: seeding prior context as session can cross the
#   memory-update throttle and fire real (paid) update_contact_memory LLM calls
#   per qualifying turn (code-review #5). 0 disables it for eval.
import os

os.environ.setdefault("SHADOW_MODE", "true")
os.environ.setdefault("MEMORY_UPDATE_INTERVAL_TURNS", "0")

import argparse
import asyncio
import csv
import json
import random
from datetime import UTC, datetime
from pathlib import Path

from sqlmodel import Session, select

from persona_rag._logging import configure_logging, get_logger
from persona_rag.config import get_settings
from persona_rag.db.engine import make_engine
from persona_rag.db.models import PersonaTurnRow
from persona_rag.eval.distribution import persona_distance
from persona_rag.generate.llm_client import active_model
from persona_rag.graph.compile import build_graph
from persona_rag.graph.nodes.load_session import SessionEntry, get_sessions
from persona_rag.models import ChatMessage

log = get_logger()


def _admin_id() -> int:
    """Bohdan's own (whitelisted) Telegram id — read from settings, never
    hardcoded, so the privacy scrub holds."""
    return get_settings().ADMIN_TELEGRAM_ID


def _sample_held_out(n: int, seed: int) -> list[PersonaTurnRow]:
    with Session(make_engine()) as sess:
        rows = list(
            sess.exec(
                select(PersonaTurnRow).where(PersonaTurnRow.eval_split == True)  # noqa: E712
            ).all()
        )
    # Need something to reply to and a real reply to compare against.
    usable = [
        r for r in rows if (r.your_reply or "").strip() and json.loads(r.incoming_context_json)
    ]
    rng = random.Random(seed)
    rng.shuffle(usable)
    return usable[:n]


def _seed_context(ctx: list[str]) -> str:
    """Inject the lead-up messages as session history; return the final
    incoming message. ctx is chronological; ctx[-1] is what we directly reply to."""
    sessions = get_sessions()
    sessions.clear()
    if len(ctx) > 1:
        msgs = [ChatMessage(role="user", content=c) for c in ctx[:-1] if c.strip()]
        if msgs:
            sessions[_admin_id()] = SessionEntry(messages=msgs, last_seen=datetime.now(UTC))
    return ctx[-1]


async def _generate(graph, turn: PersonaTurnRow) -> tuple[str, str, str]:
    ctx = json.loads(turn.incoming_context_json)
    incoming = _seed_context(ctx)
    out = await graph.ainvoke({"user_id": _admin_id(), "chat_id": 0, "incoming": incoming})
    return incoming, turn.your_reply, (out.get("reply") or "")


def _style_self_sim(real: list[str], gen: list[str]) -> float | None:
    """Mean style-embedding cosine of generated replies to Bohdan's voice
    centroid. None when the optional authorship scorer/model isn't available."""
    try:
        from persona_rag.eval.authorship import reference_vector, self_similarity
    except Exception as e:
        log.info("style_scorer_unavailable", reason=str(e)[:120])
        return None
    try:
        ref = reference_vector(real)
        return self_similarity(gen, ref)
    except Exception as e:
        log.warning("style_scorer_failed", error=str(e)[:160])
        return None


def write_pairs_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    """Write (incoming, real, generated) triples via the csv module so embedded
    quotes / commas / newlines round-trip correctly (code-review #4). The old
    json.dumps-per-cell escaped a `"` as `\\"` which csv.reader mis-parses."""
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["incoming", "real", "generated"])
        w.writerows(rows)


async def run(name: str, n: int, seed: int) -> None:
    s = get_settings()
    turns = _sample_held_out(n, seed)
    if not turns:
        log.warning("no_eval_turns", message="Run ingest first")
        return
    graph = build_graph()

    incomings: list[str] = []
    real: list[str] = []
    gen: list[str] = []
    for i, t in enumerate(turns, 1):
        try:
            inc, r, g = await _generate(graph, t)
        except Exception as e:
            log.warning("gen_failed", i=i, error=str(e)[:200])
            continue
        if not g:
            continue
        incomings.append(inc)
        real.append(r)
        gen.append(g)
        if i % 20 == 0:
            log.info("eval_progress", done=i, n=len(turns), kept=len(gen))

    if not gen:
        log.warning("no_replies_generated")
        return

    dist = persona_distance(real, gen)
    style_sim = _style_self_sim(real, gen)

    out_dir = Path("data/eval") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_pairs_csv(out_dir / "pairs.csv", list(zip(incomings, real, gen, strict=True)))

    scorecard = {
        "name": name,
        "n_generated": len(gen),
        "seed": seed,
        "params": {
            "top_k": s.TOP_K,
            "alpha": s.HYBRID_DENSE_ALPHA,
            "mmr_enabled": getattr(s, "MMR_ENABLED", None),
            "register_aware": getattr(s, "REGISTER_AWARE_ENABLED", None),
            "shape_hint": getattr(s, "SHAPE_HINT_ENABLED", None),
            "best_of_n": getattr(s, "BEST_OF_N", None),
            "paren_logit_bias": getattr(s, "PAREN_LOGIT_BIAS", None),
            "backend": getattr(s, "GENERATION_BACKEND", None),
            "model": active_model(),
            "temperature": s.TEMPERATURE,
            "score_floor": s.HYBRID_SCORE_FLOOR,
        },
        "distance": dist,
        "style_self_sim": style_sim,
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out_dir / "scorecard.json").write_text(json.dumps(scorecard, ensure_ascii=False, indent=2))
    _print_scorecard(scorecard)


def _print_scorecard(sc: dict) -> None:
    d = sc["distance"]
    rs, gs = d["real"], d["gen"]
    print(f"\n{'=' * 56}")
    print(f"  PERSONA SCORECARD — {sc['name']}  (n={sc['n_generated']})")
    print(f"{'=' * 56}")
    print(
        f"  model={sc['params']['model']}  top_k={sc['params']['top_k']}  "
        f"mmr={sc['params']['mmr_enabled']}  reg={sc['params']['register_aware']}  "
        f"temp={sc['params']['temperature']}"
    )
    print("\n  HEADLINE DISTANCES (lower = more like Bohdan):")
    print(f"    shape_js          {d['shape_js']:.4f}   (message-count distribution)")
    print(f"    len_wasserstein   {d['len_wasserstein']:.2f}   (per-bubble char length)")
    print(f"    len_ks            {d['len_ks']:.4f}")
    if sc.get("style_self_sim") is not None:
        print(f"    style_self_sim    {sc['style_self_sim']:.4f}   (HIGHER=more like me)")
    print("\n  SHAPE — % single-message replies:")
    pr, pg = d["pct_single_real"] * 100, d["pct_single_gen"] * 100
    print(f"    real {pr:5.1f}%   vs   generated {pg:5.1f}%")
    print("\n  shape histogram (bubbles per reply):")
    print(f"    {'bucket':>7} | {'real':>7} | {'gen':>7}")
    for b in range(1, 7):
        rv = rs["shape_hist"].get(str(b), rs["shape_hist"].get(b, 0.0))
        gv = gs["shape_hist"].get(str(b), gs["shape_hist"].get(b, 0.0))
        print(f"    {b:>7} | {rv * 100:6.1f}% | {gv * 100:6.1f}%")
    print(
        f"\n  per-bubble length:  real median={rs['bubble_len_median']:.0f} "
        f"mean={rs['bubble_len_mean']:.0f}   gen median={gs['bubble_len_median']:.0f} "
        f"mean={gs['bubble_len_mean']:.0f}"
    )
    print(
        f"  caps ratio:         real={rs['caps_ratio_mean']:.3f}   gen={gs['caps_ratio_mean']:.3f}"
    )
    print(
        f"  emoji rate:         real={rs['emoji_rate_mean']:.4f}   gen={gs['emoji_rate_mean']:.4f}"
    )
    print(
        f"  paren-smiley ):     real={rs.get('paren_smiley_rate', 0):.3f}   "
        f"gen={gs.get('paren_smiley_rate', 0):.3f}"
    )
    print(
        f"  latin-script rate:  real={rs.get('latin_script_rate', 0):.3f}   "
        f"gen={gs.get('latin_script_rate', 0):.3f}   (code-switch)"
    )
    print(
        f"  top-opener share:   real={rs.get('opener_top_share', 0):.3f}   "
        f"gen={gs.get('opener_top_share', 0):.3f}   (monotony)"
    )
    print(f"{'=' * 56}\n")


def main() -> None:
    configure_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=datetime.now().strftime("%Y%m%d-%H%M") + "-baseline")
    p.add_argument(
        "--model",
        default=None,
        help="Override the generation model for this eval only (e.g. gpt-4o-mini "
        "for cheap iteration). Does not touch the live bot's .env.",
    )
    p.add_argument(
        "--register",
        choices=["on", "off"],
        default=None,
        help="Toggle REGISTER_AWARE_ENABLED for this eval only (A/B the tone fix).",
    )
    args = p.parse_args()
    if args.model:
        os.environ["OPENAI_CHAT_MODEL"] = args.model
    if args.register:
        os.environ["REGISTER_AWARE_ENABLED"] = "true" if args.register == "on" else "false"
    if args.model or args.register:
        get_settings.cache_clear()
    asyncio.run(run(args.name, args.n, args.seed))


if __name__ == "__main__":
    main()
