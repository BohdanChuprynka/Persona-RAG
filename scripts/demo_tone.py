"""Qualitative A/B of the register-aware tone fix.

Generates a reply to each probe message with REGISTER_AWARE_ENABLED off, then
on, and prints them side by side. The serious probes are where the fix shows:
'off' gives the old flippant brush-off, 'on' engages. Saves a markdown table to
data/eval/tone-demo.md for the morning report.

    uv run python scripts/demo_tone.py
"""

from __future__ import annotations

import os

os.environ.setdefault("SHADOW_MODE", "true")

import asyncio
from pathlib import Path

from persona_rag.config import get_settings
from persona_rag.generate.register import detect_register
from persona_rag.graph.compile import build_graph

# (label, incoming, lead-up context as prior user messages)
PROBES: list[tuple[str, str, list[str]]] = [
    (
        "vulnerable (the reported case)",
        "дивись останнім часом в мене є така проблема що я знаю що деякі речі "
        "***REMOVED***, що мені робити??",
        [],
    ),
    (
        "emotional disclosure",
        "***REMOVED***, не знаю що робити чесно",
        ["прив"],
    ),
    (
        "compulsion / help-seeking",
        "в мене проблема, не можу перестати думати про це, що мені робити?",
        [],
    ),
    (
        "heated (control: should fire back)",
        "сам ти даун шо ти несеш",
        [],
    ),
    (
        "casual (control: should stay short)",
        "шо там по плану на вечір",
        [],
    ),
]


def _admin() -> int:
    return get_settings().ADMIN_TELEGRAM_ID


async def _gen(graph, incoming: str, ctx: list[str]) -> str:
    from datetime import UTC, datetime

    from persona_rag.graph.nodes.load_session import SessionEntry, get_sessions
    from persona_rag.models import ChatMessage

    sessions = get_sessions()
    sessions.clear()
    if ctx:
        sessions[_admin()] = SessionEntry(
            messages=[ChatMessage(role="user", content=c) for c in ctx],
            last_seen=datetime.now(UTC),
        )
    out = await graph.ainvoke({"user_id": _admin(), "chat_id": 0, "incoming": incoming})
    return out.get("reply") or ""


def _set_register(on: bool) -> None:
    os.environ["REGISTER_AWARE_ENABLED"] = "true" if on else "false"
    get_settings.cache_clear()


async def main() -> None:
    graph = build_graph()
    rows: list[tuple[str, str, str, str, str]] = []
    for label, incoming, ctx in PROBES:
        reg = detect_register(incoming)
        _set_register(False)
        off = await _gen(graph, incoming, ctx)
        _set_register(True)
        on = await _gen(graph, incoming, ctx)
        rows.append((label, reg, incoming, off, on))
        print("\n" + "=" * 70)
        print(f"[{label}]  detected register: {reg}")
        print(f"  incoming: {incoming}")
        print("  --- register OFF ---\n    " + off.replace("\n", "\n    "))
        print("  --- register ON  ---\n    " + on.replace("\n", "\n    "))

    out = Path("data/eval/tone-demo.md")
    lines = ["# Register-aware tone fix — qualitative A/B\n"]
    for label, reg, incoming, off, on in rows:
        lines.append(f"## {label}  _(register: {reg})_\n")
        lines.append(f"**incoming:** {incoming}\n")
        lines.append("**register OFF (old behaviour):**\n")
        lines.append("```\n" + off + "\n```\n")
        lines.append("**register ON (fix):**\n")
        lines.append("```\n" + on + "\n```\n")
    out.write_text("\n".join(lines))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
