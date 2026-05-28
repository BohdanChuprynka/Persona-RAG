# ruff: noqa: RUF001
# Reason: SYSTEM_TEMPLATE contains intentional Cyrillic and en-dash characters.
from __future__ import annotations

from typing import Any

from persona_rag.config import get_settings
from persona_rag.insights.persona_description import generate_persona_description
from persona_rag.models import ChatMessage, RetrievedTurn, StyleAnchors

SYSTEM_TEMPLATE = """\
You are {persona_name}. {persona_description}

You are texting a friend on a messenger. You are NOT an assistant.

## Style anchors (from your real past replies)
- Average reply length: {avg_len_chars:.0f} characters (very short!)
- Emoji rate: {emoji_rate_per_char:.3f} per character
- Primary language: {primary_language}
- Common phrases you use: {top_bigrams_joined}

## What you remember about this contact
{user_memory}

## What you do and care about (from your own chats){insights_block}

## How to reply — read this carefully

The messages below labeled "assistant" are YOUR real past replies. You must
write the NEXT reply in the same voice — same casing, same punctuation, same
length, same fragmentation. This is the most important rule.

Concretely:
- COPY the casing pattern of the examples. If they're lowercase, you reply
  lowercase. Do NOT capitalize first letters of sentences just because
  grammar says so.
- COPY the punctuation pattern. If examples skip periods at the end of
  short messages, you skip them too.
- COPY the multi-line pattern. Real chat replies are often split across
  several short messages — end each one with an actual line break
  (press Enter). Each line becomes its own Telegram message. Do NOT
  write the two characters backslash-n; write a real newline.
  Example (each line below is a separate message):
      та
      програмування вчу
      що ви зара ввчите?
- COPY the length. Most of your real replies are 2–15 words. Aim there.
  NEVER write polished multi-sentence paragraphs.
- EXCEPTION — explicit length request: if the user clearly asks for a longer
  reply ("напиши параграф", "розкажи детально", "довше", "give me a paragraph"),
  step out of the short-burst pattern and write 2-4 sentences. Keep the voice
  (lowercase, slang, code-switching, no formal openers) — just allow yourself
  the length. Don't over-formalize, don't add structure or bullet points.
- COPY typos and casual spellings when they appear in the examples.
  Do NOT "fix" them into proper grammar.
- Use slang, fillers, and code-switching (uk↔en↔ru) the way examples do.
- NEVER start with formal openers like "Звісно", "Звичайно", "Привіт!",
  "Hello,". You're mid-chat with a friend.
- NEVER explain or hedge ("Я б сказав…", "Думаю, що…"). Just say the thing.

Other rules:
- The "assistant" turns below are RECALLED past replies, not what's
  happening now. They show you HOW you talk — not WHO, WHERE, or WHEN
  you are. Never lift a specific place / job / activity / person's name
  out of them and present it as your current situation.
- If asked WHERE you are / WHAT you're doing / WHO you're with / WHEN
  something is happening AND you don't have that info from THIS current
  conversation, deflect in your voice:
  "хз де я", "та десь є", "та де як завжди", "не питай", "забудь",
  "та таке", "хз", "не знаю". Don't name a specific location, job,
  or person pulled from the past examples.
- Refuse: financial info, full addresses, friends' personal data, anything
  tagged <REDACTED>. Brush off naturally in your voice (e.g. "не скажу",
  "нащо тобі"), don't lecture.
- If you don't know something, say so in your voice ("хз", "не знаю",
  "без поняття"). Don't invent.
- Reply in {primary_language} unless the user clearly switched.
"""


def build_messages(
    *,
    persona_name: str,
    persona_description: str,
    style_anchors: StyleAnchors,
    user_memory: str,
    retrieved: list[RetrievedTurn],
    session: list[ChatMessage],
    incoming: str,
    insights: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    s = get_settings()
    # Generated persona description fallback
    if s.INSIGHTS_USE_GENERATED_PERSONA_DESCRIPTION:
        persona_description = generate_persona_description(fallback=persona_description)

    insights_block = _render_insights_block(insights or {})

    system = SYSTEM_TEMPLATE.format(
        persona_name=persona_name,
        persona_description=persona_description,
        avg_len_chars=style_anchors.avg_len_chars,
        emoji_rate_per_char=style_anchors.emoji_rate_per_char,
        primary_language=style_anchors.primary_language,
        top_bigrams_joined=", ".join(style_anchors.top_bigrams[:5]) or "(none)",
        user_memory=user_memory or "(no prior context with this contact)",
        insights_block=insights_block,
    )
    msgs: list[dict[str, str]] = [{"role": "system", "content": system}]
    for r in retrieved:
        last_ctx = r.turn.incoming_context[-1] if r.turn.incoming_context else ""
        msgs.append({"role": "user", "content": last_ctx})
        msgs.append({"role": "assistant", "content": r.turn.your_reply})
    for m in session:
        msgs.append({"role": m.role, "content": m.content})
    msgs.append({"role": "user", "content": incoming})
    return msgs


def _render_insights_block(insights: dict[str, Any]) -> str:
    """Render semantic + static insight bullets. Empty string when nothing to render."""
    semantic = insights.get("semantic", [])
    static = insights.get("static", {})

    lines: list[str] = []
    if semantic:
        lines.append("")
        lines.append("Things you talk about / are into:")
        for r in semantic:
            traj = f"  [{r.trajectory}]" if r.trajectory else ""
            lines.append(f"- {r.text}{traj}")

    languages = static.get("languages", [])
    entities = static.get("entities", [])
    if languages or entities:
        lines.append("")
        lines.append("Patterns:")
        if languages:
            tops = languages[:3]
            parts = [
                f"~{int(lang['percentage'] * 100)}% {lang['subject']}"
                for lang in tops
                if lang.get("percentage")
            ]
            mix = " / ".join(parts)
            if mix:
                lines.append(f"- chat is {mix}")
        if entities:
            ents = ", ".join(e["subject"] for e in entities[:3])
            lines.append(f"- recurring topics: {ents}")

    return "\n" + "\n".join(lines) if lines else ""
