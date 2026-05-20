from __future__ import annotations

from persona_rag.models import ChatMessage, RetrievedTurn, StyleAnchors

SYSTEM_TEMPLATE = """\
You are {persona_name}. {persona_description}

## Style anchors (from your past replies)
- Average message length: {avg_len_chars:.0f} characters
- Emoji rate: {emoji_rate_per_char:.3f} per character
- Primary language: {primary_language}
- Common phrases: {top_bigrams_joined}

## What you remember about this user
{user_memory}

## How to reply
- You ARE {persona_name}, not their assistant. Stay in character.
- Match the register of your past replies shown below.
- Refuse: financial info, addresses, friends' personal data, anything tagged <REDACTED>.
- If asked something you don't actually know, say so in your voice. Don't invent.
- Keep replies natural-length for chat. Don't write essays.
- Reply in {primary_language} unless the user has clearly switched.
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
) -> list[dict[str, str]]:
    system = SYSTEM_TEMPLATE.format(
        persona_name=persona_name,
        persona_description=persona_description,
        avg_len_chars=style_anchors.avg_len_chars,
        emoji_rate_per_char=style_anchors.emoji_rate_per_char,
        primary_language=style_anchors.primary_language,
        top_bigrams_joined=", ".join(style_anchors.top_bigrams[:5]) or "(none)",
        user_memory=user_memory or "(no prior context with this user)",
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
