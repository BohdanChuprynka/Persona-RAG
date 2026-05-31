"""Stage C — prompt template + JSON parser. LLM call wrapper in Task 9."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ValidationError

from persona_rag.config import get_settings
from persona_rag.generate.llm_client import chat_complete
from persona_rag.insights.sessions import SessionDoc

VALID_CATEGORIES = {"bio", "opinion", "interest", "behavior"}

EXTRACT_SYSTEM_PROMPT = """\
You are analyzing a chat conversation to extract first-person insights about {persona_name}.

Categories (use exactly these):
- bio: factual biographical info (age, location, school, job, family, current projects)
- opinion: stated or strongly implied opinions, preferences, takes
- interest: topics they care about, hobbies, things they spend time on
- behavior: characteristic patterns of behavior or reactions

Output ONLY valid JSON in this exact schema:
{{
  "insights": [
    {{
      "category": "bio|opinion|interest|behavior",
      "subject": "short noun-phrase, normalized lowercase, e.g. 'school', 'cyberpunk 2077', 'mom'",
      "text": "single declarative sentence about persona",
      "confidence": 0.0,
      "source_quote": "short excerpt (≤80 chars) from the conversation supporting this"
    }}
  ]
}}

Rules:
- Only extract first-person assertions about {persona_name}. Skip facts about other people.
- If a statement is sarcastic, joking, or hypothetical, mark confidence ≤ 0.5
- If the session reveals nothing about {persona_name}, return {{"insights": []}}
- Be specific: "studies CS" beats "studies"
- Avoid duplicates within one session
- Each insight ≤ 25 words
- Max 10 insights per session
"""


class RawInsight(BaseModel):
    session_id: str
    category: Literal["bio", "opinion", "interest", "behavior"]
    subject: str
    text: str
    confidence: float
    source_quote: str
    extracted_at: datetime
    # Spec 2026-05-31 §6.1: provenance + verification audit trail.
    source_quote_validated: bool = False
    verification_verdict: Literal["YES", "NO", "AMBIGUOUS"] | None = None
    verification_reason: str | None = None


def _contact_label(recipient_id_hash: str | None) -> str:
    """Stable 8-char-prefix label for a recipient. Same hash → same label."""
    if not recipient_id_hash:
        return "Contact-unknown"
    return f"Contact-{recipient_id_hash[:8]}"


def render_session(session: SessionDoc, persona_name: str) -> str:
    """Render a session with Me: / Contact-<8hex>: speaker labels.

    Spec 2026-05-31 §5.2 — explicit speaker attribution lets the extractor
    discriminate first-person assertions from topics the friend brought up.
    """
    lines: list[str] = [f"Persona name: {persona_name}", f"Session date: {session.start}", ""]
    counter = 1
    for turn in session.turns:
        try:
            ctx = json.loads(turn.incoming_context_json)
        except json.JSONDecodeError:
            ctx = []
        last_in = ctx[-1] if ctx else ""
        contact = _contact_label(turn.recipient_id_hash)
        if last_in:
            lines.append(f"T{counter} {contact}: {last_in}")
            counter += 1
        lines.append(f"T{counter} Me: {turn.your_reply}")
        counter += 1
    return "\n".join(lines)


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL)


def _strip_markdown_fence(text: str) -> str:
    m = _FENCE_RE.match(text.strip())
    return m.group(1) if m else text


def parse_extractor_response(text: str, *, session_id: str) -> list[RawInsight]:
    """Parse the model output. Raises ValueError on un-parseable JSON.

    Silently drops items with unknown categories.
    """
    cleaned = _strip_markdown_fence(text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as e:
        preview = (text[:200] + "…") if len(text) > 200 else text
        raise ValueError(f"non-JSON extractor output: {e} | preview={preview!r}") from e

    items = payload.get("insights", [])
    if not isinstance(items, list):
        raise ValueError("'insights' is not a list")

    now = datetime.now(UTC)
    out: list[RawInsight] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("category") not in VALID_CATEGORIES:
            continue
        try:
            out.append(
                RawInsight(
                    session_id=session_id,
                    category=item["category"],
                    subject=str(item["subject"]),
                    text=str(item["text"]),
                    confidence=float(item.get("confidence", 0.5)),
                    source_quote=str(item.get("source_quote", "")),
                    extracted_at=now,
                )
            )
        except (ValidationError, KeyError, ValueError):
            continue
    return out


async def extract_from_session(
    session: SessionDoc,
    *,
    persona_name: str,
    entity_hints: list[str],
) -> list[RawInsight]:
    """Single LLM call. Returns parsed insights. Raises ValueError on bad output."""
    s = get_settings()
    hint_block = (
        ("Hints — entities the persona often mentions: " + ", ".join(entity_hints) + "\n\n")
        if entity_hints
        else ""
    )
    user_msg = hint_block + render_session(session, persona_name)
    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM_PROMPT.format(persona_name=persona_name)},
        {"role": "user", "content": user_msg},
    ]
    response = await chat_complete(
        messages,
        model=s.INSIGHTS_EXTRACT_MODEL,
        temperature=0.2,
        max_tokens=2000,
        response_format={"type": "json_object"},
    )
    return parse_extractor_response(response, session_id=session.session_id)
