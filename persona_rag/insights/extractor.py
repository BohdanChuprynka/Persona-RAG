"""Stage C — prompt template + JSON parser. LLM call wrapper in Task 9."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ValidationError

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


def render_session(session: SessionDoc, persona_name: str) -> str:
    """Render a session as a numbered conversation with [friend] / [persona_name] labels."""
    lines: list[str] = [f"Persona name: {persona_name}", f"Session date: {session.start}", ""]
    counter = 1
    for turn in session.turns:
        try:
            ctx = json.loads(turn.incoming_context_json)
        except json.JSONDecodeError:
            ctx = []
        last_in = ctx[-1] if ctx else ""
        if last_in:
            lines.append(f"T{counter} [friend]: {last_in}")
            counter += 1
        lines.append(f"T{counter} [{persona_name}]: {turn.your_reply}")
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
        raise ValueError(f"non-JSON extractor output: {e}") from e

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
