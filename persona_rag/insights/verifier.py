# ruff: noqa: RUF001
# Reason: prompt template contains intentional Cyrillic examples.
"""Stage C→D verification gate — audits each RawInsight against its source quote.

Per spec docs/superpowers/specs/2026-05-31-insights-extraction-accuracy-design.md
§5.5. Uses gpt-4o-mini via chat_complete. Fail-open on API errors (returns
verdict=None) so a verifier blip never silently drops legitimate insights.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ValidationError

from persona_rag._logging import get_logger
from persona_rag.config import get_settings
from persona_rag.generate.llm_client import chat_complete
from persona_rag.insights.extractor import RawInsight
from persona_rag.insights.sessions import SessionDoc

log = get_logger()


class VerificationVerdict(BaseModel):
    verdict: Literal["YES", "NO", "AMBIGUOUS"] | None = None
    reason: str = ""


VERIFY_PROMPT = """\
You are auditing a single claim made by an insights extractor about a
person named {persona_name}.

The claim:
  category: {category}
  subject: {subject}
  text: {text}

The supporting quote (allegedly from {persona_name}'s own message):
  "{source_quote}"

Question: does this quote, said by {persona_name}, actually support the
claim about {persona_name} as the subject? Consider:
- If the quote is about another person, third party, or general topic
  rather than {persona_name}'s own activity / opinion / state → NO.
- If the quote is too vague to support the specific claim (e.g. quote is
  "ага" or "норм", claim is "plays basketball") → NO.
- If the quote could be sarcastic, hypothetical, or jokingly agreeing
  with something {persona_name} doesn't actually do → AMBIGUOUS.
- If the quote clearly establishes the claim → YES.

Output ONLY valid JSON in this exact schema:
{{"verdict": "YES" | "NO" | "AMBIGUOUS", "reason": "<one-line explanation>"}}
"""


async def verify_raw(raw: RawInsight, *, session: SessionDoc) -> VerificationVerdict:
    """Verify a single RawInsight. Fail-open on errors."""
    s = get_settings()
    prompt = VERIFY_PROMPT.format(
        persona_name=s.PERSONA_NAME,
        category=raw.category,
        subject=raw.subject,
        text=raw.text,
        source_quote=raw.source_quote,
    )
    try:
        response = await chat_complete(
            [{"role": "user", "content": prompt}],
            model=s.INSIGHTS_VERIFY_MODEL,
            temperature=0.0,
            max_tokens=120,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        log.warning("verifier_api_error", subject=raw.subject, error=str(e)[:200])
        return VerificationVerdict(verdict=None, reason=f"verifier_error: {e!s}"[:200])

    try:
        payload = json.loads(response)
        v = VerificationVerdict(**payload)
    except (json.JSONDecodeError, ValidationError) as e:
        log.warning("verifier_bad_json", subject=raw.subject, error=str(e)[:200])
        return VerificationVerdict(verdict=None, reason=f"verifier_bad_json: {e!s}"[:200])

    log.info(
        "verifier_verdict",
        subject=raw.subject,
        verdict=v.verdict,
        reason=v.reason[:120],
    )
    return v
