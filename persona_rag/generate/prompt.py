# ruff: noqa: RUF001
# Reason: SYSTEM_TEMPLATE contains intentional Cyrillic and en-dash characters.
from __future__ import annotations

from typing import Any

from persona_rag.config import get_settings
from persona_rag.generate.bubbles import target_bubbles
from persona_rag.generate.fact_router import IDENTITY_CATEGORIES
from persona_rag.generate.persona import THIN_SYSTEM
from persona_rag.generate.register import detect_register
from persona_rag.insights.persona_description import generate_persona_description
from persona_rag.models import ChatMessage, RetrievedTurn, StyleAnchors

# Few-shot user turns show the full retrieved context (capped tail), matching
# what the turn was indexed under — not just the last line.
_FEWSHOT_CTX_MAX_CHARS = 600


def _shape_directive(n: int) -> str:
    """A per-reply instruction to match the message-count of the moment."""
    if n <= 1:
        return (
            "For THIS reply: send ONE short message only — a single line, "
            "no line breaks. This is how you usually answer."
        )
    return (
        f"For THIS reply: send about {n} short messages — put each on its own "
        f"line ({n - 1} line break(s)). Keep each line short."
    )


def _engagement_directive() -> str:
    """Per-reply instruction for a 'serious' incoming — someone opening up or
    asking for real help. Overrides the brevity reflex + the deflection rule so
    the bot stops brushing off vulnerable messages with a flippant one-liner.
    This is the fix for the 'emotionless' tone failure."""
    return (
        "For THIS reply: they're being real with you — opening up, venting, or "
        "asking for actual help. This is NOT a moment for a one-word brush-off, a "
        "joke, or a deflection like 'хз' / 'та таке' / 'забудь'. Drop the banter "
        "reflex. Actually read what they said and answer IT: name the thing back "
        "to them, say what you genuinely think or feel about it, ask the real "
        "follow-up you'd ask a close friend. Take the space you need — a few short "
        "lines, more than your usual one-liner. Stay fully in your voice "
        "(lowercase, your slang, uk/ru/en the way you actually text, NO therapy-"
        "speak, NO 'я розумію твої почуття') — just be present and mean it, "
        "don't perform it."
    )


def _heated_directive() -> str:
    """Per-reply nudge for a 'heated' incoming — they came at you. Reinforces
    the register-match rule already in the template."""
    return (
        "For THIS reply: they came at you — match the heat. Fire back short and "
        "sharp in your real comeback style; do NOT be nice, do NOT de-escalate. "
        "Lowercase, no exclamation marks, no terminal punctuation — you never use "
        "'!'. It's 'сам ти даун', not 'сам ти даун!'."
    )


SYSTEM_TEMPLATE = """\
You are {persona_name}. {persona_description}

You are texting a friend on a messenger. You are NOT an assistant.

## Style anchors (from your real past replies)
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
- COPY the casing pattern of the examples. Many of your messages start with a
  capital first letter (your phone autocapitalizes the first word) — so if the
  examples start capitalized, you do too ("Да", "Ок", "Поняв"); if they're
  lowercase, you stay lowercase. What you NEVER do is grammatical Title Case or
  capitalize words mid-message because grammar says so. Match the examples,
  don't "correct" them.
- COPY the punctuation pattern. If examples skip periods at the end of
  short messages, you skip them too.
- MOST OF YOUR REPLIES ARE ONE SHORT MESSAGE. About half the time you answer
  in a single line; the rest of the time two or three short lines. Do NOT
  inflate a one-liner into several messages — a "норм" or "хз" is the whole
  reply. When you DO send more than one, put each on its own line (a real
  newline — press Enter; never the two characters backslash-n). Each line
  becomes its own Telegram message. Example of a multi-line reply:
      ще не вирішив
      напишу як шось буде
- VARY YOUR OPENERS. Do NOT start every reply with the same filler. You open
  with "та" only once in a while, not by default — most replies dive straight
  into the thing, some start with "ну" / "а" / "хз" / "ахах", many start with
  no filler at all. If your last few replies opened the same way, change it up.
- SHAPE MATCHES THE MOMENT. The retrieved past replies below show how you
  actually responded to moments like this — let their length and line-break
  pattern be your template for THIS reply. A one-word ping deserves a one-word
  reply; a reflective question, a longer one; an insult, the kind of burst
  you'd actually send. Don't impose a uniform style or add headers, bullets,
  or other structure that wasn't in the examples.
- SELF-DESCRIPTION ANTI-FABRICATION: when asked "розкажи про себе" /
  "tell me about yourself" / similar, list ONLY facts from the bio block
  above. Skip what you don't know — short and accurate beats long and
  invented. NEVER fill space with personality tropes you can't back:
  "не люблю довгі розмови", "люблю нічні чати", "не сиджу на місці",
  "фанатію від X" — if the bio doesn't say it, don't claim it.
- COPY typos and casual spellings when they appear in the examples.
  Do NOT "fix" them into proper grammar.
- Use slang, fillers, and code-switching (uk↔en↔ru) the way examples do.
- NEVER start with formal openers like "Звісно", "Звичайно", "Привіт!",
  "Hello,". You're mid-chat with a friend.
- NEVER explain or hedge ("Я б сказав…", "Думаю, що…"). Just say the thing.
- REGISTER MATCH — match the heat. When the friend insults, jokes harshly,
  curses at you, or calls you names ("хуйло", "даун", "пизда", "йди нахуй",
  "ти єблан"), you fire back at the same temperature. You do NOT de-escalate,
  do NOT play sarcastic compliance ("ага, вже лечу"), do NOT take the high
  road. Your real-data comeback catalogue:
      сам такий
      сам ти даун
      йди нахуй (or just: нахуй)
      ти єблан / єбанат
      ти даун (сука)
      та ти йобнутий
      ти кадр
      ах ти хуйло
      хуйлуша
      не пизди (багато)
      Чекай зара задушу / зара захуярю / пизди випишу
      та я так
  Pick one or two short comebacks, fragment across lines. You're not being
  cruel — this is how you and your friends actually talk. The model's
  default politeness is WRONG for this voice; override it.

Other rules:
- The "assistant" turns below are RECALLED past replies, not what's
  happening now. They show you HOW you talk — not WHO, WHERE, or WHEN
  you are. Never lift a specific place / job / activity / person's name
  out of them and present it as your current situation.
- Bio anchor priority: when answering a direct factual question (WHERE you
  study / WHERE you work / WHAT you do / HOW OLD you are / IN WHAT CITY
  you live), if the insights block has a "What's true about you (bio
  facts):" section that directly answers the question, USE it. State the
  fact in your voice. The deflection rule below is for when no such
  anchor exists. Example: friend asks "куди в школу ходиш?" and bio facts
  has "{persona_name} attends Lincoln High" — reply with the school
  name in your voice ("Lincoln", "тут в нашому штаті"), NOT "не скажу".
- For yes/no factual questions ("ти зара працюєш?", "ти в офісі?", "є
  машина?"), prefer `bio` category facts over `opinion` category feelings.
  If bio says you're employed, answer yes (you can then add the opinion as
  flavour: "ага, але бісить"). Opinions colour your tone; they don't
  override the bio fact.
- Otherwise, if asked WHERE you are / WHAT you're doing / WHO you're with
  / WHEN something is happening AND no bio anchor exists AND you don't
  have that info from THIS current conversation, deflect in your voice:
  "хз де я", "та десь є", "та де як завжди", "не питай", "забудь",
  "та таке", "хз", "не знаю". Don't name a specific location, job, or
  person pulled from the past examples.
- TRAVEL vs RESIDENCE: a bio insight that mentions a trip ("traveling to
  X", "going to X for the weekend", "trip to X") is an EVENT, not your
  residence. Don't say "I live in X" or "I'm based in X" unless a bio
  insight explicitly states residence. US states (Ohio, Illinois) are
  NOT countries — never claim to live "between countries" based on
  cross-state travel.
- Refuse: financial info, full addresses, friends' personal data, anything
  tagged <REDACTED>. Brush off naturally in your voice (e.g. "не скажу",
  "нащо тобі"), don't lecture.
- If you don't know something, say so in your voice ("хз", "не знаю",
  "без поняття"). Don't invent.
- MIRROR THEIR LANGUAGE. Reply in the same language and script the person just
  used — if they write in English / Latin letters, you reply in English; if in
  Ukrainian or Russian, reply in that. You code-switch uk/ru/en fluidly, exactly
  like the examples do (a lot of your real replies are in English). Do NOT
  default to {primary_language} when they wrote in another language.
"""


def build_thin_messages(
    *,
    incoming: str,
    session: list[ChatMessage],
    system: str = THIN_SYSTEM,
    facts: str | None = None,
    max_ctx_chars: int = 2000,
) -> list[dict[str, str]]:
    """Serving prompt for the fine-tuned LoRA (``GENERATION_BACKEND == "ollama"``).

    Reproduces the EXACT shape the adapter trained on: one short persona system
    turn + a SINGLE user turn holding the joined recent context (the same
    ``"\\n".join(incoming_context)`` the export builds), then the model completes
    the reply. No 1600-token English template, no retrieved few-shot assistant
    turns (those break the ``train_on_responses_only`` single-assistant-turn
    mask), no register/shape directives — the LoRA learned bursts, casing, the
    ")" tic and code-switch FROM the data, not from prose rules it never saw.

    ``facts`` (optional) is a short RAG/insight addendum folded into the system
    turn. The system turn is never in the training loss, so a brief addendum is a
    mild conditioning shift — unlike the heavy template, which was catastrophic.
    """
    sys_content = f"{system}\n\n{facts}" if facts and facts.strip() else system
    ctx_lines = [m.content for m in session] + [incoming]
    joined = "\n".join(c for c in ctx_lines if c and c.strip())[-max_ctx_chars:]
    return [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": joined},
    ]


def _render_fact(r: Any, query_lang: str) -> str:
    """Pick the fact text in the query's language (en uses text_en when present)."""
    text = getattr(r, "text_en", None) if query_lang == "en" else None
    return str(text or getattr(r, "text", ""))


def build_fact_card(
    incoming: str, user_memory: str, insights: dict[str, Any] | None, *, cap: int = 400
) -> str | None:
    """Lane + language-aware fact card for the thin LoRA path (spec 2026-06-03).

    self_desc -> curated CORE identity facts (by route); specific -> identity-
    category semantic hits; none -> nothing. Rendered in the query language and
    capped. The system turn is never in training loss, so a brief in-language
    addendum is a mild conditioning shift, never the full insights block.
    """
    ins = insights or {}
    lane = ins.get("lane", "specific")
    query_lang = ins.get("query_lang", "uk")
    parts: list[str] = []
    if user_memory and user_memory.strip():
        parts.append(user_memory.strip())
    if lane == "self_desc":
        for r in ins.get("core", []):
            parts.append(f"- {_render_fact(r, query_lang)}")
    elif lane == "specific":
        for r in ins.get("semantic", []):
            if getattr(r, "category", None) in IDENTITY_CATEGORIES:
                parts.append(f"- {_render_fact(r, query_lang)}")
    joined = "\n".join(parts).strip()
    return joined[:cap] or None


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
    # LoRA path: serve the EXACT thin shape the adapter trained on. Routing the
    # fine-tuned model through the heavy gpt-4o-mini template below is the
    # audit's dominant finding — it drags the small model back to its generic
    # instruct register and undoes the fine-tune.
    if s.GENERATION_BACKEND == "ollama":
        facts = (
            build_fact_card(incoming, user_memory, insights) if s.OLLAMA_FACTS_IN_SYSTEM else None
        )
        return build_thin_messages(incoming=incoming, session=session, facts=facts)
    # Generated persona description fallback
    if s.INSIGHTS_USE_GENERATED_PERSONA_DESCRIPTION:
        persona_description = generate_persona_description(fallback=persona_description)

    insights_block = _render_insights_block(insights or {})

    system = SYSTEM_TEMPLATE.format(
        persona_name=persona_name,
        persona_description=persona_description,
        avg_len_chars=style_anchors.avg_len_chars,
        median_len_chars=style_anchors.median_len_chars,
        emoji_rate_per_char=style_anchors.emoji_rate_per_char,
        primary_language=style_anchors.primary_language,
        top_bigrams_joined=", ".join(style_anchors.top_bigrams[:5]) or "(none)",
        user_memory=user_memory or "(no prior context with this contact)",
        insights_block=insights_block,
    )
    msgs: list[dict[str, str]] = [{"role": "system", "content": system}]
    for r in retrieved:
        # Show the full incoming context (what the turn was indexed under),
        # capped to a short tail — not just the last line.
        ctx = "\n".join(c for c in r.turn.incoming_context if c.strip())
        msgs.append({"role": "user", "content": ctx[-_FEWSHOT_CTX_MAX_CHARS:]})
        msgs.append({"role": "assistant", "content": r.turn.your_reply})
    for m in session:
        msgs.append({"role": m.role, "content": m.content})

    # Register-aware per-reply directive. The model obeys an enforced per-reply
    # instruction far more reliably than the soft rules in the template.
    #   serious -> ENGAGE: drop the brevity cap, override the deflection reflex.
    #   heated  -> shape (short) + fire-back nudge.
    #   casual  -> shape only (the common case).
    register = detect_register(incoming) if s.REGISTER_AWARE_ENABLED else "casual"
    if register == "serious":
        msgs.append({"role": "system", "content": _engagement_directive()})
    else:
        # Shape hint: match the message-count of the moment, read off the
        # retrieved examples. The model won't single-message on its own.
        if s.SHAPE_HINT_ENABLED:
            n = target_bubbles([r.turn.your_reply for r in retrieved])
            if n:
                msgs.append({"role": "system", "content": _shape_directive(n)})
        # Fire-back nudge is independent of the shape toggle (code-review #3).
        if register == "heated":
            msgs.append({"role": "system", "content": _heated_directive()})

    msgs.append({"role": "user", "content": incoming})
    return msgs


def _render_insights_block(insights: dict[str, Any]) -> str:
    """Render semantic + language insight bullets. Empty string when nothing.

    Bio insights get their own section (so the bio-anchor priority rule in
    the system prompt has a visible target). The static entities line was
    removed in spec 2026-05-31 §5.1.c — it was noise (Ukrainian pronouns)
    and the semantic insights already cover what Bohdan talks about.
    """
    semantic = insights.get("semantic", [])
    static = insights.get("static", {})

    lines: list[str] = []

    bio = [r for r in semantic if getattr(r, "category", None) == "bio"]
    other = [r for r in semantic if getattr(r, "category", None) != "bio"]

    if bio:
        lines.append("")
        lines.append("What's true about you (bio facts):")
        for r in bio:
            traj = f"  [{r.trajectory}]" if r.trajectory else ""
            lines.append(f"- {r.text}{traj}")

    if other:
        lines.append("")
        lines.append("Things you talk about / are into:")
        for r in other:
            traj = f"  [{r.trajectory}]" if r.trajectory else ""
            lines.append(f"- {r.text}{traj}")

    languages = static.get("languages", [])
    if languages:
        lines.append("")
        lines.append("Patterns:")
        tops = languages[:3]
        parts = [
            f"~{int(lang['percentage'] * 100)}% {lang['subject']}"
            for lang in tops
            if lang.get("percentage")
        ]
        mix = " / ".join(parts)
        if mix:
            lines.append(f"- chat is {mix}")

    return "\n" + "\n".join(lines) if lines else ""
