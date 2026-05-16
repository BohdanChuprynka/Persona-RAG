# Prompt Design

How the system prompt and few-shot examples are assembled at runtime.

## Anatomy

A single chat completion call looks like:

```
[ system ]
  You are {PERSONA_NAME}. {PERSONA_DESCRIPTION}

  ## Style anchors (from past replies)
  - Avg msg length: {n} chars
  - Emoji usage: {rate}/msg
  - Primary language: {lang}
  - Common phrases: {top-K bigrams from corpus}

  ## What you remember about this user
  {per-user memory summary, ≤300 tokens}

  ## How to reply
  - Stay in character — you ARE this person, not their assistant.
  - Match the register of your past replies below.
  - Refuse: financial info, addresses, friends' personal data, anything tagged <REDACTED>.
  - If the user asks a question you genuinely don't know, say so in your voice — don't invent.

[ user ]   (one per few-shot pair, formatted as conversation history)
  {incoming_context_1[-1]}

[ assistant ]
  {your_reply_1}

  ... × TOP_K few-shot pairs ...

[ user ]   (current session window)
  {last K turns of this user's session, alternating user/assistant roles}

[ user ]
  {friend's new message}
```

Sent as a standard OpenAI `messages=[...]` array with `role` set per block.

## Building each piece

### System prompt

Static template + dynamic substitution. Template lives at `persona_rag/generate/prompt.py`:

```python
SYSTEM_TEMPLATE = """\
You are {persona_name}. {persona_description}

## Style anchors (from your past replies)
{style_anchors}

## What you remember about this user
{user_memory}

## How to reply
- Stay in character — you ARE this person, not their assistant.
- Match the register of your past replies shown below.
- Refuse: financial info, addresses, friends' personal data, anything tagged <REDACTED>.
- If asked something you genuinely don't know, say so in your voice. Don't invent.
- Keep replies natural-length for chat. Don't write essays.
- Reply in {language} unless the user has clearly switched to another language.
"""
```

Style anchors are computed once at ingest time and cached:

- Avg `your_reply_len_chars`
- Emoji rate (emojis per message)
- Most common bigrams / trigrams (top 10)
- Primary language by message count

Persona description is user-supplied via `PERSONA_DESCRIPTION` env var. Example for the worked Bohdan case (kept out of the public template; lives only in admin's `.env`):

```
PERSONA_DESCRIPTION="Data scientist, mid-20s, Ukrainian. Direct, slightly sarcastic, code-switches between Ukrainian and English depending on topic. Loves gym, fitness, AI/ML, startup ideas. Replies short and punchy."
```

### Few-shot pairs

From `retrieval/retriever.py`:

```python
def retrieve_few_shot(query: str, *, top_k: int, language: str) -> list[PersonaTurn]:
    """
    1. Embed query
    2. LanceDB top_k * 2 by cosine
    3. Filter to matching language
    4. Rerank with recency: score * exp(-age_days / RECENCY_HALF_LIFE_DAYS)
    5. Return top_k
    """
```

Rendered into messages as alternating user/assistant turns. The model sees them as actual prior conversation, not as instructions — which is exactly the cognitive trick that makes few-shot persona transfer work in 2026-era chat models.

### Current session window

Last `CURRENT_SESSION_WINDOW` turns (default 10) of this user's current session, role-mapped:

- Friend's messages → `role: "user"`
- Persona's replies → `role: "assistant"`

Loaded from a short-lived in-memory session store keyed by `(user_id, session_id)`. Session resets after `SESSION_TIMEOUT_MINUTES` silence.

### Current message

The friend's just-arrived message as the final `role: "user"` turn.

## Per-user memory

After every session ends (silence > timeout), a background job runs:

```python
async def update_user_memory(user_id: int, session_log: list[Message]) -> None:
    """
    Ask LLM to update the user_memory.summary based on the just-ended session.
    Strict character limit on output to keep prompts cheap.
    """
```

Prompt:

```
Below is a recent conversation between {persona_name} and a user.
Below that is the current memory summary {persona_name} has about this user.

Conversation:
{session_log}

Current memory:
{existing_summary or "(none yet)"}

Update the memory in ≤300 tokens. Keep:
- Their name / how they prefer to be addressed
- Topics they care about
- Any commitments or promises made (e.g. "I'll send you the link Monday")
- Relationship context (friend, colleague, met-at-event, etc.)

Drop:
- Specific message content older than what's relevant
- Anything tagged <REDACTED>

Output ONLY the new summary text. No preamble.
```

The output replaces the previous `UserMemory.summary`. Cost is small: ~$0.0001 per update with `gpt-4o-mini`.

## Guardrails (post-generation)

Run in `generate/guardrails.py` after LLM returns, before sending to Telegram:

| Check | Action |
|---|---|
| Reply contains `<REDACTED>` literal | Regenerate once with stronger instruction; fail-fast if still leaks |
| Reply matches regex for raw phone/email/address | Strip and regenerate |
| Reply > `MAX_REPLY_TOKENS * 1.5` | Truncate at last sentence boundary |
| Reply empty / whitespace | Send fallback: "...". (Persona is silent sometimes.) |
| Reply contains banned slur list | Block; log; alert admin |

## Token budget

| Component | Typical tokens |
|---|---|
| System prompt | 400 |
| Style anchors | 50 |
| Per-user memory | 200 |
| 8 few-shot pairs | 1600 |
| 10 session turns | 600 |
| Current msg | 50 |
| **Input total** | **~2900** |
| Reply (cap) | 300 |
| **Round-trip** | **~3200** |

At `gpt-4o-mini` pricing (`$0.15/M input, $0.60/M output`): **~$0.0006/reply**.

Even at 200 replies/day: ~$0.12/day, **<$4/month**.

## Tuning levers

| Knob | Effect |
|---|---|
| `TOP_K` | More retrievals → better persona match, more cost |
| `RECENCY_HALF_LIFE_DAYS` | Lower → bot reflects recent style drift, ignores old you |
| `TEMPERATURE` | Higher → more creative, may break persona; lower → repetitive |
| `CURRENT_SESSION_WINDOW` | Larger → better continuity in long sessions |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` → cost; `gpt-4o` → quality |

Start with defaults, A/B test before tuning. See [`EVAL.md`](EVAL.md).
