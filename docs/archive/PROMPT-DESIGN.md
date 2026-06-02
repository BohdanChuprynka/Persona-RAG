# Prompt Design

How the system prompt + few-shot examples are assembled, and how prompt-caching is engineered for cost.

## Two-part layout (cacheable prefix + dynamic suffix)

OpenAI auto-caches identical prompt **prefixes** ≥1024 tokens on `gpt-4o-mini`. Cached input tokens cost ~50% less. Persona-RAG splits the prompt into a stable prefix (shared across turns in a session) and a variable suffix (changes per message).

```
┌────────────────── CACHED PREFIX (≥1024 tokens, stable per session) ──────────────────┐
│ system: persona identity + style anchors + per-user memory + behavior rules          │
└──────────────────────────────────────────────────────────────────────────────────────┘
┌────────────────── DYNAMIC SUFFIX (changes every turn) ───────────────────────────────┐
│ user/assistant alternation: top-K retrieved few-shot pairs                            │
│ user/assistant alternation: current session window                                    │
│ user: new incoming message                                                            │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

OpenAI matches the prefix by exact token prefix. As long as the system message is identical and the message order doesn't change, the cache hits.

**Implication:** put few-shot pairs and session turns in the *suffix* (user/assistant messages), not in the system prompt. Otherwise every retrieval change busts the cache.

## System prompt template

`persona_rag/generate/prompt.py`:

```python
SYSTEM_TEMPLATE = """\
You are {persona_name}. {persona_description}

## Style anchors (from your past replies)
- Average message length: {avg_len_chars} characters
- Emoji rate: {emoji_rate_per_char:.3f} per character
- Primary language: {primary_language}
- Common phrases: {top_bigrams_joined}

## What you remember about this user
{user_memory_summary}

## How to reply
- You ARE {persona_name}, not their assistant. Stay in character.
- Match the register of the example past replies you'll see below.
- Refuse: financial info, addresses, friends' personal data, anything tagged <REDACTED>.
- If asked something you don't actually know, say so in your voice. Don't invent.
- Keep replies natural-length for chat. Don't write essays.
- Reply in {primary_language} unless the user has clearly switched.
"""
```

Style anchors come from `data/style_anchors.json` (computed once at ingest). `user_memory_summary` is loaded per-user from SQLite. Both are stable within a session → cacheable.

## Few-shot retrieval

`persona_rag/retrieval/retriever.py`:

```python
def retrieve_hybrid(
    query: str,
    *,
    user_language: str,
    top_k: int = TOP_K,
    alpha: float = HYBRID_DENSE_ALPHA,
) -> list[PersonaTurn]:
    """
    1. Embed query with text-embedding-3-small
    2. Qdrant dense top_k * 4 with filter eval_split=False
    3. BM25 top_k * 4 over the same corpus
    4. Min-max normalize both score sets
    5. Fuse: alpha * dense_norm + (1 - alpha) * bm25_norm
    6. Optional language filter: prefer user_language; fall back if too few
    7. Recency rerank: final_score *= exp(-age_days / RECENCY_HALF_LIFE_DAYS)
    8. Return top_k
    """
```

Rendered into messages as alternating user/assistant turns:

```python
messages = [
    {"role": "system", "content": CACHED_PREFIX},
    # few-shot pairs (assistant=persona reply, user=what they replied to)
    {"role": "user", "content": turn1.incoming_context[-1]},
    {"role": "assistant", "content": turn1.your_reply},
    ...
    {"role": "user", "content": turn8.incoming_context[-1]},
    {"role": "assistant", "content": turn8.your_reply},
    # current session window
    {"role": "user", "content": session[-10].text},
    {"role": "assistant", "content": session[-9].text},
    ...
    # new incoming
    {"role": "user", "content": friend_msg},
]
```

The model sees this as actual prior conversation history rather than instructions — which is the cognitive trick that makes few-shot persona transfer work in 2026-era chat models.

## Per-user memory update

After every session ends (silence > `SESSION_TIMEOUT_MINUTES`), an async background node runs:

```python
async def update_user_memory(user_id: int, session_log: list[Message]) -> None:
    prompt = MEMORY_UPDATE_PROMPT.format(
        persona_name=settings.PERSONA_NAME,
        session_log=format_session(session_log),
        existing_summary=load_memory(user_id) or "(none yet)",
    )
    new_summary = await openai.chat.completions.create(
        model=settings.OPENAI_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=350,
    )
    save_memory(user_id, new_summary.choices[0].message.content)
```

Prompt template:

```
Below is a recent conversation between {persona_name} and a user.
Below that is the current memory summary {persona_name} has about this user.

Conversation:
{session_log}

Current memory:
{existing_summary}

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

The new summary replaces the previous. Cost: ~$0.0001 per update with `gpt-4o-mini`.

## Guardrails (post-generation)

Run in `generate/guardrails.py` after the LLM returns, before sending:

| Check | Action |
|---|---|
| Reply contains `<REDACTED>` literal | Regenerate once; fail to fallback if still leaks |
| Reply matches regex for raw phone/email/address | Strip + regenerate |
| Reply > `MAX_REPLY_TOKENS * 1.5` | Truncate at last sentence boundary |
| Reply empty / whitespace | Send fallback: "..." (persona is silent sometimes) |
| Reply contains banned slur list | Block; log; structlog `severity=alert` → admin |

## Token budget

| Component | Typical tokens | Cached? |
|---|---|---|
| System prompt (persona + anchors + memory + rules) | 1100 | ✅ Yes |
| 8 few-shot pairs | 1600 | ❌ No (rotates per query) |
| 10 session turns | 600 | ❌ No |
| Current msg | 50 | ❌ No |
| **Input total** | **~3350** | partial |
| Reply (cap) | 300 | — |

With cache hit on the 1100-token prefix (which dominates fixed cost since few-shot/session pairs vary):

- Without cache: `3350 * $0.15/1M = $0.0005` input + `300 * $0.60/1M = $0.00018` output ≈ **$0.00068/reply**
- With cache (50% off cached portion): `(1100 * 0.5 + 2250) * $0.15/1M + output ≈ $0.00043 + $0.00018 = $0.00061/reply` ≈ **10% total saving**

Cache shines more on long persona prompts or longer sessions. Real-world saving usually 5–15% — modest but free.

At 200 replies/day: ~$0.12/day, **<$4/month**.

## Tuning levers

| Knob | Effect |
|---|---|
| `TOP_K` | More retrievals → better persona match, more cost |
| `HYBRID_DENSE_ALPHA` | 1.0 = dense only; 0.0 = BM25 only. Default 0.7 |
| `RECENCY_HALF_LIFE_DAYS` | Lower → bot reflects recent style drift |
| `TEMPERATURE` | Higher → more creative; lower → repetitive |
| `CURRENT_SESSION_WINDOW` | Larger → better continuity, more tokens |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` (cost) vs `gpt-4o` (quality) |
| `ENABLE_PROMPT_CACHING` | Toggle the cacheable-prefix layout |

Each MLflow eval run logs these as params. See [`EVAL.md`](EVAL.md).
