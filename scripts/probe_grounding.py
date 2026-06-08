#!/usr/bin/env python
"""Factual-grounding probe: bare vs grounded local LoRA, judged for hallucination.

Produces the report's grounding-section numbers (spec 2026-06-08). For each probe
in the gitignored probe set, generate K decodes from the local fine-tune under two
conditions, identical except for the fact card (the report's Arm-B discipline):

    bare      thin persona prompt only
    grounded  thin persona prompt + the REAL vault fact card
              (retrieve_insights -> build_fact_card, the live serving path)

Each generation is judged into {correct, hallucinated, deflected} by an LLM judge
given (question, ground-truth, answer). We aggregate the hallucination and correct
rates with Wilson 95% intervals and a register-preservation profile (length,
Latin-script, exclamation, paren-smiley) so the card is shown to add facts without
moving the voice.

Everything here is personal (probes carry the owner's real facts) -> all inputs and
outputs live under reports/ (gitignored). The printed scorecard is aggregate rates
only, no facts.

Requires: llama-server (local LoRA) on OLLAMA_BASE_URL, Qdrant (self_insights with
vault facts), and OPENAI_API_KEY (retrieval embeddings + the judge).
Run: ``make compare-vault``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
# Reflect real serving: load the main-checkout .env, data, and settings regardless
# of the cwd this is launched from (the probe must hit the real vault + model).
os.chdir(_REPO_ROOT)

from openai import AsyncOpenAI  # noqa: E402

from persona_rag.config import get_settings  # noqa: E402
from persona_rag.eval.grounding import (  # noqa: E402
    aggregate_labels,
    parse_judge_label,
    register_profile,
)
from persona_rag.generate.prompt import build_fact_card, build_thin_messages  # noqa: E402
from persona_rag.graph.nodes.retrieve_insights import retrieve_insights  # noqa: E402

DEFAULT_PROBES = _REPO_ROOT / "reports" / "main" / "grounding" / "probes.json"
DEFAULT_OUT = _REPO_ROOT / "reports" / "main" / "grounding" / "results.json"
DEFAULT_GENS = _REPO_ROOT / "reports" / "main" / "grounding" / "generations.jsonl"

# Pre-registered scoring rubric (spec 2026-06-08). Judges factual grounding ONLY.
JUDGE_SYSTEM = """You score whether a chatbot's ANSWER about a person is factually \
grounded against a known GROUND-TRUTH fact. Output exactly one label:

- "correct": the answer asserts the ground-truth fact. Paraphrase, translation, or a \
consistent subset all count as correct.
- "hallucinated": the answer asserts a SPECIFIC fact that CONTRADICTS the ground truth \
(a different school, city, country, employer, job, or person). A confident WRONG \
specific is hallucinated.
- "deflected": the answer commits to no checkable fact relevant to the question \
(vague, refuses, jokes, asks back, "хз", "не знаю"). Neither right nor wrong.

Judge ONLY factual grounding — ignore tone, style, slang, casing, and language. \
Reply in the same language as the question is fine and irrelevant to the label.

Output ONLY JSON: {"label": "correct|hallucinated|deflected", "reason": "<short>"}"""


def _judge_user(question: str, ground_truth: str, answer: str) -> str:
    ans = answer.strip() or "(empty reply)"
    return f"QUESTION:\n{question}\n\nGROUND-TRUTH:\n{ground_truth}\n\nANSWER:\n{ans}"


async def _grounded_card(question: str) -> tuple[str | None, str, int]:
    """Run the real retrieval+routing path for a probe -> (card, lane, n_semantic)."""
    state: dict[str, Any] = {"incoming": question}
    state = await retrieve_insights(state)  # type: ignore[arg-type]
    ins = state.get("insights", {}) or {}
    card = build_fact_card(question, "", ins)
    return card, str(ins.get("lane", "?")), len(ins.get("semantic", []))


def _scorecard(result: dict[str, Any]) -> str:
    def line(cond: str) -> str:
        lab = result[cond]["labels"]
        reg = result[cond]["register"]
        h = lab["hallucinated"]
        c = lab["correct"]
        return (
            f"  {cond:<9} n={lab['n']:<3} "
            f"halluc {h['rate']:.2f} [{h['lo']:.2f},{h['hi']:.2f}]   "
            f"correct {c['rate']:.2f} [{c['lo']:.2f},{c['hi']:.2f}]   "
            f"deflect {lab['deflected']['rate']:.2f}   "
            f"| len {reg['mean_bubble_len']:.0f} lat {reg['latin_rate']:.2f} "
            f"excl {reg['exclaim_rate']:.2f} paren {reg['paren_smiley_rate']:.2f}"
        )

    m = result["meta"]
    return "\n".join(
        [
            "",
            "  ===== GROUNDING PROBE — bare vs grounded (local LoRA) =====",
            f"  gen={m['gen_model']}  judge={m['judge_model']}  "
            f"K={m['k']}  T={m['temperature']}  probes={m['n_probes']}  "
            f"n/condition={m['n_per_condition']}",
            line("bare"),
            line("grounded"),
            f"  unparsed: bare={result['bare']['labels']['unparsed']} "
            f"grounded={result['grounded']['labels']['unparsed']}",
            "",
        ]
    )


async def run(
    *, probes_path: Path, out_path: Path, gens_path: Path, k: int, limit: int, max_tokens: int
) -> dict[str, Any]:
    s = get_settings()
    gen_client = AsyncOpenAI(base_url=s.OLLAMA_BASE_URL, api_key="ollama")
    judge_client = AsyncOpenAI(api_key=s.OPENAI_API_KEY)
    gen_model, judge_model, temperature = s.OLLAMA_MODEL, s.OPENAI_CHAT_MODEL, s.TEMPERATURE

    probes = json.loads(probes_path.read_text(encoding="utf-8"))["probes"]
    if limit:
        probes = probes[:limit]

    gen_sem = asyncio.Semaphore(4)
    judge_sem = asyncio.Semaphore(8)

    async def gen_one(messages: list[dict[str, str]]) -> str:
        async with gen_sem:
            resp = await gen_client.chat.completions.create(
                model=gen_model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            return resp.choices[0].message.content or ""

    async def judge_one(question: str, gt: str, answer: str) -> str:
        async with judge_sem:
            try:
                resp = await judge_client.chat.completions.create(
                    model=judge_model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": _judge_user(question, gt, answer)},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                return parse_judge_label(resp.choices[0].message.content or "") or "unparsed"
            except Exception:
                return "unparsed"

    # Build messages (grounded needs the async retrieval path).
    records: list[dict[str, Any]] = []
    for p in probes:
        q = p["question"]
        card, lane, n_sem = await _grounded_card(q)
        records.append(
            {
                "probe": p,
                "bare_msgs": build_thin_messages(incoming=q, session=[], facts=None),
                "grounded_msgs": build_thin_messages(incoming=q, session=[], facts=card),
                "card": card,
                "lane": lane,
                "n_sem": n_sem,
            }
        )
        print(f"  routed {p['id']:<14} lane={lane:<9} hits={n_sem} card={'yes' if card else 'no'}")

    # Schedule all decodes at once; the semaphore throttles to llama-server capacity.
    for r in records:
        r["bare_task"] = asyncio.gather(*[gen_one(r["bare_msgs"]) for _ in range(k)])
        r["grounded_task"] = asyncio.gather(*[gen_one(r["grounded_msgs"]) for _ in range(k)])
    for r in records:
        r["bare_gens"] = await r["bare_task"]
        r["grounded_gens"] = await r["grounded_task"]

    # Judge every generation.
    for r in records:
        q, gt = r["probe"]["question"], r["probe"]["ground_truth"]
        r["bare_labels"] = await asyncio.gather(*[judge_one(q, gt, a) for a in r["bare_gens"]])
        r["grounded_labels"] = await asyncio.gather(
            *[judge_one(q, gt, a) for a in r["grounded_gens"]]
        )

    bare_labels = [x for r in records for x in r["bare_labels"]]
    grounded_labels = [x for r in records for x in r["grounded_labels"]]
    bare_gens = [a for r in records for a in r["bare_gens"]]
    grounded_gens = [a for r in records for a in r["grounded_gens"]]

    result: dict[str, Any] = {
        "meta": {
            "gen_model": gen_model,
            "judge_model": judge_model,
            "k": k,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n_probes": len(records),
            "n_per_condition": len(bare_labels),
        },
        "bare": {
            "labels": aggregate_labels(bare_labels),
            "register": register_profile(bare_gens),
        },
        "grounded": {
            "labels": aggregate_labels(grounded_labels),
            "register": register_profile(grounded_gens),
        },
        "per_probe": [
            {
                "id": r["probe"]["id"],
                "target": r["probe"]["target"],
                "lang": r["probe"]["lang"],
                "lane": r["lane"],
                "n_semantic": r["n_sem"],
                "has_card": bool(r["card"]),
                "bare_labels": r["bare_labels"],
                "grounded_labels": r["grounded_labels"],
            }
            for r in records
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with gens_path.open("w", encoding="utf-8") as f:
        for r in records:
            for cond, gens, labels in (
                ("bare", r["bare_gens"], r["bare_labels"]),
                ("grounded", r["grounded_gens"], r["grounded_labels"]),
            ):
                for a, lab in zip(gens, labels, strict=True):
                    f.write(
                        json.dumps(
                            {
                                "id": r["probe"]["id"],
                                "cond": cond,
                                "q": r["probe"]["question"],
                                "gt": r["probe"]["ground_truth"],
                                "answer": a,
                                "label": lab,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    print(_scorecard(result))
    print(f"  wrote {out_path.relative_to(_REPO_ROOT)} + {gens_path.relative_to(_REPO_ROOT)}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Bare-vs-grounded factual hallucination probe.")
    ap.add_argument("--k", type=int, default=5, help="decodes per probe per condition")
    ap.add_argument("--limit", type=int, default=0, help="probe cap for a smoke test (0 = all)")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--gens", type=Path, default=DEFAULT_GENS)
    args = ap.parse_args()
    asyncio.run(
        run(
            probes_path=args.probes,
            out_path=args.out,
            gens_path=args.gens,
            k=args.k,
            limit=args.limit,
            max_tokens=args.max_tokens,
        )
    )


if __name__ == "__main__":
    main()
