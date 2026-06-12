"""Validated author-detector eval (review fix #2 — construct validity).

Trains a char-n-gram author detector on the TRAIN split (owner replies vs
correspondents' messages), validates it by held-out ROC-AUC on the EVAL split, then
applies it to each backend's generations as a calibrated voice metric: mean P(owner)
and the share accepted as the owner. The AUC step is the point — it shows the metric
can tell the owner from other people before we read it as "sounds like him".

Two units are reported on purpose, because they say different things:
  * message-level (individual messages on both sides — comparable units): isolates
    per-message lexical authorship, which is *thin* in short casual texts.
  * reply-level (the owner's full reply vs a correspondent's context block — the unit
    a generation actually is): a strong detector that jointly captures length,
    structure and style; a length-only baseline is reported alongside so the reader
    can see how much is mere length.

Splits are the LoRA-disjoint ``train.jsonl`` / ``eval.jsonl`` (recipient-stratified),
so the detector never sees the held-out replies it scores. Privacy-safe: only
aggregate AUC / mean-probability numbers are written.

    uv run python scripts/authorship_validation.py
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from persona_rag.eval.authorship_detect import (
    acceptance,
    train_detector,
    validate_auc,
)

FINETUNE = Path("data/finetune")
COMPARE = Path("data/eval/compare")
OUT = Path("data/eval/authorship")


def _load(jsonl: Path, *, split_lines: bool) -> tuple[list[str], list[str]]:
    """(owner, other) units. ``split_lines`` → individual messages (message-level);
    else whole reply vs whole context block (reply-level)."""
    owner: list[str] = []
    other: list[str] = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        br = {m["from"]: m["value"] for m in json.loads(line)["conversations"]}
        g, h = br.get("gpt", ""), br.get("human", "")
        if split_lines:
            owner += [b.strip() for b in g.split("\n") if b.strip()]
            other += [m.strip() for m in h.split("\n") if m.strip()]
        else:
            if g.strip():
                owner.append(g.strip())
            if h.strip():
                other.append(h.strip())
    return owner, other


def _gens(name: str) -> dict[str, list[str]]:
    p = COMPARE / name / "pairs.jsonl"
    if not p.exists():
        return {}
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        "real": [r["real"] for r in rows],
        "api": [r["gen_api"] for r in rows],
        "lora": [r["gen_lora"] for r in rows],
    }


def _length_auc(owner: list[str], other: list[str], seed: int = 0) -> float:
    """ROC-AUC of a trivial 'shorter = owner' baseline — how much of the separation is
    just reply length. Computed by ranking on negative character length."""
    from sklearn.metrics import roc_auc_score

    scores = [-len(t) for t in owner] + [-len(t) for t in other]
    labels = [1] * len(owner) + [0] * len(other)
    return float(roc_auc_score(labels, scores))


def _run_level(split_lines: bool, *, with_arms: bool, rng: random.Random) -> dict[str, Any]:
    owner_tr, other_tr = _load(FINETUNE / "train.jsonl", split_lines=split_lines)
    owner_ev, other_ev = _load(FINETUNE / "eval.jsonl", split_lines=split_lines)
    rng.shuffle(other_tr)
    rng.shuffle(other_ev)
    other_tr = other_tr[: 3 * len(owner_tr)]
    other_ev = other_ev[: 3 * len(owner_ev)]
    model = train_detector(owner_tr, other_tr)
    out: dict[str, Any] = {
        "n_train": {"owner": len(owner_tr), "other": len(other_tr)},
        "validation": validate_auc(model, owner_ev, other_ev),
        "length_only_auc": _length_auc(owner_ev, other_ev),
    }
    if with_arms:
        out["arms"] = {
            arm: {k: acceptance(model, v) for k, v in g.items()}
            for arm in ("main", "armA")
            if (g := _gens(arm))
        }
    return out


def main() -> None:
    rng = random.Random(0)
    msg = _run_level(True, with_arms=False, rng=rng)
    rep = _run_level(False, with_arms=True, rng=rng)

    def auc(d: dict[str, Any]) -> str:
        v = d["validation"]
        ci = v["auc_ci"]
        lo = d["length_only_auc"]
        return f"AUC={v['auc']:.3f} CI[{ci[0]:.3f},{ci[1]:.3f}] (len-only {lo:.3f})"

    print(f"message-level: {auc(msg)}  -> per-message authorship is thin")
    print(f"reply-level:   {auc(rep)}")
    for arm, row in rep.get("arms", {}).items():
        line = "  ".join(
            f"{k}={row[k]['mean_p_owner']:.3f}"
            for k in ("real", "api", "lora")
            if not math.isnan(row[k]["mean_p_owner"])
        )
        print(f"  [{arm}] mean P(owner): {line}")

    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "detector": "char_wb TF-IDF (2,5) + balanced logistic regression; "
        "owner=1 vs correspondents=0",
        "message_level": msg,
        "reply_level": rep,
        "note": "message-level uses comparable individual messages (isolates lexical "
        "authorship); reply-level scores the owner's full reply vs a context block and "
        "jointly captures length+structure+style (length_only_auc shows the length "
        "share). mean_p_owner on generations is the voice metric. Aggregates only.",
    }
    (OUT / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
