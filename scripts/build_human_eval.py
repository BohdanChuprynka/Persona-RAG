"""Build a BLIND human-preference kit from a comparison run.

The automatic metrics are proxies; the only real measurement of "feels like
Bohdan" is a blind forced choice. This reads ``data/eval/compare/<name>/pairs.jsonl``
and emits, under ``reports/<name>/human_eval/``:

  - rater.html        self-contained blind rater (open in a browser, no server);
                      shows the incoming context + two replies in randomized
                      order, records A / B / tie, persists to localStorage,
                      downloads choices.json.
  - key.json          the un-blinding key (item_id -> which option is api/lora).
  - pairs_blind.csv   low-tech fallback (fill the `choice` column).

Score later by joining choices.json against key.json (win-rate + Wilson CI).
Backend labels are NEVER shown to the rater.

    uv run python scripts/build_human_eval.py --name main --n 100 --seed 7
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Persona blind A/B</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:760px;margin:32px auto;padding:0 16px;color:#0f172a}
 .ctx{white-space:pre-wrap;background:#f1f5f9;border-radius:10px;padding:14px 16px;margin:8px 0 18px;font-size:14px;color:#334155}
 .opt{white-space:pre-wrap;border:2px solid #e2e8f0;border-radius:10px;padding:14px 16px;margin:10px 0;cursor:pointer;font-size:15px}
 .opt:hover{border-color:#94a3b8;background:#f8fafc}
 .opt b{color:#64748b;font-size:12px;letter-spacing:.05em}
 .bar{display:flex;gap:8px;margin:16px 0}
 button{flex:1;padding:12px;border:0;border-radius:8px;background:#2563eb;color:#fff;font-size:15px;cursor:pointer}
 button.tie{background:#94a3b8}button.nav{background:#e2e8f0;color:#0f172a;flex:0 0 90px}
 .meta{display:flex;justify-content:space-between;color:#64748b;font-size:13px;margin-top:10px}
 .done{background:#16a34a}
</style></head><body>
<h2>Which reply is more like something <i>you</i> would actually send?</h2>
<div class="meta"><span id="prog"></span><span id="saved"></span></div>
<div class="ctx" id="ctx"></div>
<div class="opt" id="optA" onclick="choose('A')"><b>OPTION A</b><br><span id="a"></span></div>
<div class="opt" id="optB" onclick="choose('B')"><b>OPTION B</b><br><span id="b"></span></div>
<div class="bar">
 <button class="nav" onclick="go(-1)">◀ prev</button>
 <button class="tie" onclick="choose('tie')">can't tell / tie</button>
 <button class="nav" onclick="go(1)">next ▶</button>
</div>
<div class="bar"><button id="dl" class="done" onclick="dl()">Download choices.json</button></div>
<script>
const DATA = __DATA__;
const KEY = "persona_ab_choices";
let i = 0;
let choices = JSON.parse(localStorage.getItem(KEY) || "{}");
function render(){
 const it = DATA[i];
 document.getElementById('ctx').textContent = it.incoming;
 document.getElementById('a').textContent = it.a;
 document.getElementById('b').textContent = it.b;
 document.getElementById('prog').textContent = `Item ${i+1} / ${DATA.length}`;
 const c = choices[it.item_id];
 document.getElementById('optA').style.borderColor = c==='A' ? '#2563eb' : '#e2e8f0';
 document.getElementById('optB').style.borderColor = c==='B' ? '#2563eb' : '#e2e8f0';
 const n = Object.keys(choices).length;
 document.getElementById('saved').textContent = `${n} rated`;
}
function choose(c){ choices[DATA[i].item_id]=c; localStorage.setItem(KEY,JSON.stringify(choices)); if(i<DATA.length-1){i++;} render(); }
function go(d){ i=Math.max(0,Math.min(DATA.length-1,i+d)); render(); }
function dl(){ const blob=new Blob([JSON.stringify(choices,null,2)],{type:'application/json'});
 const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='choices.json'; a.click(); }
render();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a blind human-preference kit.")
    ap.add_argument("--name", default="main")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    base = Path("data/eval/compare") / a.name
    pairs = [
        json.loads(line)
        for line in (base / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Only items where both backends produced a non-empty reply are ratable.
    pairs = [p for p in pairs if p["gen_api"].strip() and p["gen_lora"].strip()]
    rng = random.Random(a.seed)
    rng.shuffle(pairs)
    pairs = pairs[: a.n]

    blind: list[dict[str, Any]] = []
    key: dict[str, dict[str, str]] = {}
    rows: list[list[str]] = []
    for p in pairs:
        iid = str(p["item_id"])
        a_is_api = rng.random() < 0.5
        a_text = p["gen_api"] if a_is_api else p["gen_lora"]
        b_text = p["gen_lora"] if a_is_api else p["gen_api"]
        blind.append({"item_id": iid, "incoming": p["incoming"], "a": a_text, "b": b_text})
        key[iid] = {"A": "api" if a_is_api else "lora", "B": "lora" if a_is_api else "api"}
        rows.append([iid, p["incoming"], a_text, b_text, ""])

    out = Path("reports") / a.name / "human_eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "rater.html").write_text(
        _HTML.replace("__DATA__", json.dumps(blind, ensure_ascii=False)), encoding="utf-8"
    )
    (out / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "pairs_blind.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "incoming", "option_a", "option_b", "choice(A/B/tie)"])
        w.writerows(rows)
    print(f"wrote {len(blind)} blind pairs -> {out}  (open rater.html in a browser)")


if __name__ == "__main__":
    main()
