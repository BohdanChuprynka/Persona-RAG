"""Build a BLIND LoRA-vs-real (Turing) kit from a comparison run.

The API-vs-LoRA kit (``build_human_eval.py``) answers "which is more Bohdan".
THIS kit answers the *absolute* question: pair each LoRA reply against Bohdan's
REAL reply for the same context and ask "which one is the machine?". If he can't
beat chance, the LoRA passes as him. Every catch also records a one-tap **tell**
(why) so the scorer can split the gap into *voice* (decode/training) vs
*knowledge* (missing real-world facts -> RAG) — the RAG business case.

Reads ``data/eval/compare/<name>/pairs.jsonl`` (reusing the ``real`` + ``gen_lora``
already generated — no regeneration, no API spend) and emits, under
``reports/<name>/turing/``:

  - rater.html        self-contained blind rater (open in a browser, no server);
                      shows the context + two replies in randomized order, records
                      pick (A / B / unsure) + tell, persists to localStorage,
                      downloads choices.json.
  - key.json          un-blinding key (item_id -> which slot is real / machine).
  - pairs_blind.csv   low-tech fallback.

Score later with ``scripts/score_turing_eval.py``. Labels are NEVER shown.

    uv run python scripts/build_turing_eval.py --name main --n 100 --seed 7
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from persona_rag.eval.compare import build_turing_kit

_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Persona Turing test</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:760px;margin:32px auto;padding:0 16px;color:#0f172a}
 .ctx{white-space:pre-wrap;background:#f1f5f9;border-radius:10px;padding:14px 16px;margin:8px 0 18px;font-size:14px;color:#334155}
 .opt{white-space:pre-wrap;border:2px solid #e2e8f0;border-radius:10px;padding:14px 16px;margin:10px 0;cursor:pointer;font-size:15px}
 .opt:hover{border-color:#94a3b8;background:#f8fafc}
 .opt b{color:#64748b;font-size:12px;letter-spacing:.05em}
 .hint{color:#64748b;font-size:13px;margin:6px 2px 2px}
 .tells{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 4px}
 .tell{border:1px solid #cbd5e1;border-radius:999px;padding:6px 12px;font-size:13px;cursor:pointer;user-select:none}
 .tell:hover{background:#f1f5f9}
 .tell.know{border-color:#f59e0b}
 .bar{display:flex;gap:8px;margin:14px 0}
 button{flex:1;padding:12px;border:0;border-radius:8px;background:#dc2626;color:#fff;font-size:15px;cursor:pointer}
 button.tie{background:#94a3b8}button.nav{background:#e2e8f0;color:#0f172a;flex:0 0 90px}
 .meta{display:flex;justify-content:space-between;color:#64748b;font-size:13px;margin-top:10px}
 .done{background:#16a34a}
</style></head><body>
<h2>Which reply is the <i>machine</i> (AI)?</h2>
<div class="meta"><span id="prog"></span><span id="saved"></span></div>
<div class="ctx" id="ctx"></div>
<div class="opt" id="optA" onclick="pick('A')"><b>OPTION A</b><br><span id="a"></span></div>
<div class="opt" id="optB" onclick="pick('B')"><b>OPTION B</b><br><span id="b"></span></div>
<div class="hint" id="hint"></div>
<div class="tells">
 <span class="tell" data-t="wording" onclick="tell('wording')">wording</span>
 <span class="tell" data-t="length" onclick="tell('length')">length</span>
 <span class="tell" data-t="punct" onclick="tell('punct')">punct/caps</span>
 <span class="tell" data-t="too-generic" onclick="tell('too-generic')">too generic</span>
 <span class="tell know" data-t="missing-facts" onclick="tell('missing-facts')">missing facts</span>
 <span class="tell" data-t="topic" onclick="tell('topic')">topic</span>
 <span class="tell" data-t="other" onclick="tell('other')">other</span>
</div>
<div class="bar">
 <button class="nav" onclick="go(-1)">prev</button>
 <button class="tie" onclick="unsure()">can't tell</button>
 <button class="nav" onclick="go(1)">next</button>
</div>
<div class="bar"><button id="dl" class="done" onclick="dl()">Download choices.json</button></div>
<script>
const DATA = __DATA__;
const STORE = "persona_turing_choices";
let i = 0;
let choices = JSON.parse(localStorage.getItem(STORE) || "{}");
function cur(){ return DATA[i]; }
function save(){ localStorage.setItem(STORE, JSON.stringify(choices)); }
function render(){
 const it = cur();
 document.getElementById('ctx').textContent = it.incoming;
 document.getElementById('a').textContent = it.a;
 document.getElementById('b').textContent = it.b;
 document.getElementById('prog').textContent = `Item ${i+1} / ${DATA.length}`;
 const c = choices[it.item_id] || {};
 document.getElementById('optA').style.borderColor = c.pick==='A' ? '#dc2626' : '#e2e8f0';
 document.getElementById('optB').style.borderColor = c.pick==='B' ? '#dc2626' : '#e2e8f0';
 const picked = c.pick==='A' || c.pick==='B';
 document.querySelectorAll('.tell').forEach(el=>{
  el.style.opacity = picked ? '1' : '0.4';
  el.style.outline = (picked && c.tell===el.dataset.t) ? '2px solid #dc2626' : 'none';
 });
 document.getElementById('hint').textContent = picked
  ? 'why do you think so? tap a tell -> goes to next' : 'tap the reply you think is the AI';
 document.getElementById('saved').textContent = `${Object.keys(choices).length} rated`;
}
function pick(opt){ const it=cur(); const c=choices[it.item_id]||{}; c.pick=opt; choices[it.item_id]=c; save(); render(); }
function tell(t){ const it=cur(); const c=choices[it.item_id]; if(!c||!(c.pick==='A'||c.pick==='B')) return;
 if(t==='other'){ const note=prompt('optional: why? (a few words)')||''; if(note) c.note=note; }
 c.tell=t; choices[it.item_id]=c; save(); if(i<DATA.length-1){i++;} render(); }
function unsure(){ const it=cur(); choices[it.item_id]={pick:'unsure',tell:null}; save(); if(i<DATA.length-1){i++;} render(); }
function go(d){ i=Math.max(0,Math.min(DATA.length-1,i+d)); render(); }
function dl(){ const blob=new Blob([JSON.stringify(choices,null,2)],{type:'application/json'});
 const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='choices.json'; a.click(); }
render();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a blind LoRA-vs-real (Turing) kit.")
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
    blind, key = build_turing_kit(pairs, a.n, a.seed)
    if not blind:
        print(f"no usable real-vs-lora pairs in {base / 'pairs.jsonl'}")
        return

    out = Path("reports") / a.name / "turing"
    out.mkdir(parents=True, exist_ok=True)
    (out / "rater.html").write_text(
        _HTML.replace("__DATA__", json.dumps(blind, ensure_ascii=False)), encoding="utf-8"
    )
    (out / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "pairs_blind.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "incoming", "option_a", "option_b", "pick(A/B/unsure)", "tell"])
        for it in blind:
            w.writerow([it["item_id"], it["incoming"], it["a"], it["b"], "", ""])
    print(f"wrote {len(blind)} blind real-vs-LoRA pairs -> {out}  (open rater.html in a browser)")


if __name__ == "__main__":
    main()
