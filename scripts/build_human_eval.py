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

from persona_rag.eval.compare import build_turing_kit

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
 .hint{color:#64748b;font-size:13px;margin:4px 0 14px}
 .hint b{color:#2563eb}
</style></head><body>
<h2>Which reply is more like something <i>you</i> would actually send?</h2>
<div class="hint">keys: <b>A</b>/<b>B</b> pick &middot; <b>space</b> = can't tell &middot; <b>&larr;/&rarr;</b> nav &middot; <b>Enter</b> = download &nbsp;(auto-advances; ~30-40 is plenty)</div>
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
document.addEventListener('keydown', function(e){
 const k = e.key.toLowerCase();
 if(k==='a'){ choose('A'); }
 else if(k==='b'){ choose('B'); }
 else if(k==='t' || e.key===' '){ e.preventDefault(); choose('tie'); }
 else if(e.key==='ArrowLeft'){ go(-1); }
 else if(e.key==='ArrowRight'){ go(1); }
 else if(e.key==='Enter'){ dl(); }
});
</script></body></html>
"""


# Turing rater: shows the incoming + ONE pair (your REAL reply vs the LoRA, order
# randomized) and asks which is the bot, then an optional one-key "tell" tag. Tags
# match compare.VOICE_TELLS / KNOWLEDGE_TELLS so score_detection can bucket them.
_TURING_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Persona Turing test</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:760px;margin:32px auto;padding:0 16px;color:#0f172a}
 .ctx{white-space:pre-wrap;background:#f1f5f9;border-radius:10px;padding:14px 16px;margin:8px 0 18px;font-size:14px;color:#334155}
 .opt{white-space:pre-wrap;border:2px solid #e2e8f0;border-radius:10px;padding:14px 16px;margin:10px 0;cursor:pointer;font-size:15px}
 .opt:hover{border-color:#94a3b8;background:#f8fafc}
 .opt.sel{border-color:#dc2626;background:#fef2f2}
 .opt b{color:#64748b;font-size:12px;letter-spacing:.05em}
 .bar{display:flex;gap:8px;margin:14px 0;flex-wrap:wrap}
 button{flex:1;padding:12px;border:0;border-radius:8px;background:#dc2626;color:#fff;font-size:15px;cursor:pointer}
 button.unsure{background:#94a3b8}button.nav{background:#e2e8f0;color:#0f172a;flex:0 0 80px}button.done{background:#16a34a}
 .tells{margin:8px 0;padding:12px;border:1px dashed #cbd5e1;border-radius:10px;display:none}
 .tells.show{display:block}
 .tell{display:inline-block;margin:4px;padding:6px 10px;border:1px solid #e2e8f0;border-radius:8px;cursor:pointer;font-size:13px;background:#fff}
 .tell.sel{background:#1d4ed8;color:#fff;border-color:#1d4ed8}
 .hint{color:#64748b;font-size:13px;margin:4px 0 14px}.hint b{color:#dc2626}
 .meta{display:flex;justify-content:space-between;color:#64748b;font-size:13px;margin-top:10px}
</style></head><body>
<h2>Which reply is the <i>bot</i>? &nbsp;<span style="color:#64748b;font-size:14px">(the other one is your real message)</span></h2>
<div class="hint">keys: <b>A</b>/<b>B</b> = that one's the bot &middot; <b>space</b> = can't tell &middot; then <b>1-6</b> = why (optional) &middot; <b>&rarr;</b> next &middot; <b>&larr;</b> back &middot; <b>D</b> download</div>
<div class="meta"><span id="prog"></span><span id="saved"></span></div>
<div class="ctx" id="ctx"></div>
<div class="opt" id="optA" onclick="pick('A')"><b>OPTION A</b><br><span id="a"></span></div>
<div class="opt" id="optB" onclick="pick('B')"><b>OPTION B</b><br><span id="b"></span></div>
<div class="bar"><button class="unsure" onclick="pick('unsure')">can't tell (space)</button></div>
<div class="tells" id="tells">
 <div style="margin-bottom:6px;color:#475569;font-size:13px">what gave the bot away? (optional)</div>
 <span class="tell" data-t="wording" onclick="setTell('wording')">1 wording</span>
 <span class="tell" data-t="length" onclick="setTell('length')">2 length</span>
 <span class="tell" data-t="punct" onclick="setTell('punct')">3 punctuation</span>
 <span class="tell" data-t="too-generic" onclick="setTell('too-generic')">4 too generic</span>
 <span class="tell" data-t="topic" onclick="setTell('topic')">5 wrong reaction/topic</span>
 <span class="tell" data-t="missing-facts" onclick="setTell('missing-facts')">6 missing facts</span>
</div>
<div class="bar">
 <button class="nav" onclick="go(-1)">&#9664; prev</button>
 <button class="nav" onclick="adv()">next &#9654;</button>
 <button class="done" onclick="dl()">Download choices.json</button>
</div>
<script>
const DATA = __DATA__;
const KEY = "persona_turing_choices";
const TELLS = {'1':'wording','2':'length','3':'punct','4':'too-generic','5':'topic','6':'missing-facts'};
let i = 0, awaitingTell = false;
let choices = JSON.parse(localStorage.getItem(KEY) || "{}");
function save(){ localStorage.setItem(KEY, JSON.stringify(choices)); }
function render(){
 const it = DATA[i];
 document.getElementById('ctx').textContent = it.incoming;
 document.getElementById('a').textContent = it.a;
 document.getElementById('b').textContent = it.b;
 document.getElementById('prog').textContent = `Item ${i+1} / ${DATA.length}`;
 const c = choices[it.item_id] || {};
 document.getElementById('optA').className = 'opt' + (c.pick==='A' ? ' sel' : '');
 document.getElementById('optB').className = 'opt' + (c.pick==='B' ? ' sel' : '');
 const show = awaitingTell || c.pick==='A' || c.pick==='B';
 document.getElementById('tells').className = 'tells' + (show ? ' show' : '');
 document.querySelectorAll('.tell').forEach(function(e){ e.className = 'tell' + (c.tell===e.dataset.t ? ' sel' : ''); });
 document.getElementById('saved').textContent = `${Object.keys(choices).length} rated`;
}
function pick(p){
 const id = DATA[i].item_id;
 if(p==='unsure'){ choices[id] = {pick:'unsure', tell:null}; save(); awaitingTell=false; adv(); return; }
 choices[id] = {pick:p, tell:(choices[id]||{}).tell || null}; save(); awaitingTell=true; render();
}
function setTell(t){
 const id = DATA[i].item_id, c = choices[id];
 if(!c || (c.pick!=='A' && c.pick!=='B')) return;
 c.tell = t; save(); adv();
}
function adv(){ awaitingTell=false; if(i<DATA.length-1){ i++; } render(); }
function go(d){ awaitingTell=false; i=Math.max(0,Math.min(DATA.length-1,i+d)); render(); }
function dl(){ const blob=new Blob([JSON.stringify(choices,null,2)],{type:'application/json'});
 const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='choices.json'; a.click(); }
document.addEventListener('keydown', function(e){
 const k = e.key.toLowerCase();
 if(k==='a'){ pick('A'); }
 else if(k==='b'){ pick('B'); }
 else if(e.key===' '){ e.preventDefault(); pick('unsure'); }
 else if(TELLS[e.key]){ setTell(TELLS[e.key]); }
 else if(k==='n' || e.key==='ArrowRight'){ adv(); }
 else if(e.key==='ArrowLeft'){ go(-1); }
 else if(k==='d'){ dl(); }
});
render();
</script></body></html>
"""


def _write_kit(
    out: Path, html: str, blind: list[dict[str, Any]], key: dict[str, dict[str, str]]
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "rater.html").write_text(
        html.replace("__DATA__", json.dumps(blind, ensure_ascii=False)), encoding="utf-8"
    )
    (out / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a blind human-eval kit (A/B preference or Turing)."
    )
    ap.add_argument("--name", default="main")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mode", choices=["ab", "turing"], default="ab")
    a = ap.parse_args()

    base = Path("data/eval/compare") / a.name
    pairs = [
        json.loads(line)
        for line in (base / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if a.mode == "turing":
        # real-vs-LoRA "which is the bot"; engine: compare.build_turing_kit / score_detection.
        blind, key = build_turing_kit(pairs, a.n, a.seed)
        out = Path("reports") / a.name / "turing"
        _write_kit(out, _TURING_HTML, blind, key)
        with (out / "pairs_blind.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "incoming", "option_a", "option_b", "pick(A/B/unsure)", "tell"])
            w.writerows([[b["item_id"], b["incoming"], b["a"], b["b"], "", ""] for b in blind])
        print(f"wrote {len(blind)} blind real-vs-LoRA items -> {out}  (open rater.html)")
        return

    # default: blind A/B preference (api vs lora); both replies must be non-empty.
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
    _write_kit(out, _HTML, blind, key)
    with (out / "pairs_blind.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "incoming", "option_a", "option_b", "choice(A/B/tie)"])
        w.writerows(rows)
    print(f"wrote {len(blind)} blind pairs -> {out}  (open rater.html in a browser)")


if __name__ == "__main__":
    main()
