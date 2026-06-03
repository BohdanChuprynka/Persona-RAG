#import "@preview/cetz:0.3.4": canvas, draw

#canvas({
  import draw: *
  let stroke = 0.6pt + rgb("#475569")
  let node(name, pos, body, fill: rgb("#eef2ff"), w: auto, size: 7.5pt) = {
    let inner = if w == auto { text(size, body) } else { box(width: w, align(center, text(size, body))) }
    content(pos, inner, frame: "rect", fill: fill, stroke: stroke, padding: 6pt, name: name)
  }
  let arr(a, b) = line(a, b, mark: (end: ">", scale: 0.6, fill: rgb("#475569")), stroke: stroke)

  // shared ingestion spine
  node("exp", (5.5, 7.4), [Telegram / Instagram export])
  node("prep", (5.5, 6.1), [PII redact → burst-collapse (300s) → session-split (6h, ≥4 turns)], w: 8.6cm, size: 7pt)
  node("turns", (5.5, 4.9), [persona turns: (context → reply)], fill: rgb("#e0e7ff"))
  arr("exp.south", "prep.north")
  arr("prep.south", "turns.north")

  // fork A — temporal split (skewed) drives the index
  node("forkA", (2.6, 3.4), [*fork A* — temporal `eval_split`\ (last 10% by time)], fill: rgb("#fef9c3"), w: 4.4cm)
  node("idxA", (2.6, 2.1), [index: Qdrant + BM25], fill: rgb("#fef9c3"))
  arr((rel: (-2.9, 0), to: "turns.south"), "forkA.north")
  arr("forkA.south", "idxA.north")

  // fork B — recipient-stratified split drives train/eval
  node("forkB", (8.4, 3.4), [*fork B* — recipient-stratified\ `eval_split_for` (hash)], fill: rgb("#dcfce7"), w: 4.4cm)
  node("trainB", (8.4, 2.1), [train.jsonl / eval.jsonl (LoRA)], fill: rgb("#dcfce7"))
  arr((rel: (2.9, 0), to: "turns.south"), "forkB.north")
  arr("forkB.south", "trainB.north")

  // the leak the guard fixes — fed by the temporal index
  node("leak", (2.6, 0.6), [retrieval can return the held-out *gold reply* → *28% leak*, removed per-item by the guard], fill: rgb("#fee2e2"), w: 5.6cm, size: 7pt)
  arr("idxA.south", "leak.north")
})
