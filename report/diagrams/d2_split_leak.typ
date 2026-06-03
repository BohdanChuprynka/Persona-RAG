#import "@preview/cetz:0.3.4": canvas, draw

#canvas({
  import draw: *
  let stroke = 0.6pt + rgb("#475569")
  let box(pos, size, body, fill: rgb("#eef2ff")) = {
    let (x, y) = pos
    let (w, h) = size
    rect((x - w / 2, y - h / 2), (x + w / 2, y + h / 2), fill: fill, stroke: stroke, radius: 3pt)
    content((x, y), text(7.5pt)[#body])
  }
  let arr(a, b) = line(a, b, mark: (end: ">"), stroke: stroke)

  // shared ingestion spine
  box((5.5, 7.4), (6.5, 0.7), [Telegram / Instagram export])
  box((5.5, 6.4), (8.2, 0.7), [PII redact → burst-collapse (300s) → session-split (6h, ≥4 turns)])
  box((5.5, 5.4), (5.2, 0.7), [persona turns: (context → reply)], fill: rgb("#e0e7ff"))
  arr((5.5, 7.05), (5.5, 6.75))
  arr((5.5, 6.05), (5.5, 5.75))

  // fork A — temporal split (the skewed one) drives the index
  box((2.7, 4.0), (4.6, 0.85), [*fork A* · temporal `eval_split`\ (last 10% by time)], fill: rgb("#fef9c3"))
  box((2.7, 2.7), (4.6, 0.7), [index: Qdrant + BM25], fill: rgb("#fef9c3"))
  arr((4.6, 5.15), (2.9, 4.45))
  arr((2.7, 3.575), (2.7, 3.05))
  content((2.7, 1.75), text(6.5pt, fill: rgb("#b45309"))[register-skewed: 1 EN contact\ = 62% of Latin → 0.47 artifact])

  // fork B — recipient-stratified split drives train/eval
  box((8.3, 4.0), (4.6, 0.85), [*fork B* · recipient-stratified\ `eval_split_for` (hash)], fill: rgb("#dcfce7"))
  box((8.3, 2.7), (4.6, 0.7), [train.jsonl / eval.jsonl (LoRA)], fill: rgb("#dcfce7"))
  arr((6.4, 5.15), (8.1, 4.45))
  arr((8.3, 3.575), (8.3, 3.05))
  content((8.3, 1.95), text(6.5pt, fill: rgb("#15803d"))[train ≈ eval ≈ 0.18 Latin — honest, reachable target])

  // the leak the guard fixes
  box((2.7, 0.85), (5.2, 0.75), [retrieval can return the held-out *gold reply* →\ *28% leak*, removed per-item (Fig. 1)], fill: rgb("#fee2e2"))
  arr((2.7, 2.35), (2.7, 1.25))
})
