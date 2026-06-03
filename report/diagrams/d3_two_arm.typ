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

  // Arm B — controlled: identical thin prompt isolates the weights.
  box((4, 6), (8.6, 0.85), [*Arm B — identical _thin_ prompt* (no retrieval / directives / levers)], fill: rgb("#eef2ff"))
  box((2, 4.7), (3, 0.7), [gpt-4o-mini], fill: rgb("#dbeafe"))
  box((6, 4.7), (3, 0.7), [LoRA (thin)], fill: rgb("#dcfce7"))
  arr((4, 5.575), (2, 5.05))
  arr((4, 5.575), (6, 5.05))
  content((9.3, 4.7), text(7pt, fill: rgb("#475569"))[isolates\ the *model*])

  // Arm A — production: shipped stack vs thin LoRA isolates the product.
  box((4, 2.9), (8.6, 0.85), [*Arm A — shipped API stack vs thin LoRA*], fill: rgb("#fef9c3"))
  box((2, 1.6), (3, 0.8), [gpt-4o-mini\ + rich RAG prompt\ + retrieval + levers], fill: rgb("#dbeafe"))
  box((6, 1.6), (3, 0.7), [LoRA (thin)], fill: rgb("#dcfce7"))
  arr((4, 2.475), (2, 2.0))
  arr((4, 2.475), (6, 1.95))
  content((9.3, 1.6), text(7pt, fill: rgb("#475569"))[isolates\ the *product*])

  // shared scoring pipeline
  box((4, 0.3), (9.5, 0.6), [shared scorer · paired bootstrap 95% CIs · leak + copy guards · per-language], fill: rgb("#f1f5f9"))
})
