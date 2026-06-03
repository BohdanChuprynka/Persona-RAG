#import "@preview/cetz:0.3.4": canvas, draw

#canvas({
  import draw: *
  let stroke = 0.6pt + rgb("#475569")
  let node(name, pos, body, fill: rgb("#eef2ff"), w: auto, size: 7.5pt) = {
    let inner = if w == auto { text(size, body) } else { box(width: w, align(center, text(size, body))) }
    content(pos, inner, frame: "rect", fill: fill, stroke: stroke, padding: 6pt, name: name)
  }
  let arr(a, b) = line(a, b, mark: (end: ">"), stroke: stroke)

  // Arm B — controlled: identical thin prompt isolates the weights.
  node("armB", (4.0, 6.0), [*Arm B — identical _thin_ prompt*\ (no retrieval / directives / levers)], fill: rgb("#eef2ff"), w: 7cm)
  node("gptB", (2.0, 4.6), [gpt-4o-mini], fill: rgb("#dbeafe"))
  node("loraB", (6.0, 4.6), [LoRA (thin)], fill: rgb("#dcfce7"))
  arr("armB.south", "gptB.north")
  arr("armB.south", "loraB.north")
  content((9.2, 4.6), align(center, text(7pt, fill: rgb("#475569"))[isolates\ the *model*]))

  // Arm A — production: shipped stack vs thin LoRA isolates the product.
  node("armA", (4.0, 2.9), [*Arm A — shipped API stack vs thin LoRA*], fill: rgb("#fef9c3"), w: 7cm)
  node("gptA", (2.0, 1.5), [gpt-4o-mini\ + rich RAG prompt\ + retrieval + levers], fill: rgb("#dbeafe"), w: 3.6cm)
  node("loraA", (6.0, 1.5), [LoRA (thin)], fill: rgb("#dcfce7"))
  arr("armA.south", "gptA.north")
  arr("armA.south", "loraA.north")
  content((9.2, 1.5), align(center, text(7pt, fill: rgb("#475569"))[isolates\ the *product*]))

  // shared scoring pipeline
  node("scorer", (4.0, -0.1), [shared scorer · paired bootstrap 95% CIs · leak + copy guards · per-language], fill: rgb("#f1f5f9"), w: 9.4cm, size: 7pt)
})
