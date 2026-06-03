#import "@preview/cetz:0.3.4": canvas, draw

#canvas({
  import draw: *
  let stroke = 0.6pt + rgb("#475569")
  let node(name, pos, body, fill: rgb("#eef2ff"), w: auto, size: 7.5pt) = {
    let inner = if w == auto { text(size, body) } else { box(width: w, align(center, text(size, body))) }
    content(pos, inner, frame: "rect", fill: fill, stroke: stroke, padding: 6pt, name: name)
  }
  let arr(a, b) = line(a, b, mark: (end: ">"), stroke: stroke)
  let darr(a, b) = line(a, b, mark: (end: ">"), stroke: (paint: rgb("#b45309"), dash: "dashed", thickness: 0.7pt))

  // train (left) -> merge -> local quantize, then down to GGUF -> serve (right)
  node("train", (2.0, 6.0), [*train* (Colab T4)\ Qwen2.5-3B · 4-bit\ QLoRA r=32 / α=64\ train\_on\_responses\_only], fill: rgb("#dcfce7"))
  node("merge", (5.5, 6.0), [merge 16-bit])
  node("conv", (9.1, 6.0), [local — convert\_hf\_to\_gguf\ + llama-quantize], fill: rgb("#dbeafe"))
  node("gguf", (9.1, 4.3), [GGUF *Q5\_K\_M*], fill: rgb("#dbeafe"))
  node("serve", (9.1, 3.0), [*serve* — llama-server], fill: rgb("#dbeafe"))
  arr("train.east", "merge.west")
  arr("merge.east", "conv.west")
  arr("conv.south", "gguf.north")
  arr("gguf.south", "serve.north")

  // the THIN_SYSTEM invariant: one string fed byte-identically into BOTH train and serve
  node("thin", (3.6, 1.0), [*`THIN_SYSTEM`* — one Ukrainian persona anchor,\ the byte-identical string used in BOTH train and serve], fill: rgb("#fef9c3"), w: 6.6cm, size: 7pt)
  darr("thin.north-west", "train.south")
  darr("thin.north-east", "serve.west")
  content((3.6, 0.05), text(6.8pt, fill: rgb("#b45309"))[train == serve invariant])
})
