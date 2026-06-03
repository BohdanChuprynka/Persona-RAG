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

  // training (left) -> merge -> local quantize -> GGUF -> serve (right)
  box((2.1, 5.4), (3.6, 1.15), [*train* (Colab T4)\ Qwen2.5-3B · 4-bit\ QLoRA r=32 / α=64\ train\_on\_responses\_only], fill: rgb("#dcfce7"))
  box((5.2, 5.4), (2.4, 0.7), [merge 16-bit])
  box((8.2, 5.4), (3.4, 0.95), [local · convert\_hf\_to\_gguf\ + llama-quantize], fill: rgb("#dbeafe"))
  box((10.9, 3.9), (2.6, 0.7), [GGUF *Q5\_K\_M*], fill: rgb("#dbeafe"))
  box((10.9, 2.6), (2.6, 0.7), [*serve* · llama-server], fill: rgb("#dbeafe"))
  arr((3.9, 5.4), (4.0, 5.4))
  arr((6.4, 5.4), (6.5, 5.4))
  arr((9.4, 4.92), (10.6, 4.25))
  arr((10.9, 3.55), (10.9, 2.95))

  // the THIN_SYSTEM invariant: one string into BOTH train and serve
  box((5.5, 1.4), (6.8, 0.8), [*`THIN_SYSTEM`* — one Ukrainian persona anchor,\ the byte-identical string used in BOTH train and serve], fill: rgb("#fef9c3"))
  arr((4.0, 1.8), (2.1, 4.82))
  arr((7.5, 1.6), (10.4, 2.42))
  content((5.5, 0.6), text(7pt, fill: rgb("#b45309"))[train == serve invariant])
})
