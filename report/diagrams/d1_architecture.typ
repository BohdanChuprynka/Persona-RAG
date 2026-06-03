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
  let darr(a, b) = line(a, b, mark: (end: ">"), stroke: (paint: rgb("#475569"), dash: "dashed", thickness: 0.6pt))

  // incoming
  box((1.8, 7), (2.8, 0.7), [Telegram user])
  box((6.4, 7), (4.6, 0.7), [aiogram bot · chat handler])
  arr((3.2, 7), (4.1, 7))

  // the per-message LangGraph pipeline
  box((5.7, 5.7), (10.2, 1.0), text(6.5pt)[*LangGraph pipeline* (per message)\ auth → retrieve → memory → insights → build\_prompt → *generate* → guardrails → send])
  arr((6.4, 6.65), (5.7, 6.2))

  // the two backends the eval compares
  box((3, 4.0), (4.4, 1.15), [*API path* · gpt-4o-mini\ rich ~1600-tok RAG prompt\ + retrieval few-shot + logit-bias], fill: rgb("#dbeafe"))
  box((8.4, 4.0), (4.4, 1.15), [*local path* · llama.cpp llama-server\ Qwen2.5-3B LoRA · GGUF q5\_k\_m\ thin prompt], fill: rgb("#dcfce7"))
  arr((5.7, 5.22), (3, 4.58))
  arr((5.7, 5.22), (8.4, 4.58))
  content((5.7, 4.75), text(6.5pt, fill: rgb("#475569"))[generate])

  // retrieval stores feed the API path only
  box((3, 2.2), (5.6, 0.95), text(7pt)[Retrieval stores\ Qdrant · BM25 · SQLite · style\_anchors.json], fill: rgb("#f1f5f9"))
  darr((3, 2.675), (3, 3.42))
  content((8.4, 2.4), text(7pt, fill: rgb("#16a34a"))[`GENERATION_BACKEND=ollama`\ skips retrieval (thin prompt)])
})
