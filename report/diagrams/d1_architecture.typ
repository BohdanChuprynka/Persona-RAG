#import "@preview/cetz:0.3.4": canvas, draw

#canvas({
  import draw: *
  let stroke = 0.6pt + rgb("#475569")
  // auto-fitting node: the frame grows to the text, so it can never overflow.
  let node(name, pos, body, fill: rgb("#eef2ff"), w: auto, size: 7.5pt) = {
    let inner = if w == auto { text(size, body) } else { box(width: w, align(center, text(size, body))) }
    content(pos, inner, frame: "rect", fill: fill, stroke: stroke, padding: 6pt, name: name)
  }
  let arr(a, b) = line(a, b, mark: (end: ">", scale: 0.6, fill: rgb("#475569")), stroke: stroke)
  let darr(a, b) = line(a, b, mark: (end: ">", scale: 0.6, fill: rgb("#475569")), stroke: (paint: rgb("#475569"), dash: "dashed", thickness: 0.6pt))

  // ingress
  node("tg", (1.5, 7.4), [Telegram user])
  node("bot", (6.0, 7.4), [aiogram bot · chat handler])
  arr("tg.east", "bot.west")

  // the per-message pipeline
  node("pipe", (5.0, 5.7), [*LangGraph pipeline* (per message)\ auth → retrieve → memory → insights → build\_prompt → *generate* → guardrails → send], w: 10cm, size: 6.8pt)
  arr("bot.south", "pipe.north")

  // the two backends the eval compares
  node("api", (2.5, 3.7), [*API path* — gpt-4o-mini\ rich \~1600-tok RAG prompt\ + retrieval few-shot + logit-bias], fill: rgb("#dbeafe"), w: 4.6cm)
  node("local", (8.2, 3.7), [*local path* — llama-server (llama.cpp)\ Qwen2.5-3B LoRA · GGUF q5\_k\_m\ thin prompt], fill: rgb("#dcfce7"), w: 4.8cm)
  arr((rel: (-2.5, 0), to: "pipe.south"), "api.north")
  arr((rel: (3.2, 0), to: "pipe.south"), "local.north")

  // retrieval stores feed the API path only
  node("stores", (2.5, 1.7), [Retrieval stores\ Qdrant · BM25 · SQLite · style\_anchors.json], fill: rgb("#f1f5f9"), w: 5.0cm)
  darr("stores.north", "api.south")
  content((8.2, 1.7), box(width: 5.2cm, align(center, text(7pt, fill: rgb("#16a34a"))[`GENERATION_BACKEND=ollama`\ skips retrieval (thin prompt)])))
})
