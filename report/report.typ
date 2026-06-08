// Replicating a Texting Voice — main document.
// Build from the repo root:  bash report/build.sh
// (renders figures via scripts/plot_report.py, then compiles this file).

#set document(title: "Replicating a Texting Voice", author: "Bohdan Chuprynka")
#set page(paper: "a4", margin: (x: 2.2cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#set figure(numbering: "1")
#show figure.caption: set text(9pt)
#show link: set text(rgb("#2563eb"))

// Monospace that pairs with New Computer Modern (NCM Mono is not installed on the
// build host); the default mono falls back to a heavy sans that clashes with the serif.
#show raw: set text(font: "PT Mono")
// Block code reads as an intentional, framed artifact rather than raw terminal text.
#show raw.where(block: true): it => block(
  width: 100%,
  fill: rgb("#f6f8fa"),
  stroke: 0.5pt + rgb("#d0d7de"),
  radius: 4pt,
  inset: (x: 9pt, y: 8pt),
  above: 1.0em,
  below: 1.0em,
  text(size: 8.5pt, it),
)

// ---- title block -------------------------------------------------------------
#align(center)[
  #text(19pt, weight: "bold")[Replicating a Texting Voice]
  #v(0.35em)
  #text(11.5pt)[Building and honestly evaluating a fine-tuned persona model of one
  person, against a production RAG\ + GPT-4o-mini baseline]
  #v(0.6em)
  #text(10pt)[Bohdan Chuprynka · June 2026]
]

#v(0.4em)
#align(center)[
  #block(width: 88%, inset: (x: 6pt))[
    #set par(justify: true)
    #set text(9.5pt)
    #align(left)[*Abstract.* Can a small local fine-tune replicate one person's
    texting voice — and how would you prove it? We build Persona-RAG, a Telegram bot
    that answers in its owner's voice, and ask whether a fine-tuned Qwen2.5-3B LoRA
    texts more like him than the shipped gpt-4o-mini product (a \~1600-token
    retrieval-augmented prompt with decode levers). Trust comes first: we document a
    \~90% train/test leak in the original evaluation, found and fixed, then score both
    backends on a recipient-stratified, model-disjoint hold-out with paired bootstrap
    intervals under a pre-registered acceptance rule. A controlled arm isolates the
    weights; a production arm pits the full API stack against the thin fine-tune, with
    a per-item retrieval guard that removes a measured 28% gold-answer leak. On a level
    field the fine-tune is decisively closer on reply length (Cliff's δ = 0.95, closer
    on 292 of 299 messages) and matches the no-"!" register; fully equipped, the
    production machinery only pulls the API back to a _voice tie_ — at \$0.37 per
    thousand replies and an \~11× token tax against \$0 local — which cost, privacy, and
    offline capability break toward the local model. On the dimensions we can measure,
    then, the conclusion is clear: the local fine-tune is at least the equal of the
    fully-equipped product and decisively beats the bare model — the better replica of
    the register, at zero cost. A thin, intent-routed grounding layer then addresses the
    fine-tune's one structural weakness — that it cannot know facts it was never told:
    distilling identity facts from the owner's own notes lifts its correct-identity-
    answer rate from 0.05 to 0.33 with the voice intact, making the voice/knowledge
    split actionable. What these surface metrics cannot certify is the
    subjective "feels like me"; that pre-registered _primary_ check (a blind human
    win-rate) is unresolved here, because the only rater with standing (the author) is
    recall-biased and privacy precludes outside raters — a real limitation, not a gap
    in the metric verdict.]
  ]
]

#v(0.6em)

#include "parts/part1.typ"
#include "parts/part2.typ"
#include "parts/part3.typ"
#include "parts/part4.typ"
#include "parts/part5.typ"
#include "parts/appendix.typ"

#bibliography("refs.bib", style: "ieee", title: "References")
