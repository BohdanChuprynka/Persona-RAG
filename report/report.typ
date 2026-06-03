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
    #align(left)[*Abstract.* #emph[Written in the final pass (Task 12), once every number
    is locked. It states the question (can a fine-tune replicate one person's texting
    voice, and how do you prove it?), the \~90% eval leak found and fixed, the two-arm
    controlled/production design with a proven retrieval guard, the verdict (the LoRA
    ships), and the open Turing frontier.]]
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
