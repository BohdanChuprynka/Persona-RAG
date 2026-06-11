= Related work <sec-related>

The system touches five literatures; positioning against each also sharpens what is,
and is not, novel here.

*Persona and personalized dialogue.* Persona-conditioned generation is well studied:
@li2016persona inject a learned speaker embedding into a seq2seq decoder, and
PersonaChat @zhang2018personachat conditions agents on a handful of natural-language
persona sentences. Those personas are short, declarative, and often crowd-authored; the
task here is the inverse — recover one real individual's _idiolect_ (code-switch ratio,
burst shape, punctuation tics) from their own dense chat logs, with no written persona
description and no second speaker to imitate. The evaluation burden differs accordingly:
persona-dialogue work scores consistency or engagingness, whereas the question here is
metric fidelity to a specific person's surface statistics.

*Stylometry and authorship verification.* Deciding whether two texts share an author is
the classical stylometry problem @stamatatos2009survey, with a mature toolbox of
character- and function-word features and verification protocols. That literature is
directly relevant in two ways: it is the principled source of an _automatic_ "is this
the same author?" score — a far better proxy for "voice" than the surface distances used
here — and its verification framing (same-author vs different-author under a decision
threshold) is exactly what a rigorous voice metric should adopt. This report's surface
metrics are a deliberately lightweight stand-in; promoting a held-out authorship-
verification score to a headline number is named as the clearest measurement upgrade
(@sec-learned).

*Style transfer and controllable generation.* Text style transfer learns to re-render
content in a target style, often without parallel data @shen2017styletransfer. The
contrast is threefold: there is no discrete, labelled style to transfer _to_ (the
"style" is one private person's), no content/style pair to disentangle, and the target
register is bilingual code-switching rather than a sentiment or formality axis. The
fine-tune learns the joint distribution directly rather than editing along a style
dimension.

*LLM-as-judge.* The grounding probe (@sec-grounding) uses a `gpt-4o-mini` judge to label
factual correctness, following the now-standard LLM-as-judge paradigm @zheng2023judging.
That paradigm's documented failure modes — position and verbosity bias, self-preference —
are why the judge is restricted to a three-class factual rubric (never tone or language,
where LLM judges are least reliable) and spot-checked against hand labels (agreement on
11 of 12). The blind _human_ panels remain the design's primary instrument precisely
because an LLM judge cannot certify "feels like me".

*Memorization and code-switching.* The "not memorization" analysis (@sec-relative)
echoes work extracting verbatim training data from language models
@carlini2021extracting: a fine-tune on a small personal corpus is exactly the regime
where memorization is a live risk, which is why the copy/near-copy rate is read against a
measured human-reuse floor rather than against zero. Finally, the corpus is 87% Cyrillic
with heavy Ukrainian/Russian/English code-switching; evaluation therefore cannot lean on
English-centric metrics, and follows the caution in the code-switching literature
@dogruoz2021survey that such mixing is a first-class register feature, not noise to be
normalized away.
