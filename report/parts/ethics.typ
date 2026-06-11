= Ethics and responsible use <sec-ethics>

Cloning a real person's voice from their private messages is not an ethically neutral
engineering task, and the privacy constraints that block the human evaluation
(@sec-absolute) are the same ones that make this section necessary. The posture
throughout is _self-modeling by a consenting subject_, with the harder obligations named
rather than assumed away.

*Consent and scope.* The subject, the author, and the data owner are the same person,
who consents to modeling his own voice. That consent does _not_ extend to the other
parties in the corpus: the training unit is a `(context → reply)` pair, and the context
is the correspondents' own messages. PII redaction removes identifiers; it does _not_
constitute consent from those correspondents, who never agreed to be modeled even as
conditioning context. This is a real limitation, recorded here as one rather than
resolved: a stricter build would train on the owner's _replies_ alone, or obtain
explicit consent for retained context. The method is scoped to self-use; it carries no
license to clone any other individual, and applying it to a non-consenting person's
messages would be a misuse the authors disavow.

*Misuse and safeguards.* A model that reproduces a named person's voice — and, with the
grounding layer, asserts their identity facts — is a plausible impersonation and
social-engineering vector. The deployment is therefore built to keep that capability
contained rather than portable: inference is fully local (a quantized GGUF served on the
owner's machine, no third-party API in the loop and no phone-home), the bot enforces
sender whitelisting with admin approval before it will reply, a post-generation guardrail
redacts PII and caps length, and a shadow mode logs candidate replies for offline review
without sending. These are framed as safeguards, not just features. The grounding layer
is deliberately conservative for the same reason: it deflects rather than guesses on
facts it was not given, because a confident wrong assertion about a real person is the
costlier error (@sec-grounding).

*Data ownership and lifecycle.* All training data, indices, generations, and human-rating
artifacts are personal; they are stored locally and held out of version control
(gitignored), and the report surfaces only aggregate statistics, never raw messages.
Because the data is the owner's own and lives in one place, deletion is a local operation
— removing the corpus, the derived indices, and the adapter — and an opt-out for any
correspondent who asks is in scope by the same mechanism.

*Identity, not just style.* Finally, a code-switching bilingual voice encodes more than
stylistic surface: language choice and mixing are bound up with identity and
relationship. Replicating them touches the person, not only their phrasing, which is part
of why the certifying judgment is reserved for a human and why this work is framed as a
single-subject study rather than a generalizable cloning recipe.
