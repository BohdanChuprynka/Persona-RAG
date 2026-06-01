"""Best-of-N candidate selection by style cosine.

When BEST_OF_N > 1 the generate node samples several candidates; we keep the one
closest to Bohdan's voice centroid (research item 3 — the style embedding used
as an inference-time selector/reward). Pure ``pick_best`` is unit-tested; the
style path degrades gracefully to the first candidate if the scorer/model is
unavailable, so enabling best-of-N can never crash generation.
"""

from __future__ import annotations

from persona_rag._logging import get_logger

log = get_logger()


def pick_best(candidates: list[str], scores: list[float]) -> str:
    """Candidate with the highest score; ties keep the first. Defensive: returns
    the first candidate if lengths mismatch, "" if there are no candidates."""
    if not candidates:
        return ""
    if len(candidates) != len(scores):
        return candidates[0]
    best_i = 0
    best = scores[0]
    for i in range(1, len(candidates)):
        if scores[i] > best:
            best = scores[i]
            best_i = i
    return candidates[best_i]


def select_best_style(candidates: list[str]) -> str:
    """Pick the most on-voice candidate by style cosine to Bohdan's centroid."""
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]
    try:
        from persona_rag.eval.authorship import cached_reference_vector, self_similarity

        ref = list(cached_reference_vector())
        scores = [self_similarity([c], ref) for c in candidates]
        best = pick_best(candidates, scores)
        log.info(
            "best_of_n_selected",
            n=len(candidates),
            scores=[round(x, 4) for x in scores],
        )
        return best
    except Exception as e:
        log.warning("best_of_n_fallback", error=str(e)[:160])
        return candidates[0]
