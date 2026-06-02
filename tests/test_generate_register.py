"""Tests for incoming-message register detection.

The system had one register (terse banter) and no notion of emotional tone, so
a friend opening up got a flippant 2-bubble brush-off. Register detection lets
generation adapt: heated -> fire back, serious -> engage and take space, casual
-> short.
"""

from __future__ import annotations

from persona_rag.generate.register import detect_register


class TestDetectRegister:
    def test_insult_is_heated(self) -> None:
        assert detect_register("сам ти даун") == "heated"

    def test_profanity_is_heated(self) -> None:
        assert detect_register("хуйлуша іди нахуй") == "heated"

    def test_short_ping_is_casual(self) -> None:
        assert detect_register("шо там") == "casual"

    def test_short_question_is_casual(self) -> None:
        assert detect_register("як справи?") == "casual"

    def test_long_help_seeking_is_serious(self) -> None:
        msg = (
            "слухай, в мене така проблема що я постійно відкладаю важливі справи, "
            "ніяк не можу зібратись, що мені робити??"
        )
        assert detect_register(msg) == "serious"

    def test_explicit_problem_is_serious(self) -> None:
        assert detect_register("в мене проблема, не можу перестати, що мені робити?") == "serious"

    def test_emotional_disclosure_is_serious(self) -> None:
        assert detect_register("чесно, я зовсім розгубився, не знаю що робити") == "serious"

    def test_heated_beats_serious(self) -> None:
        # an insult wins even if long — fire back, don't counsel
        msg = "ти реально даун, я не можу з тобою, що ти робиш взагалі"
        assert detect_register(msg) == "heated"

    def test_distress_profanity_is_serious_not_heated(self) -> None:
        # profanity describing one's OWN bad state is pain, not an attack on me.
        # this is the exact case the bot used to brush off — must engage.
        assert detect_register("мені так хуєво останнім часом, не знаю що робити") == "serious"

    def test_reflective_self_doubt_is_serious(self) -> None:
        msg = "не розумію навіщо я це все роблю, який в цьому сенс взагалі"
        assert detect_register(msg) == "serious"

    def test_dismissive_casual_not_heated(self) -> None:
        # "та похуй" = "whatever" — casual dismissal, not a directed insult
        assert detect_register("та похуй, шо там по плану") == "casual"

    def test_empty_is_casual(self) -> None:
        assert detect_register("") == "casual"

    # Precision guards (code-review #2): casual / technical / negated / resolved
    # text that merely contains a marker substring must NOT become serious.
    def test_technical_question_is_casual(self) -> None:
        assert detect_register("не знаю як це працює в коді") == "casual"

    def test_playful_hypothetical_is_casual(self) -> None:
        assert detect_register("що робити якщо я виграю") == "casual"

    def test_negated_fear_is_casual(self) -> None:
        assert detect_register("та норм, не страшно") == "casual"

    def test_negated_hard_is_casual(self) -> None:
        assert detect_register("мені не важко це зробити") == "casual"

    def test_negated_problem_is_casual(self) -> None:
        assert detect_register("ну це не проблема") == "casual"

    def test_resolved_problem_is_casual(self) -> None:
        assert detect_register("я вирішив проблеми вже") == "casual"

    def test_bored_filler_is_casual(self) -> None:
        assert detect_register("відос набридло") == "casual"
