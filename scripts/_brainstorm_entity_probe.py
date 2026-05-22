"""Fourth probe — what entities / high-signal terms exist in the corpus?
If clustering is broken, maybe entity-anchored extraction is the way.
"""

# ruff: noqa: RUF001, SIM905
# Reason: intentional Cyrillic stopwords; multi-line string.split() is convenient.
from __future__ import annotations

import re
from collections import Counter

from sqlmodel import Session, select

from persona_rag.db.engine import make_engine
from persona_rag.db.models import PersonaTurnRow

# Cyrillic + Latin tokens, length >= 3
TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЇїІіЄєҐґЁё]{3,}")

# Stopwords (uk/ru/en mixed) — common chat noise to drop
STOPWORDS = set(
    """
    я ти ми ви вони він вона воно мене тебе нас вас його її них мені тобі
    і та що це той ця те так ні там тут тоді тому чи бо але а як яка які
    був була було були буде буду будемо будуть є нема немає є їм їх им
    же ж ж б би уже вже все всё ще вже все ще ось он от да нет неа угу ага
    ок окей ну при по на у в з с до від до по за про щоб щоби якщо
    бы был быть просто только теперь сейчас потом будет есть нет ещё уже
    тоже даже только когда чтобы потому что или но если ладно почему как что
    the and you are for not but with this that have was been from they were
    will would could should about they your what when where which there here
    just like get got just one two three been very some this then them than
    very really because also even still always never ever ago more less now
    мне меня тебе тебя нас вас его её их свой свою своих свои наш ваш мой
    тот эта это эту тех этих тех этих тех тут там везде нигде куда откуда
    дуже трохи багато мало кілько скільки чого нічого нічого щось хтось когось
    нічого почему зачем кто который чем чему этим тем этим тем
    кстати кста короче типу ого блять бля пиздец нахуй ебать сука хуй
    """.split()
)


def main() -> None:
    print("loading rows…")
    with Session(make_engine()) as s:
        rows = list(
            s.exec(select(PersonaTurnRow).where(PersonaTurnRow.eval_split == False)).all()  # noqa: E712
        )
    print(f"  {len(rows)} rows")

    # 1) Capitalized tokens (likely proper nouns / brands / acronyms)
    cap_counter: Counter[str] = Counter()
    # 2) Lower-case content tokens (lexical / topical mass)
    low_counter: Counter[str] = Counter()
    # 3) Latin tokens (English / brand / code)
    latin_counter: Counter[str] = Counter()
    # 4) Hashtags & URLs
    hashtag_counter: Counter[str] = Counter()
    url_count = 0

    HASHTAG_RE = re.compile(r"#\w+")
    URL_RE = re.compile(r"https?://\S+")

    for r in rows:
        text = r.your_reply
        for h in HASHTAG_RE.findall(text):
            hashtag_counter[h] += 1
        if URL_RE.search(text):
            url_count += 1
        for tok in TOKEN_RE.findall(text):
            lower = tok.lower()
            if lower in STOPWORDS:
                continue
            if re.match(r"^[A-Za-z]+$", tok):
                latin_counter[lower] += 1
            elif tok[0].isupper() and len(tok) > 3:
                cap_counter[tok] += 1
            else:
                low_counter[lower] += 1

    print(f"\nurls in persona replies: {url_count}")
    print("\ntop 30 hashtags:")
    for h, n in hashtag_counter.most_common(30):
        print(f"  {n:5d}  {h}")
    print("\ntop 40 capitalized tokens (likely proper nouns):")
    for t, n in cap_counter.most_common(40):
        if n >= 5:
            print(f"  {n:5d}  {t}")
    print("\ntop 40 latin tokens (likely english / brands / tech):")
    for t, n in latin_counter.most_common(40):
        if n >= 5:
            print(f"  {n:5d}  {t}")
    print("\ntop 60 lower-case content tokens (lexical mass):")
    for t, n in low_counter.most_common(60):
        if n >= 10:
            print(f"  {n:5d}  {t}")


if __name__ == "__main__":
    main()
