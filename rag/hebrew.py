"""Hebrew text normalisation and repair of PDFs that store text in visual (reversed) order."""
from __future__ import annotations

import re
import unicodedata

HEB_LETTER = r"א-ת"
FINAL_LETTERS = set("ךםןףץ")
NIQQUD_RE = re.compile(r"[֑-ׇ]")  # cantillation + vowel points (keeps letters, maqaf is ־ -> handled below)
HEB_WORD_RE = re.compile(f"[{HEB_LETTER}]{{2,}}")
LTR_RUN_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9.,:%/+\-_' ]*[A-Za-z0-9%])?")


def normalize(text: str) -> str:
    """Unicode-normalise, drop niqqud, unify quote marks and whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("־", "-")          # maqaf -> hyphen
    text = NIQQUD_RE.sub("", text)
    text = text.replace("״", '"').replace("׳", "'")   # gershayim / geresh
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = re.sub(r"[‎‏‪-‮⁦-⁩]", "", text)  # bidi control chars
    text = text.replace(" ", " ")
    text = re.sub(r"[-]", "•", text)  # Symbol/Wingdings bullets mapped to the private-use area
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?<=\S) +([.,:;!?])(?=\s|$)", r"\1", text)      # "איימה ." -> "איימה."
    text = re.sub(r"^(\d{1,2}) ?\.(?=\S)", r"\1. ", text)             # "1 .האדם" -> "1. האדם"
    return text


def hebrew_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "א" <= c <= "ת") / len(letters)


def reversed_score(text: str) -> float:
    """Fraction of Hebrew words that look reversed.

    Final letters (ך ם ן ף ץ) only appear at the END of a correctly ordered word. If they show up at the start
    of words much more often than at the end, the text was extracted in visual order.
    """
    start = end = 0
    for w in HEB_WORD_RE.findall(text):
        if w[0] in FINAL_LETTERS:
            start += 1
        if w[-1] in FINAL_LETTERS:
            end += 1
    total = start + end
    return start / total if total else 0.0


def is_reversed(text: str) -> bool:
    return hebrew_ratio(text) > 0.3 and reversed_score(text) > 0.6


def fix_reversed_line(line: str) -> str:
    """Reverse a visually ordered line back to logical order, keeping LTR runs (English, numbers) intact."""
    rev = line[::-1]
    rev = LTR_RUN_RE.sub(lambda m: m.group(0)[::-1], rev)
    # mirror brackets that got flipped by the reversal
    return rev.translate(str.maketrans("()[]{}<>", ")(][}{><"))
