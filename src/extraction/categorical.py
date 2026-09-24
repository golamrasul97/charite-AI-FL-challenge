"""Smoking status and drug allergy.

Both fields are single-valued. Every phrase is matched in the note and the
earliest surviving match wins, so the value reflects the first statement about
the patient. Mentions about relatives ("father is a current smoker") are
dropped with the same family cues as the diagnosis negation.

Smoking phrases are written to be mutually exclusive: "no tobacco use" (never)
and "ongoing tobacco use" (current) cannot both match the same text, and the
bare "smoker" fallback is blocked when "non-", "ex-", "former" or "never"
precedes it.
"""
from __future__ import annotations

import re

from .negation import is_about_relative

_I = re.IGNORECASE

SMOKING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("never", re.compile(
        r"\bnever[\s-]+(?:smoked|smoker|used\s+tobacco)\b|\bnon[\s-]?smoker\b|\blifelong\s+non\b"
        r"|\bno\s+(?:history\s+of\s+)?(?:tobacco|smoking)\b"
        r"|\b(?:denies|does\s+not|doesn't)\s+smok\w*|\bnichtraucher\b", _I)),
    ("former", re.compile(
        r"\bex[\s-]?smoker\b|\b(?:former|previous|past|prior)\s+smoker\b"
        r"|\b(?:stopped|quit|gave\s+up|ceased)\s+(?:smoking|tobacco)\b"
        r"|\bformer\s+tobacco\b|\bex[\s-]?raucher\b", _I)),
    ("current", re.compile(
        r"\bcurrent(?:ly)?\s+smok\w*|\bactive(?:ly)?\s+smok\w*"
        r"|\bongoing\s+(?:tobacco|smoking)\b|\bstill\s+smokes\b"
        r"|(?<!non-)(?<!non)(?<!ex-)(?<!ex\s)(?<!former\s)(?<!never\s)\bsmoker\b"
        r"|(?<!\w)smokes\b", _I)),
)

ALLERGY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("penicillin", re.compile(r"\b(?:penicillins?|amoxicillin|beta[\s-]?lactams?)\b", _I)),
    ("nsaid", re.compile(
        r"\bNSAIDs?\b|\bnon[\s-]?steroidal\s+anti[\s-]?inflammatory|\b(?:ibuprofen|diclofenac|naproxen)\b",
        _I)),
    ("iodinated_contrast", re.compile(
        r"\b(?:iodinated\s+)?contrast(?:\s+(?:medium|media|dye|agent))?\b|\biodine\b", _I)),
    ("none", re.compile(
        r"\bNKD?A\b|\bno\s+known\s+(?:drug\s+|medication\s+)?allerg\w*"
        r"|\bno\s+(?:drug|medication)\s+allerg\w*|\bno\s+allerg\w*"
        r"|\ballerg\w*\s*[:=]\s*(?:none|nil)\b", _I)),
)
# An allergen word only counts in an allergy context in the same sentence: this
# keeps "CT with contrast" or "penicillin was given" from becoming an allergy.
_ALLERGY_CONTEXT_RE = re.compile(
    r"allerg\w*|reaction|intoleran\w*|hypersensitiv\w*|anaphyla\w*"
    r"|\b(?:rash|hives|urticaria|angio[\s-]?o?edema|bronchospasm)\b",
    _I,
)
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)|\n")
_NEGATED_ALLERGEN_RE = re.compile(r"\bno\s+(?:known\s+)?$", _I)


def _sentence(note: str, start: int, end: int) -> str:
    begin = 0
    # Scan the whole note, not note[:start]: with an end position the "$" in the
    # lookahead would treat the mention's own start as the end of the text.
    for match in _SENTENCE_END_RE.finditer(note):
        if match.end() > start:
            break
        begin = match.end()
    after = _SENTENCE_END_RE.search(note, end)
    return note[begin : after.start() if after else len(note)]


def classify_smoking(note: str) -> str | None:
    hits: list[tuple[int, str]] = []
    for value, pattern in SMOKING_PATTERNS:
        for match in pattern.finditer(note):
            # Only the family cues apply: "no tobacco use" carries its own "no".
            if not is_about_relative(note, match.start(), match.end()):
                hits.append((match.start(), value))
    return min(hits)[1] if hits else None


def classify_allergy(note: str) -> str | None:
    hits: list[tuple[int, str]] = []
    for value, pattern in ALLERGY_PATTERNS:
        for match in pattern.finditer(note):
            if value != "none":
                if not _ALLERGY_CONTEXT_RE.search(_sentence(note, match.start(), match.end())):
                    continue
                if _NEGATED_ALLERGEN_RE.search(note[max(0, match.start() - 12) : match.start()]):
                    continue  # "no penicillin allergy"
            hits.append((match.start(), value))
    if not hits:
        return None
    # A specific allergen beats "none" ("no known drug allergies except penicillin").
    specific = [hit for hit in hits if hit[1] != "none"]
    return min(specific or hits)[1]
