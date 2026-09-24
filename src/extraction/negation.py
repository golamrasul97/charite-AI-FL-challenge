"""Clause-scoped negation and context filtering (a small NegEx-style rule set).

A mention is rejected when a cue sits in the same *clause* and no scope
terminator ("but", "however", ...) separates the cue from the mention.

A clause ends at a sentence end, ``;``, ``|`` or a newline. Colons, dashes and
brackets do not end a clause, so the cue in "Family history: HTN" still covers
the HTN. Post-cues additionally stop at a comma (see ``_post_scope``). The small scope keeps "No known drug allergies. Diagnoses: HTN." from
cancelling the hypertension, and "Dx: denies COPD; HTN" keeps the HTN.

Cues come in three kinds:

* pre-cues must come *before* the mention ("denies a history of COPD");
* post-cues must come *after* it ("AF was considered but not confirmed");
* anywhere-cues reject the mention from either side ("mother had AF").

Every cue below except the relatives and the medication-specific ones occurs in
the training notes; the rest come from general clinical phrasing.
"""
from __future__ import annotations

import re

# "b.i.d." and "0.58" are not sentence ends: a full stop only ends a clause
# when whitespace (or the end of the note) follows it.
_CLAUSE_END_RE = re.compile(r"[;|\n]|[.!?](?=\s|$)")
_TERMINATOR_RE = re.compile(r"\b(?:but|however|although|though|except|yet)\b", re.IGNORECASE)


def _cue(*phrases: str) -> re.Pattern[str]:
    body = "|".join(p.replace(" ", r"\s+") for p in phrases)
    return re.compile(rf"\b(?:{body})\b", re.IGNORECASE)


_FAMILY = _cue(
    "family history", "FHx", "mother", "father", "brother", "sister", "sibling",
    "parents?", "relatives?", "grandmother", "grandfather", "aunt", "uncle",
)
_COMMON_PRE = (
    "no", "not", "denies", "denied", "without", "negative for", "free of",
    "no evidence of", "ruled out", "rule out", "r/o", "excluded", "absence of",
)

# Considered-only concepts are not active diagnoses (DATA_DICTIONARY.md), so
# "suspected ACS" and "?AF" are dropped like negations.
# A "?" directly before the term ("?AF") is the usual shorthand for "query".
DIAGNOSIS_PRE = re.compile(
    _cue(*_COMMON_PRE, "suspected", "possible", "query").pattern + r"|\?\s*$", re.IGNORECASE
)
DIAGNOSIS_POST = _cue(
    "not confirmed", "was considered", "considered", "ruled out", "excluded",
    "unlikely", "was not", "is not", "resolved", "not present", "negative", "neg",
)
MEDICATION_PRE = _cue(
    *_COMMON_PRE, "stopped", "discontinued", "held", "withheld", "ceased",
    "previously on", "declined", "avoid", r"allerg\w*", r"intoleran\w*", r"hypersensitiv\w*",
)
MEDICATION_POST = _cue(
    "discussed", "not started", "was not", "is not", "discontinued", "stopped",
    "held", "withheld", "ceased", "declined", "planned", "to be started",
    "will be started", "considered", "not tolerated", r"allerg\w*", r"intoleran\w*",
)


def clause_bounds(note: str, start: int, end: int) -> tuple[int, int]:
    """Return the ``[begin, finish)`` offsets of the clause around a mention."""
    begin = 0
    # Scan the whole note, not note[:start]: with an end position the "$" in the
    # lookahead would treat the mention's own start as the end of the text.
    for match in _CLAUSE_END_RE.finditer(note):
        if match.end() > start:
            break
        begin = match.end()
    after = _CLAUSE_END_RE.search(note, end)
    finish = after.start() if after else len(note)
    return begin, finish


def _cue_in_scope(text: str, cue: re.Pattern[str], before: bool) -> bool:
    """True when ``cue`` occurs in ``text`` with no terminator towards the mention.

    ``text`` is the part of the clause before the mention (``before=True``) or
    after it (``before=False``).
    """
    for match in cue.finditer(text):
        between = text[match.end():] if before else text[: match.start()]
        if not _TERMINATOR_RE.search(between):
            return True
    return False


def is_negated(
    note: str,
    start: int,
    end: int,
    pre: re.Pattern[str] = DIAGNOSIS_PRE,
    post: re.Pattern[str] = DIAGNOSIS_POST,
) -> bool:
    """True when the mention ``note[start:end]`` is negated or not about the patient."""
    begin, finish = clause_bounds(note, start, end)
    left, right = note[begin:start], _post_scope(note[end:finish])
    return (
        _cue_in_scope(left, pre, before=True)
        or _cue_in_scope(right, post, before=False)
        or is_about_relative(note, start, end)
    )


def _post_scope(right: str) -> str:
    """A post-cue only covers the list item it follows, so it stops at a comma.

    In "HTN, T2DM, AF ruled out" the "ruled out" belongs to AF alone. A pre-cue
    keeps the whole clause ("no evidence of COPD, CKD" negates both).
    """
    return right.split(",", 1)[0]


def is_about_relative(note: str, start: int, end: int) -> bool:
    """True when the clause attributes the mention to a family member."""
    begin, finish = clause_bounds(note, start, end)
    return _cue_in_scope(note[begin:start], _FAMILY, before=True) or _cue_in_scope(
        _post_scope(note[end:finish]), _FAMILY, before=False
    )


def is_negated_medication(note: str, start: int, end: int) -> bool:
    """Medication variant: also rejects stopped, planned and allergy mentions."""
    return is_negated(note, start, end, pre=MEDICATION_PRE, post=MEDICATION_POST)
