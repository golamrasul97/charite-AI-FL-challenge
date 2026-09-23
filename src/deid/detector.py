"""Layered PII detection.

Each layer proposes candidate spans with a confidence priority; a single
resolution step then removes overlaps and produces the sorted, non-overlapping
list the evaluator expects. Layers are independent and individually testable:

    1. high-precision shapes      e-mail, phone, identifier, date
    2. context labelling          which date is a birth date, which is a visit
    3. cue-based names            "Attending physician: Dr. X" -> X
    4. propagation                every repeat of a known clinician name, and
                                  names recovered from an e-mail local part
    5. addresses                  cue-delimited, with shape fallbacks
    6. conflict resolution        priority, then overlap removal, then sort

Priorities encode how much evidence a span has, not how sensitive it is: a span
found by an explicit cue outranks one found by shape alone.
"""
from __future__ import annotations

import re
from typing import Any, NamedTuple

from . import patterns as pat


class Candidate(NamedTuple):
    start: int
    end: int
    label: str
    priority: int
    source: str


# Explicit cue beats bare shape; a longer span wins a tie at equal priority.
P_EMAIL = 100
P_PHONE = 95
P_ID_CUE = 90
P_DATE_CUE = 88
P_ADDRESS_CUE = 84
P_CLINICIAN_CUE = 80
P_PATIENT_CUE = 79
P_CLINICIAN_TITLE = 77
P_CLINICIAN_REPEAT = 76
P_PATIENT_ANCHOR = 74
P_CLINICIAN_EMAIL = 72
P_ID_SHAPE = 70
P_DATE_INFERRED = 66
P_ADDRESS_SHAPE = 62
P_PATIENT_BANNER = 58

UMLAUT_FOLDS = (("oe", "[oö]e?"), ("ue", "[uü]e?"), ("ae", "[aä]e?"), ("ss", "(?:ss|ß)"))


def _trim(note: str, start: int, end: int) -> tuple[int, int]:
    """Drop surrounding whitespace and trailing punctuation from a span."""
    while start < end and note[start].isspace():
        start += 1
    while end > start and (note[end - 1].isspace() or note[end - 1] in ",.;:|-"):
        end -= 1
    return start, end


# --------------------------------------------------------------------------- #
# Layer 1: high-precision shapes
# --------------------------------------------------------------------------- #
def _emails(note: str) -> list[Candidate]:
    return [
        Candidate(m.start(), m.end(), "EMAIL", P_EMAIL, "shape")
        for m in pat.EMAIL_RE.finditer(note)
    ]


def _phones(note: str) -> list[Candidate]:
    found = [
        Candidate(m.start(), m.end(), "PHONE_NUMBER", P_PHONE, "international")
        for m in pat.PHONE_INTERNATIONAL_RE.finditer(note)
    ]
    for match in pat.PHONE_LOCAL_RE.finditer(note):
        start, end = _trim(note, match.start(1), match.end(1))
        if sum(character.isdigit() for character in note[start:end]) >= 7:
            found.append(Candidate(start, end, "PHONE_NUMBER", P_PHONE - 5, "cue"))
    return found


def _identifiers(note: str) -> list[Candidate]:
    found: list[Candidate] = []
    for match in pat.ID_CUE_RE.finditer(note):
        start, end = _trim(note, match.start(1), match.end(1))
        if any(character.isdigit() for character in note[start:end]):
            found.append(Candidate(start, end, "PATIENT_ID", P_ID_CUE, "cue"))
    for match in pat.ID_SHAPE_RE.finditer(note):
        found.append(Candidate(match.start(), match.end(), "PATIENT_ID", P_ID_SHAPE, "shape"))
    return found


# --------------------------------------------------------------------------- #
# Layer 2: dates, labelled by context
# --------------------------------------------------------------------------- #
def _year_of(text: str) -> int:
    years = re.findall(r"\d{4}", text)
    return int(years[0]) if years else 0


def _dates(note: str) -> list[Candidate]:
    labelled: list[Candidate] = []
    unlabelled: list[tuple[int, int]] = []
    for match in pat.DATE_RE.finditer(note):
        preceding = note[max(0, match.start() - 30) : match.start()]
        if pat.DOB_CUE_RE.search(preceding):
            labelled.append(Candidate(match.start(), match.end(), "DATE_OF_BIRTH", P_DATE_CUE, "cue"))
        elif pat.ENCOUNTER_CUE_RE.search(preceding):
            labelled.append(Candidate(match.start(), match.end(), "ENCOUNTER_DATE", P_DATE_CUE, "cue"))
        else:
            unlabelled.append((match.start(), match.end()))

    if not unlabelled:
        return labelled

    # A date with no cue is a birth date only if the note does not already have
    # one and this is the earliest date present; otherwise it is the encounter
    # date (the header date in the "EMR EXPORT // id // date" layout).
    has_birth_date = any(item.label == "DATE_OF_BIRTH" for item in labelled)
    earliest = min(unlabelled, key=lambda span: (_year_of(note[span[0] : span[1]]), span[0]))
    for start, end in unlabelled:
        is_birth = not has_birth_date and (start, end) == earliest and len(unlabelled) > 1
        label = "DATE_OF_BIRTH" if is_birth else "ENCOUNTER_DATE"
        labelled.append(Candidate(start, end, label, P_DATE_INFERRED, "inferred"))
    return labelled


# --------------------------------------------------------------------------- #
# Layer 3: cue-based names
# --------------------------------------------------------------------------- #
def _patient_names(note: str) -> list[Candidate]:
    found = [
        Candidate(*_trim(note, m.start(1), m.end(1)), "PATIENT_NAME", P_PATIENT_CUE, "cue")
        for m in pat.PATIENT_NAME_CUE_RE.finditer(note)
    ]
    for pattern in pat.PATIENT_NAME_ANCHOR_RES:
        for match in pattern.finditer(note):
            start, end = _trim(note, match.start(1), match.end(1))
            found.append(Candidate(start, end, "PATIENT_NAME", P_PATIENT_ANCHOR, "anchor"))

    # Last resort: the line directly under an all-capitals banner often opens
    # with the patient's name and no cue at all.
    lines = note.split("\n")
    if len(lines) > 1 and pat.BANNER_RE.match(lines[0].strip()):
        offset = len(lines[0]) + 1
        match = re.match(rf"({pat.NAME})", lines[1])
        if match:
            start, end = _trim(note, offset + match.start(1), offset + match.end(1))
            found.append(Candidate(start, end, "PATIENT_NAME", P_PATIENT_BANNER, "banner"))
    return found


def _clinician_names(note: str) -> list[Candidate]:
    found = [
        Candidate(*_trim(note, m.start(1), m.end(1)), "CLINICIAN_NAME", P_CLINICIAN_CUE, "cue")
        for m in pat.CLINICIAN_CUE_RE.finditer(note)
    ]
    found += [
        Candidate(*_trim(note, m.start(1), m.end(1)), "CLINICIAN_NAME", P_CLINICIAN_TITLE, "title")
        for m in pat.TITLE_NAME_RE.finditer(note)
    ]
    return found


# --------------------------------------------------------------------------- #
# Layer 4: propagation
# --------------------------------------------------------------------------- #
def _repeats(note: str, known: list[Candidate]) -> list[Candidate]:
    """Label every other occurrence of an already-detected clinician name.

    In the training labels a clinician name is labelled at every occurrence
    (181 spans across 120 notes), and the number of occurrences of the string
    always equals the number of labelled spans, so this never over-labels.
    """
    found: list[Candidate] = []
    for name in sorted({note[item.start : item.end] for item in known}):
        for match in re.finditer(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", note):
            found.append(
                Candidate(match.start(), match.end(), "CLINICIAN_NAME", P_CLINICIAN_REPEAT, "repeat")
            )
    return found


def _name_from_email(local_part: str) -> str | None:
    """Turn "philipp.krueger" into a regex matching "Philipp Krüger".

    German names are transliterated in e-mail addresses (ö->oe, ü->ue), which
    accounts for 10 of the 120 training notes, so the search has to fold them
    back.
    """
    words = [word for word in re.split(r"[._-]+", local_part) if len(word) > 1]
    if len(words) < 2:
        return None
    parts: list[str] = []
    for word in words:
        pattern = re.escape(word)
        for plain, folded in UMLAUT_FOLDS:
            pattern = pattern.replace(plain, folded)
        parts.append(pattern)
    return r"\b" + r"[ \t]+".join(parts) + r"\b"


def _clinicians_from_emails(note: str, emails: list[Candidate]) -> list[Candidate]:
    found: list[Candidate] = []
    for email in emails:
        local_part = note[email.start : email.end].split("@")[0]
        pattern = _name_from_email(local_part)
        if not pattern:
            continue
        for match in re.finditer(pattern, note, re.IGNORECASE):
            if match.start() >= email.start and match.end() <= email.end:
                continue  # the address itself
            found.append(
                Candidate(match.start(), match.end(), "CLINICIAN_NAME", P_CLINICIAN_EMAIL, "email")
            )
    return found


# --------------------------------------------------------------------------- #
# Layer 5: addresses
# --------------------------------------------------------------------------- #
def _addresses(note: str) -> list[Candidate]:
    found: list[Candidate] = []
    for match in pat.ADDRESS_CUE_RE.finditer(note):
        start = match.end()
        window = note[start : start + pat.ADDRESS_MAX_CHARS]
        terminator = pat.ADDRESS_END_RE.search(window)
        end = start + (terminator.start() if terminator else len(window))
        start, end = _trim(note, start, end)
        if end - start >= 8 and any(character.isdigit() for character in note[start:end]):
            found.append(Candidate(start, end, "ADDRESS", P_ADDRESS_CUE, "cue"))
    for pattern in pat.ADDRESS_FALLBACK_RES:
        for match in pattern.finditer(note):
            start, end = _trim(note, match.start(), match.end())
            found.append(Candidate(start, end, "ADDRESS", P_ADDRESS_SHAPE, "shape"))
    return found


# --------------------------------------------------------------------------- #
# Layer 6: conflict resolution
# --------------------------------------------------------------------------- #
def resolve(candidates: list[Candidate]) -> list[Candidate]:
    """Keep the best non-overlapping subset, deterministically.

    Sorted by priority, then by length, then by position so that the result
    never depends on the order layers happened to run in.
    """
    ordered = sorted(candidates, key=lambda c: (-c.priority, -(c.end - c.start), c.start, c.label))
    accepted: list[Candidate] = []
    for candidate in ordered:
        if candidate.end <= candidate.start:
            continue
        if any(candidate.start < other.end and candidate.end > other.start for other in accepted):
            continue
        accepted.append(candidate)
    return sorted(accepted, key=lambda c: (c.start, c.end))


def detect(note: str) -> list[Candidate]:
    """All detected spans, with the provenance of each, for debugging."""
    emails = _emails(note)
    clinicians = _clinician_names(note)
    candidates = [
        *emails,
        *_phones(note),
        *_identifiers(note),
        *_dates(note),
        *_addresses(note),
        *_patient_names(note),
        *clinicians,
        *_repeats(note, clinicians),
        *_clinicians_from_emails(note, emails),
    ]
    return resolve(candidates)


def detect_pii(note: str) -> list[dict[str, Any]]:
    """The public interface: sorted, non-overlapping spans for the evaluator."""
    return [
        {"start": span.start, "end": span.end, "label": span.label} for span in detect(note)
    ]
