"""Regular expressions and cue vocabularies for de-identification.

Everything here is derived from the 120 training notes plus general clinical
document conventions. Patterns are deliberately a little wider than the training
data (extra date shapes, ID shapes and cue words that do not occur in train) so
that the unseen formatting variants in the hidden test set still match; the
narrow, high-precision work is done by the cue vocabularies rather than by the
shapes themselves.
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #
# Every name in the training labels is exactly two capitalised words, but three
# is allowed here for middle names and particles in unseen formats. German
# umlauts appear in the data (König, Krüger); the transliterated spellings
# (koenig, krueger) appear in e-mail local parts.
# The base word must contain lower-case letters, so that an all-capitals banner
# line ("SYNTHETIC BERLIN CLINICAL SUMMARY") can never parse as a name. The
# second alternative admits O'Donnell and McArthur-style capitals.
_NAME_WORD = (
    r"[A-ZÄÖÜ](?:[a-zäöüßé]+|['’][A-ZÄÖÜ][a-zäöüß]+)"
    r"(?:[-'’][A-ZÄÖÜa-zäöüß][a-zäöüß]+)*"
)
# A space, never \s: a newline ends the name, otherwise "Deepa Naidu\nDx" parses
# as a three-word name and every boundary is wrong.
NAME = rf"{_NAME_WORD}(?:[ \t]{_NAME_WORD}){{1,2}}"

# Titles sit outside the labelled span: "Attending physician: Dr. Emil Brandt"
# labels "Emil Brandt" only. Both "Dr." and "Dr" occur in the training data.
TITLE = r"(?:Dr|Prof|Mr|Ms|Mrs|Sri|Smt)\.?\s+"

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# --------------------------------------------------------------------------- #
# Phone numbers
# --------------------------------------------------------------------------- #
# Training data is always international (+49 30 0000 8420, +91 00000 96466).
# The cue-anchored form additionally catches national numbers, which the brief
# lists as an expected unseen variant.
PHONE_INTERNATIONAL_RE = re.compile(r"\+\d{1,3}(?:[ -]?\(?\d{2,6}\)?){1,4}[ -]?\d{2,6}")
PHONE_CUES = ("phone", "telephone", "mobile", "contact", "ph", "tel", "cell", "mob")
PHONE_LOCAL_RE = re.compile(
    r"\b(?:" + "|".join(PHONE_CUES) + r")\b\s*[:.\-]?\s*(\(?\d[\d ()/-]{7,16}\d)",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #
_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
DATE_RE = re.compile(
    r"\b(?:"
    r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}"          # 14.02.2026, 16/08/1976
    rf"|\d{{1,2}}[ -]{_MONTH}[ -]\d{{2,4}}"      # 25-May-1969, 03 Jan 1971
    rf"|{_MONTH}\.?[ -]\d{{1,2}},?[ -]\d{{4}}"   # Jan 3, 1971
    r"|\d{4}-\d{2}-\d{2}"                        # ISO 1971-01-03
    r")\b"
)
DOB_CUE_RE = re.compile(r"(?:d\.?o\.?b\.?|born|date of birth|geb\.?|birth\s*date)[^A-Za-z0-9]{0,4}$", re.IGNORECASE)
ENCOUNTER_CUE_RE = re.compile(
    r"(?:encounter(?:\s*date)?|admission|admitted|doa|visit(?:\s*date)?|seen(?: on)?|"
    r"date of admission|presented|attendance)[^A-Za-z0-9]{0,4}$",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------- #
# Patient identifiers
# --------------------------------------------------------------------------- #
ID_CUES = (
    "mrn", "uhid", "case id", "patient id", "hospital no", "hospital number",
    "record no", "record number", "reg no", "registration no", "ip no", "ipd no",
    "op no", "nhs no", "file no", "chart no", "patient no", "id",
)
ID_CUE_RE = re.compile(
    r"\b(?:" + "|".join(cue.replace(" ", r"\s+") for cue in ID_CUES) + r")\b\s*[:#.\-]?\s*"
    r"([A-Z][A-Z0-9]*(?:[-/][A-Z0-9]+)*\d[A-Z0-9-/]*|\d{5,12})",
    re.IGNORECASE,
)
# Shapes seen in training (B-406380, HYD371017, CHN-3601789) generalised to any
# short alphabetic prefix followed by four or more digits, with optional
# separators, so a new site prefix still matches.
ID_SHAPE_RE = re.compile(
    r"\b(?:"
    r"[A-Z]{1,5}[-/]?\d{4,10}"          # B-406380, HYD371017, CHN-3601789
    r"|[A-Z]{2,5}[-/]\d{2,6}[-/]\d{2,6}"  # BER/2026/12, MRN-12-34567
    r")\b"
)

# --------------------------------------------------------------------------- #
# Addresses
# --------------------------------------------------------------------------- #
ADDRESS_CUE_RE = re.compile(
    r"(?:\b(?:address|residence|resides at|residing at|home|domicile|lives at)\b\s*[:\-]?\s*"
    r"|\)\s*from\s+)",
    re.IGNORECASE,
)
# An address runs to the next structural delimiter: pipe, newline, semicolon, or
# a full stop followed by a new clause. Verified against all 120 training spans.
ADDRESS_END_RE = re.compile(r"\s*(?:\||\n|;|\.\s|$)")
ADDRESS_MAX_CHARS = 90
# No-cue fallbacks: German "<street> <number>, <postcode> <city>" and Indian
# "<flat/house ...>, <city> <6-digit PIN>".
ADDRESS_FALLBACK_RES = (
    re.compile(rf"{_NAME_WORD}(?:[ \t]{_NAME_WORD})*[ \t]\d{{1,4}}[a-z]?,\s*\d{{5}}\s+{_NAME_WORD}"),
    re.compile(r"(?:Flat|Plot|House|Door|No\.?)\s*[^|\n;]{5,70},\s*[A-ZÄÖÜ][a-zA-Z]+\s+\d{6}\b"),
)

# --------------------------------------------------------------------------- #
# Name cues
# --------------------------------------------------------------------------- #
# Case-sensitive on purpose: "The patient denies a history of COPD" must not
# match, and a capitalised cue is what actually precedes a name in the data.
PATIENT_NAME_CUE_RE = re.compile(rf"\b(?:Patient(?:\s*name)?|Name|Pt|Pat)\b\s*[:\-]?\s+(?:{TITLE})?({NAME})")

CLINICIAN_CUES = (
    "attending physician", "attending", "consultant", "treating clinician",
    "treating physician", "responsible doctor", "responsible clinician",
    "electronically signed by", "signed by", "signed", "reviewed by", "author",
    "physician", "clinician", "seen by", "under care of", "dictated by",
    "verified by", "reported by", "referring doctor", "referred by",
)
CLINICIAN_CUE_RE = re.compile(
    r"\b(?:" + "|".join(cue.replace(" ", r"\s+") for cue in CLINICIAN_CUES) + r")\b"
    rf"\s*[:\-]?\s*(?:{TITLE})?({NAME})",
    re.IGNORECASE,
)
# A title alone identifies a clinician even with no cue word ("Dr. Sarah Klein
# reviewed the case."), which no training note needs but an unseen layout may.
TITLE_NAME_RE = re.compile(rf"\b{TITLE}({NAME})")

# "Nandini Chowdhury / HYD841642", "Clara Scholz, born 23.06.1976",
# "Pt Lakshmi Krishnan (DOB ...)": the name is identified by what follows it.
PATIENT_NAME_ANCHOR_RES = (
    re.compile(rf"^({NAME})\s*(?=,\s*(?:born|geb|d\.?o\.?b))", re.MULTILINE | re.IGNORECASE),
    re.compile(rf"^({NAME})\s*(?=/\s*[A-Z0-9-]*\d)", re.MULTILINE),
    re.compile(rf"\b({NAME})\s*(?=\(\s*(?:d\.?o\.?b|born))", re.IGNORECASE),
)
# A banner line such as "SYNTHETIC BERLIN CLINICAL SUMMARY" is not PII, and the
# name often opens the line directly below it.
BANNER_RE = re.compile(r"^[A-Z][A-Z0-9 /|.,#-]{6,}$")
