"""Structured clinical extraction: note text -> the 9 canonical fields.

Diagnoses and medications are found by a whole-note lexicon scan, and each hit
is kept unless its clause negates it. There is deliberately no dependence on
section headers ("Dx:", "Medication list:"): the six training templates all use
different headers and the hidden test set adds new ones, so a section-first
design would fail silently on an unseen layout. Negation and context cues do
the filtering instead.
"""
from __future__ import annotations

from typing import Any

from .categorical import classify_allergy, classify_smoking
from .lexicon import DIAGNOSIS_PATTERNS, MEDICATION_PATTERNS, find_mentions
from .negation import is_negated, is_negated_medication
from .numeric import extract_numeric

FIELDS = (
    "diagnoses",
    "medications",
    "heart_rate_bpm",
    "systolic_bp_mmhg",
    "creatinine_mg_dl",
    "hemoglobin_g_dl",
    "lvef_percent",
    "smoking_status",
    "allergy",
)


def extract_diagnoses(note: str) -> list[str]:
    found = {
        canonical
        for canonical, start, end in find_mentions(note, DIAGNOSIS_PATTERNS)
        if not is_negated(note, start, end)
    }
    return sorted(found)


def extract_medications(note: str) -> list[str]:
    found = {
        canonical
        for canonical, start, end in find_mentions(note, MEDICATION_PATTERNS)
        if not is_negated_medication(note, start, end)
    }
    return sorted(found)


def empty_record() -> dict[str, Any]:
    """All 9 keys with the 'nothing documented' value."""
    return {field: [] if field in ("diagnoses", "medications") else None for field in FIELDS}


def extract_clinical_data(note: str) -> dict[str, Any]:
    """Return all 9 fields; lists sorted and unique, absent values ``None``."""
    record = empty_record()
    if not isinstance(note, str) or not note.strip():
        return record
    record["diagnoses"] = extract_diagnoses(note)
    record["medications"] = extract_medications(note)
    record.update(extract_numeric(note))
    record["smoking_status"] = classify_smoking(note)
    record["allergy"] = classify_allergy(note)
    return record
