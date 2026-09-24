"""Vital signs, laboratory values and LVEF, normalised to the standard units.

Each field is found by a cue ("HR", "serum creatinine", "LVEF", ...), a short
gap of words (no digits, no clause punctuation), then the number and an optional
unit. The first plausible value in the note wins. Values outside a physiological
range are logged and ignored rather than returned.

Unit handling follows ``DATA_DICTIONARY.md``: creatinine 1 mg/dL = 88.4 µmol/L,
haemoglobin 1 g/dL = 10 g/L. Where the unit is missing it is inferred from the
magnitude (creatinine above 20 cannot be mg/dL; haemoglobin above 25 cannot be
g/dL); that rule never fires on the training data, which always gives a unit.
"""
from __future__ import annotations

import logging
import re
from typing import Callable

logger = logging.getLogger(__name__)

NUMBER = r"(\d{1,4}(?:[.,]\d+)?)"
# Words between cue and number: "approximately", "=", "of", "measured at". The
# gap cannot contain digits or ; . , | so it never crosses into the next field.
_GAP = r"[A-Za-z \t:=~()]{0,30}?"
# A gap containing another field's cue or a negation belongs to something else:
# "pulse and BP 120/80" must not give HR 120, "EF not documented" gives nothing.
_FOREIGN_IN_GAP_RE = re.compile(
    r"\b(?:BP|blood|pressure|HR|heart|pulse|rate|creatinine|Hb|ha?emoglobin|EF|ejection|"
    r"not|no|unknown|pending)\b",
    re.IGNORECASE,
)

PLAUSIBLE = {
    "heart_rate_bpm": (20.0, 250.0),
    "systolic_bp_mmhg": (50.0, 260.0),
    "creatinine_mg_dl": (0.1, 15.0),
    "hemoglobin_g_dl": (3.0, 25.0),
    "lvef_percent": (5.0, 85.0),
}

HEART_RATE_RE = re.compile(
    rf"(?<![A-Za-z])(?:HR|heart\s+rate|pulse(?:\s+rate)?|ventricular\s+rate)(?![A-Za-z])({_GAP}){NUMBER}",
    re.IGNORECASE,
)
# "BP 110/61 mmHg", "blood pressure 140 over 86", "systolic BP 132".
BLOOD_PRESSURE_RE = re.compile(
    rf"(?<![A-Za-z])(?:BP|blood\s+pressure|NIBP)(?![A-Za-z])({_GAP}){NUMBER}\s*(?:/|over)\s*\d{{2,3}}",
    re.IGNORECASE,
)
SYSTOLIC_RE = re.compile(
    rf"(?<![A-Za-z])(?:SBP|systolic(?:\s+(?:BP|blood\s+pressure))?)(?![A-Za-z])({_GAP}){NUMBER}",
    re.IGNORECASE,
)
# "creatine kinase" and "creatinine clearance" are different tests.
CREATININE_RE = re.compile(
    rf"(?<![A-Za-z])(?:creatinine|creat|s?Cr|kreatinin)(?![A-Za-z])(?!\s+clearance)({_GAP}){NUMBER}"
    r"\s*(mg\s*/\s*dl|mg\s*/\s*100\s*ml|[µμu]mol\s*/\s*l|micromol\s*/\s*l|mmol\s*/\s*l)?",
    re.IGNORECASE,
)
# \bHb\b excludes HbA1c; "haemoglobin A1c" is excluded explicitly.
HEMOGLOBIN_RE = re.compile(
    rf"(?<![A-Za-z])(?:ha?emoglobin|Hb|Hgb)(?![A-Za-z])(?!\s*A1c)({_GAP}){NUMBER}"
    r"\s*(g\s*/\s*dl|g\s*/\s*l|mmol\s*/\s*l)?",
    re.IGNORECASE,
)
# (?<![A-Za-z])EF never matches the EF inside "HFrEF". A range "35-40%" gives its midpoint.
LVEF_RE = re.compile(
    rf"(?<![A-Za-z])(?:LVEF|EF|ejection\s+fraction)(?![A-Za-z])({_GAP}){NUMBER}"
    r"(?:\s*(?:-|–|to)\s*(\d{1,2}))?\s*(?:%|per\s*cent|percent)?",
    re.IGNORECASE,
)


def _to_float(text: str) -> float:
    return float(text.replace(",", "."))  # decimal comma: "1,04 mg/dL"


def _plain(match: re.Match[str]) -> float:
    return _to_float(match.group(2))


def _first_plausible(
    note: str,
    pattern: re.Pattern[str],
    field: str,
    convert: Callable[[re.Match[str]], float],
) -> float | None:
    low, high = PLAUSIBLE[field]
    for match in pattern.finditer(note):
        if _FOREIGN_IN_GAP_RE.search(match.group(1)):
            continue
        value = convert(match)
        if low <= value <= high:
            return value
        logger.warning("%s: implausible value %r ignored (%r)", field, value, match.group(0))
    return None


def _creatinine_mg_dl(match: re.Match[str]) -> float:
    value, unit = _to_float(match.group(2)), (match.group(3) or "").lower()
    if unit.startswith("mmol"):
        return value * 1000.0 / 88.4
    if "mol" in unit or (not unit and value > 20.0):
        return value / 88.4
    return value


def _hemoglobin_g_dl(match: re.Match[str]) -> float:
    value, unit = _to_float(match.group(2)), (match.group(3) or "").lower().replace(" ", "")
    if unit.startswith("mmol"):
        return value * 1.611  # 1 mmol/L (German Hb convention) = 1.611 g/dL
    if unit == "g/l" or (not unit and value > 25.0):
        return value / 10.0
    return value


def _lvef(match: re.Match[str]) -> float:
    low = _to_float(match.group(2))
    return (low + float(match.group(3))) / 2.0 if match.group(3) else low


def extract_numeric(note: str) -> dict[str, int | float | None]:
    """Return the five numeric fields; ``None`` when absent or implausible."""
    heart_rate = _first_plausible(note, HEART_RATE_RE, "heart_rate_bpm", _plain)
    systolic = _first_plausible(note, BLOOD_PRESSURE_RE, "systolic_bp_mmhg", _plain)
    if systolic is None:
        systolic = _first_plausible(note, SYSTOLIC_RE, "systolic_bp_mmhg", _plain)
    creatinine = _first_plausible(note, CREATININE_RE, "creatinine_mg_dl", _creatinine_mg_dl)
    hemoglobin = _first_plausible(note, HEMOGLOBIN_RE, "hemoglobin_g_dl", _hemoglobin_g_dl)
    lvef = _first_plausible(note, LVEF_RE, "lvef_percent", _lvef)

    return {
        "heart_rate_bpm": int(round(heart_rate)) if heart_rate is not None else None,
        "systolic_bp_mmhg": int(round(systolic)) if systolic is not None else None,
        "creatinine_mg_dl": round(creatinine, 2) if creatinine is not None else None,
        "hemoglobin_g_dl": round(hemoglobin, 2) if hemoglobin is not None else None,
        "lvef_percent": int(round(lvef)) if lvef is not None else None,
    }
