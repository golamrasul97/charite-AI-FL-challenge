"""Surface-form lexicons for diagnoses and medications.

Every surface form seen in the 120 training notes is listed, plus generic
variants from general clinical usage (common abbreviations, British spellings,
brand names, German terms) so that unseen formats still map to the canonical
vocabulary in ``schemas/prediction.schema.json``.

Each canonical concept has two tuples of regex fragments:

* ``ci`` - matched case-insensitively (full terms, brand names);
* ``cs`` - matched case-sensitively (short upper-case abbreviations such as
  ``AF`` or ``ASA``, which would otherwise collide with ordinary words).

Every fragment is wrapped in alphanumeric boundaries, so ``CAD`` never matches
inside ``COAD`` and ``EF`` never matches inside ``HFrEF``. A space in a fragment
matches any run of whitespace or hyphens ("community-acquired pneumonia").
"""
from __future__ import annotations

import re
from typing import Iterator

DIAGNOSES: dict[str, dict[str, tuple[str, ...]]] = {
    "atrial_fibrillation": {
        "ci": ("atrial fibrillation", "a-?fib", "vorhofflimmern"),
        "cs": ("AF", "AFib"),
    },
    "heart_failure": {
        # "heart failure with reduced EF", "chronic/congestive cardiac failure",
        # "systolic heart failure" all contain one of the first three terms.
        "ci": (
            "heart failure",
            "cardiac failure",
            "(?:LV|left ventricular) failure",
            "HF[rpm]{1,2}EF",
            "herzinsuffizienz",
        ),
        "cs": ("CHF", "HF"),
    },
    "hypertension": {
        # Pulmonary/portal/intracranial/ocular hypertension are different diseases.
        "ci": (
            r"(?<!pulmonary\s)(?<!portal\s)(?<!intracranial\s)(?<!ocular\s)hypertension",
            "high blood pressure",
            "hypertensive (?:heart )?disease",
            "(?:arterielle )?hypertonie",
            "bluthochdruck",
        ),
        "cs": ("HTN",),
    },
    "type_2_diabetes": {
        # The training labels map bare "diabetes mellitus" to type 2, so bare
        # "diabetes" counts unless the note says type 1, gestational or insipidus.
        "ci": (
            "type (?:2|II) (?:DM|diabetes)",
            r"(?<!type\s1\s)(?<!type\sI\s)(?<!gestational\s)diabetes"
            r"(?! insipidus)(?!(?: mellitus)?,? type (?:1|I)(?![I\d]))",
            "NIDDM",
        ),
        "cs": ("T2DM", "DM2", r"DM\s?II", r"DM(?!\s?(?:1|I\b))(?!\s*type\s*(?:1|I\b))"),
    },
    "chronic_kidney_disease": {
        "ci": (
            "chronic (?:kidney|renal) (?:disease|failure|insufficiency|impairment|dysfunction)",
            "end stage (?:kidney|renal) disease",
            "(?:chronische )?niereninsuffizienz",
        ),
        "cs": ("CKD", "ESRD", "ESKD"),
    },
    "coronary_artery_disease": {
        "ci": (
            "coronary (?:artery |heart )?disease",
            "isch(?:a)?emic heart disease",
            "koronare herzkrankheit",
        ),
        "cs": ("CAD", "IHD", "KHK"),
    },
    "acute_coronary_syndrome": {
        "ci": (
            "acute coronary syndrome",
            "unstable angina",
            "acute (?:myocardial infarction|MI)",
        ),
        "cs": ("ACS", "NSTEMI", "STEMI"),
    },
    "pneumonia": {
        # \w* admits bronchopneumonia and pleuropneumonia.
        "ci": (r"\w*pneumonia", r"\w*pneumonie", "infective consolidation"),
        "cs": ("CAP",),
    },
    "copd": {
        "ci": (
            "chronic obstructive (?:pulmonary|airways?|lung) disease",
            "chronic airways? obstruction",
            "obstructive (?:lung|airways?) disease",
            "emphysema",
        ),
        # COAD must stay here: it is COPD, not coronary artery disease.
        "cs": ("COPD", "COAD"),
    },
}

MEDICATIONS: dict[str, dict[str, tuple[str, ...]]] = {
    "apixaban": {"ci": ("apixaban", "eliquis"), "cs": ("APX",)},
    "rivaroxaban": {"ci": ("rivaroxaban", "xarelto"), "cs": ()},
    # Phenprocoumon (Marcumar) is a different vitamin-K antagonist, but the
    # training labels map "phenprocoumon/warfarin therapy" to warfarin.
    "warfarin": {"ci": ("warfarin", "coumadin", "phenprocoumon", "marcumar"), "cs": ()},
    "metoprolol": {"ci": ("metoprolol", "lopressor", "toprol", "beloc"), "cs": ()},
    "bisoprolol": {"ci": ("bisoprolol", "concor", "cardicor"), "cs": ()},
    "furosemide": {"ci": ("furosemide?", "frusemide", "lasix"), "cs": ()},
    "ramipril": {"ci": ("ramipril", "tritace", "altace", "delix"), "cs": ()},
    "amlodipine": {"ci": ("amlodipine?", "norvasc"), "cs": ()},
    "metformin": {"ci": ("metformin", "glucophage", "glycomet"), "cs": ()},
    "insulin": {
        "ci": (
            r"insulin(?! resistance)(?! sensitivity)",
            "lantus",
            "levemir",
            "toujeo",
            "tresiba",
            "humalog",
            "novorapid",
        ),
        "cs": (),
    },
    "atorvastatin": {"ci": ("atorvastatin", "atorva", "lipitor", "sortis"), "cs": ()},
    "aspirin": {
        "ci": ("aspirin", "acetylsalicylic acid", "ecosprin"),
        # ASS is the German abbreviation (Acetylsalicylsäure).
        "cs": ("ASA", "ASS"),
    },
    "clopidogrel": {"ci": ("clopidogrel", "plavix"), "cs": ()},
    "amiodarone": {"ci": ("amiodarone?", "cordarone"), "cs": ()},
    "digoxin": {"ci": ("digoxin", "lanoxin"), "cs": ()},
    "azithromycin": {"ci": ("azithromycin", "azithro", "zithromax", "azithral"), "cs": ()},
}


def _wrap(fragment: str) -> str:
    """Add alphanumeric boundaries and make spaces match whitespace or hyphens."""
    body = fragment.replace(" ", r"[\s-]+")
    return rf"(?<![A-Za-z0-9])(?:{body})(?![A-Za-z0-9])"


def compile_lexicon(lexicon: dict[str, dict[str, tuple[str, ...]]]) -> list[tuple[str, re.Pattern[str]]]:
    """Compile a lexicon into ``(canonical, pattern)`` pairs in a fixed order."""
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for canonical in sorted(lexicon):
        forms = lexicon[canonical]
        for fragment in forms["ci"]:
            compiled.append((canonical, re.compile(_wrap(fragment), re.IGNORECASE)))
        for fragment in forms["cs"]:
            compiled.append((canonical, re.compile(_wrap(fragment))))
    return compiled


DIAGNOSIS_PATTERNS = compile_lexicon(DIAGNOSES)
MEDICATION_PATTERNS = compile_lexicon(MEDICATIONS)


def find_mentions(
    note: str, patterns: list[tuple[str, re.Pattern[str]]]
) -> Iterator[tuple[str, int, int]]:
    """Yield ``(canonical, start, end)`` for every lexicon hit in ``note``."""
    for canonical, pattern in patterns:
        for match in pattern.finditer(note):
            yield canonical, match.start(), match.end()
