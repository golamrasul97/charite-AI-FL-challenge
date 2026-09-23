"""Tests for src/deid.

The notes in the "unseen formats" section are hand-written and hand-labelled.
They do not come from the training data: they exist because train and validation
share the same six templates, so a perfect score on both says nothing about the
hidden test set, which the challenge states uses new formatting variants.
"""
from __future__ import annotations

import pytest

from evaluator.evaluate import PII_LABELS
from evaluator.evaluate import render_deidentified as evaluator_render
from src.deid import detect_pii, render_deidentified

BERLIN = (
    "SYNTHETIC BERLIN CLINICAL SUMMARY\n"
    "Patient: Tobias Berger | DOB: 25.05.1960 | Case ID: B-904427\n"
    "Address: Birkenpfad 37, 13353 Berlin | Telephone: +49 30 0000 2893\n"
    "Encounter date: 11.05.2025 | Attending physician: Dr. Emil Brandt\n"
    "Diagnoses: Vorhofflimmern (AF). LVEF 53%. Age/Sex: 65/M.\n"
    "Electronically signed by Dr. Emil Brandt; contact emil.brandt@synthetic-clinic.example."
)


def spans_with_text(note: str) -> set[tuple[str, str]]:
    return {(span["label"], note[span["start"] : span["end"]]) for span in detect_pii(note)}


# --------------------------------------------------------------------------- #
# Output contract
# --------------------------------------------------------------------------- #
def test_spans_are_in_bounds_sorted_and_non_overlapping() -> None:
    spans = detect_pii(BERLIN)
    assert spans, "expected at least one span"
    previous_end = -1
    for span in spans:
        assert span["label"] in PII_LABELS
        assert 0 <= span["start"] < span["end"] <= len(BERLIN)
        assert span["start"] >= previous_end, "spans must be sorted and non-overlapping"
        previous_end = span["end"]


def test_rendering_matches_the_evaluators_own_function() -> None:
    spans = detect_pii(BERLIN)
    assert render_deidentified(BERLIN, spans) == evaluator_render(BERLIN, spans)


def test_no_pii_text_survives_rendering() -> None:
    rendered = render_deidentified(BERLIN, detect_pii(BERLIN))
    for leaked in ("Tobias Berger", "Emil Brandt", "B-904427", "Birkenpfad", "25.05.1960"):
        assert leaked not in rendered


# --------------------------------------------------------------------------- #
# Span conventions
# --------------------------------------------------------------------------- #
def test_titles_are_excluded_from_clinician_spans() -> None:
    for label, text in spans_with_text(BERLIN):
        if label == "CLINICIAN_NAME":
            assert text == "Emil Brandt"
            assert not text.startswith("Dr")


def test_a_repeated_clinician_name_yields_one_span_per_occurrence() -> None:
    spans = [span for span in detect_pii(BERLIN) if span["label"] == "CLINICIAN_NAME"]
    assert len(spans) == 2


def test_clinical_values_and_banners_are_not_redacted() -> None:
    rendered = render_deidentified(BERLIN, detect_pii(BERLIN))
    assert "SYNTHETIC BERLIN CLINICAL SUMMARY" in rendered
    assert "LVEF 53%" in rendered
    assert "Age/Sex: 65/M" in rendered
    assert "Vorhofflimmern (AF)" in rendered


def test_dates_are_labelled_by_context() -> None:
    found = spans_with_text(BERLIN)
    assert ("DATE_OF_BIRTH", "25.05.1960") in found
    assert ("ENCOUNTER_DATE", "11.05.2025") in found


def test_phone_span_includes_the_country_code() -> None:
    assert ("PHONE_NUMBER", "+49 30 0000 2893") in spans_with_text(BERLIN)


def test_email_propagates_to_a_clinician_name_across_transliteration() -> None:
    """laura.koenig -> Laura König; 10 training notes depend on this."""
    note = "Findings reviewed. Contact laura.koenig@synthetic-clinic.example for queries.\nLaura König."
    assert ("CLINICIAN_NAME", "Laura König") in spans_with_text(note)


# --------------------------------------------------------------------------- #
# Unseen formats: hand-written, hand-labelled
# --------------------------------------------------------------------------- #
UNSEEN_UK = (
    "ROYAL INFIRMARY OUTPATIENT LETTER\n"
    "Patient name: Margaret O'Donnell   Hospital No: RI/2024/88431\n"
    "d.o.b. 03 Jan 1971   Date of admission: 2025-11-14\n"
    "Address: 12 Rosebery Crescent, Edinburgh EH12 5JP\n"
    "Tel: 0131 496 0821\n"
    "Seen by Dr Alan Whitfield, consultant cardiologist.\n"
    "Impression: stable angina. HR 68 bpm; BP 128/76 mmHg."
)
UNSEEN_UK_EXPECTED = {
    ("PATIENT_NAME", "Margaret O'Donnell"),
    ("PATIENT_ID", "RI/2024/88431"),
    ("DATE_OF_BIRTH", "03 Jan 1971"),
    ("ENCOUNTER_DATE", "2025-11-14"),
    ("ADDRESS", "12 Rosebery Crescent, Edinburgh EH12 5JP"),
    ("PHONE_NUMBER", "0131 496 0821"),
    ("CLINICIAN_NAME", "Alan Whitfield"),
}

UNSEEN_ISO = (
    "SYNTHETIC LISBON CLINICAL EXTRACT\n"
    "Pt Marta Fernandes | Reg No: LIS-88-01422 | d.o.b 1968-12-03\n"
    "Admitted 2026-02-17 | Ph 21 456 7890\n"
    "Resides at Rua das Flores 128, 1200-195 Lisboa\n"
    "Under care of Dr. Henrique Salgado\n"
    "Problem list: COPD; hypertension. Creatinine 1.10 mg/dL.\n"
    "Verified by Henrique Salgado (henrique.salgado@synthetic-clinic.example)."
)
UNSEEN_ISO_EXPECTED = {
    ("PATIENT_NAME", "Marta Fernandes"),
    ("PATIENT_ID", "LIS-88-01422"),
    ("DATE_OF_BIRTH", "1968-12-03"),
    ("ENCOUNTER_DATE", "2026-02-17"),
    ("ADDRESS", "Rua das Flores 128, 1200-195 Lisboa"),
    ("PHONE_NUMBER", "21 456 7890"),
    ("CLINICIAN_NAME", "Henrique Salgado"),
    ("EMAIL", "henrique.salgado@synthetic-clinic.example"),
}

UNSEEN_PROSE = (
    "SYNTHETIC FEDERATED NODE NOTE\n"
    "Annika Lindqvist / NOR-4471902 / born 1954-08-30 / seen 2026-03-05. "
    "Home: Storgata 44, 0182 Oslo. Mobile 912 34 567. "
    "Diagnoses [chronic kidney disease]. Measurements: pulse 74 per minute. "
    "Responsible doctor Bjorn Haugen, email bjorn.haugen@synthetic-hospital.example."
)
UNSEEN_PROSE_EXPECTED = {
    ("PATIENT_NAME", "Annika Lindqvist"),
    ("PATIENT_ID", "NOR-4471902"),
    ("DATE_OF_BIRTH", "1954-08-30"),
    ("ENCOUNTER_DATE", "2026-03-05"),
    ("ADDRESS", "Storgata 44, 0182 Oslo"),
    ("PHONE_NUMBER", "912 34 567"),
    ("CLINICIAN_NAME", "Bjorn Haugen"),
    ("EMAIL", "bjorn.haugen@synthetic-hospital.example"),
}


@pytest.mark.parametrize(
    ("note", "expected"),
    [
        pytest.param(UNSEEN_UK, UNSEEN_UK_EXPECTED, id="uk-letter"),
        pytest.param(UNSEEN_ISO, UNSEEN_ISO_EXPECTED, id="iso-dates"),
        pytest.param(UNSEEN_PROSE, UNSEEN_PROSE_EXPECTED, id="prose-slashes"),
    ],
)
def test_unseen_formats_are_fully_detected(note: str, expected: set[tuple[str, str]]) -> None:
    found = spans_with_text(note)
    assert expected <= found, f"missed: {sorted(expected - found)}"


@pytest.mark.parametrize("note", [UNSEEN_UK, UNSEEN_ISO, UNSEEN_PROSE])
def test_unseen_formats_keep_clinical_content(note: str) -> None:
    """Recall must not be bought with precision: clinical text stays."""
    rendered = render_deidentified(note, detect_pii(note))
    for kept in ("HR 68 bpm", "COPD", "chronic kidney disease", "pulse 74 per minute"):
        if kept in note:
            assert kept in rendered


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_detection_is_deterministic() -> None:
    assert detect_pii(BERLIN) == detect_pii(BERLIN)


def test_empty_note_is_handled() -> None:
    assert detect_pii("") == []


def test_a_title_alone_identifies_a_clinician() -> None:
    """No cue word, only a title -- a layout the training data never uses."""
    note = "Impression: stable. Dr. Sarah Klein reviewed the case on 04.02.2026."
    assert ("CLINICIAN_NAME", "Sarah Klein") in spans_with_text(note)
