"""Tests for src/extraction.

The notes in the "unseen formats" section are hand-written and hand-labelled.
They do not come from the training data: train and validation share the same
six templates, so a perfect score on both says little about the hidden test
set, which the challenge states uses new formatting variants.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluator.evaluate import NUMERIC_TOLERANCES, score_extraction
from src.extraction import FIELDS, extract_clinical_data

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "schemas" / "prediction.schema.json").read_text(encoding="utf-8"))


def dx(note: str) -> list[str]:
    return extract_clinical_data(note)["diagnoses"]


def rx(note: str) -> list[str]:
    return extract_clinical_data(note)["medications"]


def _enum(field: str) -> set[str]:
    """Canonical vocabulary for a field, read from the published schema."""
    def find(node: object) -> object:
        if isinstance(node, dict):
            if field in node.get("properties", {}):
                return node["properties"][field]
            for value in node.values():
                hit = find(value)
                if hit is not None:
                    return hit
        return None

    spec = find(SCHEMA)
    assert isinstance(spec, dict)
    return set(spec["items"]["enum"] if "items" in spec else spec["enum"]) - {None}


# --------------------------------------------------------------------------- #
# Output contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("note", ["", "   ", "SYNTHETIC HEADER ONLY", "Medication: none documented."])
def test_all_nine_keys_present_and_no_empty_strings(note: str) -> None:
    record = extract_clinical_data(note)
    assert tuple(record) == FIELDS
    assert all(value != "" for value in record.values())
    assert record["diagnoses"] == [] and record["medications"] == []


def test_lists_are_sorted_unique_and_canonical() -> None:
    note = "Dx: HTN; hypertension; AF; atrial fibrillation. Rx: Lasix; furosemide; Eliquis."
    record = extract_clinical_data(note)
    assert record["diagnoses"] == ["atrial_fibrillation", "hypertension"]
    assert record["medications"] == ["apixaban", "furosemide"]
    assert set(record["diagnoses"]) <= _enum("diagnoses")
    assert set(record["medications"]) <= _enum("medications")


def test_lexicon_only_emits_schema_vocabulary() -> None:
    from src.extraction.lexicon import DIAGNOSES, MEDICATIONS

    assert set(DIAGNOSES) == _enum("diagnoses")
    assert set(MEDICATIONS) == _enum("medications")


# --------------------------------------------------------------------------- #
# Diagnosis surface forms
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Vorhofflimmern (AF)", "atrial_fibrillation"),
        ("AF with variable ventricular response", "atrial_fibrillation"),
        ("HFrEF", "heart_failure"),
        ("congestive cardiac failure", "heart_failure"),
        ("LV failure", "heart_failure"),
        ("high blood pressure", "hypertension"),
        ("hypertensive disease", "hypertension"),
        ("type II diabetes", "type_2_diabetes"),
        ("DM2", "type_2_diabetes"),
        ("diabetes mellitus", "type_2_diabetes"),
        ("DM type 2", "type_2_diabetes"),
        ("CKD-3", "chronic_kidney_disease"),
        ("chronic renal dysfunction", "chronic_kidney_disease"),
        ("IHD", "coronary_artery_disease"),
        ("stable ischemic heart disease", "coronary_artery_disease"),
        ("NSTEMI", "acute_coronary_syndrome"),
        ("bronchopneumonia", "pneumonia"),
        ("community-acquired pneumonia", "pneumonia"),
        ("infective consolidation", "pneumonia"),
        ("chronic airway obstruction", "copd"),
    ],
)
def test_diagnosis_surface_forms(text: str, expected: str) -> None:
    assert dx(f"Diagnoses: {text}.") == [expected]


def test_coad_is_copd_not_coronary_artery_disease() -> None:
    assert dx("Problem list: COAD.") == ["copd"]


@pytest.mark.parametrize(
    "text",
    ["pulmonary hypertension", "type 1 diabetes", "diabetes mellitus type 1", "DM1", "diabetes insipidus"],
)
def test_lookalike_diseases_are_not_mapped(text: str) -> None:
    assert dx(f"Diagnoses: {text}.") == []


# --------------------------------------------------------------------------- #
# Negation: one test per cue, plus positive controls
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "sentence",
    [
        "The patient denies a history of COPD.",
        "No COPD.",
        "No evidence of COPD.",
        "COPD was considered but not confirmed.",
        "COPD ruled out.",
        "COPD excluded on spirometry.",
        "Family history includes COPD.",
        "Mother had COPD.",
        "No clinical features of COPD.",
        "Suspected COPD.",
        "?COPD.",
    ],
)
def test_negated_diagnosis_is_dropped(sentence: str) -> None:
    assert dx(f"Diagnoses: HTN. {sentence}") == ["hypertension"]


@pytest.mark.parametrize(
    "note",
    [
        "No known drug allergies. Diagnoses: COPD.",   # negation stops at the sentence end
        "Dx: denies chest pain; COPD.",                # ... and at a semicolon
        "Pt denies chest pain but has known COPD.",    # "but" ends the negation scope
        "Known COPD.",
    ],
)
def test_positive_controls_keep_the_diagnosis(note: str) -> None:
    assert dx(note) == ["copd"]


def test_post_cue_only_covers_the_list_item_it_follows() -> None:
    assert dx("PMH: HTN, COPD, AF ruled out.") == ["copd", "hypertension"]


@pytest.mark.parametrize(
    "sentence",
    [
        "Apixaban was discussed but was not started.",
        "Apixaban discontinued.",
        "Apixaban stopped last week.",
        "Allergic to apixaban.",
        "No apixaban.",
    ],
)
def test_negated_medication_is_dropped(sentence: str) -> None:
    assert rx(f"Rx: ramipril. {sentence}") == ["ramipril"]


def test_stopped_drug_does_not_cancel_the_new_one() -> None:
    assert rx("Warfarin discontinued; now on apixaban 5 mg bd.") == ["apixaban"]


# --------------------------------------------------------------------------- #
# Medications: brand -> generic, prefixes and doses
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Tab Eliquis 5 mg BD", "apixaban"),
        ("APX 5 mg BID", "apixaban"),
        ("Xarelto 20 OD", "rivaroxaban"),
        ("Lasix 40 mg", "furosemide"),
        ("frusemide", "furosemide"),
        ("ASA 75 mg", "aspirin"),
        ("ecosprin 75 mg", "aspirin"),
        ("acetylsalicylic acid 100 mg", "aspirin"),
        ("atorva 40", "atorvastatin"),
        ("azithro 500", "azithromycin"),
        ("inj insulin glargine", "insulin"),
        ("basal insulin", "insulin"),
        ("phenprocoumon/warfarin therapy", "warfarin"),
        ("warfarin titrated to INR", "warfarin"),
        ("Tab metoprolol XL 50 mg OD", "metoprolol"),
        ("metoprolol succinate", "metoprolol"),
    ],
)
def test_medication_surface_forms(text: str, expected: str) -> None:
    assert rx(f"Medication list: {text}.") == [expected]


def test_insulin_resistance_is_not_insulin() -> None:
    assert rx("Findings suggest insulin resistance.") == []


# --------------------------------------------------------------------------- #
# Numeric fields and units
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "field", "expected"),
    [
        ("HR 72 bpm", "heart_rate_bpm", 72),
        ("pulse 87/min", "heart_rate_bpm", 87),
        ("ventricular rate 78 per minute", "heart_rate_bpm", 78),
        ("BP 110/61 mmHg", "systolic_bp_mmhg", 110),
        ("blood pressure 140 over 86", "systolic_bp_mmhg", 140),
        ("creatinine 0.97 mg/dL", "creatinine_mg_dl", 0.97),
        ("creatinine 1,04 mg/dL", "creatinine_mg_dl", 1.04),
        ("serum creatinine 124 µmol/L", "creatinine_mg_dl", 1.40),
        ("serum creatinine 59 µmol/L", "creatinine_mg_dl", 0.67),
        ("hemoglobin 12.3 g/dL", "hemoglobin_g_dl", 12.3),
        ("Hb 133 g/L", "hemoglobin_g_dl", 13.3),
        ("LVEF 58%", "lvef_percent", 58),
        ("EF=45%", "lvef_percent", 45),
        ("ejection fraction approximately 54 per cent", "lvef_percent", 54),
    ],
)
def test_numeric_values_and_unit_conversion(text: str, field: str, expected: float) -> None:
    value = extract_clinical_data(f"Data: {text}.")[field]
    assert value is not None
    assert abs(value - expected) < 1e-9 or abs(value - expected) <= NUMERIC_TOLERANCES[field] / 10


def test_ef_inside_hfref_is_not_an_lvef() -> None:
    record = extract_clinical_data("Dx: HFrEF; heart failure with reduced EF; diabetes mellitus.")
    assert record["lvef_percent"] is None
    assert record["diagnoses"] == ["heart_failure", "type_2_diabetes"]


def test_ejection_fraction_not_documented_is_null() -> None:
    note = "Echocardiography: echocardiographic ejection fraction not documented. Allergy: NKDA 45%."
    assert extract_clinical_data(note)["lvef_percent"] is None


def test_missing_vitals_are_null() -> None:
    record = extract_clinical_data("On assessment: vital signs not captured in the export.")
    assert record["heart_rate_bpm"] is None and record["systolic_bp_mmhg"] is None


def test_hba1c_and_creatinine_clearance_are_not_matched() -> None:
    record = extract_clinical_data("HbA1c 7.2%. Creatinine clearance 45 mL/min.")
    assert record["hemoglobin_g_dl"] is None and record["creatinine_mg_dl"] is None


def test_implausible_value_is_skipped_for_the_next_plausible_one() -> None:
    assert extract_clinical_data("HR 300 bpm (artefact), repeat HR 88.")["heart_rate_bpm"] == 88


def test_a_neighbouring_field_is_not_borrowed() -> None:
    assert extract_clinical_data("pulse and BP 120/80 recorded.")["heart_rate_bpm"] is None


# --------------------------------------------------------------------------- #
# Categorical fields
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("never smoked", "never"),
        ("non-smoker", "never"),
        ("no tobacco use", "never"),
        ("ex-smoker", "former"),
        ("former smoker", "former"),
        ("stopped smoking several years ago", "former"),
        ("current smoker", "current"),
        ("actively smokes cigarettes", "current"),
        ("ongoing tobacco use", "current"),
    ],
)
def test_smoking_status(text: str, expected: str) -> None:
    assert extract_clinical_data(f"Smoking: {text}.")["smoking_status"] == expected


def test_relative_who_smokes_is_ignored() -> None:
    note = "Father is a current smoker. Smoking: never smoked."
    assert extract_clinical_data(note)["smoking_status"] == "never"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("NKDA", "none"),
        ("no known drug allergies", "none"),
        ("no medication allergy documented", "none"),
        ("allergic to penicillin", "penicillin"),
        ("penicillin causes rash", "penicillin"),
        ("NSAID allergy", "nsaid"),
        ("allergic to non-steroidal anti-inflammatory drugs", "nsaid"),
        ("ibuprofen-associated urticaria", "nsaid"),
        ("iodinated contrast allergy", "iodinated_contrast"),
        ("contrast medium reaction", "iodinated_contrast"),
    ],
)
def test_allergy(text: str, expected: str) -> None:
    assert extract_clinical_data(f"Allergy: {text}.")["allergy"] == expected


def test_contrast_without_allergy_context_is_not_an_allergy() -> None:
    assert extract_clinical_data("CT with contrast performed. NKDA.")["allergy"] == "none"


# --------------------------------------------------------------------------- #
# Unseen formats (hand-written, hand-labelled; not from train or validation)
# --------------------------------------------------------------------------- #
UNSEEN = [
    (
        # UK discharge letter: prose, British spelling, list with commas.
        "DISCHARGE LETTER\n"
        "Background: ischaemic heart disease, CKD 4, type 2 DM. Family history of AF.\n"
        "Presented with bronchopneumonia. Pulmonary hypertension excluded on echo.\n"
        "Obs: HR 104, BP 98/60. Bloods: Hb 102 g/L, creat 210 umol/L. LVEF 35-40%.\n"
        "Discharge meds: Plavix 75 mg, Lipitor 80 mg, Lasix 40 mg, Zithromax 500 mg. "
        "Clopidogrel held pre-op? No - continued. Aspirin stopped (GI bleed).\n"
        "Allergies: amoxicillin (hives). Social: ex-smoker, 30 pack-years.",
        {
            "diagnoses": ["chronic_kidney_disease", "coronary_artery_disease", "pneumonia", "type_2_diabetes"],
            "medications": ["atorvastatin", "azithromycin", "clopidogrel", "furosemide"],
            "heart_rate_bpm": 104,
            "systolic_bp_mmhg": 98,
            "creatinine_mg_dl": 2.38,
            "hemoglobin_g_dl": 10.2,
            "lvef_percent": 38,
            "smoking_status": "former",
            "allergy": "penicillin",
        },
    ),
    (
        # Key-value export with German terms and units.
        "KV-EXPORT\n"
        "Diagnosen: Vorhofflimmern; arterielle Hypertonie; Herzinsuffizienz\n"
        "Medikation: Marcumar nach INR; Beloc Zok 95 mg; ASS 100\n"
        "Vitalwerte: HR 76; RR 150/90\n"
        "Labor: Kreatinin 1,6 mg/dl; Hb 8.1 mmol/L\n"
        "Echo: EF 42 %\n"
        "Raucher: nein - Nichtraucher. Allergien: keine (NKDA)",
        {
            "diagnoses": ["atrial_fibrillation", "heart_failure", "hypertension"],
            "medications": ["aspirin", "metoprolol", "warfarin"],
            "heart_rate_bpm": 76,
            "systolic_bp_mmhg": None,  # "RR" is ambiguous (respiratory rate) and not used
            "creatinine_mg_dl": 1.6,
            "hemoglobin_g_dl": 13.05,
            "lvef_percent": 42,
            "smoking_status": "never",
            "allergy": "none",
        },
    ),
    (
        # Terse telegraphic ward note with negations and missing values.
        "WARD NOTE 2025-03-01\n"
        "PMH: HTN, T2DM (on metformin 1g bd), ?AF - Holter negative, AF ruled out.\n"
        "No COPD, no CKD. Suspected ACS -> troponin neg, ACS excluded.\n"
        "Meds: metformin, amlodipine 10mg, ramipril; digoxin was discussed but not started.\n"
        "HR not recorded. SBP 162. EF not documented. Smoking: currently smokes 10/day. "
        "Allergy: NSAIDs (angioedema).",
        {
            "diagnoses": ["hypertension", "type_2_diabetes"],
            "medications": ["amlodipine", "metformin", "ramipril"],
            "heart_rate_bpm": None,
            "systolic_bp_mmhg": 162,
            "creatinine_mg_dl": None,
            "hemoglobin_g_dl": None,
            "lvef_percent": None,
            "smoking_status": "current",
            "allergy": "nsaid",
        },
    ),
]


@pytest.mark.parametrize(("note", "expected"), UNSEEN, ids=["uk_letter", "german_kv", "terse_ward"])
def test_unseen_formats(note: str, expected: dict) -> None:
    record = extract_clinical_data(note)
    for field, value in expected.items():
        if field in NUMERIC_TOLERANCES and value is not None and record[field] is not None:
            assert abs(record[field] - value) <= NUMERIC_TOLERANCES[field], field
        else:
            assert record[field] == value, field


# --------------------------------------------------------------------------- #
# Regression on the labelled training data
# --------------------------------------------------------------------------- #
def test_training_extraction_score() -> None:
    records = [json.loads(line) for line in (ROOT / "data" / "train.jsonl").open(encoding="utf-8")]
    truth = {r["case_id"]: r["labels"] for r in records}
    predictions = {
        r["case_id"]: {"extracted_clinical_data": extract_clinical_data(r["note_text"])} for r in records
    }
    report = score_extraction(truth, predictions, errors=[])
    assert report["score"] >= 0.99, report
