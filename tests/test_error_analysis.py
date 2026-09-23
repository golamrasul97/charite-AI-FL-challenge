"""Tests for scripts/error_analysis.py.

The load-bearing one is test_summary_matches_evaluator: the script's count table
has to agree with `make evaluate`, otherwise later phases would be tuned against
a second, subtly different scorer.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from evaluator.evaluate import evaluate

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def _load_error_analysis() -> Any:
    """Import the script by path; scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location(
        "error_analysis", REPO_ROOT / "scripts" / "error_analysis.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


error_analysis = _load_error_analysis()


def test_load_split_shapes() -> None:
    train_inputs, train_gt = error_analysis.load_split("train")
    assert len(train_inputs) == len(train_gt) == 120
    assert set(train_inputs) == set(train_gt)

    validation_inputs, validation_gt = error_analysis.load_split("validation")
    assert len(validation_inputs) == len(validation_gt) == 30

    for _, ground_truth in (("train", train_gt), ("validation", validation_gt)):
        sample = next(iter(ground_truth.values()))
        assert {"pii_entities", "extracted_clinical_data", "readmission_30d"} <= set(sample)


def test_unknown_split_rejected() -> None:
    with pytest.raises(ValueError):
        error_analysis.load_split("test")


def test_summary_matches_evaluator(tmp_path: Path) -> None:
    """The script's totals must equal the official evaluator's on the same predictions."""
    inputs, ground_truth = error_analysis.load_split("validation")
    predictions = error_analysis.predict(inputs)

    predictions_path = tmp_path / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for case_id, record in predictions.items():
            handle.write(
                json.dumps({**record, "case_id": case_id, "readmission_probability": 0.5}) + "\n"
            )

    official = evaluate(
        DATA_DIR / "validation_inputs.jsonl",
        DATA_DIR / "validation_ground_truth.jsonl",
        predictions_path,
    )["metrics"]
    ours = error_analysis.summarize(inputs, ground_truth, predictions)

    assert ours["deidentification"] == official["deidentification"]
    assert ours["structured_extraction"] == official["structured_extraction"]
    assert ours["validation_messages"] == []


def test_missed_span_reasons() -> None:
    note = "Signed: Emil Brandt on 01.02.1970"
    missed = (8, 19, "CLINICIAN_NAME")

    assert "not detected" in error_analysis.describe_missed_span(missed, [], note)
    assert "wrong label" in error_analysis.describe_missed_span(
        missed, [(8, 19, "PATIENT_NAME")], note
    )
    assert "boundary" in error_analysis.describe_missed_span(
        missed, [(0, 12, "CLINICIAN_NAME")], note
    )


def test_evidence_prefers_a_matching_sentence() -> None:
    """Sentences are split on ':' too, so the segment is the text, not the header."""
    note = "Dx: pneumonia.\nRx: azithromycin."
    assert error_analysis.evidence_for(note, "pneumonia", "diagnoses") == ["pneumonia."]


def test_evidence_falls_back_to_the_section_for_abbreviations() -> None:
    """HTN shares no words with hypertension, so the Dx section is the useful pointer."""
    note = "Problem list: HTN; CKD.\nObs: HR 70 bpm."
    evidence = error_analysis.evidence_for(note, "hypertension", "diagnoses")
    assert evidence == ["(section) Problem list: HTN; CKD."]


def test_evidence_requires_two_words_of_a_multiword_term() -> None:
    """"ischemic heart disease" must not be offered as evidence for chronic_kidney_disease."""
    note = "Known diagnoses: ischemic heart disease."
    evidence = error_analysis.evidence_for(note, "chronic_kidney_disease", "diagnoses")
    assert evidence == ["(section) Known diagnoses: ischemic heart disease."]


def test_cli_runs_on_both_splits(capsys: pytest.CaptureFixture[str]) -> None:
    for split in ("train", "validation"):
        assert error_analysis.main(["--split", split, "--summary-only"]) == 0
        captured = capsys.readouterr().out
        assert "de-identification score" in captured
        assert "structured extraction score" in captured


def test_cli_rejects_unknown_case(capsys: pytest.CaptureFixture[str]) -> None:
    assert error_analysis.main(["--split", "validation", "--case", "NOPE-0001"]) == 2
