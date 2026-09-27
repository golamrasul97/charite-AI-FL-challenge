"""End-to-end tests for run_submission.py (the standard challenge command)."""
from __future__ import annotations

import builtins
import io
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

import run_submission

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "schemas" / "prediction.schema.json").read_text(encoding="utf-8"))
TRAIN = ROOT / "data" / "train.jsonl"


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run the standard command once on 6 validation inputs (no answers in them)."""
    tmp = tmp_path_factory.mktemp("submission")
    lines = (ROOT / "data" / "validation_inputs.jsonl").read_text(encoding="utf-8").splitlines()[:6]
    inputs = tmp / "inputs.jsonl"
    inputs.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output, artifacts = tmp / "predictions.jsonl", tmp / "artifacts"
    subprocess.run(
        [sys.executable, "run_submission.py", "--train", str(TRAIN), "--input", str(inputs),
         "--output", str(output), "--artifacts-dir", str(artifacts)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return {
        "inputs": [json.loads(line) for line in lines],
        "predictions": [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()],
        "experiment": json.loads((artifacts / "experiment_summary.json").read_text(encoding="utf-8")),
        "privacy": json.loads((artifacts / "privacy_summary.json").read_text(encoding="utf-8")),
    }


def test_one_schema_valid_prediction_per_input_case(run: dict) -> None:
    assert [p["case_id"] for p in run["predictions"]] == [r["case_id"] for r in run["inputs"]]
    for prediction in run["predictions"]:
        jsonschema.validate(prediction, SCHEMA)
        assert 0.0 <= prediction["readmission_probability"] <= 1.0


def test_experiment_summary_has_every_required_item(run: dict) -> None:
    summary = run["experiment"]
    # SUBMISSION_SCHEMA.md: algorithms and feature sets per local model; federated algorithm, rounds,
    # local epochs, optimizer, client weighting, seeds; local/federated/centralized metrics overall and
    # by site; communication payload and what leaves each client; convergence and non-IID; limitations.
    for site, model in summary["local_models"].items():
        assert model["algorithm"] and model["feature_set"]
    federated = summary["federated_model"]
    for key in ("algorithm", "rounds", "local_epochs", "optimizer", "client_weighting"):
        assert key in federated
    assert summary["random_seeds"]
    for model in ("local", "federated", "centralized"):
        assert {"auc", "brier"} <= set(summary["metrics"][model]["overall"])
        assert set(summary["metrics"][model]["by_site"]) == set(summary["local_models"])
    assert summary["communication"]["what_leaves_each_client"] and "bytes_per_round" in summary["communication"]
    assert summary["convergence"]["train_log_loss_by_round"] and summary["non_iid_analysis"]["observations"]
    assert summary["limitations"]


def test_privacy_summary_has_every_required_item(run: dict) -> None:
    # SUBMISSION_SCHEMA.md: mechanism and status; protected asset; adversary and trust assumptions;
    # privacy parameters; utility or runtime comparison; exact claim and non-guarantees; attack surface.
    required = {"mechanism", "implementation_status", "protected_asset", "adversary", "trust_assumptions",
                "parameters", "utility_analysis", "runtime_analysis", "privacy_claim", "not_guaranteed",
                "remaining_attack_surface", "limitations"}
    assert required <= set(run["privacy"])


def test_the_submission_never_opens_a_ground_truth_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    real_builtin_open, real_io_open = builtins.open, io.open

    def spy_builtin(file, *args, **kwargs):  # type: ignore[no-untyped-def]
        opened.append(str(file))
        return real_builtin_open(file, *args, **kwargs)

    def spy_io(file, *args, **kwargs):  # type: ignore[no-untyped-def]
        opened.append(str(file))
        return real_io_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy_builtin)
    monkeypatch.setattr(io, "open", spy_io)
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text((ROOT / "data" / "validation_inputs.jsonl").read_text(encoding="utf-8").splitlines()[0] + "\n")
    monkeypatch.setattr(sys, "argv", ["run_submission.py", "--train", str(TRAIN), "--input", str(inputs),
                                      "--output", str(tmp_path / "out.jsonl"), "--artifacts-dir", str(tmp_path / "a")])
    run_submission.main()
    assert opened, "the spy saw no file reads"
    assert not [path for path in opened if "ground_truth" in path]


# --------------------------------------------------------------------------- #
# A failing step still gives a valid record
# --------------------------------------------------------------------------- #
NOTE = {"case_id": "X-1", "hospital_id": "BERLIN_NODE", "note_text": "Patient: Anna Keller. HFrEF. LVEF 30%.",
        "structured_features": {"age_years": 70, "sex": "female", "prior_admissions_12m": 0,
                                "length_of_stay_days": 4, "emergency_admission": True}}


class BrokenModel:
    def predict_proba(self, rows):  # type: ignore[no-untyped-def]
        raise RuntimeError("broken")


def test_failed_deidentification_redacts_the_whole_note(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(note: str) -> list:
        raise RuntimeError("boom")

    monkeypatch.setattr(run_submission, "detect_pii", fail)
    record = run_submission.predict_case(NOTE, BrokenModel(), fallback_probability=0.34)
    jsonschema.validate(record, SCHEMA)
    assert record["deidentified_text"] == "[PATIENT_NAME]"     # no word of the note survives
    assert record["readmission_probability"] == 0.34            # prediction fell back too


def test_failed_extraction_gives_empty_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(note: str) -> dict:
        raise RuntimeError("boom")

    monkeypatch.setattr(run_submission, "extract_clinical_data", fail)
    record = run_submission.predict_case(NOTE, BrokenModel(), fallback_probability=0.34)
    jsonschema.validate(record, SCHEMA)
    assert record["extracted_clinical_data"]["diagnoses"] == []
    assert record["extracted_clinical_data"]["lvef_percent"] is None


def test_invalid_json_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    path.write_text('{"case_id": "A"}\nnot json\n\n{"case_id": "B"}\n', encoding="utf-8")
    assert [r["case_id"] for r in run_submission.read_jsonl(path)] == ["A", "B"]
