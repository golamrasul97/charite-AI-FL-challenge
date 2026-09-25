"""Tests for src/readmission.py (features and the three training models)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src import readmission

ROOT = Path(__file__).resolve().parents[1]
TRAIN = [json.loads(line) for line in (ROOT / "data" / "train.jsonl").open(encoding="utf-8")]


def record(hospital: str, label: int, age: float, note: str = "") -> dict:
    return {
        "hospital_id": hospital,
        "structured_features": {"age_years": age, "prior_admissions_12m": 0, "length_of_stay_days": 5},
        "note_text": note,
        "labels": {"readmission_30d": label},
    }


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def test_feature_dict_has_the_9_features() -> None:
    features = readmission.feature_dict(TRAIN[0])
    assert sorted(features) == sorted([
        "age", "prior_admissions_12m", "heart_failure", "chronic_kidney_disease", "atrial_fibrillation",
        "lvef", "lvef_missing", "hemoglobin", "n_medications",
    ])
    assert all(np.isfinite(v) for v in features.values())


def test_scaling_uses_fixed_constants() -> None:
    extracted = {"diagnoses": ["heart_failure"], "medications": [], "lvef_percent": 35, "hemoglobin_g_dl": None}
    features = readmission.feature_dict(record("A", 0, age=75), extracted)
    assert features["age"] == pytest.approx((75 - 60) / 15)
    assert features["lvef"] == pytest.approx((35 - 55) / 10) and features["lvef_missing"] == 0.0
    assert features["hemoglobin"] == 0.0  # missing -> the typical value
    assert features["heart_failure"] == 1.0 and features["chronic_kidney_disease"] == 0.0


def test_features_of_one_patient_do_not_depend_on_other_patients() -> None:
    """No statistics over the data: the same record always gives the same features."""
    alone = readmission.feature_dict(TRAIN[5])
    readmission.feature_dict(TRAIN[6])
    assert readmission.feature_dict(TRAIN[5]) == alone


def test_extreme_values_are_clipped() -> None:
    extracted = {"diagnoses": [], "medications": [], "lvef_percent": 5, "hemoglobin_g_dl": 40}
    features = readmission.feature_dict(record("A", 0, age=200), extracted)
    assert features["age"] == 3.0 and features["hemoglobin"] == 3.0 and features["lvef"] == -3.0


def test_malformed_structured_values_do_not_crash() -> None:
    bad = {"structured_features": {"age_years": "n/a", "prior_admissions_12m": None}, "note_text": ""}
    assert all(np.isfinite(v) for v in readmission.feature_dict(bad).values())


# --------------------------------------------------------------------------- #
# Centralized and local
# --------------------------------------------------------------------------- #
def test_centralized_is_a_pipeline_with_valid_probabilities() -> None:
    model = readmission.train_centralized(TRAIN)
    assert [name for name, _ in model.steps] == ["vectorizer", "classifier"]
    p = model.predict_proba([readmission.feature_dict(r) for r in TRAIN[:10]])[:, 1]
    assert np.all((p >= 0) & (p <= 1))


def test_local_trains_one_model_per_hospital() -> None:
    models = readmission.train_local(TRAIN)
    assert sorted(models) == ["BERLIN_NODE", "CHENNAI_NODE", "HYDERABAD_NODE"]


# --------------------------------------------------------------------------- #
# FedAvg
# --------------------------------------------------------------------------- #
def test_average_weights_matches_hand_calculation() -> None:
    updates = [(np.array([[1.0, 0.0]]), np.array([2.0]), 10), (np.array([[4.0, 3.0]]), np.array([0.0]), 30)]
    coef, intercept = readmission.average_weights(updates)
    np.testing.assert_allclose(coef, [[(10 * 1 + 30 * 4) / 40, 30 * 3 / 40]])
    np.testing.assert_allclose(intercept, [10 * 2 / 40])


def test_local_update_returns_only_weights_and_row_count() -> None:
    hospital_data = {"A": ([readmission.feature_dict(r) for r in TRAIN[:20]], readmission.labels(TRAIN[:20]))}
    model = readmission.fedavg(hospital_data, rounds=1)
    result = readmission.local_update(model, *hospital_data["A"])
    assert len(result) == 3
    coef, intercept, n = result
    assert coef.shape == (1, 9) and intercept.shape == (1,) and n == 20


def test_fedavg_sends_rows_of_one_hospital_per_update(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[int] = []
    original = readmission.local_update

    def spy(global_model, x_train, y_train, local_updates=1):
        received.append(len(y_train))
        return original(global_model, x_train, y_train, local_updates)

    monkeypatch.setattr(readmission, "local_update", spy)
    groups = readmission.by_hospital(TRAIN)
    hospital_data = {h: ([readmission.feature_dict(r) for r in rows], readmission.labels(rows)) for h, rows in groups.items()}
    readmission.fedavg(hospital_data, rounds=2)
    sizes = [len(rows) for rows in groups.values()]
    assert received == sizes * 2  # every call saw exactly one hospital's rows, never the pool


def test_federated_model_is_close_to_centralized() -> None:
    x = [readmission.feature_dict(r) for r in TRAIN]
    y = readmission.labels(TRAIN)
    fed = readmission.train_federated(TRAIN).predict_proba(x)[:, 1]
    cen = readmission.train_centralized(TRAIN).predict_proba(x)[:, 1]
    assert abs(roc_auc_score(y, fed) - roc_auc_score(y, cen)) < 0.02


def test_federated_training_is_deterministic() -> None:
    first = readmission.get_weights(readmission.train_federated(TRAIN))
    second = readmission.get_weights(readmission.train_federated(TRAIN))
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_unknown_hospital_can_still_be_scored() -> None:
    model = readmission.train_federated(TRAIN)
    new = record("NEW_NODE", 0, age=70, note="HFrEF. LVEF 30%.")
    p = model.predict_proba([readmission.feature_dict(new)])[0, 1]
    assert 0.0 <= p <= 1.0
