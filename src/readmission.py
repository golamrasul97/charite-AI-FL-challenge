"""Readmission models: DictVectorizer + LogisticRegression, like the starter.

Three models, each a plain sklearn Pipeline:

* ``train_centralized``  all training rows pooled in one place (reference only)
* ``train_local``        one Pipeline per hospital, trained on that hospital's rows
* ``train_federated``    FedAvg: each round every hospital fits the shared model on
                          its own rows and sends back only the weights; the weights
                          are averaged, weighted by hospital size (n_k / N)

``run_submission.py`` uses the federated model.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.extraction import extract_clinical_data

# Selected on train cross-validation by scripts/run_experiments.py (results/settings_selection.csv).
C = 10.0           # inverse L2 strength of LogisticRegression
ROUNDS = 50        # FedAvg rounds
LOCAL_UPDATES = 1  # local updates per hospital per round (one optimizer step over all its rows)
SEED = 7


def _number(value: Any, default: float) -> float:
    """Float value, or ``default`` when missing or not a number."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) and not isinstance(value, bool) else default


def _scaled(value: Any, typical: float, step: float) -> float:
    """(value - typical) / step, clipped to [-3, 3] so one unusual value cannot dominate."""
    return float(np.clip((_number(value, typical) - typical) / step, -3.0, 3.0))


def feature_dict(record: dict[str, Any], extracted: dict[str, Any] | None = None) -> dict[str, float]:
    """9 features from the structured fields and the note.

    Chosen by a feature study on the training data (5 folds x 5 seeds, settings chosen inside each fold):
    the diagnoses CKD, heart failure and atrial fibrillation plus prior admissions
    and age carry most of the signal. Length of stay, emergency admission, acute
    coronary syndrome and creatinine were dropped: they added nothing or made the
    model worse (CV AUC 0.797 with them, 0.837 without).

    Every value is scaled with fixed constants, (value - typical) / step, never
    with statistics computed from the data, so no hospital's values leak into
    another hospital's preprocessing. A missing LVEF becomes 0 plus a "missing"
    flag, so "LVEF not documented" differs from "LVEF 55%".
    """
    if extracted is None:
        extracted = extract_clinical_data(str(record.get("note_text", "")))
    structured = record.get("structured_features") or {}
    diagnoses = set(extracted.get("diagnoses") or [])
    lvef = extracted.get("lvef_percent")
    return {
        "age": _scaled(structured.get("age_years"), 60, 15),
        "prior_admissions_12m": min(_number(structured.get("prior_admissions_12m"), 0), 5) / 2,
        "heart_failure": 1.0 if "heart_failure" in diagnoses else 0.0,
        "chronic_kidney_disease": 1.0 if "chronic_kidney_disease" in diagnoses else 0.0,
        "atrial_fibrillation": 1.0 if "atrial_fibrillation" in diagnoses else 0.0,
        "lvef": _scaled(lvef, 55, 10),
        "lvef_missing": 1.0 if lvef is None else 0.0,
        "hemoglobin": _scaled(extracted.get("hemoglobin_g_dl"), 13, 2),
        "n_medications": float(np.clip((len(extracted.get("medications") or []) - 3) / 2, -3.0, 3.0)),
    }


def make_pipeline(**classifier_options: Any) -> Pipeline:
    return Pipeline(
        [
            ("vectorizer", DictVectorizer(sparse=False)),
            ("classifier", LogisticRegression(C=C, max_iter=1000, random_state=SEED, **classifier_options)),
        ]
    )


def labels(records: list[dict[str, Any]]) -> list[int]:
    return [int(record["labels"]["readmission_30d"]) for record in records]


def by_hospital(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group records by ``hospital_id`` (sorted, so the order is deterministic)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record.get("hospital_id", "UNKNOWN")), []).append(record)
    return dict(sorted(groups.items()))


# --------------------------------------------------------------------------- #
# Centralized and local
# --------------------------------------------------------------------------- #
def fit_pipeline(x_train: list[dict[str, float]], y_train: list[int]) -> Pipeline:
    model = make_pipeline()
    model.fit(x_train, y_train)
    return model


def train_centralized(train_records: list[dict[str, Any]]) -> Pipeline:
    x_train = [feature_dict(record) for record in train_records]
    y_train = labels(train_records)
    return fit_pipeline(x_train, y_train)


def train_local(train_records: list[dict[str, Any]]) -> dict[str, Pipeline]:
    return {hospital: train_centralized(rows) for hospital, rows in by_hospital(train_records).items()}


# --------------------------------------------------------------------------- #
# Federated (FedAvg)
# --------------------------------------------------------------------------- #
def get_weights(model: Pipeline) -> tuple[np.ndarray, np.ndarray]:
    classifier = model.named_steps["classifier"]
    return classifier.coef_.copy(), classifier.intercept_.copy()


def set_weights(model: Pipeline, coef: np.ndarray, intercept: np.ndarray) -> None:
    classifier = model.named_steps["classifier"]
    classifier.coef_, classifier.intercept_, classifier.classes_ = coef.copy(), intercept.copy(), np.array([0, 1])


def local_update(
    global_model: Pipeline, x_train: list[dict[str, float]], y_train: list[int], local_updates: int = LOCAL_UPDATES
) -> tuple[np.ndarray, np.ndarray, int]:
    """One hospital's step: start from the shared weights, fit on its own rows.

    Returns only (coef, intercept, n). The rows never leave this function.
    """
    vectorizer = global_model.named_steps["vectorizer"]  # shared, fixed column order
    classifier = LogisticRegression(C=C, max_iter=local_updates, warm_start=True, random_state=SEED)
    coef, intercept = get_weights(global_model)
    classifier.coef_, classifier.intercept_, classifier.classes_ = coef, intercept, np.array([0, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)  # a few iterations per round is intended
        classifier.fit(vectorizer.transform(x_train), y_train)
    return classifier.coef_.copy(), classifier.intercept_.copy(), len(y_train)


def average_weights(updates: list[tuple[np.ndarray, np.ndarray, int]]) -> tuple[np.ndarray, np.ndarray]:
    """FedAvg aggregation: sum over hospitals of (n_k / N) * weights_k."""
    total = sum(n for _, _, n in updates)
    coef = sum(c * n / total for c, _, n in updates)
    intercept = sum(b * n / total for _, b, n in updates)
    return coef, intercept


def fedavg(
    hospital_data: dict[str, tuple[list[dict[str, float]], list[int]]],
    rounds: int = ROUNDS,
    local_updates: int = LOCAL_UPDATES,
    on_round: Any = None,
) -> Pipeline:
    """FedAvg over ``{hospital: (x_train, y_train)}``; returns the shared Pipeline.

    The shared model starts at zero weights. Its column order comes from the
    feature names alone (``feature_dict`` always returns the same 9 keys), not
    from any hospital's data. ``on_round(r, model)`` lets the experiments record
    a convergence curve.
    """
    some_x = next(iter(hospital_data.values()))[0]
    global_model = make_pipeline()
    global_model.named_steps["vectorizer"].fit([dict.fromkeys(some_x[0], 0.0)])
    n_features = len(global_model.named_steps["vectorizer"].feature_names_)
    set_weights(global_model, np.zeros((1, n_features)), np.zeros(1))
    for round_number in range(1, rounds + 1):
        updates = [local_update(global_model, x, y, local_updates) for x, y in hospital_data.values()]
        set_weights(global_model, *average_weights(updates))
        if on_round is not None:
            on_round(round_number, global_model)
    return global_model


def train_federated(train_records: list[dict[str, Any]]) -> Pipeline:
    hospital_data = {
        hospital: ([feature_dict(record) for record in rows], labels(rows))
        for hospital, rows in by_hospital(train_records).items()
    }
    return fedavg(hospital_data)
