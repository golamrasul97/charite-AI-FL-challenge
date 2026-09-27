"""experiment_summary.json: the federated-learning experiment, measured from --train only.

``run_submission.py`` may read only --train and --input, so every number here is
recomputed at each run with repeated cross-validation on the training data
(local, federated and centralized models, overall and by site). The full study,
including the validation comparison, lives in ``results/`` (scripts/run_experiments.py).
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from src import readmission

SUMMARY_SEEDS = (7, 19, 43)  # 3 seeds x 5 folds keeps the run fast; results/ uses 5 seeds


def stratified_folds(hospitals: np.ndarray, y: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Fold number (0..k-1) per row, so every fold keeps each hospital's size and readmission rate."""
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), dtype=int)
    offset = 0
    for hospital in sorted(set(hospitals.tolist())):
        for label in (0, 1):
            idx = np.flatnonzero((hospitals == hospital) & (y == label))
            idx = idx[rng.permutation(len(idx))]
            fold[idx] = (np.arange(len(idx)) + offset) % k
            offset += len(idx)
    return fold


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    both = len(set(y.tolist())) == 2
    return {
        "auc": round(float(roc_auc_score(y, p)), 3) if both else None,
        "average_precision": round(float(average_precision_score(y, p)), 3) if y.any() else None,
        "brier": round(float(brier_score_loss(y, p)), 3),
        "log_loss": round(float(log_loss(y, p, labels=[0, 1])), 3),
    }


def cross_validated_metrics(train_records: list[dict[str, Any]], seeds: tuple[int, ...] = SUMMARY_SEEDS) -> dict[str, Any]:
    """Local, federated and centralized models under the same folds; metrics averaged over seeds."""
    x = [readmission.feature_dict(r) for r in train_records]
    y = np.asarray(readmission.labels(train_records))
    sites = np.asarray([str(r.get("hospital_id", "UNKNOWN")) for r in train_records])
    names = sorted(set(sites.tolist()))
    collected: dict[str, list[dict[str, dict[str, float]]]] = {"local": [], "federated": [], "centralized": []}
    seconds = {k: 0.0 for k in collected}
    for seed in seeds:
        fold = stratified_folds(sites, y, 5, seed)
        oof = {k: np.zeros(len(y)) for k in collected}
        for f in range(5):
            train, test = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            x_test = [x[i] for i in test]
            by_site = {s: ([x[i] for i in train[sites[train] == s]], y[train[sites[train] == s]].tolist()) for s in names}
            start = time.perf_counter()
            for s in names:
                t = test[sites[test] == s]
                oof["local"][t] = readmission.fit_pipeline(*by_site[s]).predict_proba([x[i] for i in t])[:, 1]
            seconds["local"] += time.perf_counter() - start
            start = time.perf_counter()
            oof["federated"][test] = readmission.fedavg(by_site).predict_proba(x_test)[:, 1]
            seconds["federated"] += time.perf_counter() - start
            start = time.perf_counter()
            oof["centralized"][test] = readmission.fit_pipeline([x[i] for i in train], y[train].tolist()).predict_proba(x_test)[:, 1]
            seconds["centralized"] += time.perf_counter() - start
        for k in collected:
            collected[k].append({"overall": _metrics(y, oof[k]), **{s: _metrics(y[sites == s], oof[k][sites == s]) for s in names}})

    def average(model: str, scope: str) -> dict[str, float]:
        values = [rep[scope] for rep in collected[model]]
        return {m: round(float(np.mean([v[m] for v in values])), 3) for m in values[0] if values[0][m] is not None}

    return {
        model: {"overall": average(model, "overall"), "by_site": {s: average(model, s) for s in names},
                "seconds_per_training": round(seconds[model] / (5 * len(seeds)), 4)}
        for model in collected
    }


def convergence(train_records: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Training log loss of the federated model after each round (on all training rows)."""
    x = [readmission.feature_dict(r) for r in train_records]
    y = readmission.labels(train_records)
    by_site = {s: ([readmission.feature_dict(r) for r in rows], readmission.labels(rows))
               for s, rows in readmission.by_hospital(train_records).items()}
    curve: list[dict[str, float]] = []
    readmission.fedavg(by_site, on_round=lambda r, m: curve.append(
        {"round": r, "train_log_loss": round(float(log_loss(y, m.predict_proba(x)[:, 1], labels=[0, 1])), 4)}))
    return curve


def summary(train_records: list[dict[str, Any]]) -> dict[str, Any]:
    groups = readmission.by_hospital(train_records)
    n_features = len(readmission.feature_dict(train_records[0]))
    values_per_message = n_features + 1
    curve = convergence(train_records)
    metrics = cross_validated_metrics(train_records)
    features = sorted(readmission.feature_dict(train_records[0]))
    return {
        "implementation": "FedAvg over the hospital nodes in --train; the submitted readmission_probability comes "
                          "from the federated model",
        "random_seeds": {"model": readmission.SEED, "cross_validation": list(SUMMARY_SEEDS),
                         "full_study_in_results": [7, 19, 43, 101, 202]},
        "features": {
            "feature_set": features,
            "source": "structured_features (age, prior admissions) and extracted_clinical_data (heart failure, "
                      "chronic kidney disease, atrial fibrillation, LVEF, haemoglobin, number of medications)",
            "scaling": "fixed values written in the code, e.g. (age - 60) / 15, capped at +/-3; missing LVEF -> 0 "
                       "plus a missing flag; no statistics are shared between hospital nodes",
        },
        "local_models": {
            site: {"algorithm": "logistic regression (sklearn Pipeline(DictVectorizer, LogisticRegression), "
                                f"C={readmission.C}, L2)", "feature_set": features,
                   "training_rows": len(rows),
                   "readmission_rate": round(float(np.mean(readmission.labels(rows))), 3)}
            for site, rows in groups.items()
        },
        "federated_model": {
            "algorithm": "FedAvg", "model": "logistic regression, same features and settings as the local models",
            "rounds": readmission.ROUNDS, "local_epochs": readmission.LOCAL_UPDATES,
            "local_epochs_meaning": "one optimizer update over all of the client's patients per round",
            "optimizer": "L-BFGS (sklearn LogisticRegression, warm start from the server's weights)",
            "client_weighting": "number_of_training_cases (n_k / N)",
            "starting_weights": "zeros",
        },
        "centralized_model": {"algorithm": "logistic regression, same features and settings, all training rows "
                                           "pooled (reference only; not allowed between real hospitals)"},
        "metrics": {
            "method": "repeated 5-fold cross-validation on --train, stratified by site and label "
                      f"({len(SUMMARY_SEEDS)} seeds), recomputed at every run; validation labels are never read",
            **metrics,
        },
        "communication": {
            "client_to_server_per_round": f"{values_per_message} float64 values (coef_ and intercept_) and the row count",
            "server_to_client_per_round": f"{values_per_message} float64 values (the averaged model)",
            "bytes_per_round": 2 * len(groups) * values_per_message * 8,
            "bytes_total": 2 * len(groups) * values_per_message * 8 * readmission.ROUNDS,
            "raw_rows_transferred": False,
            "what_leaves_each_client": "only its model weights and its number of training cases; patient rows, "
                                       "features and labels stay inside the hospital node",
        },
        "convergence": {
            "train_log_loss_by_round": [curve[i] for i in (0, 4, 9, 19, 29, 39, 49) if i < len(curve)],
            "observations": [
                "The training loss falls quickly in the first rounds and then levels off.",
                "Performance is stable across random starting weights (results/seed_stability.csv), but single "
                "probabilities can differ between starts because one L-BFGS update per client per round does not "
                "settle exactly on the optimum.",
                "More local updates per round lower the federated AUC slightly (client drift; "
                "results/local_updates_comparison.csv).",
            ],
        },
        "non_iid_analysis": {
            "sites": {site: {"training_rows": len(rows), "readmission_rate": round(float(np.mean(readmission.labels(rows))), 3)}
                      for site, rows in groups.items()},
            "observations": [
                "Readmission rates differ between hospital nodes: " + ", ".join(
                    f"{site} {np.mean(readmission.labels(rows)):.2f}" for site, rows in groups.items()) + ".",
                "Berlin is the hardest site for every model; readmission relates to the features differently there.",
                "Local models are weakest; the federated model matches the centralized one overall and by site.",
                "Trained on two hospital nodes, the federated model still ranks the third well "
                "(results/leave_one_site_out.csv).",
                "Local fine-tuning of the federated model per hospital made every site worse (tested, not used).",
            ],
        },
        "full_results": "results/ (scripts/run_experiments.py): model_comparison_cv.csv, model_comparison_validation.csv, "
                        "convergence.csv, local_updates_comparison.csv, local_models_on_other_sites.csv, "
                        "leave_one_site_out.csv, seed_stability.csv, settings_selection.csv, "
                        "settings_selection_check.csv, feature_set_comparison.csv, prior_admissions_check.csv, "
                        "model_family_comparison.csv",
        "limitations": [
            "120 training patients and 41 readmissions: differences below about 0.03 AUC are noise.",
            "The 9 features were chosen on the training data, so training-data scores are somewhat optimistic.",
            "Local models reuse the federated model's settings.",
            "The simulation runs in one process; real deployment needs networking, authentication and dropout handling.",
            "Model coefficients are not clinical findings (synthetic data, overlapping features).",
        ],
    }
