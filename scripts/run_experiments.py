#!/usr/bin/env python3
"""Reproduce every readmission / federated-learning number in REPORT.md.

    python scripts/run_experiments.py

Uses the models in ``src/readmission.py`` and writes (all committed):
  results/settings_selection.csv           C x rounds for the federated model, train CV (how the settings were chosen)
  results/settings_selection_check.csv     honest score of "choose C inside each training part, then train FedAvg"
  results/model_comparison_cv.csv          local / federated / centralized models, overall and by site
  results/model_comparison_validation.csv  the same models trained on all of train, scored on validation
  results/local_models_on_other_sites.csv  each hospital's local model scored on every site (3 x 3)
  results/local_updates_comparison.csv     federated vs centralized model for 1 / 2 / 5 local updates per round
  results/feature_set_comparison.csv       structured features only vs all 9 features
  results/convergence.csv                  test AUC and log loss of the federated model after each round
  results/federated_experiment.json        settings, features, communication payload, runtimes

Protocol: 5-fold cross-validation on train, stratified by hospital and label,
repeated with seeds 7, 19, 43, 101, 202; mean and standard deviation over the
repeats. Settings are chosen on train CV only. Validation labels are read only
here, for the final comparison. Not part of the submission path.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import readmission  # noqa: E402

RESULTS = ROOT / "results"
SEEDS = (7, 19, 43, 101, 202)
MODELS = ("local", "federated", "centralized")
METRICS = ("auc", "ap", "brier", "log_loss")
C_GRID = (1.0, 3.0, 10.0, 30.0, 100.0)
ROUNDS_GRID = (20, 50, 100)
STRUCTURED_ONLY = ("age", "prior_admissions_12m")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})


def stratified_folds(hospitals: np.ndarray, y: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Fold number (0..k-1) per row, so every fold keeps each hospital's size and prevalence."""
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


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "auc": float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else float("nan"),
        "ap": float(average_precision_score(y, p)) if y.any() else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def mean_sd(per_repeat: list[dict[str, float]]) -> dict[str, float]:
    row: dict[str, float] = {}
    for metric in METRICS:
        values = np.asarray([r[metric] for r in per_repeat])
        row[f"{metric}_mean"], row[f"{metric}_sd"] = float(values.mean()), float(values.std(ddof=1))
    return row


def predict(model: Any, x: list[dict[str, float]]) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


class Data:
    """Feature dicts (computed once), labels and hospital per training row."""

    def __init__(self, records: list[dict[str, Any]], keep: tuple[str, ...] | None = None) -> None:
        dicts = [readmission.feature_dict(r) for r in records]
        self.x = [{k: v for k, v in d.items() if keep is None or k in keep} for d in dicts]
        self.y = np.asarray(readmission.labels(records))
        self.hospitals = np.asarray([r["hospital_id"] for r in records])

    def rows(self, idx: np.ndarray) -> tuple[list[dict[str, float]], list[int]]:
        return [self.x[i] for i in idx], self.y[idx].tolist()

    def by_hospital(self, idx: np.ndarray) -> dict[str, tuple[list[dict[str, float]], list[int]]]:
        return {h: self.rows(idx[self.hospitals[idx] == h]) for h in sorted(set(self.hospitals[idx].tolist()))}


# --------------------------------------------------------------------------- #
# Cross-validation of the three models
# --------------------------------------------------------------------------- #
def cross_validate(data: Data, rounds: int, local_updates: int = 1, curve: bool = False) -> dict[str, Any]:
    hospitals = sorted(set(data.hospitals.tolist()))
    scores: dict[str, list[dict[str, dict[str, float]]]] = {s: [] for s in MODELS}
    transfer: list[dict[tuple[str, str], float]] = []
    auc_curve, loss_curve = np.zeros(rounds), np.zeros(rounds)
    runtime = {s: 0.0 for s in MODELS}
    n_folds = 5 * len(SEEDS)

    for seed in SEEDS:
        fold = stratified_folds(data.hospitals, data.y, 5, seed)
        oof = {s: np.zeros(len(data.y)) for s in MODELS}
        cross = {(a, b): np.zeros(len(data.y)) for a in hospitals for b in hospitals}
        for f in range(5):
            train, test = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            x_test, y_test = data.rows(test)

            start = time.perf_counter()
            for hospital, (x, y) in data.by_hospital(train).items():
                local = readmission.fit_pipeline(x, y)
                for target in hospitals:
                    idx = test[data.hospitals[test] == target]
                    cross[(hospital, target)][idx] = predict(local, data.rows(idx)[0])
            runtime["local"] += time.perf_counter() - start

            def record_round(r: int, model: Any) -> None:
                p = np.clip(predict(model, x_test), 1e-7, 1 - 1e-7)
                auc_curve[r - 1] += roc_auc_score(y_test, p) / n_folds
                loss_curve[r - 1] += log_loss(y_test, p, labels=[0, 1]) / n_folds

            start = time.perf_counter()
            federated = readmission.fedavg(
                data.by_hospital(train), rounds, local_updates, on_round=record_round if curve else None
            )
            runtime["federated"] += time.perf_counter() - start
            oof["federated"][test] = predict(federated, x_test)

            start = time.perf_counter()
            oof["centralized"][test] = predict(readmission.fit_pipeline(*data.rows(train)), x_test)
            runtime["centralized"] += time.perf_counter() - start

        for hospital in hospitals:
            mask = data.hospitals == hospital
            oof["local"][mask] = cross[(hospital, hospital)][mask]
        for model_type in MODELS:
            per_site = {"overall": metrics(data.y, oof[model_type])}
            for hospital in hospitals:
                mask = data.hospitals == hospital
                per_site[hospital] = metrics(data.y[mask], oof[model_type][mask])
            scores[model_type].append(per_site)
        transfer.append({
            (a, b): metrics(data.y[data.hospitals == b], cross[(a, b)][data.hospitals == b])["auc"]
            for a in hospitals for b in hospitals
        })

    return {
        "scores": scores,
        "transfer": transfer,
        "curve": {"auc": auc_curve.tolist(), "log_loss": loss_curve.tolist()},
        "runtime_seconds_per_repeat": {s: runtime[s] / len(SEEDS) for s in MODELS},
    }


def model_rows(result: dict[str, Any], setting: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for model_type in MODELS:
        for site in result["scores"][model_type][0]:
            row: dict[str, Any] = {"setting": setting} if setting else {}
            row.update({"model": model_type, "site": site})
            row.update(mean_sd([rep[site] for rep in result["scores"][model_type]]))
            rows.append(row)
    return rows


def with_c(c: float, run: Callable[[], Any]) -> Any:
    """Run ``run()`` with readmission.C temporarily set to ``c``."""
    saved, readmission.C = readmission.C, c
    try:
        return run()
    finally:
        readmission.C = saved


# --------------------------------------------------------------------------- #
# Choosing C and the number of rounds
# --------------------------------------------------------------------------- #
def choose(losses: dict[tuple[float, int], float]) -> tuple[float, int]:
    """Lowest FedAvg CV log loss; settings within 0.002 of the best count as tied,
    and ties go to fewer rounds (less communication, smaller privacy cost), then
    to the smaller C (stronger regularisation)."""
    best = min(losses.values())
    tied = [key for key, value in losses.items() if value <= best + 0.002]
    return min(tied, key=lambda key: (key[1], key[0]))


def select(data: Data) -> tuple[tuple[float, int], list[dict[str, Any]]]:
    rows, losses = [], {}
    for c, rounds in product(C_GRID, ROUNDS_GRID):
        result = with_c(c, lambda: cross_validate(data, rounds))
        fed = mean_sd([rep["overall"] for rep in result["scores"]["federated"]])
        cen = mean_sd([rep["overall"] for rep in result["scores"]["centralized"]])
        losses[(c, rounds)] = fed["log_loss_mean"]
        rows.append({"C": c, "rounds": rounds, "local_updates": 1,
                     "federated_auc": fed["auc_mean"], "federated_auc_sd": fed["auc_sd"],
                     "federated_brier": fed["brier_mean"], "federated_log_loss": fed["log_loss_mean"],
                     "centralized_auc": cen["auc_mean"], "centralized_log_loss": cen["log_loss_mean"]})
    chosen = choose(losses)
    for row in rows:
        row["selected"] = (row["C"], row["rounds"]) == chosen
    return chosen, rows


def settings_selection_check(data: Data, rounds: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Outer 5-fold x 5 seeds; inside each outer training part, C is chosen by
    3-fold CV with the same rule, then scored on the untouched outer fold."""
    per_repeat, picks = [], {}
    for seed in SEEDS:
        outer = stratified_folds(data.hospitals, data.y, 5, seed)
        oof = np.zeros(len(data.y))
        for f in range(5):
            train, test = np.flatnonzero(outer != f), np.flatnonzero(outer == f)
            inner = stratified_folds(data.hospitals[train], data.y[train], 3, seed + 1000)
            losses = {}
            for c in C_GRID:
                p = np.zeros(len(train))
                for j in range(3):
                    fit_idx, val_idx = train[inner != j], train[inner == j]
                    model = with_c(c, lambda: readmission.fedavg(data.by_hospital(fit_idx), rounds))
                    p[inner == j] = predict(model, data.rows(val_idx)[0])
                losses[(c, rounds)] = log_loss(data.y[train], np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])
            c, _ = choose(losses)
            picks[f"C={c}"] = picks.get(f"C={c}", 0) + 1
            model = with_c(c, lambda: readmission.fedavg(data.by_hospital(train), rounds))
            oof[test] = predict(model, data.rows(test)[0])
        per_site = {"overall": metrics(data.y, oof)}
        for hospital in sorted(set(data.hospitals.tolist())):
            mask = data.hospitals == hospital
            per_site[hospital] = metrics(data.y[mask], oof[mask])
        per_repeat.append(per_site)
    rows = [{"site": site, **mean_sd([rep[site] for rep in per_repeat])} for site in per_repeat[0]]
    return rows, dict(sorted(picks.items()))


# --------------------------------------------------------------------------- #
# Validation: train on all of train, score on validation (read only here)
# --------------------------------------------------------------------------- #
def validation_rows(train_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    inputs = {r["case_id"]: r for r in read_jsonl(ROOT / "data" / "validation_inputs.jsonl")}
    truth = read_jsonl(ROOT / "data" / "validation_ground_truth.jsonl")
    records = [inputs[t["case_id"]] for t in truth]
    y = np.asarray([int(t["readmission_30d"]) for t in truth])
    hospitals = np.asarray([r["hospital_id"] for r in records])
    x = [readmission.feature_dict(r) for r in records]

    local_models = readmission.train_local(train_records)
    local = np.asarray([predict(local_models[h], [xi])[0] for h, xi in zip(hospitals, x)])
    preds = {
        "local": local,
        "federated": predict(readmission.train_federated(train_records), x),
        "centralized": predict(readmission.train_centralized(train_records), x),
    }
    rows = []
    for model_type in MODELS:
        site_masks = {"overall": np.ones(len(y), dtype=bool)} | {h: hospitals == h for h in sorted(set(hospitals))}
        for site, mask in site_masks.items():
            rows.append({"model": model_type, "site": site, "n": int(mask.sum()), "positives": int(y[mask].sum()),
                         **metrics(y[mask], preds[model_type][mask])})
    return rows


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    started = time.perf_counter()
    train_records = read_jsonl(ROOT / "data" / "train.jsonl")
    data = Data(train_records)
    hospitals = sorted(set(data.hospitals.tolist()))

    (c, rounds), grid = select(data)
    write_csv(RESULTS / "settings_selection.csv", grid)
    print(f"selected C={c}, rounds={rounds}")
    if (c, rounds) != (readmission.C, readmission.ROUNDS):
        raise SystemExit(f"selected C={c}, rounds={rounds} differ from src/readmission.py "
                         f"(C={readmission.C}, ROUNDS={readmission.ROUNDS}); update the constants there")

    check, picks = settings_selection_check(data, rounds)
    write_csv(RESULTS / "settings_selection_check.csv", check)

    main_run = cross_validate(data, rounds, curve=True)
    write_csv(RESULTS / "model_comparison_cv.csv", model_rows(main_run))
    write_csv(RESULTS / "local_models_on_other_sites.csv", [
        {"trained_on": a, **{f"auc_on_{b}": float(np.mean([t[(a, b)] for t in main_run["transfer"]])) for b in hospitals}}
        for a in hospitals
    ])
    write_csv(RESULTS / "convergence.csv", [
        {"round": r + 1, "test_auc": a, "test_log_loss": l}
        for r, (a, l) in enumerate(zip(main_run["curve"]["auc"], main_run["curve"]["log_loss"]))
    ])

    updates_comparison = []
    for updates in (1, 2, 5):
        rows = model_rows(cross_validate(data, rounds, local_updates=updates), f"local_updates={updates}")
        updates_comparison += [r for r in rows if r["site"] == "overall" and r["model"] != "local"]
    write_csv(RESULTS / "local_updates_comparison.csv", updates_comparison)

    comparison = []
    for name, keep in (("structured_only_2", STRUCTURED_ONLY), ("all_9", None)):
        rows = model_rows(cross_validate(Data(train_records, keep), rounds), name)
        comparison += [r for r in rows if r["site"] == "overall"]
    write_csv(RESULTS / "feature_set_comparison.csv", comparison)

    write_csv(RESULTS / "model_comparison_validation.csv", validation_rows(train_records))

    n_params = len(data.x[0]) + 1
    summary = {
        "model": "sklearn Pipeline(DictVectorizer, LogisticRegression)",
        "settings": {"C": c, "rounds": rounds, "local_updates": readmission.LOCAL_UPDATES, "seed": readmission.SEED},
        "selection_rule": " ".join(choose.__doc__.split()),
        "grid": {"C": list(C_GRID), "rounds": list(ROUNDS_GRID)},
        "settings_selection_check_picks": picks,
        "features": sorted(data.x[0]),
        "seeds": list(SEEDS),
        "cv": "5-fold, stratified by hospital and label, repeated for each seed",
        "communication": {
            "values_per_message": n_params,
            "bytes_per_round": 2 * len(hospitals) * n_params * 8,
            "total_bytes": 2 * len(hospitals) * n_params * 8 * rounds,
            "hospital_to_server": "coef_ and intercept_ of the locally fitted model, and the row count n_k",
            "server_to_hospital": "the averaged coef_ and intercept_",
            "raw_rows_transferred": False,
        },
        "train_size": {h: int((data.hospitals == h).sum()) for h in hospitals},
        "train_prevalence": {h: float(data.y[data.hospitals == h].mean()) for h in hospitals},
        "runtime_seconds_per_cv_repeat": main_run["runtime_seconds_per_repeat"],
        "total_runtime_seconds": time.perf_counter() - started,
    }
    (RESULTS / "federated_experiment.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for row in model_rows(main_run):
        if row["site"] == "overall":
            print(f"{row['model']:12s} AUC {row['auc_mean']:.3f}±{row['auc_sd']:.3f}  "
                  f"Brier {row['brier_mean']:.3f}  log loss {row['log_loss_mean']:.3f}")
    print(f"{'check':12s} AUC {check[0]['auc_mean']:.3f}±{check[0]['auc_sd']:.3f}  "
          f"Brier {check[0]['brier_mean']:.3f}  picks {picks}")
    print(f"done in {summary['total_runtime_seconds']:.1f} s -> {RESULTS}")


if __name__ == "__main__":
    main()
