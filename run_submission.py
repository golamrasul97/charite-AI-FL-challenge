#!/usr/bin/env python3
"""Challenge entry point: de-identification, extraction, FedAvg readmission model.

Reads only ``--train`` and ``--input``; never opens a ground-truth file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src import readmission
from src.deid import detect_pii, render_deidentified
from src.extraction import extract_clinical_data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    args = parser.parse_args()

    train_records = read_jsonl(args.train)
    evaluation_records = read_jsonl(args.input)
    model = readmission.train_federated(train_records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in evaluation_records:
            note = record["note_text"]
            spans = detect_pii(note)
            extracted = extract_clinical_data(note)
            features = readmission.feature_dict(record, extracted)
            probability = float(model.predict_proba([features])[0, 1])
            prediction = {
                "case_id": record["case_id"],
                "pii_entities": spans,
                "deidentified_text": render_deidentified(note, spans),
                "extracted_clinical_data": extracted,
                "readmission_probability": float(np.clip(np.nan_to_num(probability, nan=0.5), 0.0, 1.0)),
            }
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    n_params = len(model.named_steps["vectorizer"].feature_names_) + 1
    n_hospitals = len(readmission.by_hospital(train_records))
    # Minimal summary; the full version with live CV metrics is added in Phase 6.
    experiment_summary = {
        "implementation": "fedavg",
        "federated_model": {
            "algorithm": "FedAvg",
            "model": "sklearn Pipeline(DictVectorizer, LogisticRegression)",
            "C": readmission.C,
            "rounds": readmission.ROUNDS,
            "local_iterations_per_round": readmission.LOCAL_UPDATES,
            "client_weighting": "number_of_training_cases",
            "seed": readmission.SEED,
            "features": list(model.named_steps["vectorizer"].feature_names_),
        },
        "submission_model": "federated_model",
        "communication": {
            "client_to_server": "coef_ and intercept_ (float64, length d+1) and row count n_k",
            "bytes_per_round": 2 * n_hospitals * n_params * 8,
            "raw_rows_transferred": False,
        },
        "results": "see results/ (scripts/run_experiments.py) for local / FedAvg / centralized metrics",
    }
    (args.artifacts_dir / "experiment_summary.json").write_text(
        json.dumps(experiment_summary, indent=2), encoding="utf-8"
    )
    privacy_summary = {
        "mechanism": None,
        "threat_model": "TODO",
        "privacy_guarantee": "TODO",
        "utility_analysis": "TODO",
        "limitations": "TODO",
    }
    (args.artifacts_dir / "privacy_summary.json").write_text(
        json.dumps(privacy_summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
