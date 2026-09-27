#!/usr/bin/env python3
"""Challenge entry point: de-identification, extraction, federated readmission model.

Reads only ``--train`` and ``--input``; never opens a ground-truth file.
Writes one prediction per input case plus experiment_summary.json and
privacy_summary.json. A note that makes one step fail still gets a valid record
(see ``predict_case``) and a logged warning, so one bad case cannot stop the run.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src import experiment_summary, privacy, readmission
from src.deid import detect_pii, render_deidentified
from src.extraction import extract_clinical_data
from src.extraction.extractor import empty_record

log = logging.getLogger("run_submission")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one JSON object per line; a line that is not valid JSON is skipped with a warning."""
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                log.warning("%s line %d is not valid JSON and was skipped: %s", path, number, error)
    return records


def predict_case(record: dict[str, Any], model: Any, fallback_probability: float) -> dict[str, Any]:
    """One output record. Each step falls back safely if it fails:
    de-identification -> the whole note is redacted (nothing can leak);
    extraction -> all fields empty (null / []);
    prediction -> the training readmission rate."""
    case_id = str(record.get("case_id", ""))
    note = str(record.get("note_text") or "")
    try:
        spans = detect_pii(note)
    except Exception as error:  # noqa: BLE001 - keep the run going, redact everything
        log.warning("%s: de-identification failed (%s); the whole note is redacted", case_id, error)
        spans = [{"start": 0, "end": len(note), "label": "PATIENT_NAME"}] if note else []
    try:
        extracted = extract_clinical_data(note)
    except Exception as error:  # noqa: BLE001
        log.warning("%s: extraction failed (%s); fields left empty", case_id, error)
        extracted = empty_record()
    try:
        probability = float(model.predict_proba([readmission.feature_dict(record, extracted)])[0, 1])
        if not np.isfinite(probability):
            raise ValueError("probability is not finite")
    except Exception as error:  # noqa: BLE001
        log.warning("%s: prediction failed (%s); the training readmission rate is used", case_id, error)
        probability = fallback_probability
    return {
        "case_id": case_id,
        "pii_entities": spans,
        "deidentified_text": render_deidentified(note, spans),
        "extracted_clinical_data": extracted,
        "readmission_probability": float(np.clip(probability, 0.0, 1.0)),
    }


def write_json(path: Path, build: Any, name: str) -> None:
    """Write a summary; if building it fails, write the error instead of stopping the run."""
    try:
        content = build()
    except Exception as error:  # noqa: BLE001
        log.warning("%s could not be computed: %s", name, error)
        content = {"error": f"{name} could not be computed: {error}"}
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    train_records = read_jsonl(args.train)
    evaluation_records = read_jsonl(args.input)
    model = readmission.train_federated(train_records)
    fallback = float(np.mean(readmission.labels(train_records)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in evaluation_records:
            handle.write(json.dumps(predict_case(record, model, fallback), ensure_ascii=False) + "\n")

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.artifacts_dir / "experiment_summary.json",
               lambda: experiment_summary.summary(train_records), "experiment_summary.json")
    write_json(args.artifacts_dir / "privacy_summary.json",
               lambda: privacy.summary(train_records), "privacy_summary.json")


if __name__ == "__main__":
    main()
