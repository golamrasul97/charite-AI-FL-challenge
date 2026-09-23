#!/usr/bin/env python3
"""Per-field error analysis for the de-identification and extraction stages.

The evaluator reports aggregate scores; this script reports the concrete strings
that were missed or invented, so that lexicons and patterns can be developed
against evidence instead of against a moving benchmark number.

    python scripts/error_analysis.py                      # train split (default)
    python scripts/error_analysis.py --split validation   # 30 labelled cases
    python scripts/error_analysis.py --case TRN-0007      # one case, full detail
    python scripts/error_analysis.py --summary-only       # just the count table

Scoring is delegated to ``evaluator.evaluate``: tolerances, normalisation and the
span bookkeeping are imported rather than re-implemented, so the summary table
cannot drift away from ``make evaluate``.

This is a development tool. It reads ground truth on purpose and is never
imported by ``run_submission.py``; the submission path stays label-free.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluator.evaluate import (  # noqa: E402  (path set up above)
    CATEGORICAL_FIELDS,
    LIST_FIELDS,
    NUMERIC_TOLERANCES,
    PII_LABELS,
    index_by_case,
    normalize_list,
    normalize_token,
    numeric_value,
    read_jsonl,
    render_deidentified,
    score_extraction,
    score_pii,
    validate_spans,
)

# The single place that names the implementation under analysis. Phase 2 and 3
# repoint these at src.deid / src.extraction; nothing else in this file changes.
from src.baseline import detect_pii as _detect_pii  # noqa: E402
from src.baseline import extract_clinical_data as _extract_clinical_data  # noqa: E402

DATA_DIR = REPO_ROOT / "data"

Span = tuple[int, int, str]

# Display-only cues: used to locate the line of the note a reviewer should look
# at. They are deliberately not extraction logic and must not grow into a
# lexicon -- that belongs in src/extraction/ (Phase 3).
NUMERIC_CUES: dict[str, tuple[str, ...]] = {
    "heart_rate_bpm": ("hr", "pulse", "heart rate", "ventricular rate"),
    "systolic_bp_mmhg": ("bp", "blood pressure", "systolic"),
    "creatinine_mg_dl": ("creatinine", "creat", "renal"),
    "hemoglobin_g_dl": ("hemoglobin", "haemoglobin", "hb", "haemogram"),
    "lvef_percent": ("lvef", "ef", "ejection fraction"),
}
CATEGORICAL_CUES: dict[str, tuple[str, ...]] = {
    "smoking_status": ("smok", "tobacco", "cigarette", "nicotine", "pack year"),
    "allergy": ("allerg", "nkda", "penicillin", "nsaid", "contrast", "intoleran"),
}
# Section headers, not clinical terms: where to look when a missed item is written
# as an abbreviation or a brand name and so shares no words with its canonical form.
SECTION_CUES: dict[str, tuple[str, ...]] = {
    "diagnoses": ("dx", "diagnos", "problem", "impression", "history of"),
    "medications": ("med", "rx", "therapy", "drug", "treatment"),
}
SECTION_CUE_WINDOW = 30  # a header, not an incidental mention later in the clause
EVIDENCE_MAX_CHARS = 220


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_split(split: str) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return (inputs_by_case, ground_truth_by_case) in the evaluator's shape.

    The two splits are stored differently: train.jsonl carries its labels inline
    under "labels", while the validation labels live in a separate file that has
    to be joined on case_id.
    """
    if split == "train":
        records = read_jsonl(DATA_DIR / "train.jsonl")
        inputs = index_by_case(records, "train inputs")
        ground_truth = {
            case_id: {
                "case_id": case_id,
                "hospital_id": record.get("hospital_id", "UNKNOWN"),
                **record.get("labels", {}),
            }
            for case_id, record in inputs.items()
        }
        return inputs, ground_truth

    if split == "validation":
        inputs = index_by_case(read_jsonl(DATA_DIR / "validation_inputs.jsonl"), "validation inputs")
        ground_truth = index_by_case(
            read_jsonl(DATA_DIR / "validation_ground_truth.jsonl"), "validation ground truth"
        )
        missing = sorted(set(ground_truth) - set(inputs))
        if missing:
            raise ValueError(f"Ground-truth cases missing from inputs: {missing[:5]}")
        return inputs, ground_truth

    raise ValueError(f"Unknown split {split!r}")


def predict(
    inputs: dict[str, dict[str, Any]],
    detect: Callable[[str], list[dict[str, Any]]] = _detect_pii,
    extract: Callable[[str], dict[str, Any]] = _extract_clinical_data,
) -> dict[str, dict[str, Any]]:
    """Run the current de-id and extraction implementation over every case."""
    predictions: dict[str, dict[str, Any]] = {}
    for case_id, record in inputs.items():
        note = str(record["note_text"])
        spans = detect(note)
        predictions[case_id] = {
            "case_id": case_id,
            "pii_entities": spans,
            "deidentified_text": render_deidentified(note, spans),
            "extracted_clinical_data": extract(note),
        }
    return predictions


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def flatten(text: str) -> str:
    return text.replace("\n", "\\n").replace("\t", " ")


def context_snippet(note: str, start: int, end: int, width: int) -> str:
    """The span in guillemets, with `width` characters either side."""
    left = note[max(0, start - width) : start]
    right = note[end : end + width]
    prefix = "..." if start - width > 0 else ""
    suffix = "..." if end + width < len(note) else ""
    return f"{prefix}{flatten(left)}<<{flatten(note[start:end])}>>{flatten(right)}{suffix}"


def split_sentences(note: str) -> list[str]:
    parts = re.split(r"(?<=[.;:])\s+|\n+", note)
    return [part.strip() for part in parts if part.strip()]


def split_clauses(note: str) -> list[str]:
    """Like split_sentences but keeps semicolon lists intact, so that a section
    such as ``Dx: ACS; HFrEF; HTN`` stays in one piece."""
    parts = re.split(r"(?<=\.)\s+|\n+", note)
    return [part.strip() for part in parts if part.strip()]


def truncate(text: str, limit: int = EVIDENCE_MAX_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def sentences_matching(note: str, pattern: re.Pattern[str], limit: int = 2) -> list[str]:
    hits = [sentence for sentence in split_sentences(note) if pattern.search(sentence)]
    return hits[:limit]


def cue_pattern(cues: Iterable[str]) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(re.escape(cue) for cue in cues) + r")", re.IGNORECASE)


def evidence_for(note: str, canonical: str, field: str, limit: int = 2) -> list[str]:
    """Sentences that plausibly carry `canonical`, else the relevant section lines.

    A multi-word term must match at least two of its words, so that
    chronic_kidney_disease is not "explained" by the word "disease" in
    "ischemic heart disease". When nothing matches -- the usual case for
    abbreviations (HTN, AF) and brand names (Lasix, Eliquis) -- fall back to the
    note's diagnosis or medication section, which is where the surface form
    almost certainly is. The fallback is keyed on section headers only; mapping
    surface forms to canonical terms is Phase 3's job, not this script's.
    """
    words = [word for word in canonical.split("_") if len(word) > 3] or canonical.split("_")
    required = min(2, len(words))
    scored: list[tuple[int, str]] = []
    for sentence in split_sentences(note):
        hits = sum(1 for word in words if re.search(rf"\b{re.escape(word)}", sentence, re.IGNORECASE))
        if hits >= required:
            scored.append((hits, sentence))
    if scored:
        scored.sort(key=lambda pair: -pair[0])
        return [truncate(sentence) for _, sentence in scored[:limit]]

    pattern = cue_pattern(SECTION_CUES[field])
    clauses: list[str] = []
    for clause in split_clauses(note):
        match = pattern.search(clause)
        if match and match.start() < SECTION_CUE_WINDOW:
            clauses.append(f"(section) {truncate(clause)}")
    return clauses[:limit]


def header(title: str) -> str:
    return f"\n{'=' * 78}\n{title}\n{'=' * 78}"


def subheader(title: str) -> str:
    return f"\n--- {title} " + "-" * max(0, 74 - len(title))


# --------------------------------------------------------------------------- #
# PII
# --------------------------------------------------------------------------- #
def describe_missed_span(missed: Span, pred_spans: list[Span], note: str) -> str:
    """Say whether a missed span leaked entirely, or was only mis-bounded/mis-labelled."""
    start, end, _ = missed
    same_offsets = [span for span in pred_spans if span[0] == start and span[1] == end]
    if same_offsets:
        return f"wrong label: predicted {same_offsets[0][2]}"
    overlapping = [span for span in pred_spans if span[0] < end and span[1] > start]
    if overlapping:
        shown = ", ".join(f"{span[2]} {note[span[0]:span[1]]!r}" for span in overlapping[:2])
        return f"boundary: overlaps predicted {shown}"
    return "not detected (leaked)"


def report_pii(
    inputs: dict[str, dict[str, Any]],
    ground_truth: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    limit: int,
    width: int,
) -> None:
    missed: dict[str, list[tuple[str, Span, str, str]]] = defaultdict(list)
    spurious: dict[str, list[tuple[str, Span, str]]] = defaultdict(list)
    render_mismatches: list[str] = []
    errors: list[str] = []

    for case_id, gt in ground_truth.items():
        note = str(inputs[case_id]["note_text"])
        true_spans = validate_spans(gt.get("pii_entities", []), note, f"GT/{case_id}", errors)
        pred_spans = validate_spans(
            predictions.get(case_id, {}).get("pii_entities", []), note, case_id, errors
        )
        true_set = {(span["start"], span["end"], span["label"]) for span in true_spans}
        pred_set = {(span["start"], span["end"], span["label"]) for span in pred_spans}
        pred_list = sorted(pred_set)

        for span in sorted(true_set - pred_set):
            reason = describe_missed_span(span, pred_list, note)
            missed[span[2]].append((case_id, span, context_snippet(note, span[0], span[1], width), reason))
        for span in sorted(pred_set - true_set):
            spurious[span[2]].append((case_id, span, context_snippet(note, span[0], span[1], width)))

        rendered = predictions.get(case_id, {}).get("deidentified_text")
        if not isinstance(rendered, str) or rendered != render_deidentified(note, pred_spans):
            render_mismatches.append(case_id)

    print(header("PII: missed spans (false negatives = leakage)"))
    if not any(missed.values()):
        print("  none")
    for label in PII_LABELS:
        items = missed.get(label, [])
        if not items:
            continue
        print(subheader(f"{label}  ({len(items)} missed)"))
        for case_id, span, snippet, reason in items[:limit]:
            print(f"  {case_id} [{span[0]}:{span[1]}] {reason}")
            print(f"      {snippet}")
        if len(items) > limit:
            print(f"  ... {len(items) - limit} more")

    print(header("PII: false alarms (false positives)"))
    if not any(spurious.values()):
        print("  none")
    for label in PII_LABELS:
        items = spurious.get(label, [])
        if not items:
            continue
        print(subheader(f"{label}  ({len(items)} spurious)"))
        for case_id, span, snippet in items[:limit]:
            print(f"  {case_id} [{span[0]}:{span[1]}]")
            print(f"      {snippet}")
        if len(items) > limit:
            print(f"  ... {len(items) - limit} more")

    if render_mismatches:
        print(subheader(f"deidentified_text does not match the evaluator's rendering ({len(render_mismatches)})"))
        for case_id in render_mismatches[:limit]:
            print(f"  {case_id}")
    if errors:
        print(subheader(f"span validation messages ({len(errors)})"))
        for message in errors[:limit]:
            print(f"  {message}")


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def report_list_fields(
    inputs: dict[str, dict[str, Any]],
    ground_truth: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    limit: int,
) -> None:
    for field in LIST_FIELDS:
        missed: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
        extra: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
        for case_id, gt in ground_truth.items():
            note = str(inputs[case_id]["note_text"])
            true_values = set(normalize_list(gt.get("extracted_clinical_data", {}).get(field)))
            pred_values = set(
                normalize_list(
                    predictions.get(case_id, {}).get("extracted_clinical_data", {}).get(field)
                )
            )
            for value in sorted(true_values - pred_values):
                missed[value].append((case_id, evidence_for(note, value, field)))
            for value in sorted(pred_values - true_values):
                extra[value].append((case_id, evidence_for(note, value, field)))

        print(header(f"{field}: missed (in truth, not predicted)"))
        if not missed:
            print("  none")
        for value, items in sorted(missed.items(), key=lambda pair: (-len(pair[1]), pair[0])):
            print(subheader(f"{value}  ({len(items)} cases)"))
            for case_id, sentences in items[:limit]:
                print(f"  {case_id}")
                if sentences:
                    for sentence in sentences:
                        print(f"      {flatten(sentence)}")
                else:
                    print("      (nothing found -- neither the canonical words nor a section header)")
            if len(items) > limit:
                print(f"  ... {len(items) - limit} more")

        print(header(f"{field}: extra (predicted, not in truth)"))
        if not extra:
            print("  none")
        for value, items in sorted(extra.items(), key=lambda pair: (-len(pair[1]), pair[0])):
            print(subheader(f"{value}  ({len(items)} cases)"))
            for case_id, sentences in items[:limit]:
                print(f"  {case_id}")
                for sentence in sentences:
                    print(f"      {flatten(sentence)}")
            if len(items) > limit:
                print(f"  ... {len(items) - limit} more")


def report_numeric_fields(
    inputs: dict[str, dict[str, Any]],
    ground_truth: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    limit: int,
) -> None:
    print(header("numeric fields: outside the evaluator's tolerance"))
    any_problem = False
    for field, tolerance in NUMERIC_TOLERANCES.items():
        problems: list[tuple[str, Any, Any, str, list[str]]] = []
        for case_id, gt in ground_truth.items():
            note = str(inputs[case_id]["note_text"])
            true_value = numeric_value(gt.get("extracted_clinical_data", {}).get(field))
            pred_value = numeric_value(
                predictions.get(case_id, {}).get("extracted_clinical_data", {}).get(field)
            )
            if true_value is None and pred_value is None:
                continue
            if true_value is None:
                kind = "invented (truth is null)"
            elif pred_value is None:
                kind = "missing (truth has a value)"
            elif abs(pred_value - true_value) <= tolerance:
                continue
            else:
                kind = f"off by {abs(pred_value - true_value):.2f} (tolerance {tolerance:g})"
            problems.append(
                (case_id, true_value, pred_value, kind, sentences_matching(note, cue_pattern(NUMERIC_CUES[field])))
            )

        if not problems:
            continue
        any_problem = True
        print(subheader(f"{field}  ({len(problems)} cases outside tolerance)"))
        for case_id, true_value, pred_value, kind, sentences in problems[:limit]:
            print(f"  {case_id}  truth={true_value}  pred={pred_value}  {kind}")
            for sentence in sentences:
                print(f"      {flatten(sentence)}")
        if len(problems) > limit:
            print(f"  ... {len(problems) - limit} more")
    if not any_problem:
        print("  none")


def report_categorical_fields(
    inputs: dict[str, dict[str, Any]],
    ground_truth: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    limit: int,
) -> None:
    print(header("categorical fields: mismatches"))
    any_problem = False
    for field in CATEGORICAL_FIELDS:
        problems: list[tuple[str, str, str, list[str]]] = []
        for case_id, gt in ground_truth.items():
            note = str(inputs[case_id]["note_text"])
            true_value = normalize_token(gt.get("extracted_clinical_data", {}).get(field))
            pred_value = normalize_token(
                predictions.get(case_id, {}).get("extracted_clinical_data", {}).get(field)
            )
            if true_value == pred_value:
                continue
            problems.append(
                (case_id, true_value, pred_value, sentences_matching(note, cue_pattern(CATEGORICAL_CUES[field])))
            )

        if not problems:
            continue
        any_problem = True
        print(subheader(f"{field}  ({len(problems)} mismatches)"))
        for case_id, true_value, pred_value, sentences in problems[:limit]:
            print(f"  {case_id}  truth={true_value}  pred={pred_value}")
            for sentence in sentences:
                print(f"      {flatten(sentence)}")
        if len(problems) > limit:
            print(f"  ... {len(problems) - limit} more")
    if not any_problem:
        print("  none")


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def summarize(
    inputs: dict[str, dict[str, Any]],
    ground_truth: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Component metrics straight from the evaluator, so the table cannot drift."""
    errors: list[str] = []
    return {
        "deidentification": score_pii(inputs, ground_truth, predictions, errors),
        "structured_extraction": score_extraction(ground_truth, predictions, errors),
        "validation_messages": errors,
    }


def print_summary(split: str, n_cases: int, metrics: dict[str, Any]) -> None:
    pii = metrics["deidentification"]
    extraction = metrics["structured_extraction"]
    character = pii["character_detection"]
    entity = pii["exact_entity"]

    print(header(f"SUMMARY  split={split}  cases={n_cases}"))
    print(f"de-identification score      {pii['score']:.4f}")
    print(
        f"  character    P {character['precision']:.3f}  R {character['recall']:.3f}"
        f"  F1 {character['f1']:.3f}   leakage {character['leakage_rate']:.3f}"
        f"  ({character['fn_chars']} chars leaked)"
    )
    print(
        f"  exact entity P {entity['precision']:.3f}  R {entity['recall']:.3f}"
        f"  F1 {entity['f1']:.3f}   macro F1 {entity['macro_f1']:.3f}"
    )
    print(
        f"  label-char   F1 {pii['label_aware_character']['f1']:.3f}"
        f"   render-exact {pii['deidentified_text_render_exact_rate']:.3f}"
    )
    print(f"  {'label':<16}{'tp':>6}{'fp':>6}{'fn':>6}{'P':>8}{'R':>8}{'F1':>8}")
    for label in PII_LABELS:
        counts = entity["per_label"][label]
        print(
            f"  {label:<16}{counts['tp']:>6}{counts['fp']:>6}{counts['fn']:>6}"
            f"{counts['precision']:>8.3f}{counts['recall']:>8.3f}{counts['f1']:>8.3f}"
        )

    print(f"\nstructured extraction score  {extraction['score']:.4f}")
    for field in LIST_FIELDS:
        counts = extraction[field]
        print(
            f"  {field:<16}tp {counts['tp']:>4}  fp {counts['fp']:>4}  fn {counts['fn']:>4}"
            f"   P {counts['precision']:.3f}  R {counts['recall']:.3f}  F1 {counts['f1']:.3f}"
        )
    print(f"  {'numeric field':<20}{'tol':>7}{'within':>9}{'score':>8}{'MAE':>9}")
    for field, values in extraction["numeric"].items():
        mae = values["mean_absolute_error_nonmissing_pairs"]
        mae_text = f"{mae:>9.2f}" if mae is not None else f"{'-':>9}"
        print(
            f"  {field:<20}{values['tolerance']:>7g}{values['within_tolerance_rate']:>9.3f}"
            f"{values['score']:>8.3f}{mae_text}"
        )
    for field in CATEGORICAL_FIELDS:
        values = extraction["categorical"][field]
        print(f"  {field:<20}accuracy {values['accuracy']:.3f}  ({values['correct']}/{values['n']})")

    if metrics["validation_messages"]:
        print(f"\nvalidation messages ({len(metrics['validation_messages'])}):")
        for message in metrics["validation_messages"][:10]:
            print(f"  - {message}")

    print(
        "\nnote: readmission prediction is a model output, not a per-note rule, and is"
        "\n      therefore scored by make evaluate rather than here."
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--case", help="restrict the analysis to a single case_id")
    parser.add_argument("--limit", type=int, default=15, help="examples printed per group (default 15)")
    parser.add_argument("--context", type=int, default=30, help="context characters around a span (default 30)")
    parser.add_argument("--summary-only", action="store_true", help="print only the count table")
    args = parser.parse_args(argv)

    inputs, ground_truth = load_split(args.split)
    if args.case:
        if args.case not in ground_truth:
            print(f"Unknown case_id {args.case!r} in split {args.split!r}", file=sys.stderr)
            return 2
        inputs = {args.case: inputs[args.case]}
        ground_truth = {args.case: ground_truth[args.case]}

    predictions = predict(inputs)

    if not args.summary_only:
        report_pii(inputs, ground_truth, predictions, args.limit, args.context)
        report_list_fields(inputs, ground_truth, predictions, args.limit)
        report_numeric_fields(inputs, ground_truth, predictions, args.limit)
        report_categorical_fields(inputs, ground_truth, predictions, args.limit)

    print_summary(args.split, len(ground_truth), summarize(inputs, ground_truth, predictions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
