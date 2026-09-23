# AI usage log

One entry per working session. Source material for `AI_USAGE.md` (Phase 6).
Tool: Claude Code (Claude Opus 5), CLI, unless noted otherwise.

---

## 2026-09-23 — Phase 0: repo, environment, baseline

**Task.** Set up a clean working state and measure the starter baseline before any code changes.

**What the assistant did.**
- Verified the integrity of the read-only challenge assets. The command in the brief uses GNU
  `sha256sum -c --quiet`; the `sha256sum` on this machine (macOS 26.5.1, `/sbin/sha256sum`) is a BSD
  variant that does not accept `-c`, so the equivalent check was run with `shasum`:
  `grep -E '\./(data|evaluator|schemas)/' MANIFEST.sha256 | shasum -a 256 -c --quiet && echo OK`
  → **OK**, 10 files checked (`data/`, `evaluator/`, `schemas/`). No protected file was modified.
- `cp README.md docs/CHALLENGE_README.md` — preserves the challenge-provided README before it is
  replaced by our own in Phase 6.
- Created this log file.
- Ran `make install`, `make test`, `make evaluate`.

**Environment.** macOS 26.5.1 (arm64), Python 3.13.9 in `.venv`. Installed: numpy 2.5.3,
scikit-learn 1.9.1, scipy 1.18.1, pytest 9.1.1. Challenge version 1.0.0 (released 2026-09-03).
Note: the brief's Docker target is `python:3.11-slim`; local development runs on 3.13, so the
Phase 6 Docker smoke test must confirm the numbers still hold on 3.11.

**Tests.** `make test` → 3 passed (13.2 s).

**Baseline on validation (30 cases), reproduced exactly as stated in the brief:**

| Component | Score | Weight |
|---|---|---|
| De-identification | 0.7279 | 15 pts |
| Structured extraction | 0.8797 | 15 pts |
| Readmission prediction | 0.7726 | 10 pts |
| **Automated composite** | **0.7960** | **31.84 / 40** |

Component detail worth carrying into later phases (from `outputs/validation_report.json`):

- **De-id** — precision 1.00 everywhere, recall is the whole problem. Character recall 0.543
  (leakage rate 0.457), exact-entity recall 0.588, render-exact 1.00. Three labels score 0.0:
  `PATIENT_NAME` (0/30), `ADDRESS` (0/30), `CLINICIAN_NAME` (0/45) — the starter simply does not
  detect them. `DATE_OF_BIRTH`, `ENCOUNTER_DATE`, `PHONE_NUMBER`, `PATIENT_ID`, `EMAIL` are all
  at F1 1.00. So Phase 2 is essentially "add names and addresses without losing precision".
- **Extraction** — diagnoses recall 0.519 (P 1.00, F1 0.684), medications recall 0.864 (F1 0.927).
  Numeric: HR, SBP, LVEF at 1.00; creatinine 0.833 and hemoglobin 0.733 within tolerance — matches
  the unit-conversion gaps (µmol/L, g/L) flagged in the brief. Categorical accuracy 1.00 on both
  `smoking_status` and `allergy`.
- **Readmission** — ROC-AUC 0.819, AP 0.658, Brier 0.213, log loss 0.608, prevalence 0.20.
  Per site: Berlin AUC 0.810 (n=10), Chennai 1.000 (n=10), Hyderabad **0.222** (n=10).
  Two observations for Phase 4: the Brier of 0.213 is worse than predicting the constant
  prevalence (0.2 × 0.8 = 0.16), confirming the brief's note that `class_weight="balanced"` breaks
  calibration; and the per-site AUCs on n=10 are far too noisy to draw conclusions from — the
  repeated-CV protocol on train is the number to trust, not this.

**Verification.** Baseline matches the brief's stated 31.84 / 40 exactly, so the environment is
sound and the measurement is reproducible. Manifest check passed. Working tree otherwise untouched:
the only change is the new `docs/` directory.

**What Rasul changed or rejected.** Nothing — Phase 0 was environment setup and measurement only,
with no design decision to take. Reviewed and accepted the `shasum` substitution for the brief's
`sha256sum -c` (macOS ships a BSD variant with no check mode) and confirmed the baseline matched
the brief's stated 31.84 / 40 before moving on.

**State at end of session.** Tests green, baseline recorded, nothing else modified. No commit made
by the assistant (per the brief, Rasul commits). Phase 1 not started.

---

## 2026-09-23 — Phase 1: error-analysis script

**Task.** Build `scripts/error_analysis.py`, the diagnostic tool every later phase runs against,
plus its tests. No change to `src/` — this phase adds a lens, not a fix.

**What the assistant generated.**
- `scripts/error_analysis.py` (~420 lines). Loads a labelled split, runs the current de-id and
  extraction implementation, and prints four detail sections plus a count table:
  PII misses and false alarms per label with 30 characters of context; diagnoses/medications
  missed and extra with the sentence they came from; numeric fields outside the evaluator's
  tolerance; categorical mismatches. Flags: `--split {train,validation}`, `--case`, `--limit`,
  `--context`, `--summary-only`.
- `tests/test_error_analysis.py` (9 tests).

**Design decisions worth recording.**
- *Scoring is delegated, not re-implemented.* The summary table calls `evaluator.evaluate`'s own
  `score_pii` / `score_extraction` on in-memory dicts, and imports `NUMERIC_TOLERANCES`,
  `PII_LABELS`, `validate_spans`, `normalize_list`, `normalize_token` and `render_deidentified`.
  The script therefore cannot drift into being a second, subtly different scorer. The test
  `test_summary_matches_evaluator` asserts equality against a real `evaluate()` run.
- *The two splits have different shapes.* `train.jsonl` carries labels inline under `labels`;
  validation needs `validation_inputs.jsonl` joined to `validation_ground_truth.jsonl` on
  `case_id`. `load_split` normalises both to the evaluator's `(inputs, ground_truth)` form.
- *Ground-truth access.* This script reads the answer key deliberately. That is consistent with
  hard rule 2, which constrains `run_submission.py`; nothing in the submission path imports
  `scripts/`. Phase 6 adds the test that enforces the boundary.
- *Missed spans are classified*, not just listed: `not detected (leaked)` vs `wrong label` vs
  `boundary: overlaps predicted …`. For Phase 2 these need different fixes, and the 25% exact-entity
  term of the de-id score is decided by boundaries.
- *Evidence lookup for missed list items.* A missed `hypertension` is usually written `HTN`, and a
  missed `furosemide` as `Lasix`, so searching for the canonical words finds nothing. Two rules were
  added after seeing the first output: a multi-word term must match at least two of its words
  (otherwise "ischemic heart disease" was offered as evidence for `chronic_kidney_disease`), and
  when nothing matches, fall back to the note's diagnosis/medication *section* clause. Section
  headers only — mapping surface forms to canonical terms is Phase 3's job and stays out of this
  script. Two cue lists were trimmed after they produced noise: `assessment` (it heads the vitals
  section in this corpus) and `discharge` (it matched the `SYNTHETIC CHENNAI DISCHARGE NOTE`
  banner).

**Verification.**
- `make test` → **12 passed in 0.49 s** (3 pre-existing + 9 new).
- Runtime: 0.52 s for the full 120-case train split; 0.68 s for validation.
- `make evaluate` → **31.84 / 40**, unchanged, confirming `src/` was not touched.
- The done-criterion holds: `--split validation --summary-only` reproduces
  `outputs/validation_report.json` exactly (de-id 0.7279, extraction 0.8797, and every per-label
  tp/fp/fn), and the equality is asserted in a test rather than eyeballed.

**Worklist the tool produced (train split, 120 cases — the evidence Phases 2 and 3 will work from).**

De-identification, score 0.7303, character leakage 0.453 (8327 characters):

| Label | Missed | Note |
|---|---|---|
| `CLINICIAN_NAME` | 181 | every occurrence, incl. repeats; `Dr.` sits outside the true span |
| `PATIENT_NAME` | 120 | one per note |
| `ADDRESS` | 120 | one per note |
| others | 0 | DOB, encounter date, phone, ID, email are all at F1 1.00 |

Zero false alarms on either split, so Phase 2 is purely a recall problem — precision is the thing
to protect, not to win back.

Extraction, score 0.8483 on train (0.8797 on validation):

- Diagnoses recall 0.413 (93 tp / 132 fn). Biggest misses: `type_2_diabetes` 34, `hypertension` 21,
  `coronary_artery_disease` 19, `atrial_fibrillation` 16, `heart_failure` 15, `chronic_kidney_disease` 10,
  `copd` 9, `acute_coronary_syndrome` 4, `pneumonia` 4. Nearly all are abbreviations (HTN, CAD, AF,
  HFrEF, CKD, T2DM) or synonyms (`diabetes mellitus`, `chronic renal dysfunction`,
  `congestive cardiac failure`, `infective consolidation`, `Vorhofflimmern`).
- Medications recall 0.822 (236 tp / 51 fn): `aspirin` 18 (`ecosprin`, `acetylsalicylic acid`),
  `furosemide` 15 (`Lasix`), `rivaroxaban` 8 (`Xarelto`), `apixaban` 6 (`Eliquis`),
  `azithromycin` 3 (`azithro`), `atorvastatin` 1.
- False positives exist on train though not on validation: 3 × `atrial_fibrillation` from
  "Atrial fibrillation was considered but not confirmed" and 1 × `apixaban` from "Apixaban was
  discussed but was not started". Direct evidence for the negation work in Phase 3, and a reminder
  that validation's 30 cases hide failure modes that train's 120 expose.
- Numeric: creatinine outside tolerance in 29 cases and hemoglobin in 27 — every one of them a
  `null` where the note gives µmol/L or g/L. Unit conversion, not parsing. HR, SBP and LVEF are
  clean; categorical is 120/120 on both fields.

**What Rasul changed or rejected.** Asked for the design of the script before any code was written,
reviewed it, and approved the plan as proposed. Reviewed the generated output afterwards and
accepted the script as delivered; nothing was rejected. Two points were explicitly considered and
accepted rather than waved through: the evidence-lookup heuristics (the two-word rule and the
section fallback, including the removal of the `assessment` and `discharge` cues), and the decision
to delegate scoring to `evaluator.evaluate` — which guarantees agreement with `make evaluate` but
means this tool can only detect disagreement with the evaluator, never a bug inside it.
Separately, decided to fill Phase 0's review line in this session's commit rather than amend the
already-pushed commit, so that the log grows session by session.

**State at end of session.** Tests green, `make evaluate` unchanged at 31.84 / 40, `src/` untouched.
Phase 2 not started.
