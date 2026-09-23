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

**What Rasul changed or rejected.** _(to fill in on review)_

**State at end of session.** Tests green, baseline recorded, nothing else modified. No commit made
by the assistant (per the brief, Rasul commits). Phase 1 not started.
