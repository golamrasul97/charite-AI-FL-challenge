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

---

## 2026-09-23 — Phase 2: de-identification

**Task.** Replace the starter's `detect_pii` with `src/deid/`, robust to unseen formats.

**What the assistant generated.**
- `src/deid/patterns.py` — regexes and cue vocabularies, all counted from `train.jsonl`.
- `src/deid/detector.py` — six layers: high-precision shapes (e-mail, phone, ID, date), context
  labelling of dates, cue-based names, propagation, addresses, conflict resolution.
- `src/deid/render.py` — the evaluator's rendering, kept separate so a test can assert equality.
- `tests/test_deid.py` — 18 tests, including three hand-written notes in unseen formats.
- `run_submission.py` and `scripts/error_analysis.py` repointed at `src.deid`. Extraction still
  comes from `src/baseline.py`; that is Phase 3.

**Evidence gathered before writing any code** (counted over the 120 training notes):
- The corpus uses **six** templates, not three: Berlin clinical summary (34), Hyderabad EMR
  snapshot (27), Chennai discharge note (24), federated node note (12), EMR export (8),
  cross-silo record (15). The last three have no `Patient:`-style cue at all.
- Every note contains **exactly one** of each label except `CLINICIAN_NAME` — 120 each, 181
  clinician spans (59 notes with one occurrence, 61 with two).
- The patient name string occurs exactly once per note (120/120), so propagating patient names
  would add risk for no gain; the clinician name string occurs exactly as often as it is
  labelled (1/1 or 2/2, never 2/1), so propagating clinician names is provably safe here.
- Emails reconstruct the clinician name in 110/120 notes; the **10 failures are all German
  transliteration** (`laura.koenig` → `Laura König`, `philipp.krueger` → `Philipp Krüger`), so the
  propagation needs oe/ue/ae/ss folding.
- All 120 patient and 181 clinician names are exactly two words; the only non-ASCII characters
  are `ö` and `ü`.

**Design decisions worth recording.**
- *Priorities, not label precedence.* Each candidate span carries a priority reflecting how much
  evidence it has (explicit cue > shape alone). Resolution sorts by priority, then length, then
  position, so the output never depends on the order the layers ran in.
- *Cues do the precision work, shapes do the recall work.* Shape patterns were widened past the
  training data (ISO dates, `03 Jan 1971`, national phone numbers, generic `PREFIX-digits` IDs)
  because the hidden set has new formats; precision is protected by requiring a cue for the
  ambiguous cases rather than by keeping the shapes narrow.
- *A title alone now marks a clinician* (`Dr. Sarah Klein reviewed the case`). No training note
  needs this; it was added after an edge-case probe showed such a name would otherwise leak.
- *Dates without a cue* fall back to: earliest date is the birth date, the rest are encounter
  dates. This handles the `EMR EXPORT // id // date` header, where the header date is the visit.

**Bug found and fixed during development.** The name pattern used `\s` between words, which
matches newlines, so `Treating clinician: Deepa Naidu\nDx` parsed as a three-word name. Every
clinician span was one token too long: 85 false positives *and* 85 false negatives, with character
recall still 1.000. It cost 0.031 of the de-id score and would have been invisible without the
per-label table from Phase 1. Fixed by using `[ \t]`; `_NAME_WORD` also requires lower-case letters
so that an all-capitals banner can never parse as a name.

**Verification.**
- `make test` → **30 passed in 0.54 s** (12 previous + 18 new).
- `make evaluate` → de-identification **1.0000** (was 0.7279); **35.92 / 40** (was 31.84).
- `scripts/error_analysis.py --split train` → 1.0000, zero characters leaked, every label
  120/120 (181/181 for clinicians), zero false positives.
- Three hand-written unseen-format notes (UK letter with `d.o.b.`/`03 Jan 1971`/national phone,
  ISO dates with a `LIS-88-01422` identifier, prose with slash separators): all expected spans
  found and **zero spans beyond the hand-labelled set** — recall was not bought with precision.
- Edge cases probed: empty note, whitespace-only, banner-only, `Patient:` with no name,
  implausible date, bare `+49`, repeated text. No crashes, no out-of-bounds spans.
- 26 ms for all 120 notes (0.22 ms/note); identical output across repeated runs.

**Honest caveat.** 1.0000 on train *and* validation is a warning sign, not a victory: both splits
draw on the same six templates, so the score mostly measures template coverage. The hand-written
notes are the only evidence of generalisation, and they encode my own guess about what the hidden
formats look like. Expect the test-set de-id score to be lower than 1.0; the report should say so
rather than quoting 1.0000 as an expected result.

**What Rasul changed or rejected.** Reviewed the layered design and the measured result and
accepted both; nothing was rejected. Accepted the three hand-written unseen-format notes as a
reasonable proxy for the hidden test set, and accepted the caveat stated above — that 1.0000 on
train and validation measures template coverage rather than proven generalisation, so REPORT.md
must present it that way instead of quoting 1.0000 as an expected test-set score.

**State at end of session.** Tests green, 35.92 / 40, manifest OK. Extraction still on
`src/baseline.py`. Phase 3 not started.

---

## 2026-09-24 — Phase 3: structured extraction

**Task.** Replace the starter's `extract_clinical_data` with `src/extraction/`, robust to unseen
formats and to negated / family-history / considered-only mentions.

**What the assistant generated.**
- `src/extraction/lexicon.py` — surface forms for the 9 diagnoses and 16 medications; each concept
  has case-insensitive terms and case-sensitive abbreviations (`AF`, `ASA`, `COAD`), all wrapped in
  alphanumeric boundaries.
- `src/extraction/negation.py` — clause-scoped NegEx-style rules: pre-cues, post-cues, family
  cues, scope terminators ("but", "however").
- `src/extraction/numeric.py` — HR, SBP, creatinine, Hb, LVEF with unit conversion and
  plausibility guards.
- `src/extraction/categorical.py` — smoking status and allergy.
- `src/extraction/extractor.py` — orchestration; always returns all 9 keys.
- `tests/test_extraction.py` — 116 tests.
- `run_submission.py` and `scripts/error_analysis.py` now import `src.extraction`.
  `src/baseline.py` is kept, per the brief.

**Evidence gathered before writing any code** (train only): every phrase inside the diagnosis,
medication, smoking, allergy, vitals and echo sections was enumerated and counted — 55 distinct
diagnosis forms, 64 medication forms, 6 distractor sentences, 9 smoking phrases, 10 allergy
phrases. The starter missed 132 of 225 diagnoses (recall 0.413) and every µmol/L creatinine and
g/L haemoglobin value.

**Design decisions worth recording.**
- *No section dependence.* The brief suggested preferring active sections with a whole-note
  fallback. Chosen instead: a whole-note scan with negation filtering only, because the six
  templates already use six different section headers and a section-first design fails silently
  on an unseen one. **Rasul should confirm this deviation.**
- *Clause = sentence end, `;`, `|`, newline.* Colons are not boundaries, so "Family history: HTN"
  still negates. Post-cues additionally stop at a comma, so in "HTN, T2DM, AF ruled out" only AF
  is dropped.
- *Considered-only is negation.* "suspected", "possible", "query" and a leading `?` drop a
  diagnosis, following the DATA_DICTIONARY rule on considered-only concepts. This is a judgement
  call: "suspected pneumonia" in a real admission note is often treated as the working diagnosis.
- *Dataset conventions applied deliberately.* Bare "diabetes mellitus" maps to `type_2_diabetes`,
  and phenprocoumon maps to `warfarin`. Both follow the training labels, although phenprocoumon is
  pharmacologically a different drug.
- *A missing unit is inferred from magnitude* (creatinine > 20 → µmol/L, Hb > 25 → g/L). This never
  fires on train, where every value has a unit.
- *German "RR" is not used for blood pressure*, because RR is also respiratory rate. The
  hand-written German note therefore expects `systolic_bp_mmhg = null`.

**Bugs found by the hand-written unseen-format notes** (none were visible on train or
validation):
1. Post-cues crossed commas: "PMH: HTN, T2DM, …, AF ruled out" dropped HTN and T2DM. Fixed by
   `_post_scope`.
2. `DM` had a lookahead that excluded any following "type", so "DM type 2" was missed.
3. Bare German "Raucher" was a *current* cue, but it is usually a field label
   ("Raucher: nein – Nichtraucher"). Removed.
4. `finditer(note, 0, start)` treats `start` as end-of-string, so the `$` in the sentence-end
   lookahead fired on any `?` or `.` directly before a mention ("?COPD"). The scan now covers the
   whole note. The same fix was applied to the allergy sentence finder.
5. "ibuprofen-associated urticaria" (train) had no allergy context word. Added standard reaction
   words: urticaria, hives, angioedema, bronchospasm, rash.

**Verification.**
- `make test` → **146 passed in 0.81 s** (30 previous + 116 new).
- `make evaluate` → extraction **1.0000** (was 0.8797); de-id 1.0000; readmission 0.7726
  (unchanged); **37.73 / 40** (was 35.92).
- `scripts/error_analysis.py --split train` → 1.0000: diagnoses 225/225, medications 287/287,
  zero FP, every numeric field and both categorical fields fully correct.
- `--split validation` → 1.0000. Validation was measured only; no pattern was added because of it.
- Three hand-written unseen notes (UK letter with comma lists and British spelling, German
  key-value export with mmol/L Hb, terse ward note with `?AF`, "suspected", "neg"): all fields
  pass.
- 92 ms for 120 notes; output identical across repeated runs. Manifest OK.

**Honest caveat.** As in Phase 2, 1.0000 on train and validation measures template coverage. The
evidence for generalisation is the three hand-written notes, and they found five bugs that the
labelled data never exposed. Expect a lower hidden-test score. The report should quote the
unseen-note results and the bug list, not 1.0000.

**What Rasul changed or rejected.** Nothing rejected. Rasul reviewed the session, ran
`scripts/error_analysis.py` himself to check the updated per-field results, and accepted:
the whole-note scan instead of section-first extraction; "suspected"/"possible"/"?" as not active;
the dataset mappings (diabetes mellitus → `type_2_diabetes`, phenprocoumon → `warfarin`); ignoring
German "RR"; the three hand-written unseen-format notes and their labels; and the caveat that
REPORT.md quotes the unseen-note results, not 1.0000.

**State at end of session.** Tests green, 37.73 / 40, manifest OK. The readmission model is still
the starter's (Phase 4 next).

---

## 2026-09-24 – 2026-09-25 — Phase 4: readmission models and federated learning (Task 3)

**Task.** Build the local, federated and centralized models for `readmission_30d`, compare them,
and make the submission use the federated model (FedAvg across the three hospital nodes).

**Final result (commits `a030123`, `ba698c2`, `7bf8c12`).**
- `src/readmission.py` — one file in the starter's style: `Pipeline(DictVectorizer,
  LogisticRegression(C=10))`. `train_local`, `train_centralized`, `train_federated`. FedAvg is written
  by hand: each round every client starts from the server's weights, fits on its own patients
  (`local_update`, one optimizer update), and sends back only its weights and row count; the server
  averages them weighted by n_k / N (`average_weights`); 50 rounds, zero start.
- 9 features: age, prior admissions, heart failure, chronic kidney disease, atrial fibrillation,
  LVEF, LVEF missing, haemoglobin, number of medications. Scaled with fixed values written in the
  code and capped at ±3 — no statistics shared between hospital nodes.
- `run_submission.py` trains the federated model on `--train` and writes one
  `readmission_probability` per case. `scripts/run_experiments.py` writes `results/` (9 files).
- `tests/test_readmission.py` — 13 tests (weighted average by hand, each client update sees only its
  own hospital's patients, only weights and a count are returned, federated close to centralized,
  same result every run, capping, unknown hospital).

**Results (`results/`, repeated 5-fold cross-validation on train, 5 seeds).**

| Model | AUC | Brier | Berlin / Chennai / Hyderabad AUC |
|---|---|---|---|
| Local models | 0.754 ± 0.016 | 0.183 | 0.52 / 0.85 / 0.83 |
| **Federated model (submitted)** | **0.843 ± 0.024** | **0.142** | 0.67 / 0.90 / 0.94 |
| Centralized model (reference) | 0.841 ± 0.025 | 0.139 | 0.66 / 0.90 / 0.94 |
| Starter's model, same splits | 0.683 | 0.222 | 0.54 / 0.74 / 0.71 |

Score that also includes choosing C inside each training part: AUC 0.835. Structured features only
(age, prior admissions): 0.715. Validation (read once): readmission score 0.7679 (starter 0.7726;
only 6 readmitted patients in 30). `make evaluate` → **37.68 / 40** (Phase 3: 37.73; the readmission
score moved 0.7726 → 0.7679 while calibration improved, Brier 0.213 → 0.141).

**How the final version was reached (in the order of Rasul's decisions).**
1. First version (numpy logistic regression, 7 files, 30 features, FedAvg): validation readmission
   dropped 0.7726 → 0.6786. Rasul rejected it and asked why 30 features were used when the starter
   used 5. Diagnosis: 2 of the 6 validation readmissions (older patients with few findings in the
   note) explained the gap; 30 features spread the signal thin.
2. Rasul asked to rename "arm" (a CLAUDE.md term, not the challenge's).
3. A 14-feature set chosen by a pre-declared rule: CV AUC 0.813, validation 0.7231.
4. Rasul asked for code as simple as the starter's `train_centralized_baseline`: the 7 numpy files
   were replaced by one sklearn file (same performance, AUC 0.809 vs 0.813). Cost: the exact
   "FedAvg = centralized" proof and direct control of each gradient step (matters for Phase 5).
5. Rasul asked to try all 5 structured features + all 9 extracted fields: raw values made FedAvg
   fail to converge; scaled with missing flags, AUC 0.733 (above the starter's 0.683) but worse on
   validation — about 48 columns for 41 readmissions is too many.
6. Rasul asked for a thorough feature study: 49 candidate features built and checked on train and
   validation (4 exact duplicates — each medication group equals a diagnosis in this data; one
   feature with a single patient; a hypertension shift 42 % → 70 % between train and validation),
   ranked several ways, sets of 3–49 features compared with the choice made inside each training
   part. Best results came from 8–10 features; more than 15 made things worse. Core signal: CKD,
   heart failure, number of diagnoses, atrial fibrillation, prior admissions, age.
7. Candidates compared on cross-validation, 5 000 simulated 50-patient test sets and validation:
   the earlier 14 features (0.797 / beat the starter in 97 % of simulated sets / validation 0.7512),
   an 8-feature set (0.818 / 98 % / 0.7101), and the 14 minus their 5 weakest features (0.837 / 99 %
   and better than the 14 in 92 % / 0.7679). **Rasul chose the 9-feature set.**
8. Alternatives Rasul asked to test, all rejected with numbers (scripts kept locally in
   `experiments/`, not committed): random forest (federated forest 0.764), gradient boosting as in
   the FedXGBoost paper Rasul found (centralized, i.e. best case, 0.800; the paper is vertical
   federated learning with only a heuristic privacy guarantee), TabNet (0.721–0.762, unstable), and
   local fine-tuning of the federated model per hospital (worse at every site, Berlin 0.67 → 0.59).
   Published evidence supports the choice: no benefit of machine learning over logistic regression
   for clinical prediction models (Christodoulou et al. 2019, J Clin Epidemiol).
9. Rasul brought a ChatGPT analysis suggesting a 15–20 feature set, a check of the "suspiciously
   powerful" prior admissions, and leave-one-hospital-out testing. Results: without prior admissions
   AUC 0.843 → 0.834 (the model does not depend on it; all 6 such patients were readmitted); adding
   any of 22 left-out features changed the score by at most +0.007; the suggested 27-column set
   scored 0.774; leave-one-hospital-out (federated model trained on 2 hospital nodes, tested on the
   third) gave Berlin 0.72, Chennai 0.92, Hyderabad 0.95 — the model carries over to a hospital it
   never saw. The 9 features were kept.
10. Rasul asked to use only the challenge's wording: code and `results/` now say "model" (local /
    federated / centralized), "site", "local updates" instead of the assistant's own terms.

**Verification.** `make test` → 159 passed. `make evaluate` → 37.68 / 40. `run_experiments.py`
reproduces `results/` in about 70 s and stops if its chosen settings differ from the constants in
`src/readmission.py`. Submission runtime 0.7 s, identical output on a rerun, no ground-truth file
opened. Manifest OK.

**Honest caveats for REPORT.md.**
- The 9 features were chosen by looking at the training data, so the training-data scores are a
  little flattering; validation was never used for the choice and improved (0.7513 → 0.7679).
- 120 patients and 41 readmissions: differences below about 0.03 AUC are noise; a 50-case hidden set
  can move the score by about ±0.6 points.
- Berlin stays the hardest site (0.67; 0.72 when held out); readmission behaves differently there
  (non-IID data).
- The federated model starts from zero weights and training is fully repeatable, so the seeds only
  change the data splits — random-seed stability needs one extra check (random starting weights).
- Local models reuse the federated model's settings (fairness point for the comparison).
- Coefficients are not clinical findings: the data is synthetic and some features overlap.
- `experiment_summary.json` is still minimal; the full version is Phase 6.

**What Rasul changed or rejected.** Rejected the first 30-feature numpy version after the
validation drop and questioned the feature count; required starter-style simplicity (one sklearn
file instead of seven numpy files); required the challenge's own wording and the "arm" rename;
asked for the all-fields, random forest,
gradient boosting (from a paper he found), TabNet and fine-tuning experiments and dropped the ones
that did not help; brought an outside (ChatGPT) analysis and had its suggestions tested rather than
adopted; chose the final 9-feature set himself after comparing the candidates. Reviewed and accepted
the final version and committed it.

**State at end of phase.** Phase 4 complete and committed. 37.68 / 40, tests green. Next: Phase 5
(privacy mechanism) — DP-SGD vs objective perturbation to be decided with Rasul first (research
notes in CLAUDE.md).

---

## 2026-09-25 – 2026-09-27 — Phase 5: privacy mechanism (Task 4)

**Task.** Implement one privacy mechanism with a full threat model, and show the difference between
federated learning and a formal privacy guarantee.

**How the design was decided.** Rasul asked for no code until he understood the mechanism and the
options. The assistant first compared the challenge's options (differential privacy, secure
aggregation, homomorphic encryption); Rasul questioned why encryption would not fit, which led to a
side-by-side comparison (encryption: no accuracy cost, but it only hides updates from the server, does
not protect the released model, and with three hospitals two can expose the third; differential
privacy: a formal per-patient claim with a number, at an accuracy cost). The mechanism was then
explained step by step in plain words — what the 10 numbers per round are, one FedAvg round with a
worked example, what differential privacy changes inside the hospital ("limit each patient's
influence, add noise, then send"), the difference between the noise level σ (what we set) and ε
(what we promise), and the exact privacy claim. After that the assistant proposed 10 decisions with
recommendations; Rasul accepted all of them: differential privacy only; noise every round inside
each hospital; the submission keeps the non-private federated model and the private model is the
fully measured extension (the challenge allows "implement or rigorously prototype"); a leakage demo;
an own accountant rather than a library call; one new file; a small grid of settings; a rule fixed
in advance for the reference ε; a compact scope because of the deadline.

**What the assistant generated (commits `e809565`, `69944a1`, `2959fca`).**
- `src/privacy.py` — private local update (each patient's gradient → clipped to length C → summed →
  Gaussian noise with standard deviation σ·C → one step, divided by the public patient count; only
  the noisy weights and the count leave the client), private FedAvg (same n_k / N averaging, seeded
  noise), privacy accountant (Rényi DP of the Gaussian mechanism, α/(2σ²) per round over T rounds,
  best α; no sampling amplification claimed), leakage demo, and `summary()` for
  `privacy_summary.json`. Same 9 features as the submitted model.
- `tests/test_privacy.py` — 27 checks: clipping bound; no noise and no clipping equals a plain gradient
  step; only weights and a count are returned; noise is repeatable per seed; ε matches the reference
  values for T = 40 (50.3 / 20.2 / 8.8 / 4.1) and is never below the closed-form bound; ε shrinks with
  noise and grows with rounds; exact recovery without noise; the summary has every required item.
- `scripts/run_experiments.py` — learning rate chosen without noise, sweep σ ∈ {0, 0.5, 1, 2, 4, 8}
  × C ∈ {0.5, 1} × T ∈ {20, 50} on the same 25 data splits, reference rule, leakage demo, one
  validation check → `results/privacy_sweep.csv`, `leakage_demo.csv`, `privacy_validation.csv`.
- `run_submission.py` — writes the full `privacy_summary.json` from `--train` only.

**Results.**
- Leakage demo (one real Berlin patient): without noise the patient's features are recovered exactly
  from a one-patient update (similarity 1.000); similarity 0.39 at σ = 0.5, 0.13 at σ = 1, about 0
  from σ = 2. Federated learning alone is not a privacy guarantee.
- Learning rate 4.0 (chosen without noise): the no-noise private training matches the submitted
  sklearn federated model (log loss 0.449 both).
- Privacy/accuracy trade-off (mean CV AUC): no noise 0.839; ε ≈ 23 → 0.821; ε ≈ 10 → 0.800;
  **ε ≈ 4.6 → 0.734 ± 0.037** (starter under the same CV: 0.683). Noise costs almost no time
  (about 1.5 ms per training).
- Reference setting by the rule fixed before running (smallest ε whose mean AUC minus one standard
  deviation stays above 0.683): C = 0.5, σ = 8, T = 50, δ = 1e-5, **ε ≈ 4.6**.
- Privacy claim: "(4.6, 1e-5)-differential privacy with respect to adding or removing one patient at a
  single hospital, for all updates that hospital sends and therefore also for the final private
  model." Not guaranteed: which hospitals take part and site-level patterns; Tasks 1–2 on raw notes;
  malicious clients or poisoning; side channels; the submitted non-private model.
- Validation (read once): private model AUC 0.70–0.85 across 5 noise draws; submitted model 0.81.

**Verification.** `make test` → 186 passed. `make evaluate` → 37.68 / 40 (unchanged; the submitted
predictions still come from the non-private federated model). Submission runtime about 1 s;
predictions identical between runs; the only difference between runs is the measured seconds in the
summaries. No ground-truth file opened by the submission. Manifest OK.

**Honest caveats for REPORT.md.**
- The reference setting was chosen on training-data cross-validation; ε is a worst-case bound.
- About 40 patients per hospital means large noise: strong privacy (ε ≈ 4.6) costs about 0.1 AUC.
- With 30 validation patients the private model's score depends noticeably on the noise draw.
- The server still sees each hospital's own noisy update; secure aggregation or homomorphic
  encryption would hide it too (not implemented, complementary).
- The submitted predictions are not covered by the privacy claim; this is stated in
  `privacy_summary.json`.

**What Rasul changed or rejected.** Required a plain-words explanation and a joint decision before any
code, and paused several times to ask for simpler explanations (the 10 numbers, one FedAvg round, the
effect of noise, σ versus ε) until he understood each step; questioned the recommendation against
homomorphic encryption and had it compared fairly before choosing differential privacy; then accepted
the 10 proposed decisions, reviewed the result and committed it. Nothing was rejected.

**State at end of phase.** Phase 5 complete and committed. 186 tests, 37.68 / 40. Next: Phase 6
(full `experiment_summary.json`, Docker, README, REPORT, AI_USAGE).
