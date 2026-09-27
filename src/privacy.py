"""Task 4 privacy mechanism: differential privacy inside each hospital node.

The federated model in ``src/readmission.py`` keeps patient rows at home, but
every round each client still sends numbers computed from its patients, and a
single patient's values can be read back exactly from an update (see
``leakage_demo``). Federated learning alone is therefore not a formal privacy
guarantee. This file adds one:

Private local update (inside each client, every round):
  1. each patient's gradient ("how this patient wants the weights to move");
  2. clip it to length at most ``clip`` (C), so no patient can move the sum by more than C;
  3. add the clipped gradients up and add Gaussian noise with standard deviation noise * C;
  4. divide by the client's (public) patient count and take one gradient step.
Only the noisy weights and the count leave the client. The server averages
them exactly as in FedAvg (weights n_k / N).

Privacy accountant: each round is one Gaussian mechanism with sensitivity C
(add/remove one patient) and noise multiplier sigma. Renyi DP of order alpha is
alpha / (2 sigma^2) per round, adds up over T rounds, and converts to
(epsilon, delta) as  epsilon = min over alpha of  T*alpha/(2 sigma^2) + ln(1/delta)/(alpha - 1)
(Mironov 2017; the moments accountant of Abadi et al. 2016). Training is full
batch, so no sampling amplification is claimed.

The same 9 features and fixed scaling as ``src/readmission.py`` are used.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from src import readmission

# Reference setting for the privacy claim, chosen by the rule in scripts/run_experiments.py
# (smallest epsilon whose mean train-CV AUC minus one sd stays above the starter's 0.683).
CLIP = 0.5          # C: largest allowed length of one patient's gradient
NOISE = 8.0         # sigma: noise standard deviation = NOISE * CLIP
ROUNDS = 50         # FedAvg rounds = noisy steps per client (T)
# Step size chosen without noise (lowest CV log loss). At 4.0 the no-noise model already matches
# the submitted sklearn federated model (log loss 0.449 both), so larger steps cannot help.
LEARNING_RATE = 4.0
L2 = 0.001          # same regularisation strength as C = 10 in src/readmission.py (1 / (C * N))
DELTA = 1e-5
STARTER_CV_AUC = 0.683  # the starter's model under the same cross-validation (results/)
ALPHAS = np.concatenate([np.linspace(1.01, 10, 900), np.linspace(10.1, 256, 2460)])


# --------------------------------------------------------------------------- #
# Data: the 9 features as numbers, with a constant 1 for the intercept
# --------------------------------------------------------------------------- #
def feature_names() -> list[str]:
    return sorted(readmission.feature_dict({"structured_features": {}}, {"diagnoses": [], "medications": []}))


def to_matrix(feature_dicts: list[dict[str, float]]) -> np.ndarray:
    """Rows of the 9 features in a fixed order, plus a last column of 1 (intercept)."""
    names = feature_names()
    x = np.array([[d[name] for name in names] for d in feature_dicts], dtype=float).reshape(-1, len(names))
    return np.hstack([x, np.ones((x.shape[0], 1))])


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # same as 1 / (1 + exp(-z)), without overflow


def predict(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return sigmoid(x @ w)


# --------------------------------------------------------------------------- #
# The private local update and the private federated model
# --------------------------------------------------------------------------- #
def clip_rows(gradients: np.ndarray, clip: float) -> np.ndarray:
    """Scale every row down to length at most ``clip`` (rows already shorter are unchanged)."""
    lengths = np.linalg.norm(gradients, axis=1, keepdims=True)
    return gradients * np.minimum(1.0, clip / np.maximum(lengths, 1e-12))


def private_local_update(
    w: np.ndarray, x: np.ndarray, y: np.ndarray, rng: np.random.Generator,
    clip: float = CLIP, noise: float = NOISE, learning_rate: float = LEARNING_RATE, l2: float = L2,
) -> tuple[np.ndarray, int]:
    """One client's noisy step. Returns only (new weights, patient count)."""
    per_patient = (predict(w, x) - y)[:, None] * x              # 1. one gradient per patient
    clipped = clip_rows(per_patient, clip)                       # 2. limit each patient
    noisy_sum = clipped.sum(axis=0) + rng.normal(0.0, noise * clip, size=w.shape)  # 3. add up + noise
    penalty = l2 * np.append(w[:-1], 0.0)                        # no penalty on the intercept
    new_w = w - learning_rate * (noisy_sum / len(y) + penalty)   # 4. step (count is public)
    return new_w, len(y)


def private_fedavg(
    hospital_data: dict[str, tuple[np.ndarray, np.ndarray]], seed: int = readmission.SEED,
    clip: float = CLIP, noise: float = NOISE, rounds: int = ROUNDS,
    learning_rate: float = LEARNING_RATE, l2: float = L2,
) -> np.ndarray:
    """FedAvg with private local updates over ``{hospital: (x, y)}``; returns the final weights.

    Every hospital draws its noise from its own seeded generator, so runs are repeatable.
    """
    n_columns = next(iter(hospital_data.values()))[0].shape[1]
    w = np.zeros(n_columns)
    generators = [np.random.default_rng([seed, k]) for k in range(len(hospital_data))]
    for _ in range(rounds):
        updates = [
            private_local_update(w, x, y, rng, clip, noise, learning_rate, l2)
            for (x, y), rng in zip(hospital_data.values(), generators)
        ]
        total = sum(n for _, n in updates)
        w = sum(w_k * n / total for w_k, n in updates)
    return w


def hospital_arrays(records: list[dict[str, Any]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        hospital: (to_matrix([readmission.feature_dict(r) for r in rows]), np.asarray(readmission.labels(rows), float))
        for hospital, rows in readmission.by_hospital(records).items()
    }


def train_private_federated(train_records: list[dict[str, Any]], **settings: Any) -> np.ndarray:
    return private_fedavg(hospital_arrays(train_records), **settings)


# --------------------------------------------------------------------------- #
# Privacy accountant
# --------------------------------------------------------------------------- #
def epsilon(noise: float, rounds: int, delta: float = DELTA) -> float:
    """Epsilon after ``rounds`` Gaussian steps with noise multiplier ``noise`` (RDP, best alpha)."""
    if noise <= 0:
        return float("inf")
    values = rounds * ALPHAS / (2 * noise**2) + np.log(1 / delta) / (ALPHAS - 1)
    return float(values.min())


def epsilon_closed_form(noise: float, rounds: int, delta: float = DELTA) -> float:
    """The same bound with the best alpha found by calculus; the grid result is never below it."""
    return rounds / (2 * noise**2) + 2 * np.sqrt(rounds * np.log(1 / delta) / (2 * noise**2))


# --------------------------------------------------------------------------- #
# Leakage demo: why federated learning alone is not a privacy guarantee
# --------------------------------------------------------------------------- #
def reconstruct_from_update(w_before: np.ndarray, w_after: np.ndarray, learning_rate: float, l2: float) -> np.ndarray:
    """What an honest-but-curious server can compute from one client's update.

    With one patient, the gradient is (p - y) * x and its intercept part is (p - y)
    because the last column of x is 1, so dividing recovers x exactly.
    """
    gradient = (w_before - w_after) / learning_rate - l2 * np.append(w_before[:-1], 0.0)
    return gradient / gradient[-1]


def leakage_demo(x_patient: np.ndarray, y_patient: float, noises: tuple[float, ...], seed: int = 7,
                 repeats: int = 200) -> list[dict[str, float]]:
    """Recover one patient's features from a one-patient update, without and with noise."""
    w0 = np.full(x_patient.shape[0], 0.1)
    rows = []
    for noise in noises:
        rng = np.random.default_rng(seed)
        errors, similarities = [], []
        for _ in range(repeats if noise > 0 else 1):
            # clip is set so large that it never changes the gradient; only the noise differs
            w1, _ = private_local_update(w0, x_patient[None, :], np.array([y_patient]), rng,
                                         clip=1e6 if noise == 0 else CLIP, noise=noise)
            guess = reconstruct_from_update(w0, w1, LEARNING_RATE, L2)[:-1]
            true = x_patient[:-1]
            errors.append(float(np.max(np.abs(guess - true))))
            similarities.append(float(guess @ true / (np.linalg.norm(guess) * np.linalg.norm(true) + 1e-12)))
        rows.append({"noise": noise, "max_abs_error": float(np.median(errors)),
                     "cosine_similarity": float(np.median(similarities))})
    return rows


# --------------------------------------------------------------------------- #
# privacy_summary.json (written by run_submission.py from --train only)
# --------------------------------------------------------------------------- #
def cross_validated_auc(train_records: list[dict[str, Any]], private: bool, seeds: tuple[int, ...] = (7, 19, 43),
                        folds: int = 5) -> tuple[float, float, float]:
    """Mean AUC, mean Brier and seconds per training, from repeated cross-validation on --train."""
    groups = hospital_arrays(train_records)
    x = np.vstack([g[0] for g in groups.values()]); y = np.concatenate([g[1] for g in groups.values()])
    site = np.concatenate([[h] * len(g[1]) for h, g in groups.items()])
    aucs, briers, seconds = [], [], []
    for seed in seeds:
        order = np.random.default_rng(seed).permutation(len(y))
        fold = np.empty(len(y), dtype=int); fold[order] = np.arange(len(y)) % folds
        p = np.zeros(len(y))
        for f in range(folds):
            train = {h: (x[(site == h) & (fold != f)], y[(site == h) & (fold != f)]) for h in groups}
            start = time.perf_counter()
            w = private_fedavg(train, seed=seed, noise=NOISE if private else 0.0, clip=CLIP if private else 1e6)
            seconds.append(time.perf_counter() - start)
            p[fold == f] = predict(w, x[fold == f])
        aucs.append(roc_auc_score(y, p)); briers.append(brier_score_loss(y, p))
    return float(np.mean(aucs)), float(np.mean(briers)), float(np.mean(seconds))


def summary(train_records: list[dict[str, Any]]) -> dict[str, Any]:
    eps = epsilon(NOISE, ROUNDS)
    private_auc, private_brier, private_s = cross_validated_auc(train_records, private=True)
    plain_auc, plain_brier, plain_s = cross_validated_auc(train_records, private=False)  # same training, no noise
    return {
        "mechanism": "Differential privacy inside each hospital node: each patient's gradient is clipped to "
                     "length C, the clipped gradients are summed, Gaussian noise (sd = sigma * C) is added, "
                     "then one gradient step is taken. Noise is added before anything leaves the client.",
        "implementation_status": "Implemented and evaluated (src/privacy.py, tests/test_privacy.py, "
                                 "results/privacy_sweep.csv). The submitted readmission_probability values come "
                                 "from the non-private federated model (src/readmission.py); the private model "
                                 "is the measured privacy extension.",
        "protected_asset": "One patient's record (features and readmission label) and whether that patient is in "
                           "a hospital's training data.",
        "adversary": "Honest-but-curious server; the other hospital nodes; anyone who sees the final model.",
        "trust_assumptions": [
            "Each hospital follows the protocol: clips every patient and adds the noise honestly.",
            "Noise comes from an unpredictable random source.",
            "Hospital patient counts are public (used to divide the noisy sum).",
            "Each patient belongs to one hospital only.",
        ],
        "parameters": {
            "clip_C": CLIP, "noise_multiplier_sigma": NOISE, "rounds_T": ROUNDS, "local_updates_per_round": 1,
            "learning_rate": LEARNING_RATE, "l2": L2, "delta": DELTA, "epsilon": round(eps, 2),
            "neighbouring_relation": "add or remove one patient at one hospital (sensitivity C; it would be 2C "
                                     "under replace-one)",
            "accountant": "Renyi DP of the Gaussian mechanism, alpha/(2 sigma^2) per round, summed over T rounds, "
                          "converted to (epsilon, delta) with the best alpha; full batch, no sampling amplification",
        },
        "privacy_claim": f"({eps:.1f}, {DELTA:g})-differential privacy with respect to adding or removing one "
                         "patient at a single hospital, for all updates that hospital sends and therefore also for "
                         "the final private model (post-processing).",
        "not_guaranteed": [
            "Which hospitals take part, and hospital-level patterns (e.g. readmission rates per site).",
            "De-identification and extraction (Tasks 1-2) run on raw notes and are not covered.",
            "A hospital that does not add noise or sends manipulated updates (malicious clients, poisoning).",
            "Side channels such as timing, message size or logs; predictions for new patients.",
            "The submitted non-private federated model.",
        ],
        "utility_analysis": {
            "method": "repeated cross-validation on --train (3 seeds x 5 folds), recomputed at every run; "
                      "full sweep over sigma, C and rounds in results/privacy_sweep.csv",
            "same_training_without_noise_auc": round(plain_auc, 3),
            "same_training_without_noise_brier": round(plain_brier, 3),
            "private_federated_auc": round(private_auc, 3), "private_federated_brier": round(private_brier, 3),
            "starter_auc_repeated_cv_on_train": STARTER_CV_AUC,
            "more_results": "results/privacy_sweep.csv (epsilon vs AUC/Brier/time), results/leakage_demo.csv "
                            "(one patient recovered exactly without noise), results/privacy_validation.csv",
        },
        "runtime_analysis": {
            "seconds_per_training_without_noise": round(plain_s, 4),
            "seconds_per_training_private": round(private_s, 4),
            "note": "Clipping and noise add almost no computation; the cost of privacy is accuracy, not time.",
        },
        "remaining_attack_surface": [
            "The server still sees each hospital's own noisy update; secure aggregation or homomorphic encryption "
            "would hide it as well (not implemented; complementary to differential privacy).",
            "With only three hospitals the averaged model reveals site-level information.",
            "Epsilon is a worst-case bound; small hospitals (about 40 patients) need large noise, so strong privacy "
            "costs noticeable accuracy.",
        ],
        "limitations": [
            "Hospital sizes are treated as public.",
            "The reference setting was chosen on training-data cross-validation.",
            "No privacy for the non-private model that produces the submitted predictions.",
        ],
    }
