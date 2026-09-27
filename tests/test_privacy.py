"""Tests for src/privacy.py (differential privacy inside each hospital node)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src import privacy

ROOT = Path(__file__).resolve().parents[1]
TRAIN = [json.loads(line) for line in (ROOT / "data" / "train.jsonl").open(encoding="utf-8")]


def small_hospital(seed: int = 0, n: int = 12) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.hstack([rng.normal(0, 1, (n, 3)), np.ones((n, 1))])
    y = (rng.random(n) < 0.4).astype(float)
    return x, y


# --------------------------------------------------------------------------- #
# The private local update
# --------------------------------------------------------------------------- #
def test_clipping_limits_every_patient_and_keeps_short_rows() -> None:
    rows = np.array([[3.0, 4.0], [0.3, 0.4], [0.0, 0.0]])
    clipped = privacy.clip_rows(rows, clip=1.0)
    assert np.all(np.linalg.norm(clipped, axis=1) <= 1.0 + 1e-12)
    np.testing.assert_allclose(clipped[0], [0.6, 0.8])      # long row shrunk, same direction
    np.testing.assert_allclose(clipped[1], [0.3, 0.4])      # short row unchanged


def test_without_noise_and_clipping_it_is_a_plain_gradient_step() -> None:
    x, y = small_hospital()
    w = np.array([0.2, -0.1, 0.3, 0.05])
    new_w, n = privacy.private_local_update(w, x, y, np.random.default_rng(0), clip=1e6, noise=0.0,
                                            learning_rate=0.5, l2=0.01)
    gradient = x.T @ (privacy.predict(w, x) - y) / len(y) + 0.01 * np.append(w[:-1], 0.0)
    np.testing.assert_allclose(new_w, w - 0.5 * gradient, atol=1e-12)
    assert n == len(y)


def test_update_returns_only_weights_and_patient_count() -> None:
    x, y = small_hospital()
    result = privacy.private_local_update(np.zeros(4), x, y, np.random.default_rng(1))
    assert isinstance(result, tuple) and len(result) == 2
    assert result[0].shape == (4,) and result[1] == len(y)


def test_noise_changes_the_update_and_is_repeatable_with_the_same_seed() -> None:
    x, y = small_hospital()
    quiet, _ = privacy.private_local_update(np.zeros(4), x, y, np.random.default_rng(3), noise=0.0)
    noisy_a, _ = privacy.private_local_update(np.zeros(4), x, y, np.random.default_rng(3), noise=8.0)
    noisy_b, _ = privacy.private_local_update(np.zeros(4), x, y, np.random.default_rng(3), noise=8.0)
    assert not np.allclose(quiet, noisy_a)
    np.testing.assert_array_equal(noisy_a, noisy_b)


def test_private_federated_training_is_repeatable_and_finite() -> None:
    first = privacy.train_private_federated(TRAIN, seed=19)
    second = privacy.train_private_federated(TRAIN, seed=19)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (len(privacy.feature_names()) + 1,) and np.all(np.isfinite(first))


# --------------------------------------------------------------------------- #
# Privacy accountant
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("noise", "expected"), [(1, 50.3), (2, 20.2), (4, 8.8), (8, 4.1)])
def test_epsilon_matches_reference_values_for_40_rounds(noise: float, expected: float) -> None:
    assert privacy.epsilon(noise, 40) == pytest.approx(expected, abs=0.05)


@pytest.mark.parametrize("noise", [0.5, 1, 2, 4, 8])
@pytest.mark.parametrize("rounds", [20, 50, 200])
def test_epsilon_is_never_below_the_closed_form_and_close_to_it(noise: float, rounds: int) -> None:
    grid, exact = privacy.epsilon(noise, rounds), privacy.epsilon_closed_form(noise, rounds)
    assert grid >= exact - 1e-9
    assert grid <= exact * 1.01


def test_epsilon_shrinks_with_more_noise_and_grows_with_more_rounds() -> None:
    by_noise = [privacy.epsilon(s, 50) for s in (0.5, 1, 2, 4, 8)]
    by_rounds = [privacy.epsilon(4, t) for t in (10, 20, 50, 100)]
    assert by_noise == sorted(by_noise, reverse=True)
    assert by_rounds == sorted(by_rounds)
    assert privacy.epsilon(0.0, 50) == float("inf")


# --------------------------------------------------------------------------- #
# Leakage demo
# --------------------------------------------------------------------------- #
def test_without_noise_one_patient_is_recovered_exactly_and_noise_stops_it() -> None:
    x, y = privacy.hospital_arrays(TRAIN)["BERLIN_NODE"]
    rows = {r["noise"]: r for r in privacy.leakage_demo(x[0], y[0], (0.0, 2.0, 8.0))}
    assert rows[0.0]["max_abs_error"] < 1e-9 and rows[0.0]["cosine_similarity"] > 0.999999
    assert rows[2.0]["cosine_similarity"] < 0.3 and rows[8.0]["cosine_similarity"] < 0.3


# --------------------------------------------------------------------------- #
# privacy_summary.json content
# --------------------------------------------------------------------------- #
def test_summary_contains_every_item_the_challenge_asks_for() -> None:
    summary = privacy.summary(TRAIN)
    required = {"mechanism", "implementation_status", "protected_asset", "adversary", "trust_assumptions",
                "parameters", "privacy_claim", "not_guaranteed", "utility_analysis", "runtime_analysis",
                "remaining_attack_surface", "limitations"}
    assert required <= set(summary)
    assert summary["parameters"]["epsilon"] == pytest.approx(privacy.epsilon(privacy.NOISE, privacy.ROUNDS), abs=0.01)
    json.dumps(summary)  # must be writable as JSON
