"""Monte Carlo estimates converge to the truth, at the rate the theory predicts.

Two separate claims, tested separately. The estimate lands inside its error band, and the
error falls like ``n^{-1/2}``. The second is the one that lets a reader budget a simulation,
so it is fitted and asserted rather than eyeballed.
"""

import numpy as np
import pytest

from src.evaluation.convergence import (
    error_curve,
    fit_convergence_rate,
    logspace_checkpoints,
    rmse_by_n,
    running_estimate,
)
from src.montecarlo.integration import estimate_pi, expectation, integrate, pi_estimates_at

N_DARTS = 200_000
N_REPLICATES = 200
CHECKPOINT_MAX = 20_000


def test_pi_estimate_lands_within_its_error_band(rng: np.random.Generator) -> None:
    result = estimate_pi(N_DARTS, rng)
    # The estimator is a mean of iid draws, so a 4-standard-error band fails about once in
    # 16_000 runs. The seed is fixed, so this is deterministic anyway.
    assert abs(result.value - np.pi) < 4 * result.se


def test_pi_standard_error_matches_the_binomial_prediction(rng: np.random.Generator) -> None:
    """Each dart is a Bernoulli(pi/4) trial scored as 4 * hit, so se = 4*sqrt(p(1-p)/n)."""
    result = estimate_pi(N_DARTS, rng)
    p = np.pi / 4
    predicted = 4 * np.sqrt(p * (1 - p) / N_DARTS)
    assert result.se == pytest.approx(predicted, rel=0.05)


def test_running_trace_ends_at_the_final_estimate(rng: np.random.Generator) -> None:
    result = estimate_pi(10_000, rng)
    assert result.running[-1] == pytest.approx(result.value)
    assert result.n == 10_000


def test_error_falls_at_the_square_root_rate(rng: np.random.Generator) -> None:
    """The headline rate. Measured across replicates, because one run is far too noisy."""
    checkpoints = logspace_checkpoints(CHECKPOINT_MAX, n_points=25, n_min=100)
    estimates = pi_estimates_at(checkpoints, N_REPLICATES, rng)
    rmse = rmse_by_n(estimates, np.pi)
    slope = fit_convergence_rate(checkpoints, rmse)
    # Theory says exactly -0.5. With 200 replicates the fitted slope is stable to about
    # +/-0.03; 0.06 leaves room without letting a genuinely wrong rate through.
    assert slope == pytest.approx(-0.5, abs=0.06)


def test_more_samples_really_do_help(rng: np.random.Generator) -> None:
    """100x the darts should buy about 10x the accuracy. Not more, which is the honest part."""
    checkpoints = np.array([200, 20_000])
    estimates = pi_estimates_at(checkpoints, N_REPLICATES, rng)
    rmse = rmse_by_n(estimates, np.pi)
    ratio = rmse[0] / rmse[1]
    assert ratio == pytest.approx(10.0, rel=0.35)


def test_integrate_recovers_a_known_integral(rng: np.random.Generator) -> None:
    """int_0^pi sin(x) dx = 2, a case with an exact answer to score against."""
    result = integrate(np.sin, 0.0, np.pi, 200_000, rng)
    assert abs(result.value - 2.0) < 4 * result.se


def test_integrate_recovers_an_integral_with_no_elementary_antiderivative(
    rng: np.random.Generator,
) -> None:
    """int_0^1 exp(-x^2) dx = sqrt(pi)/2 * erf(1). The case the method exists for."""
    from scipy import special

    truth = float(np.sqrt(np.pi) / 2 * special.erf(1.0))
    result = integrate(lambda x: np.exp(-(x**2)), 0.0, 1.0, 200_000, rng)
    assert abs(result.value - truth) < 4 * result.se


def test_expectation_of_a_function_of_a_random_variable(rng: np.random.Generator) -> None:
    """E[X^2] = 1 for a standard normal."""
    result = expectation(lambda x: x**2, lambda n, r: r.normal(0.0, 1.0, size=n), 200_000, rng)
    assert abs(result.value - 1.0) < 4 * result.se


def test_running_estimate_matches_a_hand_computed_mean() -> None:
    draws = np.array([1.0, 3.0, 2.0, 6.0])
    assert np.allclose(running_estimate(draws), [1.0, 2.0, 2.0, 3.0])


def test_error_curve_shrinks_as_draws_accumulate(rng: np.random.Generator) -> None:
    result = estimate_pi(5_000, rng)
    errors = error_curve(result.running, np.pi)
    assert errors.shape == result.running.shape
    # The error must be smaller late than early; compare averages to avoid single-point noise.
    assert errors[-500:].mean() < errors[:500].mean()


def test_checkpoints_are_increasing_and_bounded() -> None:
    checkpoints = logspace_checkpoints(20_000, n_points=25, n_min=100)
    assert checkpoints[0] >= 100
    assert checkpoints[-1] == 20_000
    assert np.all(np.diff(checkpoints) > 0)
