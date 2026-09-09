"""Rejection sampling hits its target distribution and its theoretical acceptance rate."""

import numpy as np
import pytest
from scipy import stats

from src.draw.rejection import check_envelope, rejection_sample

N = 20_000
ALPHA = 0.01

# Target: standard normal restricted to [-4, 4]. Proposal: uniform on the same interval.
# The envelope constant is then m = 8 * max(phi) = 8 / sqrt(2*pi), so the theoretical
# acceptance rate is 1/m = sqrt(2*pi)/8 ~= 0.313.
LO, HI = -4.0, 4.0
M = (HI - LO) / np.sqrt(2.0 * np.pi)


def target_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)


def proposal_pdf(x: np.ndarray) -> np.ndarray:
    return np.full_like(x, 1.0 / (HI - LO))


def proposal_sampler(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(LO, HI, size=n)


def test_rejection_sampling_matches_target(rng: np.random.Generator) -> None:
    result = rejection_sample(
        target_pdf,
        proposal_sampler,
        proposal_pdf,
        M,
        N,
        rng,
        check_grid=np.linspace(LO, HI, 1000),
    )
    assert result.samples.size == N
    # Truncated at +/-4 the normal loses 6e-5 of its mass, far below KS resolution here.
    assert stats.kstest(result.samples, stats.norm.cdf).pvalue > ALPHA


def test_acceptance_rate_matches_theory(rng: np.random.Generator) -> None:
    result = rejection_sample(target_pdf, proposal_sampler, proposal_pdf, M, N, rng)
    expected = 1.0 / M
    # Acceptance is a Bernoulli count over n_proposed draws; 4 standard errors is a tight
    # but safe band at this size.
    se = np.sqrt(expected * (1 - expected) / result.n_proposed)
    assert abs(result.acceptance_rate - expected) < 4 * se


def test_envelope_check_catches_a_bad_constant() -> None:
    """An envelope that does not dominate produces the wrong distribution, silently."""
    grid = np.linspace(LO, HI, 1000)
    with pytest.raises(ValueError, match="does not dominate"):
        check_envelope(target_pdf, proposal_pdf, m=1.0, grid=grid)


def test_valid_envelope_passes_the_check() -> None:
    check_envelope(target_pdf, proposal_pdf, M, np.linspace(LO, HI, 1000))


def test_rejects_envelope_constant_below_one() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        rejection_sample(
            target_pdf, proposal_sampler, proposal_pdf, 0.5, 10, np.random.default_rng(0)
        )
