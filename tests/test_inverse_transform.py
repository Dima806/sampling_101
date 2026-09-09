"""Samples from the from-scratch inverse-transform sampler match their target.

Scored with a Kolmogorov-Smirnov test against the exact distribution. scipy supplies the
reference CDF and the test; it never supplies the sampler.
"""

import numpy as np
import pytest
from scipy import stats

from src.draw.inverse_transform import (
    exponential_inv_cdf,
    numeric_inverse_cdf,
    sample_via_inverse_cdf,
)

N = 20_000
RATE = 1.5
# A KS test at n = 20_000 detects tiny discrepancies, so a correct sampler should clear a
# conventional 0.01 threshold comfortably. Anything below it means a real defect.
ALPHA = 0.01


def test_exponential_inverse_cdf_matches_target(rng: np.random.Generator) -> None:
    samples = sample_via_inverse_cdf(exponential_inv_cdf(RATE), N, rng)
    result = stats.kstest(samples, stats.expon(scale=1.0 / RATE).cdf)
    assert result.pvalue > ALPHA


def test_exponential_mean_is_right(rng: np.random.Generator) -> None:
    samples = sample_via_inverse_cdf(exponential_inv_cdf(RATE), N, rng)
    # The mean of an exponential is 1/rate, with standard error (1/rate)/sqrt(N).
    # Allow 4 standard errors.
    se = (1.0 / RATE) / np.sqrt(N)
    assert abs(samples.mean() - 1.0 / RATE) < 4 * se


def test_numeric_inverse_cdf_matches_a_normal(rng: np.random.Generator) -> None:
    """The general path: only the density is supplied, and the inverse is built numerically."""

    def pdf(x: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * x**2)  # unnormalised standard normal

    inv_cdf = numeric_inverse_cdf(pdf, lo=-6.0, hi=6.0, grid=8192)
    samples = sample_via_inverse_cdf(inv_cdf, N, rng)
    assert stats.kstest(samples, stats.norm.cdf).pvalue > ALPHA


def test_numeric_inverse_cdf_handles_an_unnormalised_density(rng: np.random.Generator) -> None:
    """Scaling the density must not change the samples: normalisation is handled internally."""
    scaled = numeric_inverse_cdf(lambda x: 7.5 * np.exp(-x), lo=0.0, hi=20.0, grid=4096)
    samples = sample_via_inverse_cdf(scaled, N, rng)
    assert stats.kstest(samples, stats.expon.cdf).pvalue > ALPHA


def test_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="rate must be positive"):
        exponential_inv_cdf(0.0)
    with pytest.raises(ValueError, match="need hi > lo"):
        numeric_inverse_cdf(lambda x: np.ones_like(x), lo=1.0, hi=1.0)
    with pytest.raises(ValueError, match="integrates to zero"):
        numeric_inverse_cdf(lambda x: np.zeros_like(x), lo=0.0, hi=1.0)
