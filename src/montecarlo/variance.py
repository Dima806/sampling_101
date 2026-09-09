"""Variance diagnostics: how to tell a converging estimate from a stuck one.

None of these change an estimate. They tell you whether to believe it, which for
importance sampling matters more than the number itself: a bad proposal produces a
plausible point estimate and a standard error that understates the true error by orders of
magnitude. The weight diagnostics catch that before the estimate does.
"""

from dataclasses import dataclass

import numpy as np

from src.montecarlo.importance import effective_sample_size


@dataclass(frozen=True)
class WeightDiagnostics:
    """Health check for a set of importance weights."""

    ess: float
    ess_fraction: float
    max_weight_share: float
    coefficient_of_variation: float


def weight_diagnostics(weights: np.ndarray) -> WeightDiagnostics:
    """Summarise how unevenly a set of importance weights is distributed.

    A healthy proposal keeps ``ess_fraction`` high and ``max_weight_share`` small. The
    coefficient of variation of the weights is the classical warning sign: it is directly
    related to the inflation of the estimator's variance.
    """
    total = float(np.sum(weights))
    ess = effective_sample_size(weights)
    mean = float(np.mean(weights))
    cv = float(np.std(weights) / mean) if mean > 0 else float("nan")
    return WeightDiagnostics(
        ess=ess,
        ess_fraction=ess / weights.size,
        max_weight_share=float(np.max(weights)) / total if total > 0 else float("nan"),
        coefficient_of_variation=cv,
    )


def variance_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """Ratio ``var(a) / var(b)``: how much variance one design saves over another."""
    var_b = float(np.var(b, ddof=1))
    if var_b <= 0:
        return float("inf")
    return float(np.var(a, ddof=1)) / var_b


def running_standard_error(values: np.ndarray) -> np.ndarray:
    """Standard error of the running mean after each draw.

    Computed from cumulative first and second moments, so the whole trace costs one pass
    instead of one pass per checkpoint.
    """
    n = values.size
    k = np.arange(1, n + 1, dtype=np.float64)
    mean = np.cumsum(values) / k
    mean_square = np.cumsum(values**2) / k
    # Sample variance from the two moments, with Bessel's correction. Clipped at zero
    # because cancellation can push the difference slightly negative.
    variance = np.maximum(mean_square - mean**2, 0.0) * k / np.maximum(k - 1.0, 1.0)
    return np.sqrt(variance / k)


def sqrt_n_band(ns: np.ndarray, sigma: float, z: float = 1.96) -> np.ndarray:
    """The ``z * sigma / sqrt(n)`` error band that Monte Carlo estimates should live inside."""
    return z * sigma / np.sqrt(np.asarray(ns, dtype=np.float64))
