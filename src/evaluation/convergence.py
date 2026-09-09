"""Does the estimate actually approach the truth, and how fast?

The headline claim of Monte Carlo is that the error falls like ``n^{-1/2}``. That is a
statement about the spread of the estimator across repeated experiments, not about the
wiggle of one run, so measuring it needs replicates. :func:`fit_convergence_rate` recovers
the exponent as the slope of a log-log fit, and the tests assert it lands near -0.5.
"""

import numpy as np


def running_estimate(draws: np.ndarray) -> np.ndarray:
    """Running mean of ``draws`` after each observation."""
    n = draws.size
    if n == 0:
        raise ValueError("draws must not be empty")
    return np.cumsum(draws) / np.arange(1, n + 1)


def error_curve(running: np.ndarray, truth: float) -> np.ndarray:
    """Absolute error of a running estimate against a known truth."""
    return np.abs(running - truth)


def logspace_checkpoints(n_max: int, n_points: int = 25, n_min: int = 100) -> np.ndarray:
    """Log-spaced, strictly increasing sample sizes at which to score an estimator.

    Log spacing is what makes the log-log slope well conditioned: linear checkpoints would
    crowd every point into the flat right-hand end of the curve.
    """
    if n_min < 2 or n_max <= n_min:
        raise ValueError(f"need 2 <= n_min < n_max, got n_min={n_min}, n_max={n_max}")
    raw = np.logspace(np.log10(n_min), np.log10(n_max), n_points)
    return np.unique(np.round(raw).astype(np.int64))


def rmse_by_n(estimates: np.ndarray, truth: float) -> np.ndarray:
    """Root mean squared error at each checkpoint, over replicates.

    ``estimates`` is shaped ``(n_replicates, n_checkpoints)``; the return value is one
    error per checkpoint, averaged across the independent runs.
    """
    if estimates.ndim != 2:
        raise ValueError(f"estimates must be 2-D (replicates, checkpoints), got {estimates.shape}")
    return np.sqrt(np.mean((estimates - truth) ** 2, axis=0))


def fit_convergence_rate(ns: np.ndarray, errors: np.ndarray) -> float:
    """Slope of ``log(error)`` against ``log(n)``.

    Monte Carlo predicts -0.5. The value is asserted in the tests, because "the error
    shrinks with more samples" is a much weaker claim than "the error shrinks at this rate",
    and only the second one lets a reader budget a simulation.
    """
    ns = np.asarray(ns, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    if ns.size != errors.size:
        raise ValueError("ns and errors must have the same length")
    if np.any(ns <= 0) or np.any(errors <= 0):
        raise ValueError("log-log fit needs strictly positive ns and errors")
    slope, _ = np.polyfit(np.log(ns), np.log(errors), deg=1)
    return float(slope)
