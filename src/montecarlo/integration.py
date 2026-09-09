"""Monte Carlo integration: hard integrals become a loop that samples and averages.

Any integral is an expectation in disguise. To integrate ``f`` over ``[lo, hi]``, draw
uniform points, average ``f`` over them, multiply by the width. To find the area of a
circle, throw darts at the square around it and count. The estimate is unbiased and its
error shrinks like ``1 / sqrt(n)``, which is the method's honest and frustrating price:
100x the samples buy 10x the accuracy.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Sampler = Callable[[int, np.random.Generator], np.ndarray]


@dataclass(frozen=True)
class MCResult:
    """A Monte Carlo estimate and the running trace behind it."""

    value: float
    se: float
    running: np.ndarray

    @property
    def n(self) -> int:
        """Number of draws behind the estimate."""
        return int(self.running.size)


def mc_result(values: np.ndarray) -> MCResult:
    """Package per-draw contributions into an estimate, its standard error and its trace.

    The running mean after every draw is carried along so that convergence plots cost
    nothing extra: no notebook needs to re-run the loop just to draw a curve.
    """
    n = values.size
    if n < 2:
        raise ValueError(f"need at least 2 draws to estimate a standard error, got {n}")
    running = np.cumsum(values) / np.arange(1, n + 1)
    return MCResult(
        value=float(values.mean()),
        se=float(values.std(ddof=1) / np.sqrt(n)),
        running=running,
    )


def estimate_pi(n: int, rng: np.random.Generator) -> MCResult:
    """Estimate pi by throwing ``n`` darts at the unit square.

    A dart lands in the quarter circle with probability ``pi / 4``, so four times the hit
    rate estimates pi. Scoring each dart as ``4 * hit`` makes the per-draw contribution
    explicit, and the standard error then follows from the same formula as any other
    Monte Carlo average.
    """
    darts = rng.random((n, 2))
    inside = (darts**2).sum(axis=1) <= 1.0
    return mc_result(4.0 * inside)


def integrate(
    f: Callable[[np.ndarray], np.ndarray],
    lo: float,
    hi: float,
    n: int,
    rng: np.random.Generator,
) -> MCResult:
    """Estimate ``\\int_lo^hi f(x) dx`` as ``(hi - lo)`` times the mean of ``f`` at uniforms."""
    if hi <= lo:
        raise ValueError(f"need hi > lo, got lo={lo}, hi={hi}")
    x = rng.uniform(lo, hi, size=n)
    return mc_result((hi - lo) * np.asarray(f(x), dtype=np.float64))


def expectation(
    f: Callable[[np.ndarray], np.ndarray],
    sampler: Sampler,
    n: int,
    rng: np.random.Generator,
) -> MCResult:
    """Estimate ``E[f(X)]`` by averaging ``f`` over draws from ``sampler``."""
    x = sampler(n, rng)
    return mc_result(np.asarray(f(x), dtype=np.float64))


def pi_estimates_at(
    checkpoints: np.ndarray, n_replicates: int, rng: np.random.Generator
) -> np.ndarray:
    """Independent pi estimates at each checkpoint, shaped ``(n_replicates, len(checkpoints))``.

    A single run's error is far too noisy to read a convergence rate from. Repeating the
    experiment and taking the spread across replicates is what turns the ``1 / sqrt(n)``
    claim into something measurable.
    """
    checkpoints = np.asarray(checkpoints, dtype=np.int64)
    n_max = int(checkpoints[-1])
    out = np.empty((n_replicates, checkpoints.size), dtype=np.float64)
    for r in range(n_replicates):
        darts = rng.random((n_max, 2))
        inside = (darts**2).sum(axis=1) <= 1.0
        running = 4.0 * np.cumsum(inside) / np.arange(1, n_max + 1)
        out[r] = running[checkpoints - 1]
    return out
