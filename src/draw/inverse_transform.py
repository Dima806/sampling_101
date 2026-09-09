"""Inverse transform sampling: uniform numbers into any distribution.

If ``U`` is uniform on [0, 1) and ``F`` is a cumulative distribution function, then
``F^{-1}(U)`` has distribution ``F``. That single line is the whole method. When the
inverse is available in closed form (the exponential) it is exact and instant; when it is
not, :func:`numeric_inverse_cdf` builds the inverse numerically from the density.
"""

from collections.abc import Callable

import numpy as np

InverseCDF = Callable[[np.ndarray], np.ndarray]


def sample_via_inverse_cdf(inv_cdf: InverseCDF, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` values by feeding uniforms through an inverse CDF."""
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    return inv_cdf(rng.random(n))


def exponential_inv_cdf(rate: float) -> InverseCDF:
    """Closed-form inverse CDF of the exponential distribution.

    ``F(x) = 1 - exp(-rate * x)`` inverts to ``F^{-1}(u) = -log(1 - u) / rate``.
    """
    if rate <= 0:
        raise ValueError(f"rate must be positive, got {rate}")

    def inv_cdf(u: np.ndarray) -> np.ndarray:
        # log1p(-u) is log(1 - u) computed accurately for u near 0.
        return -np.log1p(-u) / rate

    return inv_cdf


def numeric_inverse_cdf(
    pdf: Callable[[np.ndarray], np.ndarray],
    lo: float,
    hi: float,
    grid: int = 4096,
) -> InverseCDF:
    """Build an inverse CDF numerically from an unnormalised density on ``[lo, hi]``.

    Integrates ``pdf`` by the trapezoid rule on a regular grid, normalises to a CDF, and
    inverts it by linear interpolation. This is what makes inverse transform sampling a
    general method rather than a trick for the handful of distributions with a tidy
    closed-form inverse.
    """
    if hi <= lo:
        raise ValueError(f"need hi > lo, got lo={lo}, hi={hi}")
    if grid < 2:
        raise ValueError(f"grid must have at least 2 points, got {grid}")

    x = np.linspace(lo, hi, grid)
    density = np.asarray(pdf(x), dtype=np.float64)
    if np.any(density < 0):
        raise ValueError("pdf returned negative values")

    areas = 0.5 * (density[1:] + density[:-1]) * np.diff(x)
    cdf = np.concatenate([[0.0], np.cumsum(areas)])
    total = cdf[-1]
    if total <= 0:
        raise ValueError("pdf integrates to zero over the given range")
    cdf /= total

    # np.interp needs a strictly increasing x-coordinate. Flat stretches of the CDF (where
    # the density is zero) would otherwise make the inverse ambiguous, so drop them.
    keep = np.concatenate([[True], np.diff(cdf) > 0])
    cdf_points = cdf[keep]
    x_points = x[keep]

    def inv_cdf(u: np.ndarray) -> np.ndarray:
        return np.interp(u, cdf_points, x_points)

    return inv_cdf
