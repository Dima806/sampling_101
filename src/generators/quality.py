"""Diagnostics that separate a usable generator from a broken one.

The important lesson lives in the gap between these two families of test. Marginal
uniformity (:func:`chi_square_uniformity`) is easy to pass and RANDU passes it. Structure
in consecutive tuples (:func:`plane_fraction`) is where RANDU falls apart. A generator is
not "random" because its histogram is flat.
"""

import numpy as np
from scipy import stats

# RANDU obeys x_{n+2} = 6 * x_{n+1} - 9 * x_n (mod 2**31), so 9*u_n - 6*u_{n+1} + u_{n+2}
# is an exact integer for every triple. Projecting onto this direction is the quantitative
# version of the 3-D scatter plot where the points fall on visible planes.
RANDU_PLANE_DIRECTION: tuple[float, float, float] = (9.0, -6.0, 1.0)


def chi_square_uniformity(u: np.ndarray, bins: int = 100) -> tuple[float, float]:
    """Chi-square goodness-of-fit test of ``u`` against Uniform(0, 1).

    Returns the statistic and its p-value. A small p-value is evidence against
    uniformity of the marginal distribution.
    """
    if bins < 2:
        raise ValueError(f"bins must be at least 2, got {bins}")
    counts, _ = np.histogram(u, bins=bins, range=(0.0, 1.0))
    expected = u.size / bins
    statistic = float(np.sum((counts - expected) ** 2) / expected)
    p_value = float(stats.chi2.sf(statistic, df=bins - 1))
    return statistic, p_value


def serial_correlation(u: np.ndarray, lag: int = 1) -> float:
    """Pearson correlation between the stream and itself shifted by ``lag``.

    A generator with no short-range dependence gives a value near zero.
    """
    if lag < 1:
        raise ValueError(f"lag must be at least 1, got {lag}")
    if u.size <= lag + 1:
        raise ValueError("stream is too short for the requested lag")
    return float(np.corrcoef(u[:-lag], u[lag:])[0, 1])


def overlapping_tuples(u: np.ndarray, dim: int = 3) -> np.ndarray:
    """Stack consecutive overlapping ``dim``-tuples into an ``(n - dim + 1, dim)`` array."""
    if dim < 2:
        raise ValueError(f"dim must be at least 2, got {dim}")
    rows = u.size - dim + 1
    if rows <= 0:
        raise ValueError("stream is shorter than the requested tuple size")
    return np.stack([u[i : i + rows] for i in range(dim)], axis=1)


def lattice_triples(u: np.ndarray) -> np.ndarray:
    """Consecutive triples ``(u_i, u_{i+1}, u_{i+2})`` for the 3-D structure plot."""
    return overlapping_tuples(u, dim=3)


def plane_projection(
    u: np.ndarray, direction: tuple[float, float, float] = RANDU_PLANE_DIRECTION
) -> np.ndarray:
    """Project consecutive triples onto ``direction``.

    If the generator's triples lie on a family of parallel planes normal to ``direction``,
    the projected values pile up on a handful of integers instead of spreading out.
    """
    return lattice_triples(u) @ np.asarray(direction, dtype=np.float64)


def plane_fraction(
    u: np.ndarray,
    direction: tuple[float, float, float] = RANDU_PLANE_DIRECTION,
    tol: float = 1e-9,
) -> float:
    """Fraction of triples whose projection lands within ``tol`` of an integer.

    This is the quantitative form of "the points fall on stripes": RANDU scores 1.0 on its
    own plane direction, a sound generator scores essentially 0.
    """
    projected = plane_projection(u, direction)
    return float(np.mean(np.abs(projected - np.rint(projected)) <= tol))


def plane_count(
    u: np.ndarray,
    direction: tuple[float, float, float] = RANDU_PLANE_DIRECTION,
    tol: float = 1e-9,
) -> int:
    """Number of distinct planes the triples occupy along ``direction``.

    Only meaningful alongside :func:`plane_fraction`: a count of 15 means something when
    every triple is on a plane, and nothing when none of them are.
    """
    projected = plane_projection(u, direction)
    on_plane = projected[np.abs(projected - np.rint(projected)) <= tol]
    return int(np.unique(np.rint(on_plane)).size)
