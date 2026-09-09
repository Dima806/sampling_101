"""Simple random sampling: the default, and the baseline everything else is judged against.

Every unit has the same chance of selection and no unit is selected twice. It is unbiased,
which is why it feels safe. Its weakness is variance: when the population contains a small
subgroup carrying a large share of the total, the estimate swings on whether that subgroup
happened to be caught.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Estimate:
    """A point estimate with its standard error.

    A point estimate alone is not an answer in this project: the whole argument about
    sampling designs is an argument about variance.
    """

    value: float
    se: float
    n: int

    def covers(self, truth: float, z: float = 1.96) -> bool:
        """Whether the interval ``value +/- z * se`` contains ``truth``."""
        return abs(self.value - truth) <= z * self.se


def simple_random_indices(size: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` distinct indices from ``range(size)`` without replacement.

    A partial Fisher-Yates shuffle: swap each of the first ``n`` positions with a uniformly
    chosen position at or after it. Stopping after ``n`` swaps costs ``O(n)`` rather than
    shuffling the whole population.
    """
    if not 0 <= n <= size:
        raise ValueError(f"need 0 <= n <= size, got n={n}, size={size}")
    idx = np.arange(size)
    for i in range(n):
        j = int(rng.integers(i, size))
        idx[i], idx[j] = idx[j], idx[i]
    return idx[:n]


def simple_random_mean(population: np.ndarray, n: int, rng: np.random.Generator) -> Estimate:
    """Estimate the population mean from a simple random sample of size ``n``.

    The standard error carries the finite population correction ``1 - n/N``: sampling a
    large fraction of a finite population is more precise than the textbook ``s/sqrt(n)``.
    """
    size = population.size
    if n < 2:
        raise ValueError(f"need at least 2 units to estimate a variance, got {n}")
    sample = population[simple_random_indices(size, n, rng)]
    value = float(sample.mean())
    variance = float(sample.var(ddof=1))
    se = float(np.sqrt(max(1.0 - n / size, 0.0) * variance / n))
    return Estimate(value=value, se=se, n=n)
