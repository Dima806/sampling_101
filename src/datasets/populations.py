"""Synthetic populations whose true values are known exactly.

Every population here is finite and fully enumerated, so the "true" mean is not a
theoretical quantity to be approximated: it is ``values.mean()``, computed once. That is
what lets the project measure the bias of a sampling design instead of asserting it.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class StratifiedPopulation:
    """A population split into strata of very different size and spread."""

    values: np.ndarray
    strata: np.ndarray

    @property
    def size(self) -> int:
        """Number of units in the population."""
        return int(self.values.size)

    @property
    def true_mean(self) -> float:
        """Exact finite-population mean. The target every estimator is scored against."""
        return float(self.values.mean())

    def stratum_sizes(self) -> np.ndarray:
        """Count of units in each stratum, ordered by stratum label."""
        return np.bincount(self.strata)


@dataclass(frozen=True)
class GroupedPopulation:
    """A population of rows nested inside entities (customers, sites, patients)."""

    values: np.ndarray
    groups: np.ndarray

    @property
    def size(self) -> int:
        """Number of rows in the population."""
        return int(self.values.size)

    @property
    def n_groups(self) -> int:
        """Number of distinct entities."""
        return int(np.unique(self.groups).size)

    @property
    def true_mean(self) -> float:
        """Exact finite-population mean over rows."""
        return float(self.values.mean())


def skewed_population(
    size: int,
    proportions: list[float],
    means: list[float],
    sds: list[float],
    rng: np.random.Generator,
) -> StratifiedPopulation:
    """Build a skewed population as a mixture of strata with unequal size and spread.

    The last stratum is small and extreme: a handful of very large values carrying a large
    share of the total. A simple random sample of such a population is dominated by
    whether it happens to catch those units, which is exactly the variance that stratifying
    removes.
    """
    if not (len(proportions) == len(means) == len(sds)):
        raise ValueError("proportions, means and sds must have the same length")
    weights = np.asarray(proportions, dtype=np.float64)
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("proportions must be non-negative and sum to a positive value")
    weights = weights / weights.sum()

    # Fixed counts rather than multinomial draws, so stratum sizes are exactly known.
    counts = np.floor(weights * size).astype(int)
    counts[-1] += size - int(counts.sum())

    strata = np.repeat(np.arange(counts.size), counts)
    values = np.empty(size, dtype=np.float64)
    for h in range(counts.size):
        mask = strata == h
        values[mask] = rng.normal(means[h], sds[h], size=int(mask.sum()))

    # These are quantities like spend or claim size: negative values are not meaningful.
    values = np.maximum(values, 0.0)
    return StratifiedPopulation(values=values, strata=strata)


def grouped_population(
    n_groups: int,
    group_size: int,
    rng: np.random.Generator,
    between_group_sd: float = 12.0,
    within_group_sd: float = 4.0,
    baseline: float = 100.0,
) -> GroupedPopulation:
    """Build rows nested in entities, with most of the variance between entities.

    When ``between_group_sd`` dominates ``within_group_sd``, rows from the same entity are
    near-duplicates. Splitting or sampling rows then leaks an entity across the split and
    the resulting estimate is far too optimistic.
    """
    if n_groups < 1 or group_size < 1:
        raise ValueError("n_groups and group_size must both be at least 1")
    effects = rng.normal(0.0, between_group_sd, size=n_groups)
    groups = np.repeat(np.arange(n_groups), group_size)
    noise = rng.normal(0.0, within_group_sd, size=n_groups * group_size)
    values = baseline + effects[groups] + noise
    return GroupedPopulation(values=values, groups=groups)


def response_propensity(values: np.ndarray, strength: float) -> np.ndarray:
    """Probability of responding, tilted by the value being measured.

    With ``strength > 0`` the units with large values are the ones who answer, which is the
    canonical non-response mechanism: the sampling frame is correlated with the outcome, so
    the sample mean is biased upward no matter how many units are collected.
    """
    z = (values - values.mean()) / values.std()
    # Softmax-style tilt, kept bounded so no single unit dominates the frame.
    weights = np.exp(strength * np.clip(z, -4.0, 4.0))
    return weights / weights.sum()


def normal_tail_probability(threshold: float) -> float:
    """Exact ``P(X > threshold)`` for ``X ~ N(0, 1)``.

    This is the sanctioned use of scipy: ground truth to score the sampling estimates
    against, never an implementation of something the project claims to build.
    """
    return float(stats.norm.sf(threshold))
