"""Stratified sampling: sample inside each subgroup so none can be missed.

Splitting the population into strata and sampling within each removes the "did we catch the
rare tail?" lottery from the estimate. The payoff is a strictly smaller variance whenever
the strata differ in mean, at no cost in bias. Neyman allocation goes further and spends
the sample where the variance actually is.
"""

from typing import Literal

import numpy as np

from src.survey.simple import Estimate, simple_random_indices

Allocation = Literal["proportional", "neyman"]

MIN_PER_STRATUM = 2


def _repair_allocation(alloc: np.ndarray, counts: np.ndarray, n: int) -> np.ndarray:
    """Force an allocation to sum to ``n`` while respecting the per-stratum bounds."""
    alloc = np.clip(alloc, MIN_PER_STRATUM, counts)
    while int(alloc.sum()) > n:
        slack = alloc - MIN_PER_STRATUM
        i = int(np.argmax(slack))
        if slack[i] <= 0:
            break
        alloc[i] -= 1
    while int(alloc.sum()) < n:
        room = counts - alloc
        i = int(np.argmax(room))
        if room[i] <= 0:
            break
        alloc[i] += 1
    return alloc


def _allocate(weights: np.ndarray, counts: np.ndarray, n: int) -> np.ndarray:
    """Turn continuous allocation weights into whole numbers of units summing to ``n``."""
    strata_count = weights.size
    if n < MIN_PER_STRATUM * strata_count:
        raise ValueError(
            f"need at least {MIN_PER_STRATUM} units per stratum "
            f"({MIN_PER_STRATUM * strata_count} in total), got n={n}"
        )
    if n > int(counts.sum()):
        raise ValueError("sample size exceeds the population size")

    raw = weights / weights.sum() * n
    alloc = np.floor(raw).astype(np.int64)
    # Largest-remainder rounding, so the allocation stays as close to raw as integers allow.
    order = np.argsort(-(raw - alloc))
    for k in range(int(n - alloc.sum())):
        alloc[order[k % strata_count]] += 1
    return _repair_allocation(alloc, counts, n)


def proportional_allocation(counts: np.ndarray, n: int) -> np.ndarray:
    """Allocate ``n`` units across strata in proportion to stratum size."""
    return _allocate(counts.astype(np.float64), counts, n)


def neyman_allocation(counts: np.ndarray, sds: np.ndarray, n: int) -> np.ndarray:
    """Allocate ``n`` units in proportion to ``N_h * S_h``, the variance-optimal split.

    Note this uses the true within-stratum standard deviations. In practice they come from
    a pilot sample or prior data; here the population is synthetic, so the optimum is known
    and the notebook can show how much is left on the table by ignoring it.
    """
    weights = counts.astype(np.float64) * np.asarray(sds, dtype=np.float64)
    if weights.sum() <= 0:
        raise ValueError("Neyman allocation needs at least one stratum with positive spread")
    return _allocate(weights, counts, n)


def stratified_mean(
    population: np.ndarray,
    strata: np.ndarray,
    n: int,
    rng: np.random.Generator,
    allocation: Allocation = "proportional",
) -> Estimate:
    """Estimate the population mean by sampling within every stratum.

    The estimate is the stratum-size-weighted average of the within-stratum means,
    ``sum_h W_h * ybar_h``, and its variance is the weighted sum of the within-stratum
    variances. Variation *between* strata never enters, which is precisely why the
    estimator beats simple random sampling.
    """
    labels = np.unique(strata)
    members = [np.flatnonzero(strata == h) for h in labels]
    counts = np.array([m.size for m in members], dtype=np.int64)

    if allocation == "proportional":
        alloc = proportional_allocation(counts, n)
    elif allocation == "neyman":
        sds = np.array([population[m].std(ddof=1) for m in members])
        alloc = neyman_allocation(counts, sds, n)
    else:
        raise ValueError(f"unknown allocation {allocation!r}")

    size = population.size
    value = 0.0
    variance = 0.0
    for stratum_members, count, n_h in zip(members, counts, alloc, strict=True):
        sample = population[stratum_members[simple_random_indices(int(count), int(n_h), rng)]]
        weight = count / size
        value += weight * float(sample.mean())
        fpc = max(1.0 - n_h / count, 0.0)
        variance += weight**2 * fpc * float(sample.var(ddof=1)) / n_h

    return Estimate(value=value, se=float(np.sqrt(variance)), n=int(alloc.sum()))
