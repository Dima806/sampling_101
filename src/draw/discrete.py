"""Weighted categorical sampling, the slow way and the fast way.

Sampling one of ``k`` outcomes with given probabilities is the most common draw in
applied work. The obvious method searches a cumulative sum in ``O(log k)`` per draw. Vose's
alias method does ``O(k)`` setup once and then answers every draw in constant time with one
uniform and one integer.
"""

from dataclasses import dataclass, field

import numpy as np


def _normalised(probs: np.ndarray) -> np.ndarray:
    """Validate a probability vector and scale it to sum to one."""
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"probs must be 1-D, got shape {p.shape}")
    if p.size == 0:
        raise ValueError("probs must not be empty")
    if np.any(p < 0):
        raise ValueError("probs must be non-negative")
    total = float(p.sum())
    if total <= 0:
        raise ValueError("probs must sum to a positive value")
    return p / total


def cumulative_search_sample(probs: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample ``n`` category indices by binary search on the cumulative distribution.

    The transparent baseline: ``O(log k)`` per draw. Kept for comparison against
    :class:`AliasTable`.
    """
    p = _normalised(probs)
    cumulative = np.cumsum(p)
    # Guard against the last entry landing just below 1 through floating point drift.
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, rng.random(n), side="right")


@dataclass
class AliasTable:
    """Vose's alias method: ``O(k)`` setup, ``O(1)`` per draw.

    Each of the ``k`` categories becomes a bucket holding at most two outcomes. A draw
    picks a bucket uniformly, then a single uniform decides between the bucket's own
    outcome and its alias.
    """

    probs: np.ndarray
    prob: np.ndarray = field(init=False)
    alias: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        p = _normalised(self.probs)
        k = p.size
        self.prob = p * k
        self.alias = np.zeros(k, dtype=np.int64)

        # Buckets holding less than one unit of mass need topping up; those holding more
        # give their surplus away.
        small = [i for i in range(k) if self.prob[i] < 1.0]
        large = [i for i in range(k) if self.prob[i] >= 1.0]

        while small and large:
            s = small.pop()
            g = large.pop()
            self.alias[s] = g
            self.prob[g] = self.prob[g] - (1.0 - self.prob[s])
            if self.prob[g] < 1.0:
                small.append(g)
            else:
                large.append(g)

        # Anything left is a full bucket up to floating point error.
        for i in small + large:
            self.prob[i] = 1.0
            self.alias[i] = i

    @property
    def n_categories(self) -> int:
        """Number of categories in the table."""
        return int(self.prob.size)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` category indices in constant time each."""
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        bucket = rng.integers(0, self.n_categories, size=n)
        u = rng.random(n)
        return np.where(u < self.prob[bucket], bucket, self.alias[bucket])
