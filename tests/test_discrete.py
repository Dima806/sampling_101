"""The alias table and the cumulative-search baseline sample the same distribution."""

import numpy as np
import pytest

from src.draw.discrete import AliasTable, cumulative_search_sample

N = 200_000
PROBS = np.array([0.5, 0.25, 0.15, 0.08, 0.02])


def _empirical(counts: np.ndarray) -> np.ndarray:
    return counts / counts.sum()


def test_alias_table_matches_the_requested_probabilities(rng: np.random.Generator) -> None:
    draws = AliasTable(PROBS).sample(N, rng)
    observed = _empirical(np.bincount(draws, minlength=PROBS.size))
    # Each category is a Bernoulli proportion with se = sqrt(p(1-p)/N); 4 se at the rarest
    # category (p = 0.02) is 0.0012.
    se = np.sqrt(PROBS * (1 - PROBS) / N)
    assert np.all(np.abs(observed - PROBS) < 4 * se)


def test_cumulative_search_matches_the_requested_probabilities(rng: np.random.Generator) -> None:
    draws = cumulative_search_sample(PROBS, N, rng)
    observed = _empirical(np.bincount(draws, minlength=PROBS.size))
    se = np.sqrt(PROBS * (1 - PROBS) / N)
    assert np.all(np.abs(observed - PROBS) < 4 * se)


def test_alias_table_normalises_unnormalised_weights(rng: np.random.Generator) -> None:
    """Weights, not probabilities, are the common input. They must be normalised for you."""
    weights = PROBS * 37.0
    draws = AliasTable(weights).sample(N, rng)
    observed = _empirical(np.bincount(draws, minlength=PROBS.size))
    se = np.sqrt(PROBS * (1 - PROBS) / N)
    assert np.all(np.abs(observed - PROBS) < 4 * se)


def test_alias_table_covers_every_category(rng: np.random.Generator) -> None:
    draws = AliasTable(PROBS).sample(N, rng)
    assert set(np.unique(draws)) == set(range(PROBS.size))


def test_rejects_invalid_probability_vectors() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        AliasTable(np.array([0.5, -0.5, 1.0]))
    with pytest.raises(ValueError, match="positive value"):
        AliasTable(np.zeros(3))
    with pytest.raises(ValueError, match="must not be empty"):
        AliasTable(np.array([]))
