"""Stratified sampling is unbiased and strictly less variable than simple random sampling.

The claim is about the spread of the estimator across *repeated* samples, so every test
here re-runs both designs many times. A single draw would prove nothing, and believing a
single draw is the exact reflex this project exists to correct.
"""

import numpy as np
import pytest

from src.config import Settings
from src.datasets.populations import StratifiedPopulation, skewed_population
from src.survey.simple import simple_random_mean
from src.survey.stratified import (
    neyman_allocation,
    proportional_allocation,
    stratified_mean,
)

N_REPLICATES = 200
SAMPLE_SIZE = 1000


@pytest.fixture
def population(settings: Settings, rng: np.random.Generator) -> StratifiedPopulation:
    cfg = settings.survey
    return skewed_population(
        cfg.population_size, cfg.strata_proportions, cfg.strata_means, cfg.strata_sds, rng
    )


def _repeat(estimator, population: StratifiedPopulation, rng: np.random.Generator) -> np.ndarray:
    return np.array([estimator(population, rng).value for _ in range(N_REPLICATES)])


def test_population_has_the_requested_strata(population: StratifiedPopulation) -> None:
    sizes = population.stratum_sizes()
    assert sizes.size == 4
    # The rare tail stratum is the one simple random sampling struggles with.
    assert sizes[-1] < sizes[0] / 10


def test_both_designs_are_unbiased(
    population: StratifiedPopulation, rng: np.random.Generator
) -> None:
    truth = population.true_mean
    srs = _repeat(lambda p, r: simple_random_mean(p.values, SAMPLE_SIZE, r), population, rng)
    strat = _repeat(
        lambda p, r: stratified_mean(p.values, p.strata, SAMPLE_SIZE, r), population, rng
    )
    # Mean of N_REPLICATES estimates has se = sd/sqrt(N_REPLICATES); allow 4 of them.
    for estimates in (srs, strat):
        se = estimates.std(ddof=1) / np.sqrt(N_REPLICATES)
        assert abs(estimates.mean() - truth) < 4 * se


def test_stratified_has_lower_variance_than_simple_random(
    population: StratifiedPopulation, rng: np.random.Generator
) -> None:
    """The practical payoff, and a headline success criterion of the project."""
    srs = _repeat(lambda p, r: simple_random_mean(p.values, SAMPLE_SIZE, r), population, rng)
    strat = _repeat(
        lambda p, r: stratified_mean(p.values, p.strata, SAMPLE_SIZE, r), population, rng
    )
    assert strat.var(ddof=1) < srs.var(ddof=1)


def test_neyman_allocation_beats_proportional(
    population: StratifiedPopulation, rng: np.random.Generator
) -> None:
    """Spending the sample where the variance is beats spending it where the units are."""
    proportional = _repeat(
        lambda p, r: stratified_mean(p.values, p.strata, SAMPLE_SIZE, r, "proportional"),
        population,
        rng,
    )
    neyman = _repeat(
        lambda p, r: stratified_mean(p.values, p.strata, SAMPLE_SIZE, r, "neyman"),
        population,
        rng,
    )
    assert neyman.var(ddof=1) < proportional.var(ddof=1)


def test_reported_standard_error_is_honest(
    population: StratifiedPopulation, rng: np.random.Generator
) -> None:
    """The se a single sample reports must match the spread actually observed.

    A design whose reported uncertainty does not match its real uncertainty is worse than
    useless, so this checks the formula against the empirical standard deviation.
    """
    estimates = []
    reported = []
    for _ in range(N_REPLICATES):
        est = stratified_mean(population.values, population.strata, SAMPLE_SIZE, rng)
        estimates.append(est.value)
        reported.append(est.se)
    empirical_sd = float(np.std(estimates, ddof=1))
    mean_reported_se = float(np.mean(reported))
    assert mean_reported_se == pytest.approx(empirical_sd, rel=0.15)


def test_allocations_sum_to_the_sample_size() -> None:
    counts = np.array([70_000, 20_000, 8_000, 2_000])
    sds = np.array([8.0, 15.0, 40.0, 200.0])
    assert proportional_allocation(counts, 1000).sum() == 1000
    assert neyman_allocation(counts, sds, 1000).sum() == 1000


def test_allocation_never_starves_a_stratum() -> None:
    """Every stratum needs at least 2 units or its variance is not estimable."""
    counts = np.array([99_000, 900, 90, 10])
    alloc = proportional_allocation(counts, 100)
    assert alloc.min() >= 2
    assert alloc.sum() == 100
    assert np.all(alloc <= counts)


def test_rejects_a_sample_too_small_to_cover_the_strata() -> None:
    counts = np.array([100, 100, 100, 100])
    with pytest.raises(ValueError, match="at least 2 units per stratum"):
        proportional_allocation(counts, 5)
