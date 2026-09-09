"""Selection bias: a flawless draw from the wrong frame gives a confidently wrong answer.

These failures are not fixed by more data. Each test measures the gap against a known
population mean, so the damage is quantified rather than asserted.
"""

import numpy as np
import pytest

from src.config import Settings
from src.datasets.populations import (
    GroupedPopulation,
    StratifiedPopulation,
    grouped_population,
    response_propensity,
    skewed_population,
)
from src.survey.bias import convenience_frame_mean, frame_limit_mean, response_biased_mean
from src.survey.grouped import (
    cluster_mean,
    group_leakage_rate,
    group_train_test_split,
    naive_row_split,
)
from src.survey.simple import simple_random_mean

SAMPLE_SIZE = 1000


@pytest.fixture
def population(settings: Settings, rng: np.random.Generator) -> StratifiedPopulation:
    cfg = settings.survey
    return skewed_population(
        cfg.population_size, cfg.strata_proportions, cfg.strata_means, cfg.strata_sds, rng
    )


@pytest.fixture
def grouped(settings: Settings, rng: np.random.Generator) -> GroupedPopulation:
    cfg = settings.survey
    return grouped_population(
        cfg.n_groups, cfg.group_size, rng, cfg.between_group_sd, cfg.within_group_sd
    )


def test_response_bias_inflates_the_estimate(
    settings: Settings, population: StratifiedPopulation, rng: np.random.Generator
) -> None:
    """When the people with large values are the ones who answer, the mean is too high."""
    propensity = response_propensity(population.values, settings.survey.response_strength)
    result = response_biased_mean(population.values, propensity, SAMPLE_SIZE, rng)
    unbiased = simple_random_mean(population.values, SAMPLE_SIZE, rng)

    assert result.bias > 0
    # The bias must clearly exceed the sampling noise, or it is not a demonstration of
    # anything. Measured at about 5.8 SE; 4 leaves margin. It is deliberately NOT larger:
    # a propensity strong enough to shift the mean by hundreds of percent is a caricature,
    # not a demonstration of the mechanism.
    assert result.bias > 4 * unbiased.se


def test_convenience_frame_gives_a_quantifiably_wrong_answer(
    population: StratifiedPopulation, rng: np.random.Generator
) -> None:
    """A textbook-correct random sample of a frame that excludes the tail."""
    in_frame = population.strata < 3  # the rare, high-value stratum is unreachable
    result = convenience_frame_mean(population.values, in_frame, SAMPLE_SIZE, rng)
    assert result.estimate < result.truth
    assert abs(result.relative_bias) > 0.05


def test_more_data_converges_to_the_wrong_answer(
    settings: Settings, population: StratifiedPopulation, rng: np.random.Generator
) -> None:
    """The bias is a property of the frame, not of the sample size.

    Stated exactly rather than empirically: a propensity-weighted sample mean converges to
    ``sum_i p_i x_i``, which is computable. More data drives the estimate toward that
    number and away from nothing at all.
    """
    propensity = response_propensity(population.values, settings.survey.response_strength)
    limit = frame_limit_mean(population.values, propensity)
    truth = population.true_mean

    assert limit > truth  # the frame over-samples large values

    small = response_biased_mean(population.values, propensity, 500, rng)
    large = response_biased_mean(population.values, propensity, 200_000, rng)

    # The error against the truth does not shrink...
    assert abs(large.estimate - truth) > 0.5 * abs(small.estimate - truth)
    # ...while the error against the frame limit does, to within a fraction of a percent.
    assert large.estimate == pytest.approx(limit, rel=0.02)


def test_random_row_split_leaks_entities(
    grouped: GroupedPopulation, rng: np.random.Generator
) -> None:
    """Splitting rows puts nearly every entity on both sides of the split."""
    train, test = naive_row_split(grouped.size, 0.25, rng)
    assert group_leakage_rate(grouped.groups, train, test) > 0.95


def test_group_split_leaks_nothing(grouped: GroupedPopulation, rng: np.random.Generator) -> None:
    train, test = group_train_test_split(grouped.groups, 0.25, rng)
    assert group_leakage_rate(grouped.groups, train, test) == 0.0
    # And it is still a split: both sides hold rows.
    assert train.sum() > 0
    assert test.sum() > 0


def test_cluster_sampling_is_less_precise_than_simple_random(
    grouped: GroupedPopulation, rng: np.random.Generator
) -> None:
    """Rows inside an entity repeat information, so sampling entities buys less than it looks.

    The mirror image of the stratification result: grouping structure that helps when you
    stratify on it hurts when you sample whole clusters.
    """
    n_groups = 40
    n_rows = n_groups * (grouped.size // grouped.n_groups)
    cluster = cluster_mean(grouped.values, grouped.groups, n_groups, rng)
    simple = simple_random_mean(grouped.values, n_rows, rng)
    assert cluster.se > simple.se
