"""The settings file loads, validates, and drives a reproducible generator."""

import numpy as np

from src.config import SETTINGS_PATH, load_settings, make_rng


def test_settings_file_exists() -> None:
    assert SETTINGS_PATH.is_file()


def test_settings_load_and_validate() -> None:
    settings = load_settings()
    assert settings.seed > 0
    assert settings.survey.population_size > settings.survey.sample_size
    # The four stratum parameter lists must line up or the population builder is ambiguous.
    n_strata = len(settings.survey.strata_proportions)
    assert len(settings.survey.strata_means) == n_strata
    assert len(settings.survey.strata_sds) == n_strata


def test_make_rng_is_reproducible() -> None:
    """Same seed, same stream. This is the property every figure in the repo rests on."""
    a = make_rng(123).random(10)
    b = make_rng(123).random(10)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, make_rng(124).random(10))
