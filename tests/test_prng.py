"""The good generator passes; the bad one passes the easy test and fails the real one.

This is the whole argument of notebook 01 as an assertion: marginal uniformity is not
evidence of randomness. RANDU's histogram is flat and its lag-1 correlation is negligible,
yet every single one of its triples lies on one of 15 planes.
"""

import numpy as np
import pytest

from src.config import Settings
from src.generators.prng import LCG, MINSTD, RANDU
from src.generators.quality import (
    chi_square_uniformity,
    lattice_triples,
    plane_count,
    plane_fraction,
    serial_correlation,
)

SEED = 1
N = 100_000

# RANDU's defining recurrence is x_{n+2} = 6 x_{n+1} - 9 x_n (mod 2**31), so exactly 15
# planes are reachable. This is a fact about the generator, not a fitted number.
RANDU_PLANES = 15


@pytest.fixture
def good_stream() -> np.ndarray:
    return LCG.from_params(MINSTD, seed=SEED).random(N)


@pytest.fixture
def bad_stream() -> np.ndarray:
    return LCG.from_params(RANDU, seed=SEED).random(N)


def test_lcg_is_reproducible() -> None:
    a = LCG.from_params(MINSTD, seed=7).random(50)
    b = LCG.from_params(MINSTD, seed=7).random(50)
    assert np.array_equal(a, b)


def test_lcg_output_is_in_unit_interval(good_stream: np.ndarray) -> None:
    assert good_stream.min() >= 0.0
    assert good_stream.max() < 1.0


def test_multiplicative_lcg_rejects_zero_seed() -> None:
    """A zero seed with c = 0 yields an all-zero stream. Fail loudly rather than silently."""
    with pytest.raises(ValueError, match="only zeros"):
        LCG.from_params(MINSTD, seed=0)


def test_good_generator_passes_uniformity(good_stream: np.ndarray) -> None:
    _, p_value = chi_square_uniformity(good_stream, bins=100)
    assert p_value > 0.01


def test_good_generator_has_no_serial_correlation(good_stream: np.ndarray) -> None:
    # Under independence the lag-1 correlation is ~N(0, 1/sqrt(N)); 4 standard errors at
    # N = 100_000 is 0.0126, so this is a wide bound that a sound generator clears easily.
    assert abs(serial_correlation(good_stream)) < 4.0 / np.sqrt(N)


def test_good_generator_has_no_lattice_structure(good_stream: np.ndarray) -> None:
    assert plane_fraction(good_stream) < 1e-6


def test_bad_generator_also_passes_the_easy_tests(bad_stream: np.ndarray) -> None:
    """The point of the project. RANDU looks fine until you test the right thing."""
    _, p_value = chi_square_uniformity(bad_stream, bins=100)
    assert p_value > 0.01
    assert abs(serial_correlation(bad_stream)) < 4.0 / np.sqrt(N)


def test_bad_generator_fails_the_structure_test(bad_stream: np.ndarray) -> None:
    """Every RANDU triple lies on a plane, and there are exactly 15 of them."""
    assert plane_fraction(bad_stream) == pytest.approx(1.0)
    assert plane_count(bad_stream) == RANDU_PLANES


def test_lattice_triples_shape(good_stream: np.ndarray) -> None:
    triples = lattice_triples(good_stream)
    assert triples.shape == (N - 2, 3)


def test_settings_prng_sizes_are_usable(settings: Settings) -> None:
    assert settings.prng.n_uniform >= 10_000
    assert settings.prng.chi_square_bins >= 10
