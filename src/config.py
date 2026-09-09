"""Typed access to ``config/settings.yaml``.

Every sample size, seed and distribution parameter lives in the YAML file. Notebooks read
them from here so that no magic number is buried in a cell and every figure can be
regenerated from a recorded seed.
"""

from pathlib import Path

import numpy as np
from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"


class PRNGSettings(BaseModel):
    """Sizes for the generator-quality diagnostics."""

    n_uniform: int
    chi_square_bins: int
    plane_tolerance: float


class DrawSettings(BaseModel):
    """Sizes for the from-scratch distribution samplers."""

    n_samples: int
    grid_points: int
    exponential_rate: float


class SurveySettings(BaseModel):
    """Synthetic population shape and sampling-design sizes."""

    population_size: int
    sample_size: int
    strata_proportions: list[float]
    strata_means: list[float]
    strata_sds: list[float]
    n_groups: int
    group_size: int
    between_group_sd: float
    within_group_sd: float
    response_strength: float


class MonteCarloSettings(BaseModel):
    """Dart counts and the convergence-study grid."""

    n_darts: int
    n_replicates: int
    checkpoint_max: int
    checkpoint_points: int


class ImportanceSettings(BaseModel):
    """The rare-event problem and the two proposals compared against it."""

    threshold: float
    n_brute_force: int
    n_importance: int
    good_proposal_mean: float
    good_proposal_sd: float
    bad_proposal_mean: float
    bad_proposal_sd: float


class Settings(BaseSettings):
    """Project settings, loaded from ``config/settings.yaml``."""

    model_config = SettingsConfigDict(yaml_file=SETTINGS_PATH, extra="forbid")

    seed: int
    n_simulations: int
    prng: PRNGSettings
    draw: DrawSettings
    survey: SurveySettings
    monte_carlo: MonteCarloSettings
    importance: ImportanceSettings

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Read the YAML file only; no environment or dotenv overrides."""
        return (init_settings, YamlConfigSettingsSource(settings_cls))


def load_settings() -> Settings:
    """Load and validate ``config/settings.yaml``."""
    return Settings()


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Build the project generator (PCG64) from an explicit seed.

    Passing ``None`` uses the seed recorded in the settings file, which keeps every
    notebook run reproducible. The global ``np.random`` state is never touched.
    """
    resolved = load_settings().seed if seed is None else seed
    return np.random.default_rng(resolved)
