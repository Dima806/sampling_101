"""Shared fixtures. Everything is seeded; no test depends on global random state."""

import numpy as np
import pytest

from src.config import Settings, load_settings


@pytest.fixture(scope="session")
def settings() -> Settings:
    """The project settings, read from config/settings.yaml."""
    return load_settings()


@pytest.fixture
def rng(settings: Settings) -> np.random.Generator:
    """A fresh generator seeded from the settings file, per test."""
    return np.random.default_rng(settings.seed)
