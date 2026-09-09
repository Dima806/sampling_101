"""Synthetic populations with known ground truth, so every estimate can be scored."""

from src.datasets.populations import (
    GroupedPopulation,
    StratifiedPopulation,
    grouped_population,
    normal_tail_probability,
    response_propensity,
    skewed_population,
)

__all__ = [
    "GroupedPopulation",
    "StratifiedPopulation",
    "grouped_population",
    "normal_tail_probability",
    "response_propensity",
    "skewed_population",
]
