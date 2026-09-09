"""Pseudorandom generators and the diagnostics that tell a good one from a bad one."""

from src.generators.prng import LCG, MINSTD, RANDU
from src.generators.quality import (
    RANDU_PLANE_DIRECTION,
    chi_square_uniformity,
    lattice_triples,
    overlapping_tuples,
    plane_count,
    plane_fraction,
    plane_projection,
    serial_correlation,
)

__all__ = [
    "LCG",
    "MINSTD",
    "RANDU",
    "RANDU_PLANE_DIRECTION",
    "chi_square_uniformity",
    "lattice_triples",
    "overlapping_tuples",
    "plane_count",
    "plane_fraction",
    "plane_projection",
    "serial_correlation",
]
