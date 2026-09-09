"""Rejection sampling: draw from a hard distribution using an easy one.

Propose from a distribution ``q`` you can sample, accept each proposal with probability
``p(x) / (m * q(x))``, and the survivors are distributed as ``p``. The cost is the
rejected work: the acceptance rate is exactly ``1 / m``, so a loose envelope is expensive.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Density = Callable[[np.ndarray], np.ndarray]
Sampler = Callable[[int, np.random.Generator], np.ndarray]


@dataclass(frozen=True)
class RejectionResult:
    """Accepted draws plus the bookkeeping needed to check the acceptance rate."""

    samples: np.ndarray
    n_proposed: int
    n_accepted: int

    @property
    def acceptance_rate(self) -> float:
        """Empirical acceptance rate, which should land near ``1 / m``."""
        if self.n_proposed == 0:
            return float("nan")
        return self.n_accepted / self.n_proposed


def check_envelope(
    target_pdf: Density,
    proposal_pdf: Density,
    m: float,
    grid: np.ndarray,
    tol: float = 1e-9,
) -> None:
    """Raise if ``m * q(x) >= p(x)`` fails anywhere on ``grid``.

    An envelope that dips below the target silently produces the wrong distribution, and
    the samples still look plausible. Checking is cheap; not checking is a trap.
    """
    shortfall = target_pdf(grid) - m * proposal_pdf(grid)
    worst = float(np.max(shortfall))
    if worst > tol:
        raise ValueError(
            f"envelope does not dominate the target: max p(x) - m*q(x) = {worst:.3e}. Increase m."
        )


def rejection_sample(
    target_pdf: Density,
    proposal_sampler: Sampler,
    proposal_pdf: Density,
    m: float,
    n: int,
    rng: np.random.Generator,
    check_grid: np.ndarray | None = None,
) -> RejectionResult:
    """Draw ``n`` samples from ``target_pdf`` by proposing from ``proposal_sampler``.

    Parameters
    ----------
    m
        Envelope constant, satisfying ``m * proposal_pdf(x) >= target_pdf(x)`` everywhere.
    check_grid
        Points on which to verify that condition before sampling. Strongly recommended.
    """
    if m < 1.0:
        raise ValueError(f"envelope constant must be at least 1, got {m}")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if check_grid is not None:
        check_envelope(target_pdf, proposal_pdf, m, check_grid)

    accepted: list[np.ndarray] = []
    n_proposed = 0
    n_accepted = 0

    while n_accepted < n:
        # Propose a little more than the expected shortfall so the loop rarely repeats.
        remaining = n - n_accepted
        batch = max(int(remaining * m * 1.1), 64)
        x = proposal_sampler(batch, rng)
        u = rng.random(batch)
        keep = u * m * proposal_pdf(x) <= target_pdf(x)
        accepted.append(x[keep])
        n_proposed += batch
        n_accepted += int(np.count_nonzero(keep))

    samples = np.concatenate(accepted) if accepted else np.empty(0, dtype=np.float64)
    # n_accepted counts every acceptance, including the surplus in the final batch, so the
    # ratio against n_proposed stays an unbiased estimate of 1/m.
    return RejectionResult(samples=samples[:n], n_proposed=n_proposed, n_accepted=n_accepted)
