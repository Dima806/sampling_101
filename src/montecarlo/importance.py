"""Importance sampling: estimate a probability too small to ever observe by chance.

Brute force cannot estimate ``P(X > 5.199)`` for a standard normal. The event happens once
in ten million draws, so a million-draw simulation returns zero and a confidence interval
of zero width. The fix is to draw from a proposal ``q`` that makes the event common, then
undo the distortion by weighting each draw by ``p(x) / q(x)``:

    E_p[h(X)] = E_q[h(X) * p(X) / q(X)]

The estimator is still unbiased. What changes is the variance, and that is where the method
is won or lost. A proposal shifted onto the event turns it into a coin flip and needs a few
thousand draws. A proposal that leaves the event rare inherits brute force's problem: the
estimate rests on the one or two draws that happened to land in the tail, and its spread is
an order of magnitude worse. Choosing the proposal *is* the method.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.montecarlo.integration import MCResult, mc_result

LogDensity = Callable[[np.ndarray], np.ndarray]
Sampler = Callable[[int, np.random.Generator], np.ndarray]
Indicator = Callable[[np.ndarray], np.ndarray]

LOG_TWO_PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class ISResult:
    """An importance sampling estimate with the diagnostics that say whether to trust it."""

    value: float
    se: float
    weights: np.ndarray
    contributions: np.ndarray
    ess: float

    @property
    def n(self) -> int:
        """Number of proposal draws."""
        return int(self.weights.size)

    @property
    def ess_fraction(self) -> float:
        """Effective sample size as a share of the draws actually made."""
        return self.ess / self.n

    @property
    def ess_contributions(self) -> float:
        """Effective sample size of the terms that actually enter the estimate.

        The diagnostic that matters for a rare-event probability, and a sharper one than
        :attr:`ess`. Draws falling outside the event contribute exactly zero however large
        their weight, so the weights alone understate a good proposal and fail to
        distinguish it from a bad one. This counts how many draws genuinely carry the
        answer: a healthy run scores in the hundreds, a proposal that never reaches the
        event scores about 1.
        """
        return effective_sample_size(self.contributions)

    @property
    def max_weight_share(self) -> float:
        """Share of the estimate contributed by its single largest term.

        The tell-tale of a failing proposal. When one draw in five thousand supplies most
        of the answer, the estimate is a single lucky observation wearing a standard error.
        """
        total = float(self.contributions.sum())
        if total <= 0:
            return float("nan")
        return float(self.contributions.max()) / total


def normal_logpdf(x: np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """Log density of ``N(mu, sigma^2)``, evaluated from scratch.

    Densities are handled in log space throughout. At this project's threshold of 5.199 the
    naive ratio ``p(x) / q(x)`` would in fact be fine, since ``p(5.2)`` is about 5e-7. The
    discipline earns its keep as the event gets rarer: by ``x = 40`` the target density has
    fallen below the smallest representable double, so the naive weight silently becomes 0
    and takes the estimate with it. Working in logs costs nothing and removes the cliff.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    z = (x - mu) / sigma
    return -0.5 * LOG_TWO_PI - np.log(sigma) - 0.5 * z**2


def normal_proposal(mu: float, sigma: float) -> tuple[Sampler, LogDensity]:
    """Build a ``N(mu, sigma^2)`` proposal: a sampler and its log density."""

    def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(mu, sigma, size=n)

    def logpdf(x: np.ndarray) -> np.ndarray:
        return normal_logpdf(x, mu, sigma)

    return sampler, logpdf


def threshold_indicator(threshold: float) -> Indicator:
    """Indicator of the rare event ``{x > threshold}``."""

    def indicator(x: np.ndarray) -> np.ndarray:
        return (x > threshold).astype(np.float64)

    return indicator


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size, ``(sum w)^2 / sum(w^2)``.

    How many equally weighted draws the weighted sample is worth. Equal weights give back
    ``n``; one dominant weight gives back 1.
    """
    total = float(np.sum(weights))
    if total <= 0:
        return 0.0
    return total**2 / float(np.sum(weights**2))


def brute_force_probability(
    indicator: Indicator, sampler: Sampler, n: int, rng: np.random.Generator
) -> MCResult:
    """Estimate ``P(event)`` by sampling from the target and counting. The failing baseline.

    Included so the failure is demonstrated rather than asserted: for a one-in-ten-million
    event this returns exactly 0.0, with a standard error of 0.0, which is the most
    dangerous possible output.
    """
    return mc_result(indicator(sampler(n, rng)))


def importance_estimate(
    indicator: Indicator,
    target_logpdf: LogDensity,
    proposal_sampler: Sampler,
    proposal_logpdf: LogDensity,
    n: int,
    rng: np.random.Generator,
) -> ISResult:
    """Estimate ``E_p[indicator(X)]`` by sampling from the proposal and reweighting.

    Uses the unbiased estimator ``mean(w * h)``, valid because both densities are
    normalised. The self-normalised variant ``sum(w * h) / sum(w)`` is the one to reach for
    when a density is known only up to a constant; it trades a small bias for that freedom
    and is not needed here.
    """
    x = proposal_sampler(n, rng)
    log_weights = target_logpdf(x) - proposal_logpdf(x)
    weights = np.exp(log_weights)
    contributions = indicator(x) * weights
    result = mc_result(contributions)
    return ISResult(
        value=result.value,
        se=result.se,
        weights=weights,
        contributions=contributions,
        ess=effective_sample_size(weights),
    )
