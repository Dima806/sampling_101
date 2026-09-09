"""Selection bias: when the sampling frame is correlated with the answer.

These are the failures no sample size can fix. If the units that end up in the sample were
chosen by a mechanism related to the quantity being measured, the estimate converges
happily and confidently to the wrong number. More data makes it worse, not better, because
the confidence interval shrinks around the bias.
"""

from dataclasses import dataclass

import numpy as np

from src.draw.discrete import AliasTable
from src.survey.simple import simple_random_indices


@dataclass(frozen=True)
class BiasResult:
    """What the truth was, what the biased design reported, and the gap between them."""

    truth: float
    estimate: float
    n: int

    @property
    def bias(self) -> float:
        """Signed error in the units of the measurement."""
        return self.estimate - self.truth

    @property
    def relative_bias(self) -> float:
        """Error as a fraction of the truth."""
        return self.bias / self.truth if self.truth != 0 else float("nan")


def frame_limit_mean(population: np.ndarray, propensity: np.ndarray) -> float:
    """The value a propensity-weighted sample mean converges to as ``n`` grows.

    ``sum_i p_i x_i``: the expectation of a single draw under the biased sampling
    distribution. This is the number more data buys you, and it is not the population mean.
    Having it in closed form turns "more data does not help" from a noisy empirical
    observation into an exact statement.
    """
    if propensity.shape != population.shape:
        raise ValueError("propensity must have the same shape as population")
    weights = propensity / propensity.sum()
    return float(np.dot(weights, population))


def response_biased_mean(
    population: np.ndarray,
    propensity: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> BiasResult:
    """Sample with probability proportional to ``propensity`` and take the plain mean.

    This is the survey that only hears from the people who felt like answering. The draw
    reuses the from-scratch alias table: sampling unequally is easy, and that is the
    problem. The estimator then ignores the unequal probabilities, as an unweighted mean
    of whoever replied always does.
    """
    if propensity.shape != population.shape:
        raise ValueError("propensity must have the same shape as population")
    table = AliasTable(propensity)
    chosen = table.sample(n, rng)
    return BiasResult(
        truth=float(population.mean()),
        estimate=float(population[chosen].mean()),
        n=n,
    )


def convenience_frame_mean(
    population: np.ndarray,
    in_frame: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> BiasResult:
    """Draw a correct simple random sample from the wrong population.

    Nothing about the draw is flawed. The sampling frame simply excludes part of the
    population, so the estimate is unbiased for the subgroup that could be reached and
    biased for the population that was asked about.
    """
    frame = np.flatnonzero(in_frame)
    if frame.size < n:
        raise ValueError(f"frame holds {frame.size} units, too few for a sample of {n}")
    chosen = frame[simple_random_indices(frame.size, n, rng)]
    return BiasResult(
        truth=float(population.mean()),
        estimate=float(population[chosen].mean()),
        n=n,
    )
