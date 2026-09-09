"""Importance sampling recovers a one-in-ten-million probability from a few thousand draws.

The payoff of the project, and its honest caveat. Three claims are asserted:
brute force fails outright, a well-chosen proposal succeeds cheaply, and a proposal that
leaves the event rare is dramatically worse while still looking plausible on any single run.
"""

import numpy as np
import pytest

from src.config import Settings
from src.datasets.populations import normal_tail_probability
from src.montecarlo.importance import (
    brute_force_probability,
    effective_sample_size,
    importance_estimate,
    normal_logpdf,
    normal_proposal,
    threshold_indicator,
)
from src.montecarlo.variance import weight_diagnostics

N_REPLICATES = 200


def target_logpdf(x: np.ndarray) -> np.ndarray:
    return normal_logpdf(x, 0.0, 1.0)


def standard_normal_sampler(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, 1.0, size=n)


def _spread(mu: float, sd: float, settings: Settings, rng: np.random.Generator) -> np.ndarray:
    """Repeat the estimate many times and return the estimates, for a variance comparison."""
    sampler, logpdf = normal_proposal(mu, sd)
    indicator = threshold_indicator(settings.importance.threshold)
    return np.array(
        [
            importance_estimate(
                indicator, target_logpdf, sampler, logpdf, settings.importance.n_importance, rng
            ).value
            for _ in range(N_REPLICATES)
        ]
    )


def test_the_event_really_is_one_in_ten_million(settings: Settings) -> None:
    truth = normal_tail_probability(settings.importance.threshold)
    assert truth == pytest.approx(1e-7, rel=0.01)


def test_brute_force_never_sees_the_event(settings: Settings, rng: np.random.Generator) -> None:
    """A million draws return exactly zero, with a standard error of zero.

    The most dangerous output in the project: an estimate that is confidently, precisely
    wrong, and gives no hint that anything went wrong.
    """
    result = brute_force_probability(
        threshold_indicator(settings.importance.threshold),
        standard_normal_sampler,
        settings.importance.n_brute_force,
        rng,
    )
    assert result.value == 0.0
    assert result.se == 0.0


def test_importance_sampling_recovers_the_probability(
    settings: Settings, rng: np.random.Generator
) -> None:
    """A few thousand draws, and the answer is right to within a couple of percent."""
    cfg = settings.importance
    truth = normal_tail_probability(cfg.threshold)
    sampler, logpdf = normal_proposal(cfg.good_proposal_mean, cfg.good_proposal_sd)
    result = importance_estimate(
        threshold_indicator(cfg.threshold),
        target_logpdf,
        sampler,
        logpdf,
        cfg.n_importance,
        rng,
    )
    assert result.value == pytest.approx(truth, rel=0.15)
    assert abs(result.value - truth) < 4 * result.se


def test_importance_sampling_is_unbiased_across_replicates(
    settings: Settings, rng: np.random.Generator
) -> None:
    cfg = settings.importance
    truth = normal_tail_probability(cfg.threshold)
    estimates = _spread(cfg.good_proposal_mean, cfg.good_proposal_sd, settings, rng)
    se = estimates.std(ddof=1) / np.sqrt(N_REPLICATES)
    assert abs(estimates.mean() - truth) < 4 * se


def test_a_proposal_that_leaves_the_event_rare_is_far_worse(
    settings: Settings, rng: np.random.Generator
) -> None:
    """The honest caveat: the method is only as good as the proposal.

    N(0, 1.5) barely shifts the distribution, so the event stays rare under the proposal
    too and the estimate rests on the one or two draws that reached the tail.
    """
    cfg = settings.importance
    good = _spread(cfg.good_proposal_mean, cfg.good_proposal_sd, settings, rng)
    bad = _spread(cfg.bad_proposal_mean, cfg.bad_proposal_sd, settings, rng)
    # Measured ratio is around 25x; 5x is a wide margin that still cannot pass by accident.
    assert bad.std(ddof=1) > 5 * good.std(ddof=1)


def test_the_diagnostic_flags_the_bad_proposal_before_the_estimate_does(
    settings: Settings, rng: np.random.Generator
) -> None:
    """A single run of the bad proposal looks fine. Its effective sample size does not.

    This is what makes the caveat usable in practice: you cannot see the problem in the
    point estimate, but you can see it in how few draws actually carry the answer.
    """
    cfg = settings.importance
    indicator = threshold_indicator(cfg.threshold)

    good_sampler, good_logpdf = normal_proposal(cfg.good_proposal_mean, cfg.good_proposal_sd)
    bad_sampler, bad_logpdf = normal_proposal(cfg.bad_proposal_mean, cfg.bad_proposal_sd)
    good = importance_estimate(
        indicator, target_logpdf, good_sampler, good_logpdf, cfg.n_importance, rng
    )
    bad = importance_estimate(
        indicator, target_logpdf, bad_sampler, bad_logpdf, cfg.n_importance, rng
    )

    assert good.ess_contributions > 100
    assert good.max_weight_share < 0.05

    # On this seed the bad proposal is worse than "imprecise": in 5000 draws it never
    # reaches the event at all and returns exactly 0.0, the same answer brute force gave
    # after a million draws. Its diagnostic says so; its point estimate does not.
    assert bad.ess_contributions < 10
    assert bad.value == 0.0 or bad.max_weight_share > 0.5


def test_effective_sample_size_bounds() -> None:
    """Equal weights are worth n draws; one dominant weight is worth about one."""
    assert effective_sample_size(np.ones(100)) == pytest.approx(100.0)
    dominant = np.concatenate([[1e6], np.ones(99)])
    assert effective_sample_size(dominant) < 2.0
    assert effective_sample_size(np.zeros(10)) == 0.0


def test_weight_diagnostics_report_the_shape_of_the_weights() -> None:
    diagnostics = weight_diagnostics(np.ones(50))
    assert diagnostics.ess == pytest.approx(50.0)
    assert diagnostics.ess_fraction == pytest.approx(1.0)
    assert diagnostics.coefficient_of_variation == pytest.approx(0.0)


def test_naive_density_ratio_silently_underflows_far_out_in_the_tail() -> None:
    """Why the module works in logs, demonstrated at a threshold where it actually bites.

    Not at this project's threshold of 5.199, where p(x) is about 5e-7 and the naive ratio
    is perfectly well behaved. Push the event out to x = 40 and the target density falls
    below the smallest representable double: the naive weight becomes 0, so the estimate
    becomes 0, and nothing raises.
    """
    x = np.array([40.0])
    target = np.exp(normal_logpdf(x, 0.0, 1.0))
    proposal = np.exp(normal_logpdf(x, 40.0, 1.0))

    assert target[0] == 0.0  # underflowed
    assert proposal[0] > 0.0  # the proposal is centred here, so it is fine
    assert (target / proposal)[0] == 0.0  # a weight of zero, with no warning

    # In log space the same quantity is finite and exact.
    log_weight = normal_logpdf(x, 0.0, 1.0) - normal_logpdf(x, 40.0, 1.0)
    assert np.isfinite(log_weight).all()
    assert log_weight[0] == pytest.approx(-800.0, abs=1.0)


def test_log_space_weights_are_usable_at_the_project_threshold(settings: Settings) -> None:
    """At the threshold actually used, the log weight exponentiates to a positive number."""
    cfg = settings.importance
    x = np.array([cfg.threshold + 0.5])
    log_weight = normal_logpdf(x, 0.0, 1.0) - normal_logpdf(x, cfg.good_proposal_mean, 1.0)
    weight = np.exp(log_weight)
    assert weight[0] > 0.0
    assert np.isfinite(weight).all()
