"""Sampling 101, interactive.

Four demonstrations, each one the live version of a notebook figure: a generator that
either fills the square or falls on stripes, a distribution built from uniforms, pi
converging as darts land, and a rare event that brute force cannot see.

Run with ``make run``.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.config import load_settings
from src.datasets.populations import normal_tail_probability
from src.draw.inverse_transform import (
    exponential_inv_cdf,
    numeric_inverse_cdf,
    sample_via_inverse_cdf,
)
from src.draw.rejection import rejection_sample
from src.evaluation.convergence import error_curve
from src.generators.prng import LCG, MINSTD, RANDU
from src.generators.quality import (
    chi_square_uniformity,
    lattice_triples,
    plane_count,
    plane_fraction,
    serial_correlation,
)
from src.montecarlo.importance import (
    brute_force_probability,
    importance_estimate,
    normal_logpdf,
    normal_proposal,
    threshold_indicator,
)
from src.montecarlo.integration import estimate_pi

GOOD = "#2a6f97"
BAD = "#c1440e"
TRUTH = "#444444"

# Interactive caps, kept well inside a 2-CPU box so dragging a slider stays responsive.
MAX_UNIFORM = 60_000
MAX_DARTS = 200_000
MAX_BRUTE_FORCE = 2_000_000

st.set_page_config(page_title="Sampling 101", page_icon="🎲", layout="wide")
settings = load_settings()


@st.cache_data
def lcg_stream(params_name: str, seed: int, n: int) -> np.ndarray:
    """Cached uniform stream from one of the two named generators."""
    params = RANDU if params_name == "RANDU" else MINSTD
    return LCG.from_params(params, seed=seed).random(n)


@st.cache_data
def cached_pi(n: int, seed: int) -> tuple[float, float, np.ndarray]:
    result = estimate_pi(n, np.random.default_rng(seed))
    return result.value, result.se, result.running


def randomness_inspector() -> None:
    st.subheader("Is your generator any good?")
    st.write(
        "Both generators below produce numbers that look uniform. Only one of them is "
        "actually usable. Plot consecutive pairs and triples to see the difference."
    )
    col1, col2 = st.columns([1, 3])
    with col1:
        name = st.radio("Generator", ["MINSTD (sound)", "RANDU (broken)"])
        n = st.slider("Numbers drawn", 2_000, MAX_UNIFORM, 20_000, step=2_000)
        seed = st.number_input("Seed", min_value=1, max_value=10**9, value=1)
        view = st.radio("View", ["Pairs (2-D)", "Triples (3-D)"])

    key = "RANDU" if name.startswith("RANDU") else "MINSTD"
    u = lcg_stream(key, int(seed), n)
    colour = BAD if key == "RANDU" else GOOD
    shown = min(n, 4000)

    with col2:
        if view.startswith("Pairs"):
            fig = go.Figure(
                go.Scattergl(
                    x=u[:-1][:shown],
                    y=u[1:][:shown],
                    mode="markers",
                    marker={"size": 3, "color": colour, "opacity": 0.5},
                )
            )
            fig.update_layout(
                xaxis_title="u_i",
                yaxis_title="u_(i+1)",
                height=520,
                title="Consecutive pairs",
            )
        else:
            triples = lattice_triples(u)[:shown]
            fig = go.Figure(
                go.Scatter3d(
                    x=triples[:, 0],
                    y=triples[:, 1],
                    z=triples[:, 2],
                    mode="markers",
                    marker={"size": 1.6, "color": colour, "opacity": 0.6},
                )
            )
            fig.update_layout(height=520, title="Consecutive triples: rotate to find the planes")
        st.plotly_chart(fig, width="stretch")

    statistic, p_value = chi_square_uniformity(u, bins=settings.prng.chi_square_bins)
    c1, c2, c3 = st.columns(3)
    c1.metric("Uniformity test p-value", f"{p_value:.3f}", help="Both generators pass this.")
    c2.metric("Lag-1 correlation", f"{serial_correlation(u):+.5f}")
    c3.metric(
        "Triples on a lattice plane",
        f"{plane_fraction(u):.1%}",
        help="This is the test that separates them.",
    )
    if plane_fraction(u) > 0.5:
        st.error(
            f"Every triple lies on one of {plane_count(u)} planes. The histogram is flat, "
            f"the correlation is zero, and the generator is still unusable "
            f"(chi-square statistic {statistic:.1f})."
        )
    else:
        st.success("No lattice structure detected. Marginals uniform, triples fill the cube.")


def distribution_sampler() -> None:
    st.subheader("Building a distribution out of uniform numbers")
    st.write(
        "Inverse transform sampling pushes uniforms through the inverse CDF. Rejection "
        "sampling proposes from something easy and throws away the excess. Both are built "
        "from scratch in `src/draw/`."
    )
    col1, col2 = st.columns([1, 3])
    with col1:
        method = st.radio("Method", ["Inverse transform", "Rejection"])
        target = st.selectbox("Target", ["Exponential", "Normal", "Bimodal"])
        n = st.slider("Samples", 1_000, 50_000, 10_000, step=1_000)
        seed = st.number_input("Seed ", min_value=1, max_value=10**9, value=int(settings.seed))

    rng = np.random.default_rng(int(seed))
    rate = settings.draw.exponential_rate

    if target == "Exponential":
        lo, hi = 0.0, 6.0

        def pdf(x: np.ndarray) -> np.ndarray:
            return rate * np.exp(-rate * x)

    elif target == "Normal":
        lo, hi = -4.0, 4.0

        def pdf(x: np.ndarray) -> np.ndarray:
            return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    else:
        lo, hi = -4.0, 6.0

        def pdf(x: np.ndarray) -> np.ndarray:
            left = np.exp(-0.5 * ((x + 1.5) / 0.7) ** 2)
            right = np.exp(-0.5 * ((x - 3.0) / 1.1) ** 2)
            return 0.6 * left + 0.4 * right

    acceptance: float | None = None
    if method == "Inverse transform":
        if target == "Exponential":
            inv_cdf = exponential_inv_cdf(rate)
        else:
            inv_cdf = numeric_inverse_cdf(pdf, lo, hi, grid=settings.draw.grid_points)
        samples = sample_via_inverse_cdf(inv_cdf, n, rng)
    else:
        grid = np.linspace(lo, hi, 500)
        m = float(np.max(pdf(grid)) * (hi - lo)) * 1.05
        result = rejection_sample(
            pdf,
            lambda k, r: r.uniform(lo, hi, size=k),
            lambda x: np.full_like(x, 1.0 / (hi - lo)),
            m,
            n,
            rng,
            check_grid=grid,
        )
        samples = result.samples
        acceptance = result.acceptance_rate

    with col2:
        grid = np.linspace(lo, hi, 400)
        fig = go.Figure()
        fig.add_histogram(
            x=samples,
            histnorm="probability density",
            nbinsx=70,
            name="samples",
            marker_color=GOOD,
            opacity=0.6,
        )
        fig.add_scatter(
            x=grid,
            y=pdf(grid) / np.trapezoid(pdf(grid), grid),
            mode="lines",
            name="target density",
            line={"color": TRUTH, "width": 3},
        )
        fig.update_layout(height=480, title=f"{n:,} samples via {method.lower()}")
        st.plotly_chart(fig, width="stretch")

    if acceptance is not None:
        c1, c2 = st.columns(2)
        c1.metric("Acceptance rate", f"{acceptance:.1%}")
        c2.metric("Proposals per sample", f"{1 / acceptance:.2f}")
        st.caption(
            "Rejection sampling pays for a loose envelope in wasted proposals. Inverse "
            "transform never throws anything away, when you can invert the CDF."
        )


def pi_by_darts() -> None:
    st.subheader("Estimating pi by throwing darts")
    st.write(
        "A dart in the unit square lands inside the quarter circle with probability pi/4. "
        "Count the hits, multiply by four. No geometry, no calculus, just counting."
    )
    col1, col2 = st.columns([1, 3])
    with col1:
        n = st.slider("Darts", 1_000, MAX_DARTS, 50_000, step=1_000)
        seed = st.number_input("Seed  ", min_value=1, max_value=10**9, value=int(settings.seed))

    value, se, running = cached_pi(n, int(seed))
    ns = np.arange(1, n + 1)

    with col2:
        fig = go.Figure()
        fig.add_scatter(x=ns, y=running, mode="lines", name="estimate", line={"color": GOOD})
        fig.add_hline(y=float(np.pi), line={"color": TRUTH, "dash": "dash"})
        band = 1.96 * 4 * np.sqrt((np.pi / 4) * (1 - np.pi / 4) / ns)
        fig.add_scatter(
            x=ns,
            y=np.pi + band,
            mode="lines",
            name="95% band",
            line={"color": TRUTH, "width": 1, "dash": "dot"},
        )
        fig.add_scatter(
            x=ns,
            y=np.pi - band,
            mode="lines",
            showlegend=False,
            line={"color": TRUTH, "width": 1, "dash": "dot"},
        )
        fig.update_layout(
            height=460,
            xaxis_type="log",
            xaxis_title="darts thrown",
            yaxis_title="estimate",
            yaxis_range=[np.pi - 0.35, np.pi + 0.35],
            title="The estimate settling into its band",
        )
        st.plotly_chart(fig, width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("Estimate", f"{value:.5f}", delta=f"{value - np.pi:+.5f} vs pi")
    c2.metric("Standard error", f"{se:.5f}")
    c3.metric("Darts for one more digit", f"{n * 100:,}", help="Error falls like 1/sqrt(n).")
    st.caption(
        "The curve flattens because accuracy costs quadratically: 100x the darts buy 10x "
        "the precision. That is the price of Monte Carlo, and the reason importance "
        "sampling exists."
    )


def rare_event_estimator() -> None:
    st.subheader("A probability too small to sample")
    st.write(
        "Estimating P(X > t) for a standard normal. Push the threshold out far enough and "
        "brute force stops working entirely: the event never happens, so the answer is "
        "zero with no uncertainty attached."
    )
    col1, col2 = st.columns([1, 3])
    with col1:
        threshold = st.slider("Threshold t", 3.0, 6.0, float(settings.importance.threshold), 0.1)
        n_brute = st.select_slider(
            "Brute-force draws", [10_000, 100_000, 1_000_000, MAX_BRUTE_FORCE], value=1_000_000
        )
        n_is = st.slider("Importance draws", 1_000, 20_000, settings.importance.n_importance, 1000)
        proposal_mean = st.slider("Proposal centre", 0.0, 7.0, float(threshold), 0.1)
        seed = st.number_input("Seed   ", min_value=1, max_value=10**9, value=int(settings.seed))

    truth = normal_tail_probability(threshold)
    indicator = threshold_indicator(threshold)
    rng = np.random.default_rng(int(seed))

    brute = brute_force_probability(
        indicator, lambda k, r: r.normal(0.0, 1.0, size=k), int(n_brute), rng
    )
    sampler, logpdf = normal_proposal(proposal_mean, settings.importance.good_proposal_sd)
    result = importance_estimate(
        indicator, lambda x: normal_logpdf(x, 0.0, 1.0), sampler, logpdf, n_is, rng
    )

    with col2:
        fig = go.Figure()
        fig.add_scatter(
            x=np.arange(1, n_is + 1),
            y=error_curve(np.cumsum(result.contributions) / np.arange(1, n_is + 1), truth) / truth,
            mode="lines",
            name="importance sampling",
            line={"color": GOOD},
        )
        fig.add_hline(
            y=1.0,
            line={"color": BAD, "dash": "dash"},
            annotation_text="brute force: 100% error (estimate is 0)",
        )
        fig.update_layout(
            height=440,
            xaxis_type="log",
            yaxis_type="log",
            xaxis_title="draws",
            yaxis_title="relative error",
            title="Relative error against the exact answer",
        )
        st.plotly_chart(fig, width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("Exact answer", f"{truth:.3e}")
    c2.metric(
        f"Brute force ({n_brute:,} draws)",
        f"{brute.value:.3e}",
        delta="never saw the event" if brute.value == 0 else "found it",
        delta_color="inverse" if brute.value == 0 else "normal",
    )
    c3.metric(
        f"Importance sampling ({n_is:,} draws)",
        f"{result.value:.3e}",
        delta=f"{(result.value / truth - 1):+.1%} vs exact",
    )

    st.markdown("**Is this estimate trustworthy?**")
    d1, d2 = st.columns(2)
    d1.metric("Draws carrying the answer", f"{result.ess_contributions:.0f} of {n_is:,}")
    d2.metric("Largest single contribution", f"{result.max_weight_share:.1%}")
    if result.ess_contributions < 20 or result.max_weight_share > 0.3:
        st.error(
            "The proposal is not covering the event. The estimate rests on a handful of "
            "draws, so it is luck, not measurement. Move the proposal centre onto the "
            "threshold."
        )
    else:
        st.success(
            "The proposal covers the event well: thousands of draws contribute and no "
            "single one dominates."
        )
    st.caption(
        "Importance sampling is only as good as its proposal. Slide the centre back to 0 "
        "and watch a method that looked magical become no better than brute force."
    )


st.title("Sampling 101")
st.caption("Your random sample is not random, and it is costing you.")

tabs = st.tabs(
    ["Randomness inspector", "Distribution sampler", "Pi by darts", "Rare-event estimator"]
)
with tabs[0]:
    randomness_inspector()
with tabs[1]:
    distribution_sampler()
with tabs[2]:
    pi_by_darts()
with tabs[3]:
    rare_event_estimator()
