"""Shared matplotlib helpers.

Every figure a notebook draws is built here and written to ``outputs/figures/`` as a PNG.
Filenames are stable on purpose: the article pulls a fixed set of images out of that
directory and renaming them silently breaks it.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox

from src.config import FIGURE_DIR
from src.generators.quality import RANDU_PLANE_DIRECTION

GOOD_COLOUR = "#2a6f97"
BAD_COLOUR = "#c1440e"
TRUTH_COLOUR = "#444444"
ACCENT_COLOUR = "#7a9e7e"


def figure_path(name: str) -> Path:
    """Absolute path for a named figure, creating the output directory if needed."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURE_DIR / f"{name}.png"


def save_figure(fig: Figure, name: str, dpi: int = 150) -> Path:
    """Write a figure to ``outputs/figures/<name>.png`` and return the path."""
    path = figure_path(name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


# --------------------------------------------------------------------------------------
# Notebook 01: generator quality
# --------------------------------------------------------------------------------------


def plot_pair_scatter(
    u: np.ndarray, title: str, colour: str = GOOD_COLOUR, max_points: int = 4000
) -> Figure:
    """Scatter consecutive pairs ``(u_i, u_{i+1})``: the first look at generator structure."""
    pairs = np.stack([u[:-1], u[1:]], axis=1)[:max_points]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pairs[:, 0], pairs[:, 1], s=2, alpha=0.5, color=colour)
    ax.set_xlabel("$u_i$")
    ax.set_ylabel("$u_{i+1}$")
    ax.set_title(title)
    return fig


def plot_generator_comparison(
    u_good: np.ndarray,
    u_bad: np.ndarray,
    good_label: str = "MINSTD",
    bad_label: str = "RANDU",
    max_points: int = 4000,
) -> Figure:
    """Two generators side by side in 2-D. In pairs, both still look fine."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharex=True, sharey=True)
    for ax, u, label, colour in (
        (axes[0], u_good, good_label, GOOD_COLOUR),
        (axes[1], u_bad, bad_label, BAD_COLOUR),
    ):
        pairs = np.stack([u[:-1], u[1:]], axis=1)[:max_points]
        ax.scatter(pairs[:, 0], pairs[:, 1], s=2, alpha=0.5, color=colour)
        ax.set_xlabel("$u_i$")
        ax.set_title(label)
    axes[0].set_ylabel("$u_{i+1}$")
    fig.suptitle("Consecutive pairs: both generators fill the square")
    return fig


def edge_on_view(
    direction: tuple[float, float, float] = RANDU_PLANE_DIRECTION, elev: float = 5.0
) -> tuple[float, float]:
    """Camera angles that look *along* a family of planes, so they appear edge-on.

    Matplotlib's camera points along
    ``(cos(elev)cos(azim), cos(elev)sin(azim), sin(elev))``. Planes with normal
    ``direction`` collapse into visible stripes when that vector is perpendicular to the
    normal, i.e. when ``d0*cos(elev)cos(azim) + d1*cos(elev)sin(azim) + d2*sin(elev) = 0``.
    Dividing by ``cos(elev)`` and writing the left side as ``R*cos(azim - psi)`` solves for
    the azimuth exactly.

    This matters more than it sounds: at a generic viewing angle RANDU's triples look like
    an ordinary random cloud. The structure is only visible if you know where to stand.
    """
    d0, d1, d2 = direction
    radius = float(np.hypot(d0, d1))
    if radius == 0:
        raise ValueError("direction must have a non-zero component in x or y")
    psi = float(np.arctan2(d1, d0))
    target = -d2 * np.tan(np.radians(elev)) / radius
    if not -1.0 <= target <= 1.0:
        raise ValueError(f"no azimuth gives an edge-on view at elev={elev}")
    azim = float(np.degrees(psi + np.arccos(target)))
    return elev, azim


def plot_lattice_3d(
    triples: np.ndarray,
    title: str,
    max_points: int = 3000,
    colour: str = BAD_COLOUR,
    view: tuple[float, float] | None = None,
) -> Figure:
    """Scatter consecutive triples in 3-D.

    Defaults to the edge-on view of RANDU's plane family. Pass ``view`` to look from
    somewhere else, which is how you demonstrate that the angle is doing the work.
    """
    points = triples[:max_points]
    elev, azim = edge_on_view() if view is None else view
    fig = plt.figure(figsize=(6, 5.5))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, alpha=0.4, color=colour)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("$u_i$")
    ax.set_ylabel("$u_{i+1}$")
    ax.set_zlabel("$u_{i+2}$")
    ax.set_title(title)
    return fig


def plot_plane_projection(
    projection_good: np.ndarray, projection_bad: np.ndarray, max_points: int = 4000
) -> Figure:
    """Histogram of ``9u_i - 6u_{i+1} + u_{i+2}``: the quantitative form of the stripes."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, values, label, colour in (
        (axes[0], projection_good, "MINSTD", GOOD_COLOUR),
        (axes[1], projection_bad, "RANDU", BAD_COLOUR),
    ):
        ax.hist(values[:max_points], bins=240, color=colour)
        ax.set_xlabel(r"$9u_i - 6u_{i+1} + u_{i+2}$")
        ax.set_title(label)
    axes[0].set_ylabel("triples")
    fig.suptitle("Projected onto RANDU's plane normal: a continuum, or 15 spikes")
    return fig


def plot_uniformity_histograms(u_good: np.ndarray, u_bad: np.ndarray, bins: int = 50) -> Figure:
    """Marginal histograms. The test both generators pass, which is the point."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, u, label, colour in (
        (axes[0], u_good, "MINSTD", GOOD_COLOUR),
        (axes[1], u_bad, "RANDU", BAD_COLOUR),
    ):
        ax.hist(u, bins=bins, range=(0, 1), color=colour, alpha=0.8)
        ax.axhline(u.size / bins, color=TRUTH_COLOUR, ls="--", lw=1.5)
        ax.set_title(label)
        ax.set_xlabel("value")
    axes[0].set_ylabel("count")
    fig.suptitle("Marginal uniformity: both generators pass this test")
    return fig


# --------------------------------------------------------------------------------------
# Notebook 02: samplers
# --------------------------------------------------------------------------------------


def plot_samples_vs_density(
    samples: np.ndarray,
    pdf: Callable[[np.ndarray], np.ndarray],
    lo: float,
    hi: float,
    title: str,
    bins: int = 60,
) -> Figure:
    """Histogram of draws against the density they were meant to come from."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        samples,
        bins=bins,
        range=(lo, hi),
        density=True,
        alpha=0.55,
        color=GOOD_COLOUR,
        label="samples",
    )
    grid = np.linspace(lo, hi, 400)
    ax.plot(grid, pdf(grid), color=TRUTH_COLOUR, lw=2, label="target density")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_inverse_cdf_mechanism(
    cdf: Callable[[np.ndarray], np.ndarray],
    inv_cdf: Callable[[np.ndarray], np.ndarray],
    lo: float,
    hi: float,
    demo_uniforms: Sequence[float] = (0.15, 0.5, 0.85),
) -> Figure:
    """The method in one picture: a uniform on the y-axis, read across and down."""
    grid = np.linspace(lo, hi, 400)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(grid, cdf(grid), color=TRUTH_COLOUR, lw=2, label="CDF $F(x)$")
    for u in demo_uniforms:
        x = float(inv_cdf(np.array([u]))[0])
        ax.plot([lo, x], [u, u], color=GOOD_COLOUR, lw=1.2, ls="--")
        ax.plot([x, x], [0, u], color=BAD_COLOUR, lw=1.2, ls="--")
        ax.plot([x], [u], "o", color=BAD_COLOUR, ms=6)
        ax.annotate(f"u={u}", (lo, u), textcoords="offset points", xytext=(4, 4), fontsize=9)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    ax.set_title("Inverse transform: draw u upward, read x off the axis")
    ax.legend(loc="lower right")
    return fig


def plot_rejection_illustration(
    target_pdf: Callable[[np.ndarray], np.ndarray],
    lo: float,
    hi: float,
    m: float,
    n: int,
    rng: np.random.Generator,
) -> Figure:
    """Proposals under a uniform envelope, coloured by whether they were kept.

    Draws its own proposals rather than reusing the sampler's, because the rejected points
    are exactly what the sampler throws away and what this picture is about.
    """
    x = rng.uniform(lo, hi, size=n)
    height = m / (hi - lo)
    y = rng.uniform(0.0, height, size=n)
    keep = y <= target_pdf(x)

    grid = np.linspace(lo, hi, 400)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.scatter(x[keep], y[keep], s=4, alpha=0.5, color=GOOD_COLOUR, label="accepted")
    ax.scatter(x[~keep], y[~keep], s=4, alpha=0.35, color=BAD_COLOUR, label="rejected")
    ax.plot(grid, target_pdf(grid), color=TRUTH_COLOUR, lw=2.2, label="target $p(x)$")
    ax.axhline(height, color=TRUTH_COLOUR, ls="--", lw=1.5, label="envelope $m\\,q(x)$")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title(f"Rejection sampling: {keep.mean():.1%} of proposals survive")
    ax.legend(loc="upper right", fontsize=9)
    return fig


# --------------------------------------------------------------------------------------
# Notebook 03: survey designs
# --------------------------------------------------------------------------------------


def plot_population_skew(values: np.ndarray, strata: np.ndarray, upper_quantile: float = 0.995):
    """The population that breaks simple random sampling: a long, valuable tail."""
    cut = float(np.quantile(values, upper_quantile))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(values, bins=80, range=(0, cut), color=GOOD_COLOUR, alpha=0.85)
    axes[0].set_xlabel("value")
    axes[0].set_ylabel("units")
    axes[0].set_title(f"Population (clipped at the {upper_quantile:.1%} quantile)")

    labels = np.unique(strata)
    shares = [values[strata == h].sum() / values.sum() for h in labels]
    counts = [float((strata == h).mean()) for h in labels]
    width = 0.38
    positions = np.arange(labels.size)
    axes[1].bar(positions - width / 2, counts, width, label="share of units", color=GOOD_COLOUR)
    axes[1].bar(positions + width / 2, shares, width, label="share of total", color=BAD_COLOUR)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([f"stratum {h}" for h in labels])
    axes[1].set_title("The smallest stratum carries a large share of the total")
    axes[1].legend()
    return fig


def plot_estimate_spread(
    designs: dict[str, np.ndarray], truth: float, title: str, xlabel: str = "estimate"
) -> Figure:
    """Overlaid histograms of repeated estimates from competing designs.

    The picture behind the stratification claim: every design centres on the truth, and
    some are visibly narrower than others.
    """
    fig, ax = plt.subplots(figsize=(8, 4.4))
    colours = [BAD_COLOUR, GOOD_COLOUR, ACCENT_COLOUR, "#b08968"]
    for (label, values), colour in zip(designs.items(), colours, strict=False):
        ax.hist(
            values,
            bins=40,
            alpha=0.55,
            label=f"{label} (sd = {values.std(ddof=1):.3f})",
            color=colour,
        )
    ax.axvline(truth, color=TRUTH_COLOUR, ls="--", lw=1.8, label=f"truth = {truth:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("repeated samples")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_bias_comparison(
    labels: Sequence[str], estimates: Sequence[float], truth: float, title: str
) -> Figure:
    """Point estimates from several designs against the known truth."""
    positions = np.arange(len(labels))
    colours = [
        BAD_COLOUR if abs(e - truth) > 0.02 * abs(truth) else GOOD_COLOUR for e in estimates
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh(positions, list(estimates), color=colours, alpha=0.85)
    ax.axvline(truth, color=TRUTH_COLOUR, ls="--", lw=1.8, label=f"truth = {truth:.2f}")
    ax.set_yticks(positions)
    ax.set_yticklabels(list(labels))
    ax.invert_yaxis()
    ax.set_xlabel("estimated mean")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_leakage(naive_rate: float, grouped_rate: float) -> Figure:
    """Share of test entities that also appear in training, for the two split strategies."""
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.barh([0, 1], [naive_rate, grouped_rate], color=[BAD_COLOUR, GOOD_COLOUR], alpha=0.9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["random row split", "group-aware split"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("share of test entities also seen in training")
    for y, rate in ((0, naive_rate), (1, grouped_rate)):
        ax.annotate(
            f"{rate:.1%}",
            (rate, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
        )
    ax.set_title("Entity leakage")
    return fig


# --------------------------------------------------------------------------------------
# Notebook 04: Monte Carlo
# --------------------------------------------------------------------------------------


def plot_darts(n: int, rng: np.random.Generator) -> Figure:
    """The darts themselves, coloured by whether they landed in the quarter circle."""
    darts = rng.random((n, 2))
    inside = (darts**2).sum(axis=1) <= 1.0
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.scatter(
        darts[inside, 0], darts[inside, 1], s=3, alpha=0.5, color=GOOD_COLOUR, label="inside"
    )
    ax.scatter(
        darts[~inside, 0], darts[~inside, 1], s=3, alpha=0.5, color=BAD_COLOUR, label="outside"
    )
    arc = np.linspace(0, np.pi / 2, 200)
    ax.plot(np.cos(arc), np.sin(arc), color=TRUTH_COLOUR, lw=2)
    ax.set_aspect("equal")
    ax.set_title(f"{n:,} darts: {4 * inside.mean():.4f} estimates pi")
    ax.legend(loc="upper right", fontsize=9)
    return fig


def plot_convergence(
    ns: np.ndarray,
    running: np.ndarray,
    truth: float,
    title: str,
    ylabel: str = "estimate",
    band: np.ndarray | None = None,
) -> Figure:
    """Running estimate against the known truth as draws accumulate."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(ns, running, color=GOOD_COLOUR, lw=1, label="running estimate")
    ax.axhline(truth, color=TRUTH_COLOUR, ls="--", lw=1.5, label=f"truth = {truth:.5f}")
    if band is not None:
        ax.fill_between(
            ns, truth - band, truth + band, color=TRUTH_COLOUR, alpha=0.12, label="95% band"
        )
    ax.set_xscale("log")
    ax.set_xlabel("draws")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return fig


def plot_error_rate(ns: np.ndarray, errors: np.ndarray, slope: float, title: str) -> Figure:
    """Log-log error curve with the fitted convergence rate annotated."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.loglog(ns, errors, "o-", color=GOOD_COLOUR, ms=4, label="measured RMSE")
    reference = errors[0] * (ns / ns[0]) ** -0.5
    ax.loglog(ns, reference, "--", color=TRUTH_COLOUR, label=r"$n^{-1/2}$ reference")
    ax.set_xlabel("draws")
    ax.set_ylabel("RMSE")
    ax.set_title(f"{title}\nfitted slope = {slope:.3f}")
    ax.legend()
    return fig


def plot_function_under_curve(
    f: Callable[[np.ndarray], np.ndarray], lo: float, hi: float, title: str
) -> Figure:
    """The region whose area is being estimated."""
    grid = np.linspace(lo, hi, 400)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(grid, f(grid), color=TRUTH_COLOUR, lw=2)
    ax.fill_between(grid, 0, f(grid), color=GOOD_COLOUR, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(title)
    return fig


# --------------------------------------------------------------------------------------
# Notebook 05: importance sampling
# --------------------------------------------------------------------------------------


def plot_tail_problem(
    threshold: float,
    proposals: dict[str, tuple[float, float]],
    lo: float = -5.0,
    hi: float = 9.0,
) -> Figure:
    """Target, proposals and the event region, on a log density scale.

    The log scale is essential: on a linear axis the tail is indistinguishable from zero,
    which is precisely why the problem is hard to think about.
    """
    grid = np.linspace(lo, hi, 600)

    def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.semilogy(
        grid, normal_pdf(grid, 0.0, 1.0), color=TRUTH_COLOUR, lw=2.2, label="target N(0,1)"
    )
    colours = [GOOD_COLOUR, BAD_COLOUR, ACCENT_COLOUR]
    for (label, (mu, sigma)), colour in zip(proposals.items(), colours, strict=False):
        ax.semilogy(grid, normal_pdf(grid, mu, sigma), color=colour, lw=1.8, ls="--", label=label)
    ax.axvline(threshold, color="black", lw=1.2)
    ax.axvspan(threshold, hi, color="black", alpha=0.07)
    ax.annotate("the event", (threshold, 1e-2), xytext=(8, 0), textcoords="offset points")
    ax.set_ylim(1e-12, 1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("density (log scale)")
    ax.set_title("The rare event, and proposals that do or do not cover it")
    ax.legend(fontsize=9)
    return fig


def plot_importance_convergence(
    running_good: np.ndarray, running_bad: np.ndarray, truth: float, brute_force: float
) -> Figure:
    """Relative error of each proposal as draws accumulate, against brute force."""
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for running, label, colour in (
        (running_good, "good proposal", GOOD_COLOUR),
        (running_bad, "bad proposal", BAD_COLOUR),
    ):
        ns = np.arange(1, running.size + 1)
        ax.loglog(ns, np.abs(running - truth) / truth, color=colour, lw=1.2, label=label)
    ax.axhline(
        abs(brute_force - truth) / truth,
        color=TRUTH_COLOUR,
        ls="--",
        lw=1.6,
        label="brute force (estimate is 0)",
    )
    ax.set_xlabel("proposal draws")
    ax.set_ylabel("relative error")
    ax.set_title("Recovering a one-in-ten-million probability")
    ax.legend()
    return fig


def plot_contributions(good: np.ndarray, bad: np.ndarray) -> Figure:
    """Per-draw contributions to each estimate, sorted. Shows what the estimate rests on."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, values, label, colour in (
        (axes[0], good, "good proposal", GOOD_COLOUR),
        (axes[1], bad, "bad proposal", BAD_COLOUR),
    ):
        nonzero = values[values > 0]
        if nonzero.size == 0:
            ax.text(
                0.5,
                0.5,
                "no draw ever reached the event",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=colour,
                fontsize=11,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.semilogy(np.sort(nonzero)[::-1], color=colour, lw=1.6)
            ax.set_xlabel("draw (sorted by contribution)")
            ax.set_ylabel("contribution")
        ax.set_title(f"{label}: {nonzero.size} of {values.size} draws contribute")
    fig.suptitle("What each estimate actually rests on")
    return fig


# --------------------------------------------------------------------------------------
# Notebook 06: summary
# --------------------------------------------------------------------------------------


def plot_summary_table(rows: Sequence[tuple[str, str, str]], title: str) -> Figure:
    """Render the success-criteria table as a figure, so the article can use it directly."""
    fig, ax = plt.subplots(figsize=(11, 0.36 * (len(rows) + 1) + 0.6))
    ax.axis("off")
    table = ax.table(
        cellText=[list(r) for r in rows],
        colLabels=["Claim", "Measured", "Verdict"],
        cellLoc="left",
        colWidths=[0.52, 0.30, 0.15],
        # Fill the axes exactly, rather than anchoring and scaling, which leaves the
        # bottom half of the figure empty.
        bbox=Bbox.from_bounds(0.0, 0.0, 1.0, 1.0),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_facecolor("#eef2f5")
            cell.set_text_props(weight="bold")
    ax.set_title(title, pad=18, loc="left")
    return fig
