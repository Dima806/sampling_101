# Project: sampling_101

## Identity

Educational "101" portfolio project. Sampling and Monte Carlo, built from scratch on CPU.
6 notebooks + `src/` library + Streamlit app. Spec: [.llm/PRD_sampling_101.md](.llm/PRD_sampling_101.md).

Arc: what random means → sampling any distribution → biased sampling → Monte Carlo →
importance sampling → practical guide.

Thesis, three claims, each demonstrated with a measured number, not asserted in prose:

1. Convenient sampling hides bias. A skewed population sampled at random underrepresents the
   tail; stratifying fixes it *and* lowers variance.
2. Hard integrals are darts-and-count. Monte Carlo error shrinks like `1/sqrt(n)` — provable
   from a log-log slope, not a hand-wave.
3. Importance sampling recovers a one-in-ten-million probability from a few thousand draws,
   where brute force sees the event zero times in a million.

Audience: working data scientists. Every module is a from-scratch reveal of what the library
one-liner was doing. Correctness is measured against known ground truth everywhere, because
the populations are synthetic and the true values are exact.

## Current state

Complete and green. `make ci` passes (69 tests, ~8 s); `make notebooks` executes all six from
scratch in ~46 s and writes 23 PNGs to `outputs/figures/` and 6 JSON files to `outputs/results/`.
Library, tests, notebooks, Streamlit app, README all done. Not built, deliberately:
`.devcontainer/` and `.github/workflows/ci.yml`.

**Measured baselines** — if a change moves one of these materially, that is a regression, not a
new result:

| Quantity | Value |
|---|---|
| RANDU plane fraction / count | 1.0 / 15 |
| RANDU chi-square p (marginals pass) | 0.257 |
| Rejection acceptance vs theory | 0.312 vs 0.313 |
| Stratified variance reduction | 6.9× (Neyman 28.2×) |
| Population truth / biased frame limit | 52.68 / 66.09 |
| Row-split leakage vs group-aware | 100% / 0% |
| Monte Carlo log-log slope | −0.517 |
| Rare-event truth | 1.0018e-07 |
| Brute force, 1e6 draws | 0.0, se 0.0 |
| Importance sampling, 5e3 draws | +4.4%, ess_contributions 733 |
| Bad proposal | 27× spread, ess_contributions 2.0, 25% of runs return 0 |

## Stack and constraints

- Python 3.11+. `uv` only — never `pip install`, never `python -m venv`.
- Runtime deps: `numpy`, `scipy`, `pandas`, `matplotlib`, `plotly`, `streamlit`, `pydantic`,
  `pydantic-settings`, `pyyaml`.
- Dev deps: `pytest`, `ruff`, `ty`, `jupyter`, `ipykernel`, `nbconvert`, `nbclient`.
- Target box: 2-CPU / 8 GB GitHub Codespace. No GPU, no CUDA, no torch.
- Keep arrays to a few million floats. If a cell needs more than ~1 GB or 60 s, shrink `n`
  and say so in the notebook text.
- `scipy` is for *validation and reference only* (KS tests, chi-square, `scipy.stats.norm.sf`
  for exact truth). Never use it to implement a sampler the project claims to build from
  scratch. Same rule for `np.random.Generator.normal/choice/...` — fine as a baseline to
  compare against, never as the implementation.

## Layout

```
config/settings.yaml       sample sizes, distributions, n_simulations, seeds
src/config.py              pydantic-settings loader over settings.yaml
src/generators/prng.py     LCG (good params + deliberately bad params), uniform stream
src/generators/quality.py  chi-square uniformity, serial correlation, lattice/spectral diagnostics
src/draw/inverse_transform.py  inverse-CDF sampling, analytic + numeric-grid
src/draw/rejection.py      rejection sampling, envelope + acceptance rate
src/draw/discrete.py       categorical sampling, cumulative search + Vose alias method
src/survey/simple.py       simple random sampling, mean estimator + variance
src/survey/stratified.py   stratified sampling, proportional + Neyman allocation
src/survey/grouped.py      cluster/grouped sampling, entity-level draws
src/survey/bias.py         selection-bias demos (convenient frame → wrong answer)
src/montecarlo/integration.py  pi by darts, MC integration, expectations
src/montecarlo/importance.py   importance sampling, weights, ESS
src/montecarlo/variance.py     variance-reduction and convergence diagnostics
src/datasets/populations.py    skewed / grouped / rare-event synthetic populations
src/evaluation/convergence.py  running estimate vs truth as n grows, error-rate fit
src/visualisation.py       shared matplotlib helpers, saves to outputs/figures/
src/reporting.py           save_results/load_results, numpy-aware JSON to outputs/results/
app/streamlit_app.py       4-tab app
notebooks/01..06           the narrative
tests/                     one file per success criterion
```

Every `src/` subpackage has an `__init__.py` re-exporting its public names. Notebooks and the
app import from the package (`from src.montecarlo.importance import ...`), never by path hacking
beyond a single `sys.path` insert in the notebook's first cell.

## Randomness discipline

This is a project about sampling; sloppy RNG handling here is self-refuting.

- **Never** call `np.random.seed()`, `np.random.rand()`, or any global-state numpy legacy API.
- Every function that consumes randomness takes an explicit `rng: np.random.Generator` argument.
  No default `rng=None` that silently constructs a fresh generator — make the caller pass it, or
  make `seed: int` the parameter and construct `np.random.default_rng(seed)` inside.
- Seeds live in `config/settings.yaml` and are read through `src/config.py`. Notebooks print the
  seed they used. A figure that cannot be regenerated from a recorded seed is a bug.
- The from-scratch `LCG` is the object of study, not the project's RNG. Use `default_rng` (PCG64)
  for everything else — Monte Carlo draws, survey designs, test fixtures.
- Tests are seeded and deterministic. No `pytest.mark.flaky`, no "rerun if it fails". If a
  statistical assertion is genuinely marginal, widen the tolerance with a comment explaining the
  error bound, or raise `n`.

## Module contracts

Signatures below are the intended shape. Match them unless there's a concrete reason not to;
if you change one, update this file in the same commit.

### `generators/prng.py`

```python
class LCG:
    """x_{n+1} = (a * x_n + c) mod m. Yields floats in [0, 1)."""
    def __init__(self, a: int, c: int, m: int, seed: int) -> None: ...
    def random(self, n: int) -> np.ndarray: ...

RANDU = dict(a=65539, c=0, m=2**31)      # the deliberately bad one
MINSTD = dict(a=48271, c=0, m=2**31 - 1)  # the acceptable one
```

RANDU is the right bad generator: its triples fall on 15 hyperplanes in 3-D, which is the
visible-stripes picture the whole project opens with. Do not substitute a hand-rolled "bad"
generator with fuzzy structure — the demo has to be unmistakable and the diagnostic quantitative.

**The stripes are only visible from the right camera angle.** At a generic 3-D viewing angle
RANDU's triples look like an ordinary random cloud, and the figure proves nothing. Use
`visualisation.edge_on_view()`, which solves for the azimuth whose line of sight is perpendicular
to the `(9, -6, 1)` plane normal (elev 5°, azim ~56.8°). Never hand-tune this angle by eye.

### `generators/quality.py`

```python
RANDU_PLANE_DIRECTION = (9.0, -6.0, 1.0)   # RANDU: x_{n+2} = 6x_{n+1} - 9x_n (mod 2**31)

def chi_square_uniformity(u: np.ndarray, bins: int = 100) -> tuple[float, float]  # stat, p
def serial_correlation(u: np.ndarray, lag: int = 1) -> float
def overlapping_tuples(u: np.ndarray, dim: int = 3) -> np.ndarray
def lattice_triples(u: np.ndarray) -> np.ndarray    # (n-2, 3) for the 3-D plane plot
def plane_projection(u, direction=RANDU_PLANE_DIRECTION) -> np.ndarray
def plane_fraction(u, direction=..., tol=1e-9) -> float   # the headline diagnostic
def plane_count(u, direction=..., tol=1e-9) -> int        # only meaningful alongside it
```

`plane_fraction` is the quantitative test, not `plane_count`. A count of 15 means something only
when every triple is on a plane; on a sound generator the count is over noise and says nothing.

Both generators must pass the *marginal* uniformity test — that is the point. RANDU's marginals
look fine; only the pairwise/triple structure exposes it. Say this explicitly in notebook 01.

### `draw/inverse_transform.py`

```python
def sample_via_inverse_cdf(inv_cdf: Callable[[np.ndarray], np.ndarray], n: int,
                           rng: np.random.Generator) -> np.ndarray
def exponential_inv_cdf(rate: float) -> Callable   # -log(1-u)/rate, the closed-form showcase
def numeric_inverse_cdf(pdf: Callable, lo: float, hi: float, grid: int = 4096) -> Callable
```

The numeric path (cumulative-trapezoid the pdf on a grid, then interpolate) is what makes the
method general. Validate every sampler against its target with a KS test.

### `draw/rejection.py`

```python
def rejection_sample(target_pdf: Callable, proposal_sampler: Callable, proposal_pdf: Callable,
                     m: float, n: int, rng: np.random.Generator) -> RejectionResult
```

Return the accepted draws *and* `n_proposed`, so the empirical acceptance rate can be compared
against the theoretical `1/m`. `m` must dominate: assert `m * q(x) >= p(x)` on a grid and raise
if it doesn't — a silently invalid envelope produces a wrong distribution that looks plausible.

### `draw/discrete.py`

```python
class AliasTable:
    def __init__(self, probs: np.ndarray) -> None: ...  # Vose construction, O(k)
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray  # O(1) per draw
def cumulative_search_sample(probs, n, rng) -> np.ndarray  # O(log k) baseline for comparison
```

### `survey/*`

Estimators return an estimate *and* its standard error; a point estimate with no uncertainty is
not an answer in this project.

```python
@dataclass(frozen=True)
class Estimate:            # value, se, n  — plus .covers(truth, z)
    ...

def simple_random_indices(size, n, rng) -> np.ndarray   # partial Fisher-Yates, O(n)
def simple_random_mean(population, n, rng) -> Estimate
def stratified_mean(population, strata, n, rng, allocation="proportional") -> Estimate
def proportional_allocation(counts, n) -> np.ndarray
def neyman_allocation(counts, sds, n) -> np.ndarray      # oracle sds; note that in docstrings
def cluster_mean(population, groups, n_groups, rng) -> Estimate
def naive_row_split(n_rows, test_fraction, rng) -> tuple[np.ndarray, np.ndarray]
def group_train_test_split(groups, test_fraction, rng) -> tuple[np.ndarray, np.ndarray]
def group_leakage_rate(groups, train_mask, test_mask) -> float
def frame_limit_mean(population, propensity) -> float     # sum_i p_i x_i
def response_biased_mean(population, propensity, n, rng) -> BiasResult
def convenience_frame_mean(population, in_frame, n, rng) -> BiasResult
```

Allocations always give every stratum at least `MIN_PER_STRATUM = 2` units (a stratum with one
draw has no estimable variance) and sum exactly to `n`, via largest-remainder rounding plus a
repair pass.

The stratified-beats-simple claim is about **variance across repeated draws**, not one sample.
Always demonstrate it by running both designs `n_simulations` times and comparing the spread.
A single lucky draw proves nothing and would be the exact reflex error the project is about.

`bias.py` holds the selection-bias demos. Each reports the true value, the biased estimate, and
the gap. Two rules learned the hard way here:

- **Use `frame_limit_mean`, not two noisy samples, to show that more data does not help.** The
  biased estimator converges to `sum_i p_i x_i`, which is computable. Comparing an n=500 estimate
  against an n=50,000 one just compares two noisy numbers and can even look like the bias *grew*.
- **Keep the propensity mild.** `survey.response_strength = 0.2` shifts the mean about 18–26%
  (~5 SE) and roughly doubles the top stratum's share of the response frame (2% → 4.2%), which is
  a believable survey. At 1.2 the bias is +635% and three-quarters of the sample comes from the
  top stratum — a caricature a reader will rightly dismiss. Do not raise it to make a test pass.

### `montecarlo/integration.py`

```python
@dataclass(frozen=True)
class MCResult:            # value, se, running  — plus .n
    ...

def mc_result(values: np.ndarray) -> MCResult              # package per-draw contributions
def estimate_pi(n, rng) -> MCResult                        # scores each dart as 4 * hit
def integrate(f, lo, hi, n, rng) -> MCResult               # (hi-lo) * mean(f(U))
def expectation(f, sampler, n, rng) -> MCResult
def pi_estimates_at(checkpoints, n_replicates, rng) -> np.ndarray  # (reps, checkpoints)
```

`MCResult` carries the running estimate array so convergence plots are free and no notebook
re-runs the loop just to draw a curve. Every estimator funnels through `mc_result`, so scoring a
dart as `4 * hit` rather than counting hits separately means the standard error comes from one
formula everywhere.

### `montecarlo/importance.py`

```python
@dataclass(frozen=True)
class ISResult:   # value, se, weights, contributions, ess
    # plus .n, .ess_fraction, .ess_contributions, .max_weight_share
    ...

def normal_logpdf(x, mu=0.0, sigma=1.0) -> np.ndarray      # from scratch
def normal_proposal(mu, sigma) -> tuple[Sampler, LogDensity]
def threshold_indicator(threshold) -> Indicator
def effective_sample_size(weights) -> float                # (sum w)^2 / sum(w^2)
def brute_force_probability(indicator, sampler, n, rng) -> MCResult   # the failing baseline
def importance_estimate(indicator, target_logpdf, proposal_sampler, proposal_logpdf,
                        n, rng) -> ISResult
```

Work in **log space**. Weights are `exp(target_logpdf(x) - proposal_logpdf(x))`. Be accurate about
why: at the project's own threshold of 5.199 the naive `p(x)/q(x)` is fine (`p(5.2) ≈ 5.5e-7`), so
don't claim it breaks there. It breaks as the event gets rarer — by `x = 40`, `p(x)` is below the
smallest double, the naive weight silently becomes 0, and the estimate goes with it. Log space
costs nothing and removes the cliff. Note that `exp()` of the log weight underflows at that
extremity too; what survives is the finite log value.

Canonical example: `X ~ N(0,1)`, estimate `P(X > 5.199) ≈ 1e-7`. Ground truth from
`scipy.stats.norm.sf` — this is the sanctioned scipy use. Good proposal: `N(5.199, 1)` (mean-shift
/ exponential tilting), which makes the event a coin flip.

Bad proposal, shown honestly: **`N(0, 1.5)`, one that leaves the event rare**. Measured over 200
replicates at n=5000: relative spread 0.034 for the good proposal against 0.91 for this one, which
sees ~1 event per 5000 draws. It inherits brute force's problem — the estimate rests on the one
draw that happened to land in the tail.

The tempting alternative caveat — a proposal with a *lighter tail than the target*, e.g.
`N(5.199, 0.25)` — is theoretically infinite-variance but **does not fail at feasible n**: measured
relative spread 0.017, better than the good proposal, because the divergence lives beyond where
5000 draws ever reach. Do not build the caveat on it. It is worth a footnote in notebook 05 as a
case where the theory is real and the diagnostics cannot see it, but the demonstrated failure has
to be the coverage failure above.

Report `ess_contributions` (ESS of `indicator * weights`, not of the weights alone) and
max-weight-share.

**Plain ESS over the weight vector is actively misleading here — it reads backwards.** Measured:
good proposal 3.17, bad proposal 4146.25 out of 5000. The bad proposal is unshifted, so its
weights are all close to 1 and look beautifully uniform; the good proposal is shifted onto the
tail, so its weights vary over orders of magnitude and look catastrophic. Anyone reading weight-ESS
would conclude the broken proposal is healthy and the working one is failing.

The fix is to score only the draws that actually enter the estimate. Draws outside the event
contribute exactly zero however large their weight, so `ess_contributions` reads **733 vs 2.0** —
the right way round. Never use bare weight-ESS as the diagnostic for an indicator-style estimand.

Prefer the unbiased estimator `mean(w * indicator)` when the target density is normalized. Note
the self-normalized variant `sum(w*f)/sum(w)` and why it trades bias for usability, but don't use
it where the unbiased form applies.

### `evaluation/convergence.py`

```python
def running_estimate(draws: np.ndarray) -> np.ndarray
def error_curve(running: np.ndarray, truth: float) -> np.ndarray
def logspace_checkpoints(n_max, n_points=25, n_min=100) -> np.ndarray
def rmse_by_n(estimates: np.ndarray, truth: float) -> np.ndarray   # (reps, checkpoints) -> (k,)
def fit_convergence_rate(ns, errors) -> float   # log-log slope, expect ~ -0.5
```

The `-0.5` slope is a success criterion, not decoration. Fit it and assert it.

Fit it **across replicates**, never along a single run. One path's `|error|` is so noisy that the
fitted slope wanders far from -0.5 and means nothing. The working recipe is
`pi_estimates_at(logspace_checkpoints(...), n_replicates=200, rng)` → `rmse_by_n` →
`fit_convergence_rate`, which lands at -0.517. Log-spaced checkpoints matter too: linear ones
crowd every point into the flat right-hand end and the fit is ill-conditioned.

### `reporting.py`

```python
def save_results(name: str, payload: dict) -> Path    # outputs/results/<name>.json
def load_results(name: str) -> dict
def sanitise(value: Any) -> Any                       # non-finite floats -> None, recursively
class NumpyJSONEncoder(json.JSONEncoder): ...         # numpy scalars, ndarray, Path
```

**The NaN handling cannot live in the encoder.** `json` only calls `default()` for types it does
not already know, and `np.float64` subclasses `float` — so a NaN goes down the float fast path and
is written as a bare `NaN` literal, which is not legal JSON. The payload must be walked *before*
encoding, which is what `sanitise` does; `save_results` then dumps with `allow_nan=False` so
anything missed raises instead of writing an unparseable file. This is not hypothetical:
`ISResult.max_weight_share` is genuinely NaN whenever a proposal never reaches the event.

### `visualisation.py`

```python
def save_figure(fig: Figure, name: str, dpi: int = 150) -> Path   # outputs/figures/<name>.png
def edge_on_view(direction=RANDU_PLANE_DIRECTION, elev=5.0) -> tuple[float, float]
```

Plus one `plot_*` function per figure, grouped by the notebook that uses it. Notebooks call these;
they never build a figure inline.

## Notebooks

One notebook per PRD section 4.4, filenames fixed:
`01_what_random_really_means`, `02_sampling_from_any_distribution`, `03_when_your_sample_is_biased`,
`04_monte_carlo`, `05_importance_sampling`, `06_practical_guide`.

Rules:

- < 4 minutes each on 2 CPUs, enforced by `make notebooks` (240 s nbconvert timeout).
- Logic lives in `src/`. Notebooks import, call, plot, and narrate. If a cell grows past ~15 lines
  of computation, move it into a module and import it back.
- Every notebook: markdown intro stating the question, the seed printed, and a closing
  "what this bought you" cell. Prose is for a reader who knows pandas and has never heard of
  Neyman allocation.
- **Every visual is saved as a PNG and every number and finding is saved as JSON.** Figures go to
  `outputs/figures/<name>.png` via `save_figure`; results go to `outputs/results/<notebook>.json`
  via `save_results`. Filenames are stable — the article pulls from these directories and renaming
  churns its references. Notebook 06 reads the other five files back with `load_results`, so it
  quotes measured numbers rather than remembered ones.
- Each results file carries at least: `notebook`, `seed`, the measured quantities, a `findings`
  list of plain-language sentences, and a `figures` list naming every PNG the notebook wrote. The
  `figures` list must exactly match what landed on disk.
- `save_results` writes non-finite floats as `null` (NaN and Infinity are not legal JSON, and
  `max_weight_share` is genuinely NaN when a proposal never reaches the event).
- Both output directories are committed, not gitignored: they are deliverables.
- Notebooks 03 (bias) and 04 (Monte Carlo) are the showpieces; 05 (importance sampling) is the
  payoff. Spend the effort there.
- Commit notebooks **with** outputs — readers browse them on GitHub.
- The committed `.ipynb` files are the source of truth. Edit them in JupyterLab (`make lab`) and
  re-run `make notebooks`; there is no parallel `.py` copy to keep in sync, deliberately.
- **Look at the figure you just generated before believing it.** The RANDU 3-D plot rendered as an
  ordinary random cloud at the obvious viewing angle and proved nothing; it took `edge_on_view` to
  make the project's central claim visible. A figure that runs without error is not a figure that
  shows anything.

## Streamlit app

Four tabs, per PRD 4.5: randomness inspector (good vs bad generator, live), distribution sampler
(inverse-transform vs rejection building up), pi by darts (estimate and error converging), and the
rare-event estimator (brute force fails, importance sampling recovers). Cache with
`@st.cache_data`. Cap interactive `n` so a slider drag can't wedge a 2-CPU box.

## Testing

`tests/` mirrors the success criteria table in PRD section 8 — one file per claim:

| File | Asserts |
|---|---|
| `test_config.py` | settings load and validate; same seed gives the same stream |
| `test_reporting.py` | results JSON round-trips; numpy scalars encode; non-finite floats become `null` so strict parsers accept the file |
| `test_prng.py` | good LCG passes chi-square + serial correlation; RANDU passes those same marginal tests and fails the structure test with all triples on exactly 15 planes |
| `test_inverse_transform.py` | KS test between sampled and target distribution does not reject, for both the closed-form and numeric-grid paths |
| `test_rejection.py` | KS test passes; empirical acceptance rate matches `1/m`; a non-dominating envelope raises |
| `test_discrete.py` | alias table and cumulative search both reproduce the requested probabilities within 4 SE |
| `test_stratified.py` | stratified variance < simple random variance over repeated draws; Neyman beats proportional; reported SE matches the empirical spread |
| `test_bias.py` | response bias exceeds 4x the sampling noise; convenience frame off by >5%; more data converges to `frame_limit_mean` and not to the truth; random row split leaks >95% of entities, group split leaks 0; cluster SE > simple-random SE at equal rows |
| `test_monte_carlo.py` | pi within the `1/sqrt(n)` band; fitted log-log slope ≈ -0.5 over 200 replicates |
| `test_importance.py` | brute force returns exactly 0.0 from 1e6 draws; good proposal recovers `P ≈ 1e-7` from 5000; bad proposal's spread >5x worse and its `ess_contributions` flags it |

Statistical assertions use explicit tolerances derived from the standard error (e.g. "within 4 SE"),
with the derivation in a comment. Never tune a threshold until a flaky test passes — fix the seed,
raise `n`, or state the bound honestly.

`make test` and `make lint` must be green at every commit. Zero errors.

## Commands

```
make setup       first-time: uv sync --all-extras + register the kernel + mkdir outputs/figures
make lint        ruff format + ruff check --fix + ty check
make test        pytest
make notebooks   execute all six in place (--inplace), 240 s timeout each, ~46 s total
make run         streamlit app on :8501
make lab         jupyterlab on :8888
make ci          sync + lint + test
make dev         lint + test (fast loop)
```

`make notebooks` runs with `--inplace`, so outputs land in the committed notebooks and the PNG/JSON
files are rewritten. Without it nbconvert would write `*.nbconvert.ipynb` and leave the real
notebooks holding stale outputs.

Ruff: `target-version = py311`, `line-length = 99`, select
`["E","F","W","I","UP","N","B","A","SIM","PTH"]`, ignore `E501`. Type-check with `ty`, not mypy.
Type-annotate all public functions. Use `pathlib`, not `os.path` (the `PTH` rules enforce it).
`N803`/`N806` are ignored under `src/` and `tests/` so the maths can use the names the literature
uses (`M` for the envelope constant, and so on).

### Packaging

`pyproject.toml` sets `tool.uv.package = false`: this is a **virtual project**, so `uv sync`
installs the dependencies without building the repo as a distribution. `src` is imported from the
repo root instead, which `[tool.pytest.ini_options] pythonpath = ["."]` arranges for tests and the
first-cell `sys.path` insert arranges for notebooks. Do not add a `[build-system]` — publishing a
top-level `src` package into the environment is a namespace collision waiting to happen.

`pytest-cov` is in the dev extras because `make test-cov` needs it; the PRD's dependency list omits
it. Note also that `uv` resolves to the Codespace's CPython 3.14 while ruff targets `py311` and
`ty` is pinned to 3.11 — legal under `requires-python = ">=3.11"`, but a `.python-version` file
would pin it if the divergence ever matters.

## Conventions

- NumPy-style docstrings on every public function, with the math stated in one line
  (`Estimate E[f(X)] by the sample mean of f over n draws.`).
- Small dataclasses for multi-field returns (`Estimate`, `MCResult`, `ISResult`) — not bare tuples,
  not dicts.
- No `print` in `src/`; return values and let the notebook display them.
- Vectorize with numpy. A visible Python loop is acceptable only when the loop *is* the teaching
  point (the darts-accumulating narrative), and then say so in a comment.
- Keep the from-scratch implementations readable over clever. This is a teaching repo; a
  five-line transparent version beats a two-line dense one.

### Reporting honestly

This project's whole subject is not fooling yourself, so the repo holds itself to it.

- **Measure before you claim.** Three specifications in this file were refuted by running them:
  the light-tailed "bad proposal", plain weight-ESS as a diagnostic, and the log-space
  underflow rationale. Each was corrected here rather than quietly worked around.
- **Keep refuted claims visible.** Notebook 06's scoreboard marks the light-tail row
  `NOT confirmed` and explains why. Do not delete a claim that failed; record it.
- **Never tune a demonstration until it looks impressive.** A parameter that produces a
  spectacular number nobody would believe (`response_strength = 1.2`) is worse than a modest one
  that is true.
- Prefer an exact quantity to two noisy ones: `frame_limit_mean` over "compare n=500 to n=50,000",
  `edge_on_view` over an eyeballed camera angle, an integrated share over a claimed mixture weight.

## Non-goals

No GPU, no torch, no MCMC (that's `bayesian_101`), no bootstrap (that's `bootstrap_101`), no
quasi-Monte Carlo, no reservoir sampling. PRD section 11 lists these as future extensions —
cross-link them, don't build them. Don't add dependencies beyond the PRD list without asking.

## Token rules

- Caveman mode: ON (`.claude/skills/caveman/SKILL.md`). Terse, code-first, no tool narration.
  Normal prose still applies to persisted text: code comments, docstrings, notebook markdown,
  commit messages, README.
- `/compact` at notebook boundaries.
