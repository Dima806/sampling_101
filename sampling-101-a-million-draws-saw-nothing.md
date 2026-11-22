# Sampling 101: A Million Draws Saw Nothing, Five Thousand Got the Answer

### The probability was one in ten million. Brute force returned exactly zero, with a standard error of zero. Changing which distribution I sampled from recovered it from five thousand draws.

![Consecutive pairs from two generators, both filling the square](../outputs/figures/01_generator_pairs.png)

*Two random number generators. One of them is famously broken. This plot cannot tell you which.*

Every train test split starts with a call to a random number generator. So does every bootstrap resample, every minibatch, every A/B assignment, every simulation you have ever run. We write the line, we trust it, and we move on.

That trust is mostly earned. The gaps are where the expensive failures live, and they are quiet ones. A split that leaks entities across the boundary. A survey that converges, confidently and precisely, to a number that is simply wrong. A risk estimate that comes back as zero because the event never happened in the sample, and nothing in the output suggests anything went wrong.

I built a small library from scratch to look at all three, on synthetic populations where the true answer is known exactly, so every claim below is a measurement rather than an argument. Everything is CPU only and the whole thing runs in under a minute.

Let's dive in!

## A generator that passes every test you would think to run

Here is a linear congruential generator, which is about the simplest thing that produces numbers looking random. Three integers and one line of arithmetic.

```python
class LCG:
    def __init__(self, a, c, m, seed):
        self.a, self.c, self.m = a, c, m
        self._state = seed % m

    def random(self, n):
        out = np.empty(n)
        x = self._state
        for i in range(n):
            x = (self.a * x + self.c) % self.m
            out[i] = x / self.m
        self._state = x
        return out

RANDU  = {"a": 65539, "c": 0, "m": 2**31}
MINSTD = {"a": 48271, "c": 0, "m": 2**31 - 1}
```

Those two parameter sets differ only in the constants. MINSTD is a defensible choice. RANDU is the one IBM shipped in the 1960s, and it was used for a decade of published science before anyone caught it.

So catch it. Draw a hundred thousand numbers from each and run the checks you would actually run.

```python
chi_square_uniformity(bad, bins=100)   # statistic 107.75, p = 0.257
serial_correlation(bad, lag=1)         # 0.00079
```

The uniformity test passes. There is no serial correlation worth mentioning. Plot consecutive pairs, which is the picture at the top of this article, and both generators fill the square with no visible structure.

Three checks, three passes, and the generator is still unusable. Add one dimension. Plot each value against the next two, and look along the right axis:

![RANDU triples collapsing onto fifteen parallel planes](../outputs/figures/01_randu_lattice_3d.png)

*Every RANDU triple lies on one of exactly fifteen planes.*

Not most of them. All of them, 100 percent, on fifteen planes.

RANDU's constants satisfy a hidden relation, `x[n+2] = 6*x[n+1] - 9*x[n] mod 2**31`, which forces the quantity `9*u[i] - 6*u[i+1] + u[i+2]` to be a whole number for every triple. That is what a plane is, algebraically. Once you know the relation you can test for it in one line and get a number instead of a picture.

The practical lesson is boring: use your language's modern default, which in Python means `np.random.default_rng`, and never the legacy global `np.random.seed`. The other lesson is the reason for the rest of this article. The failure stayed invisible until I ran a test built to find it, and getting the picture to show the planes took a camera angle computed from that same relation. Point the camera anywhere else and you get an ordinary random cloud.

Random is a property you verify. It is not one you assume.

## What np.random.normal is actually doing

The generator gives you uniform numbers between 0 and 1. Everything after that is a transformation, and there are only two general tricks.

Inverse transform sampling is one line of mathematics. If `U` is uniform and `F` is a cumulative distribution function, then `F_inverse(U)` has distribution `F`. An exponential you can invert by hand. For anything else you integrate the density on a grid and interpolate, which makes the method work for any density you can write down, including one with no name.

```python
inv_cdf = numeric_inverse_cdf(pdf, lo=-6.0, hi=6.0, grid=8192)
samples = sample_via_inverse_cdf(inv_cdf, 50_000, rng)

stats.kstest(samples, stats.norm.cdf)   # p = 0.73
```

A goodness of fit test cannot separate those draws from real normal ones.

Rejection sampling needs even less. You evaluate the density and never integrate it. Cover the target with an envelope you can sample from, throw points under the envelope, keep the ones landing under the density.

![Accepted and rejected proposals under a uniform envelope](../outputs/figures/02_rejection_sampling.png)

*The orange points are wasted work. The gap between the envelope and the curve is the cost.*

The acceptance rate is `1/m` for envelope constant `m`, and it came out at 0.312 against a theoretical 0.313. Both methods are about ten lines. The library was never doing anything you could not do yourself.

## Unbiased is not the same as precise

Now the part that costs people money.

I built a population of 100,000 units in four strata, the kind of shape you see in customer spend or insurance claims. The top stratum is 2 percent of the units and carries about 19 percent of the total. Because the population is enumerated, the true mean is known exactly: 52.68.

Simple random sampling is unbiased. That is a real guarantee, and on its own it is close to useless, because being right on average says nothing about being right this time. I drew 300 samples of 1,000 units under three designs and looked at the spread.

![Three designs, one population, three hundred repeated samples](../outputs/figures/03_estimate_spread.png)

*All three sit on the truth. Only the width differs.*

| Design | Standard deviation | Variance vs simple random |
|---|---|---|
| Simple random | 2.53 | baseline |
| Stratified, proportional | 0.97 | 6.9x lower |
| Stratified, Neyman | 0.48 | 28.2x lower |

Stratifying means splitting the population into the strata and sampling inside each one, so the lottery over how many rare units you caught disappears. That count is now fixed by design. Same sample size, same cost, 6.9 times less variance. Neyman allocation spends the sample where the variance actually is and buys 28.2 times.

The corollary matters as much as the headline. Stratification removes variance, not bias. It will not repair a sample drawn from the wrong population.

## More data cannot fix a biased frame

Which brings us to the failure that no sample size touches.

Suppose the units with high values are a little more likely to respond to your survey. Not dramatically. In my setup the top stratum goes from 2 percent of the units to 4.2 percent of the response frame, so those units are roughly twice as likely to answer. Nothing about that is remarkable, and it is true of most real surveys.

The sample mean of the responders does not converge to 52.68. It converges to 66.09.

That second number is not an artifact of noise, it is computable in closed form: it is the mean under the sampling distribution you actually used, `sum(p_i * x_i)`. Collecting more responses drives the estimate toward it, and the confidence interval tightens around it.

| Responses | Estimate | Error vs truth | Error vs 66.09 |
|---|---|---|---|
| 500 | 63.13 | +10.45 | -2.97 |
| 5,000 | 66.10 | +13.42 | +0.00 |
| 50,000 | 66.80 | +14.11 | +0.70 |
| 500,000 | 66.01 | +13.33 | -0.08 |

Each row is a single sample, so there is sampling noise in both columns and the last one does not march tidily to zero. The shape is clear enough anyway. The estimate is orbiting 66.09 and has no interest whatsoever in 52.68. Multiplying the response count by a thousand moved the error against the truth from +10.5 to +13.3.

That is what selection bias looks like from the inside, and the sample gives you no signal at all, because there is nothing wrong with the sample. There is something wrong with who could be in it.

## The split that leaks

The same question, random with respect to what, has a version in every machine learning project.

Rows nest inside entities. Customers, patients, sessions, devices. When most of the variation is between entities rather than within them, two rows from the same entity are near duplicates, and splitting rows at random puts the same entity on both sides of the boundary.

![Entity leakage under two split strategies](../outputs/figures/03_group_leakage.png)

*400 entities, 25 rows each, a 25 percent test split.*

A random row split leaked 100 percent of the test entities into training. Every single one. A group aware split leaked none, and it is the same amount of code.

The interesting part is that the same structure cuts the other way too. Sampling whole entities, which is often the cheap option in the field, gave a standard error 3.9 times larger than a simple random sample of the same number of rows. The extra rows inside each entity are close to free duplicates, so the effective sample size is the number of entities, not the number of rows. Grouping helps when you stratify on it and hurts when you sample by it.

## Hard integrals turn into darts and counting

So far sampling has been a way to measure a population. The other use is a way to compute things that have no formula.

A dart thrown at the unit square lands inside the quarter circle with probability pi/4. Count the hits, multiply by four, and you have estimated pi with no geometry and no calculus. Two hundred thousand darts gave 3.1401 with a standard error of 0.0037. The same three lines evaluate an integral with no elementary antiderivative: for `exp(-x*x)` from 0 to 1, Monte Carlo returned 0.746428 against an exact 0.746824.

That generality is why the method is everywhere in pricing, risk and Bayesian inference. It comes at a price worth stating precisely.

![Measured error against sample size on a log log scale](../outputs/figures/04_error_rate.png)

*Two hundred independent replicates at each sample size. The fitted slope is -0.517 against a theoretical -0.5.*

Error falls like one over the square root of n. Ten times the accuracy costs a hundred times the samples, and a hundred times the accuracy costs ten thousand times the samples.

That is a disappointment, and it is also the setup for the next section, because if the event you care about is rare enough then no affordable sample size produces a single observation of it, and slow convergence stops being the problem.

## A probability too small to sample

Take a standard normal and ask for the probability that a draw exceeds 5.199. The exact answer is about 1 in 10 million.

I sampled a million draws and counted.

```
draws            : 1,000,000
events observed  : 0
estimate         : 0.000000e+00
standard error   : 0.000000e+00
```

An estimate of zero with a standard error of zero. This is the most dangerous output in the whole project, because nothing about it is flagged as uncertain. It is confident, precise and wrong, and a pipeline reporting tail risk would pass it straight through. Seeing the event even ten times would take a hundred million draws.

![Target, proposals and the event region on a log density scale](../outputs/figures/05_tail_problem.png)

*On a linear axis the tail beyond 5.199 is indistinguishable from zero, which is exactly why the problem is hard to think about.*

## Sample where the event is common, then correct for it

The idea that fixes it is one most applied practitioners never meet.

You are not obliged to sample from the distribution you care about. For any proposal `q` covering the region of interest:

```
E_p[h(X)] = E_q[h(X) * p(X) / q(X)]
```

Draw from `q`, where the event happens all the time, then weight every draw by the ratio of the two densities to undo the distortion. The estimator stays unbiased. Only the variance changes.

```python
w = np.exp(target_logpdf(x) - proposal_logpdf(x))
estimate = np.mean(indicator(x) * w)
```

Shift the normal onto the threshold, so a one in ten million event becomes roughly a coin flip, and draw five thousand times.

```
estimate       : 1.046388e-07
exact          : 1.001818e-07
relative error : +4.45%
```

Five thousand draws, after a million had failed outright. About half of them land in the event, each carrying a weight of roughly one ten millionth, and the average comes out right.

## The proposal is the method

None of this is free, and importance sampling fails quietly when the proposal is wrong.

A proposal that barely shifts the distribution leaves the event rare under `q` too, so the estimate rests on the one or two draws that happened to reach the tail.

![Two hundred repeated estimates from each proposal](../outputs/figures/05_proposal_spread.png)

*Both proposals are unbiased. Only one of them is usable.*

Over 200 repeated runs the bad proposal was 27 times more variable, and it returned exactly zero in 25 percent of them. A single run of it looks perfectly plausible, which is the trap.

So you check a diagnostic instead of the estimate, and here I got a result I did not expect. The standard diagnostic is effective sample size, which asks how many equally weighted draws your weighted sample is worth. Computed over the weights it reads 3.17 for the good proposal and 4,146 out of 5,000 for the bad one.

It is backwards. Follow it and you would throw away the proposal that works and ship the one returning zero a quarter of the time. The bad proposal is barely shifted, so its weights all sit near 1 and look beautifully uniform, while the good proposal spans orders of magnitude and looks alarming. Weight based effective sample size measures the wrong thing here, because draws outside the event contribute exactly zero no matter how well behaved their weight is.

Score only the draws that enter the estimate, the indicator times the weight, and it reads 733 against 2.0. Right way round, and it catches a failing proposal before the point estimate does.

One more, in the interest of not overselling. Textbooks warn that a proposal with a lighter tail than the target gives the weights infinite variance, which is true asymptotically. Measured at five thousand draws, that proposal had a relative spread of 0.020 against the good proposal's 0.034, meaning it performed better. The divergence lives far beyond where five thousand draws reach, so nothing available at that sample size can see it. The warning is real and it is not the failure that will bite you.

## What I would take away from this

![Every claim in the project against the number that backs it](../outputs/figures/06_summary_table.png)

*Twelve claims confirmed, one refused to reproduce and stayed in the table anyway.*

For sampling and splitting data, the question is what the randomness is with respect to. Stratify on the variable that drives the outcome, since it costs nothing and bought 6.9 times here. Group by entity whenever rows nest inside something. Ask who could not have been sampled at all, because that error survives any amount of data.

For computation, Monte Carlo turns an intractable integral into three lines and does not care about dimension, at a cost of one over the square root of n. For rare events, importance sampling reaches what brute force never will, but the proposal is the method, and the diagnostic has to measure the draws actually carrying your answer.

Then seed everything and write the seed down. Every number here comes from a seed in a config file, which is the cheapest habit on the list and the one most often skipped.

The project is on GitHub at [Dima806/sampling_101](https://github.com/Dima806/sampling_101): six notebooks, a tested library you can import, and a Streamlit app where you can drag the proposal around and watch importance sampling stop working.

If this was useful, the same build it from scratch treatment is in [Bootstrap 101](https://medium.com/data-and-beyond/bootstrap-101-one-for-loop-any-confidence-interval-45e78a0d4316) and [Curse of Dimensionality 101](https://medium.com/data-and-beyond/curse-of-dimensionality-101-your-nearest-neighbor-is-not-actually-near-5dc030385952).
