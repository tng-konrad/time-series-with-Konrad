# Time Series with Konrad: episode 7

### Conformal prediction: honest intervals around any forecaster

**KONRAD BANACHEWICZ**
JUL 06, 2026

A point forecast is a confident little lie. It hands you a single number — tomorrow's demand is 7, the temperature will be 14.2°C — and says nothing about how much to trust it. For any continuous quantity the probability of landing exactly on that number is zero, so the honest output was never a point in the first place. What you actually want is an interval: a range that contains the truth with, say, 90% probability, and that you can defend to whoever has to act on it.

There are classical ways to get there, and most of them ask you to sign a contract you can't honor. Frequentist confidence intervals assume the residuals are Gaussian with constant variance; the moment your data turn heavy-tailed, skewed, or heteroscedastic - which is to say, the moment they become real - those intervals are either dangerously narrow or uselessly wide. The Bayesian route is principled but wants priors and a likelihood you can rarely write down for a modern black-box model, and the integrals are usually intractable anyway.

**Conformal prediction (CP)** sidesteps the whole bargain. It is a lightweight wrapper that goes around *any* point forecaster - a linear AR, a gradient-boosted tree, a neural net, doesn't matter - and turns it into an interval generator with a finite-sample coverage guarantee and *no distributional assumptions whatsoever*. No Gaussian errors, no parametric noise model. If your underlying model is good, the intervals come out tight; if it's bad, they come out wide, but they still cover. You are never misled by an overconfident model - that's the whole pitch.

There is one catch, and this episode is largely about it. The guarantee rests on **exchangeability**: the assumption that the order of your data carries no information. Time series violate that assumption at their core - yesterday tells you about today (the autocorrelation we met back in episode 1), and the world drifts and breaks underneath you. So we'll do what we always do: start from the clean baseline, watch it fail, and climb a ladder of fixes, each rung adding exactly one new idea.

📓 All code lives in the companion notebook:
https://github.com/tng-konrad/time-series-with-Konrad/blob/main/m07-conformal-prediction.ipynb

*(The notebook reconstructs the methods and experiments from "A Gentle Introduction to Conformal Time Series Forecasting" - Stocker, Małgorzewicz, Fontana & Ben Taieb, 2025 - implementing each method from scratch so the moving parts stay visible.)*

## Groundwork

Conformal prediction is a wrapper, so we need two things underneath it: some data and a forecaster to wrap. The mechanism, in its plainest form, is almost insultingly simple.

Pick a **nonconformity score** - a number that says how "strange" a data point is to the fitted model. For regression the natural choice is the absolute residual: for a point with true value `y` and prediction `ŷ`, the score is `s = |y − ŷ|`. Now split your history into a block used to *fit* the model and a disjoint block used to *calibrate*. Compute the scores on the calibration block, and take their empirical `(1 − α)` quantile, call it `q`. The interval for any future point is just `[ŷ − q, ŷ + q]`. That's it.

The one subtlety worth slowing down for is the quantile itself. You don't take the plain `(1 − α)`-th fraction of `n` calibration scores; you take the score at rank `⌈(1 − α)(n + 1)⌉`. That `(n + 1)` reserves a slot for the unseen test point, treating the calibration scores *and* the future score as one pool of `n + 1` exchangeable values. This tiny correction is exactly what upgrades an asymptotic hand-wave into a **finite-sample guarantee**: with this rank, the interval covers the truth with probability at least `1 − α`, full stop - *provided the data are exchangeable*. (When `α` is so demanding, or `n` so small, that the rank exceeds `n`, the honest answer is an infinite interval. The procedure returns exactly that, rather than quietly over-claiming.)

Exchangeability is the load-bearing wall. The usual analogy is drawing tiles blindly from a bag: the order you pull them in tells you nothing, because every ordering was equally likely. Standard CP needs the calibration data and the future test data to come from one such bag. Time series are the opposite of a bag - the order is the entire point. So before we build anything, let's set up data designed to show us precisely *how* and *when* this breaks.

We use four simulated processes, because with simulation we know the ground truth and can see exactly why a method fails. Two are well-behaved and two are stress tests:

- **AR(1)**, `y_t = 0.8·y_{t−1} + e_t` - simple autocorrelation, stationary. Each value is a damped echo of the last plus fresh noise.
- **ARMA(1,1)**, `y_t = 0.5·y_{t−1} + e_t + 0.4·e_{t−1}` - slightly longer memory, still stationary. Both of these forget their past quickly, so the temporal dependence is mild.
- **Mean-Shift** - pure noise around a level that jumps permanently two-thirds of the way through, engineered so the break lands exactly on the first test point. This is *distribution shift*: calibration and test come from genuinely different processes, and no reweighting of old data can know about it.
- **GARCH(1,1)** - constant mean, but the *variance* clusters: a big move begets a big move, so calm and turbulent regimes alternate. The level is stable; the spread is not. A single fixed interval width cannot be right in both regimes.

> **[FIGURE 1 — notebook cell 20]** 2×2 grid of one realisation each of AR(1), ARMA(1,1), Mean-Shift, GARCH(1,1), with a dashed red line marking where the test block begins (point 600). AR(1)/ARMA(1,1) look like stationary noise around zero - the statistics on either side of the line match. Mean-Shift steps up to a new level *exactly at the red line*. GARCH stays centred but visibly breathes, quiet stretches giving way to bursts.

The forecaster is deliberately the simplest thing that could work: an autoregression fit by least squares (AR-LS), predicting each value from its two previous ones. Crucially, we fit it **once** on the training block and then freeze it. The star of this episode is the conformal layer, not the model, so freezing the forecaster forces the calibration machinery to do all the adapting. In particular, the frozen model will *not* chase the mean shift on its own - its residuals simply go biased - so any method that holds coverage there is genuinely earning it.

Here is exchangeability made visible. Fit AR-LS on the training block, then compute the nonconformity scores - the absolute residuals - on the calibration and test blocks separately, and overlay their histograms.

> **[FIGURE 2 — notebook cell 26]** Overlaid histograms of calibration vs test |residuals| for AR(1). The two distributions sit almost on top of each other.

For a stationary process the two distributions land almost on top of each other: the errors the model makes on calibration data are statistically the same as the errors it makes on test data. *That overlap is (approximate) exchangeability*, and it's the reason the calibration quantile will transfer to the test set. Hold this picture in mind - when we reach the mean-shift process, these two histograms will pull apart, the calibration scores will sit to the left of the test scores, and the fixed quantile computed on the left will be far too small for the right. Everything that follows is a response to that gap.

## The conformal ladder

We now climb five methods. To keep them comparable, each one wraps the *same* frozen AR-LS forecaster and runs on the *same* AR(1) realisation, using an identical fit → calibrate → predict → evaluate sequence. Only the calibration logic changes from rung to rung.

A word on the scorecard first. Validity and efficiency pull against each other - you can always cover everything with an infinitely wide interval - so we track three numbers at once. **Coverage** is the fraction of test points that actually land inside their interval; a valid 90% method sits near 0.90. **Width** is the mean interval length; among valid methods, smaller is sharper. The tie-breaker is the **Winkler score** (the interval score), which charges you the width of every interval *plus* a steep penalty of `(2/α)·distance` whenever the truth escapes. It rewards intervals that are tight *and* honest, and punishes the cheap trick of buying coverage with width. Lower Winkler is better.

### SCP — Split Conformal Prediction

The baseline, and the whole recipe in four lines: compute the absolute residuals on the calibration block, collapse them to a single threshold `q` with the conformal quantile, form the point forecast on the test block, and invert into `[ŷ − q, ŷ + q]`. Because the absolute-residual score is symmetric, the band is symmetric: the same `q` added and subtracted at every step.

> **[FIGURE 3 — notebook cell 34]** SCP intervals on the AR(1) test block: a constant-width green band around the point forecast, with missed points marked in red. Coverage printed in the title (near 0.90).

The defining feature - and the eventual limitation - is that `q` is computed once and never moves. On stationary AR(1) that's exactly right: coverage lands near the target, the misses are sprinkled evenly across time at roughly the 10% rate we asked for, no clustering, no drift. That even scattering is the visual signature of a *valid* interval on exchangeable-enough data. The width and Winkler here become our reference point - every fancier rung below has to justify its extra complexity against this. For a stable process, SCP is genuinely hard to beat.

### WCP — Weighted Conformal Prediction

**New ingredient: unequal calibration weights.** SCP treats every calibration point as equally informative. But if the process drifts, a residual from 300 steps ago is less relevant than one from yesterday. WCP keeps the same calibration set but assigns each point a weight and takes a *weighted* quantile - letting us say "the recent past matters most." We try three weighting schemes: exponential decay (a point `t` steps back gets weight `ρ^t`, with `ρ = 0.99`), a gentler linear ramp, and a blunt sliding window that keeps only the last 50 residuals and discards the rest.

> **[FIGURE 4 — notebook cell 39]** Three side-by-side interval plots, WCP with exponential, linear, and window weighting on AR(1), each with its coverage in the title.

On stationary AR(1) all three hold coverage near 0.90, with widths a touch larger than SCP's - the price of leaning on fewer effective calibration points. The window scheme is the noisiest, because its effective sample size is smallest. On stationary data WCP buys little; its value only shows up when the process drifts.

And here's the limitation to file away: reweighting can only redistribute emphasis among residuals it *already has*. If a shift introduces error magnitudes that simply aren't present anywhere in the calibration block, no choice of weights can conjure them. Hold that thought for the mean-shift experiment.

### EnbPI — Ensemble Batch Prediction Intervals

**New ingredient: a residual pool that refreshes itself.** WCP reweighted a fixed calibration set. EnbPI actively *replaces* old residuals with new ones, and dispenses with the train/calibration split entirely. It has four moving parts.

First, instead of one model it trains an ensemble of 25 AR-LS models, each on a bootstrap resample. Second, it forms **out-of-bag (OOB)** residuals: for each point, it predicts using only the ensemble members that did *not* see that point in training - a genuine out-of-sample error without ever carving off a separate calibration set. Third, for each test point it aggregates the ensemble into a forecast and takes the `(1 − α)` quantile of the current residual pool. Fourth - the key adaptation - every `s` steps it observes the realised error, appends it to the pool, and discards the oldest one. The pool thus forgets the distant past and tracks the present.

The refresh frequency `s` is the crucial knob: `s = 1` refreshes every step (fastest adaptation), `s = 100` is nearly static.

> **[FIGURE 5 — notebook cell 44]** EnbPI intervals on AR(1) (refresh `s = 10`), point forecast in red, band in green, misses marked.

On stationary AR(1), EnbPI hits nominal coverage like the others, but its band tends to be marginally *wider* than SCP's - the bootstrap resampling injects extra variance into the pool, and that variance shows up as fatter intervals. For a stationary process this is pure overhead. EnbPI's payoff is conditional on the process drifting, where its self-refreshing pool can chase a shift a fixed calibration set cannot. Note it never needed a held-out calibration split at all, which is part of its appeal - though, as the runtime chart will remind us, training and querying 25 models is not free.

### ACI — Adaptive Conformal Inference

**New ingredient: adapt the target level itself, with feedback.** Every method so far adjusted *which residuals* feed the quantile. ACI leaves the calibration set completely fixed and instead tweaks `α` online. It runs as a loop over the test set, revealing the truth after each prediction, and updates the effective miscoverage level by

`α_{t+1} = α_t + γ·(α − err_t)`,

where `err_t = 1` if the truth escaped the interval and `0` if it was covered.

The logic is a thermostat. On a miss, the bracket `(α − 1)` is negative, so `α_t` drops, which means `1 − α_t` *rises*: the next quantile is larger and the next band wider - the system reacts to under-coverage by becoming more conservative. On a hit, the bracket `α` is positive, so `α_t` creeps up and the band tightens, clawing back efficiency. Over a long run these pushes balance, and the empirical miscoverage converges to `α` *regardless of the data-generating process* - the guarantee holds even under arbitrary distribution shift, as long as feedback keeps arriving. The step size `γ` is the whole personality: large `γ` adapts fast but oscillates, small `γ` is smooth but sluggish.

The cleanest way to see this is to point ACI at the mean-shift series, where the level jumps at the very first test step:

> **[FIGURE 6 — notebook cell 49]** ACI's *effective* coverage target `1 − α_t` over the test steps on the Mean-Shift process (purple), against the nominal 0.90 (dashed red). It starts at 0.90, then climbs above it as the run of post-shift misses forces wider intervals, and hovers at whatever elevated level keeps the new regime covered.

That self-correction is something no fixed-quantile method can do: ACI literally rewrites its own objective in response to feedback. And back on stationary AR(1):

> **[FIGURE 7 — notebook cell 51]** ACI intervals on AR(1) (`γ = 0.01`). Near-identical to SCP: coverage near 0.90, comparable width.

With no shift to chase, the effective level barely strays and the band stays put. That's the ideal profile for an adaptive method - invisible when the data are calm, decisive when they are not - bought for the modest cost of recomputing a quantile each step.

### Block CP — blocking the data

**New ingredient: blocks as the unit of randomization.** Every method so far treated individual points as the atoms. Block CP argues that while consecutive points are not exchangeable, contiguous *blocks* of them roughly are. The split version we implement groups the calibration residuals into non-overlapping blocks, reduces each block to a single score (its mean), and computes the ordinary conformal quantile over those block-level scores.

There's a subtlety that predicts the result. Averaging residuals *within* a block produces values that are **less dispersed** than the individual residuals - block means cluster toward the centre. A quantile computed on these tamer block scores is therefore *smaller* than one on raw residuals, yielding **narrower** intervals. Narrower than warranted means **under-coverage**.

> **[FIGURE 8 — notebook cell 56]** Block CP intervals on AR(1) for block sizes B = 2 and B = 3, with coverage in each title. Both dip below 0.90, and the gap widens as B grows - red misses noticeably more frequent than 10%.

So this simple blocking scheme is over-confident even on well-behaved AR(1) - the one setting where SCP was perfectly valid - and it gets worse as the block size grows (more averaging, tamer scores, tighter-but-wrong intervals). This is the cautionary tale of the ladder: a method motivated by sound theory can still misbehave in practice if its calibration statistic quietly shrinks the effective error distribution. Block CP needs careful block-size tuning before you'd trust it.

## A head-to-head simulation study

The single-realisation plots build intuition, but coverage is a *statistical* property - we have to average over many independent series to judge a method fairly. So we run every method, on all four processes, repeated 50 times, scored on coverage, width, and runtime. (Twelve method configurations in all: SCP, three WCP schemes, three ACI step sizes, three EnbPI refresh rates, two block sizes.) Because the forecaster is shared and frozen across all of them, every difference in the results is attributable to the conformal layer alone.

> **[FIGURE 9 — notebook cell 63]** The central figure: a 2×2 grid, one panel per process, plotting coverage (x) against mean width (y) for every method, with a dashed red line at the 0.90 target. The ideal method sits *on* the line (valid) and *low* (sharp).

Reading the panels: on the two stationary processes (AR(1), ARMA(1,1)) almost everything clusters on the target line - SCP, all WCP variants, all ACI variants, all EnbPI variants are valid, differing only in width, with EnbPI pushed slightly higher by bootstrap variance. The glaring exception is Block CP, marooned to the *left* of the line, under-covering exactly as warned. GARCH tells the same story: the mean is stable, so the fixed-width methods cope, and only block CP falls short.

The **Mean-Shift** panel is where the methods separate, and it's the punchline of the whole episode. Anything relying on the frozen pre-shift calibration set - SCP, all three WCP schemes, block CP - collapses well to the left of 0.90, because its quantile was computed before the shift and is blind to the new systematic error. Pulled back toward the target are the methods with genuine recency-based adaptation: ACI (more so at larger `γ`) via its feedback loop, and EnbPI (more so at faster refresh `s`) via its sliding pool. Under distribution shift, the methods that *keep learning* recover most of the lost coverage; the static ones cannot.

> **[FIGURE 10 — notebook cell 65]** Bar chart of mean runtime per method (ms, log-ish spread), averaged across processes.

The third axis is cost, and the bars span more than two orders of magnitude. SCP, WCP, and block CP are nearly free - one model fit and one (weighted) quantile. ACI is moderately pricier because it recomputes a quantile at every test step. EnbPI is the clear outlier, an order of magnitude slower than everything else, because it trains, stores, and repeatedly queries 25 bootstrap models. This is the practical counterweight to the coverage plot: EnbPI's robustness to shift comes at a real computational price, and if a cheaper method (ACI, WCP-exp) achieves the same coverage, it's usually the better operational choice.

Two summary tables make it concrete. First, coverage with methods down the rows and processes across the columns:

> **[TABLE 1 — notebook cell 67]** Coverage of every method on every process. Two patterns jump out: `SCP-block` under-covers in *every* column, stationary or not; while plain SCP, the WCP schemes, and the slow-refresh methods are valid on the three stationary-mean columns but break on `Mean-Shift`. They are perfectly good until the world moves under them.

Then, the decisive Mean-Shift case, sorted by Winkler score:

> **[TABLE 2 — notebook cell 69]** Mean-Shift results sorted by Winkler (lower better), with a `valid` flag for coverage within two points of 0.90. Notice *every* `valid` entry reads `False`.

That last detail deserves an honest footnote rather than a victory lap. The shift here is deliberately severe - a full innovation standard deviation, landing on the very first test point, paired with a frozen base forecaster. With only 300 post-shift steps to recover, *no* method fully re-climbs to nominal coverage inside the window. So here the discriminator isn't the `valid` column, it's the **ranking**: the large-`γ` ACI variants and fast-refresh EnbPI claw back the most coverage and earn the lowest Winkler scores, while SCP, the WCP schemes, and especially block CP sit at the bottom, their narrow stale intervals looking efficient until the Winkler penalty for missed points punishes them hardest. The take-away is directional. Give the adapters a longer horizon - or a base forecaster that itself re-trains - and the gap to 0.90 keeps closing.

The thesis of the study, in one line: **validity is not a property of a method alone, but of a method matched to its data stream.**

## Bringing it together: conformal XGBoost on real data

Everything so far used a humble AR-LS forecaster on simulated processes, which was the right setting for isolating each idea. But it can leave a false impression that conformal prediction is wedded to a particular model or to toy data. It is not. The calibration layer never inspects the forecaster's internals - it only ever touches **residuals**. Swap the model and the machinery is unchanged.

So we do three things at once. We replace AR-LS with a **gradient-boosted tree (XGBoost)** - a flexible nonlinear learner with no probabilistic story of its own. We fit it on a *real* dataset: ten years of daily minimum temperatures in Melbourne (3,650 readings, a strong annual cycle plus short-range autocorrelation, and mildly heteroscedastic noise). And we add the comparison practitioners care about most - XGBoost's *native quantile regression* against the *conformalized* version of those same intervals.

> **[FIGURE 11 — notebook cell 74]** Daily minimum temperature in Melbourne, 1981–1990. A seasonal sawtooth - winter troughs, summer peaks, repeating annually - with a band of day-to-day fluctuation that is itself wider in some seasons than others.

The feature set is richer than two raw lags - five lags, two rolling means, and a sine/cosine encoding of the day-of-year so the tree can place the season - but none of that touches the conformal layer; it only makes the point forecast better, which makes the residuals smaller and the intervals tighter. We split strictly in time, fit XGBoost once, and freeze it, exactly as we did with AR-LS.

The natural competitor is **quantile regression**, and a tempting one. Instead of predicting the mean, train two models to minimise the *pinball loss* at the 5th and 95th percentiles, and read off a 90% band whose width *varies with the inputs* - naturally widening where the data are noisier. That input-dependent shape is genuinely valuable and something the symmetric SCP band cannot offer. The catch is the word *aim*: the estimated quantiles are themselves fitted functions with no finite-sample guarantee attached. On held-out data they typically **under-cover** - the right shape but the wrong size.

Now the payoff. Every rung of the ladder is re-applied to XGBoost *without modifying a single primitive* - the same `conformal_quantile`, the same scoring function. The only thing that changes is the source of the residuals: `y_cal − xgb.predict(X_cal)` instead of the AR-LS version. That substitution is the entire adaptation. We also add one new rung suited to the quantile setting:

- **CQR (Conformalized Quantile Regression)** conformalizes the quantile *band* rather than the point forecast. Its nonconformity score is the signed distance by which the truth escapes the predicted band - `max(lo − y, y − hi)` - positive when a calibration point falls outside, negative when it sits comfortably inside. The conformal quantile of those scores becomes a single correction `q` that is *added to the upper edge and subtracted from the lower edge*. When the raw band under-covers, the scores are large and positive and CQR widens it; if it over-covered, `q` is negative and the band shrinks. Crucially, CQR inflates the band uniformly while **preserving its input-dependent shape** - it inherits quantile regression's adaptive width and gains conformal's guarantee.

> **[TABLE 3 — notebook cell 87]** Head-to-head on the temperature test block: Raw QR, SCP, WCP-exp, Block CP, ACI, EnbPI, CQR, with coverage, width, Winkler, and gap-to-0.90.

Read the coverage column top to bottom. Raw quantile regression sits clearly under 0.90 - its gap is negative. *Every conformal row closes that gap*, landing at or essentially at target. The cleanest statement of the section is the contrast between the first row and the last: Raw QR and CQR are built from the *identical* underlying quantile models, yet only the conformalized one is valid. Among the conformal methods the familiar texture reappears - WCP-exp and ACI reach the target at the lowest Winkler, while SCP and EnbPI buy coverage with wider bands, and Block CP again sits a touch low.

> **[FIGURE 12 — notebook cell 89]** Left: coverage bar chart, the lone raw-QR bar in red falling below the 0.90 line, every conformal bar reaching it. Right: width-vs-coverage scatter, raw QR narrow but invalid (left of the line), the conformal methods clustered on or just past it.

Narrowness is only a virtue *once you are at the target*, and quantile regression alone never gets there. Finally, the mechanism made visual:

> **[FIGURE 13 — notebook cell 91]** Two stacked panels over the first 120 test days. Top: raw quantile regression, tracking the seasonal swing and breathing with local noise, but peppered with red misses because it's systematically a hair too narrow. Bottom: CQR - the *same shape* shifted outward by the single conformal correction, contour preserved, misses thinned out, coverage climbing to target.

This is the picture to remember: conformal prediction didn't replace the quantile model, it *repaired* it - keeping the adaptive shape while supplying the guarantee it lacked.

## Where this goes next

We stayed deliberately classical, but the field has climbed higher rungs than these. When residuals are themselves autocorrelated - a big error today predicting a big error tomorrow - you can treat the nonconformity scores as a forecasting problem in their own right: **SPCI** fits a quantile random forest to predict the *next* residual quantile from recent ones, and **SPCI-T** swaps that forest for a Transformer decoder that attends over the history of past errors. On the adaptive-control side, **AgACI** and **DtACI** automate ACI's one fiddly knob (`γ`) by aggregating or dynamically tuning step sizes, and **Conformal PID control** borrows the full proportional-integral-derivative loop to track coverage. And you needn't implement any of this from scratch in production: **MAPIE** wraps scikit-learn-style models (it exposes EnbPI, ACI, and CQR directly), and the **Nixtla** stack injects conformal intervals into ARIMA, gradient boosting, and neural forecasters alike. We'll likely return to the residual-modelling branch - it's the same "borrow strength from structure you already have" instinct that ran through the intermittent-demand episode, now pointed at the errors instead of the signal.

## Closing time

The experiments map cleanly onto practical advice, governed by the properties of your data stream:

- **Stable, stationary, weakly-dependent data.** Plain **SCP** is valid, sharp, and almost free. The theory backs this up - for weakly-dependent (β-mixing) stationary processes the coverage gap is provably small - so adaptive machinery is unnecessary. Don't over-engineer.
- **Anticipated non-stationarity (drift or abrupt shifts).** You need genuine adaptation, with a speed-vs-cost trade-off. **WCP** with exponential or linear decay is the cheapest robust option and handles *gradual* drift well, but cannot manufacture error magnitudes absent from its fixed calibration set, so it struggles with sharp breaks. **ACI** maintains coverage through active feedback under essentially arbitrary shifts, at the modest cost of a quantile per step. **EnbPI** is robust via its self-refreshing pool but is by far the most expensive and yields the widest bands - reach for it only when the compute budget allows.
- **Block CP** under-performed throughout, under-covering even on stationary data, and would need careful block-size tuning before deployment.
- **Any base learner, including a black box.** The conformal layer never inspects the model, only its residuals, so it wraps a gradient-boosted tree exactly as readily as a linear AR. And when the forecaster already emits quantiles, **CQR** is the natural choice: it repairs their coverage while preserving their attractive input-dependent width.

If there's one thread through all five rungs, it's this: an interval is only honest if it keeps up with the data. Conformal prediction's gift is the guarantee - cover the truth `1 − α` of the time, no matter how strange the model or the noise. The catch is that time series keep moving, and the guarantee was written for data that holds still. Every method here is a different answer to the same question - how do you keep a static promise in a world that won't stop changing? - and the right answer is the one that matches how *your* world changes.

---

*United States of Banan is a reader-supported publication. To receive new posts and support my work, consider becoming a free or paid subscriber.*
