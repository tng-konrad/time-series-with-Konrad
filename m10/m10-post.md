# Time Series with Konrad: episode 10

### State space methods: the Kalman filter, hidden states, and the grand unification

**KONRAD BANACHEWICZ**
AUG 17, 2026

For most of this series, "which model?" has meant choosing between families. Episode 2 gave you the exponential smoothers; episode 3 the ARIMA clan; episode 1 the structural decompositions. Historically that split was real — the Box–Jenkins people and the exponential-smoothing people went to different conferences, published in different journals, and regarded each other's methods with polite suspicion. One camp had rigorous statistics and black-box interpretability; the other had intuitive components and no probability model to stand on.

↪ *New to the families this episode unifies? Exponential smoothing was episode 2 (**<LINK TO EPISODE 2 HERE>**), the ARIMA clan episode 3 (**<LINK TO EPISODE 3 HERE>**), and structural trend/seasonal decomposition episode 1 (**<LINK TO EPISODE 1 HERE>**).*

**State space models (SSMs)** are the framework that quietly ended the feud. In the formulation perfected by Durbin and Koopman, ARIMA, exponential smoothing, and structural trend/seasonal models turn out to be *special cases* of one architecture — different fillings of the same two equations. This is not a philosophical nicety: once everything lives in one framework, everything gets a real likelihood, comparable AICs, proper prediction intervals, and — my favorite — native handling of missing data. No other idea in classical time series buys so much with so little.

The central move is a separation of concerns worth stating as this episode's core principle:

**What you observe is a noisy measurement of something you never observe directly — the state. Model the state's evolution and the measurement process separately, and everything else follows mechanically.**

The "everything else" is delivered by the **Kalman filter**, a recursive algorithm born in aerospace navigation (it guided Apollo before it estimated GDP trends) and adopted by statisticians as a likelihood engine. This episode builds one by hand — about forty lines — and validates it against statsmodels *to machine precision* on a real river. Then we climb: local level → local linear trend → full structural model → ARIMA and exponential smoothing re-derived as state space models → a custom model from raw matrices → the missing-data party trick → a three-way forecast shootout. Real data throughout: the annual flow of the **Nile** (the canonical Durbin–Koopman dataset, complete with a dam) and our old friend the monthly **airline passengers** series. Practical understanding over rigor, as always — though this episode is where the two get unusually close.

📓 All code lives in the companion notebook:
https://github.com/tng-konrad/time-series-with-Konrad/blob/main/m10-fable-version.ipynb

## Two equations, five matrices

Every model in this episode is an instance of the **linear Gaussian state space model**, written in Durbin–Koopman notation (which statsmodels adopts — more on that treacherous word "notation" in a moment):

**Observation equation** — how data is generated from the hidden state:

```
y_t = Z·α_t + ε_t,          ε_t ~ N(0, H)
```

**State equation** — how the hidden state evolves:

```
α_{t+1} = T·α_t + R·η_t,    η_t ~ N(0, Q)
```

Five matrices define a model, and it pays to memorize their jobs, because *every* model below is just a different way of filling them in:

- **Z (design):** which parts of the state show up in the observation.
- **T (transition):** how the state moves forward one step — the system's "memory".
- **R (selection):** which shocks hit which states. Sounds like bookkeeping; it is where one of this episode's best plot twists hides.
- **Q, H:** how big the process shocks are, and how noisy the measurement is.

*Pro tip, earned through much wasted time: notation varies violently across the literature. Hamilton's textbook calls the transition matrix F and the design matrix H′ — yes, H, the letter Durbin–Koopman use for the observation noise covariance. Control-theory texts use A and C. When translating any derivation into statsmodels code, check the notation table first; the same letter can mean opposite things in two books.*

The **Kalman filter** turns these two equations into estimates, recursively: at each step it *predicts* (push the state forward through T, widen its uncertainty) and *updates* (compare the prediction with the arriving observation and correct in proportion to how informative it is). Three per-step quantities deserve names, because they carry the whole logic:

- the **innovation** `v_t = y_t − Z·a_t` — what the new observation surprised us by; the only genuinely new information it contains;
- its variance `F_t` — state uncertainty projected through the design matrix, plus measurement noise;
- the **Kalman gain** `K_t` — how hard to yank the state toward the surprise. Noisy measurement → small gain, trust the model. Uncertain state → large gain, trust the data. The filter re-balances this trade *at every time step*, automatically.

And one bonus falls out for free. The innovations are independent Gaussians by construction, so the model's entire log-likelihood decomposes into a running sum over the filter pass — one clean term per observation. This **prediction error decomposition** is the bridge between filtering and estimation: it converts "fit this structural model" into garden-variety maximum likelihood, which is exactly what every `fit()` call below is doing under the hood.

Enough architecture. Let's meet a river.

## Groundwork: the Nile, and an engine you can read

> **[FIGURE 1 — notebook cell 20]** Annual flow volume of the Nile at Aswan, 1871–1970, with a red dashed line marking the first Aswan dam (1899). Noisy year-to-year swings around a level that visibly steps down after the dam: flows before 1899 average about 1,100, after about 850.

The **Nile dataset** — one hundred annual flow measurements, 1871–1970, bundled with statsmodels — is *the* canonical state space example, the running case study of the Durbin & Koopman book itself. And it earns the status: year-to-year swings of a hundred units are routine, occasional jumps reach three or four hundred, and around 1899 — when the first Aswan dam began operating — the entire band of values steps down. Not a trend, not seasonality: a *level shift*. A hidden "true level" that changes over time, observed through noisy annual readings — you could not invent a cleaner specimen for a model that tracks exactly that.

Our reflexive ADF test (episode 3 habits die hard) returns a curveball worth savoring:

```
ADF statistic : -4.0487
p-value       : 0.0012
verdict       : stationary (reject unit root)
```

Stationary?! We are about to model this series with a *random-walk* level — the textbook non-stationary process. The resolution teaches something. What the data mostly contains is a one-time level shift; asked "unit root or not?", ADF answers "not", because after the shift the series hovers around a stable mean. But a fixed-mean model would smear the 1899 break across the whole century. The local level model refuses the false choice: it *estimates how much the level moves* via a variance parameter, and if the level were truly constant, maximum likelihood would drive that variance to zero. Rather than pre-deciding stationarity, we let the model measure it. This is the state space philosophy in one sentence: **model non-stationarity explicitly instead of differencing it away.**

### Forty lines of Kalman filter, validated on a real river

The notebook implements the filter recursions directly in NumPy — innovation, variance, gain, stable covariance update, likelihood contribution, loop. I won't reprint the class here (it's in the notebook, commented line by line), but I insist on the validation, because it is the kind that leaves no wiggle room. We let statsmodels find the maximum-likelihood variances for the Nile local level model, feed those *same* numbers to our hand-built filter with the *same* initialization, and compare:

```
MLE variances:  measurement sigma2_eps = 15,078   level sigma2_eta = 1,479
max |state difference| : 5.9e-12
our loglik (after diffuse burn-in): -632.5442
statsmodels loglik                : -632.5442
```

The predicted state paths agree to **10⁻¹²** — our forty lines and statsmodels' optimized Cython engine are doing the same arithmetic. Two details in that snippet deserve unpacking, because both are recurring gotchas:

**Diffuse initialization.** We started the filter at `a1 = 0` with variance `P1 = 10⁷`. That absurdly large variance is the honest statement "we have no idea where the level starts" — it lets the filter *learn* the level from the first observations instead of being biased by a made-up prior. It is the standard device for non-stationary states, and statsmodels does a mathematically exact version of it automatically.

**The burn-in.** Our raw likelihood sum initially *disagreed* with statsmodels — by exactly one observation's worth. Under diffuse initialization the first innovation has essentially infinite variance, so its likelihood term is meaningless noise, and statsmodels quietly drops ("burns") it. Sum our per-observation contributions from the second observation onward and the two engines agree to four decimals. Initialization conventions are precisely the kind of detail that silently ruins cross-library comparisons; now you know where to look. *(This gotcha returns later in the episode — it is why two correct implementations of the same model can report different log-likelihoods while agreeing on every parameter.)*

> **[FIGURE 2 — notebook cell 29]** The hand-built filter tracking the Nile: noisy observations in blue, one-step-ahead predictions in red. The red line chases aggressively in the first few years (diffuse start → large gain), then settles into a smooth track that shrugs off single noisy years but follows the sustained post-1899 drop.

Each red point is the model's guess for that year *before seeing it*, using only the past. Ignore the noise, follow the signal — that balance is the Kalman gain doing its job with weights derived from the two estimated variances, not from any smoothing constant we chose. Speaking of which…

## The structural ladder

From here we use `sm.tsa.UnobservedComponents` — the high-level interface where you *name* the components and statsmodels assembles the matrices. Each rung adds exactly one component to the state vector.

### Rung 1: the local level

The simplest non-trivial state space model, and exactly what we just hand-filtered:

```
y_t   = μ_t + ε_t          (what we see: level plus noise)
μ_t+1 = μ_t + η_t          (the level itself: a random walk)
```

In matrix terms, Z = T = R = 1 — the five-matrix machinery collapsed to its smallest instance. Two variances carry all the behavior, and their fitted values *are* the interpretation:

```
sigma2.irregular ≈ 15,078      -> annual readings scatter with sd ≈ 123 around the level
sigma2.level     ≈  1,479      -> the level itself drifts with sd ≈ 38 per year
```

Their ratio — the **signal-to-noise ratio** q = σ²_η/σ²_ε ≈ 0.10 — is the single most interpretable number in the model: the level moves, but roughly *ten times more slowly* than the noise around it. This is precisely the resolution of the ADF puzzle, now quantified: the level is not constant (q > 0), but nearly all year-to-year variation is measurement noise, not signal. The summary's Ljung–Box and Jarque–Bera diagnostics read clean — two parameters are, statistically, enough for this river. And keep q in your pocket: in a few sections it reappears wearing a different letter, and the costume reveal is the best joke the framework tells.

> **[FIGURE 3 — notebook cell 34]** Observed flow (faint blue) with two level estimates: the filtered level (orange) and the smoothed level (red), dam marked in grey. The orange line reacts to the 1899 break with a visible lag; the red line concentrates the drop in the years around the dam — sliding from ~1,070 in 1890 through ~1,000 on the eve of the dam to ~870 by 1902 — and stays there.

This plot teaches the most important conceptual distinction in state space practice: **filtering versus smoothing**. The *filtered* estimate at year t uses only data up to t — what you could have known in real time. The *smoothed* estimate runs a second, backward recursion and uses the entire sample — later observations revise earlier state estimates. The difference is starkest exactly where it matters, at the break: the filter needs several post-dam years of low readings before it believes the regime changed (in real time, two low years might just be noise), while the smoother, knowing the future, pins the drop where it happened.

The practical rule falls straight out, and it has a familiar episode-8 flavor: **forecasting and real-time monitoring live in filtered world; historical analysis lives in smoothed world.** Admiring how well the *smoothed* level "would have detected" the break in real time is a hindsight error — the smoother has seen the future.

↪ *"Never let the model see the future" is the exact discipline episode 8 built its whole validation framework on — the smoother-as-hindsight trap is data leakage wearing a different costume.* → **<LINK TO EPISODE 8 HERE>**

### Rung 2: local linear trend — add a slope

The Nile has a wandering level but no direction. Plenty of series do have direction, so the next rung gives the level a velocity: a second state, the slope ν_t, itself a random walk.

```
y_t   = μ_t + ε_t
μ_t+1 = μ_t + ν_t + ξ_t        (position += velocity)
ν_t+1 = ν_t + ζ_t              (velocity drifts too)
```

For this we need a trending series, and the series has one on staff: the monthly **airline passengers** data from episode 2.

> **[FIGURE 4 — notebook cell 37]** Two panels: raw airline passengers 1949–1960 (seasonal swings growing with the level) and the same series in logs (swings now constant-width, trend gently linear).

We model in logs — the raw series has *multiplicative* seasonality (swings grow with the level), our structural models are additive machines, and the log is the standard bridge. All forecasts get exponentiated back before scoring, so every RMSE below is in honest passenger units. The split is the usual chronological one: last 24 months held out, two full seasonal cycles, so a model that fakes seasonality gets caught.

> **[FIGURE 5 — notebook cell 39]** The local linear trend forecast: a smooth green exponential ramp through a zigzagging holdout, with a 95% interval fanning out dramatically. RMSE ≈ 101 passengers.

```
Local linear trend: {'MAE': 76.41, 'RMSE': 100.68}
```

Deliberately underwhelming, and the picture explains why better than the number: the model has correctly learned the level and the slope, and knows *absolutely nothing* about the seasonal sawtooth it was never given a state for. The actual holdout zigzags around the ramp, summer peaks sailing far above. Note also how fast the interval fans out — with two integrated states, uncertainty compounds quickly. Honest, if humbling. Each rung's failure is the next rung's motivation, and this one spells its cure in capital letters.

### Rung 3: the basic structural model — add seasonality

One keyword — `seasonal=12` — and the state vector grows from 2 to 13: eleven extra states carrying the seasonal pattern, constrained to sum to zero over any full year, each allowed to evolve slowly via its own shock. This is the classic trend + seasonal + noise decomposition of episode 1 — but as *one coherent probability model*, estimated in one pass, with a likelihood and error bars.

> **[FIGURE 6 — notebook cell 42]** The basic structural model forecast: green dashed line riding the seasonal sawtooth through both holdout years, peaks and troughs inside a sanely narrow interval. RMSE ≈ 33.

```
Basic structural model: {'MAE': 27.72, 'RMSE': 32.58}

estimated variances:
  sigma2.irregular     0.000141
  sigma2.level         0.000797
  sigma2.trend         0.000000
  sigma2.seasonal      0.000037
```

A third of the trend-only model's error. And the estimated variances repay a slow read — one of them is a small masterpiece: **σ²_trend = 0.000000**. Maximum likelihood has driven the slope's shock variance to zero, which is the model saying "the data prefers a *fixed* growth rate on the log scale" — steady percentage growth, entirely plausible for 1950s aviation. We *offered* a stochastic slope; the likelihood declined the stochasticity. This is the state space framework doing model selection from the inside, and when a variance hits a boundary like this, the model is telling you something structural. Listen.

> **[FIGURE 7 — notebook cell 44]** `plot_components` output: smoothed level (effectively the seasonally-adjusted log series), the trend panel flat at a constant — as its zero variance dictates — and the seasonal panel repeating its annual shape with only gentle evolution. Every component wears a confidence band.

The structural model's party trick: the fitted model *is* a decomposition, so it can draw its own anatomy. Unlike the moving-average decompositions of episode 1, every curve here is an estimated state with a posterior variance — **decomposition with error bars**.

## The grand unification

The ladder so far built models *from* state space parts. The framework's deeper claim is that the models you already know are *already* state space models in disguise. Two reveals, one per old episode.

### ARIMA is a state space model

Every ARIMA(p, d, q) can be written in state space form: build a state vector big enough to "remember" the relevant past — dimension max(p, q+1) — put the AR coefficients down the first column of the transition matrix, and route a *single* shock into the states through a selection matrix loaded with the MA coefficients. For the ARIMA(1,1,1), expanding the differencing into the AR polynomial gives concrete, satisfying matrices:

```
T = | 1+φ  1 |      R = | 1 |      Z = ( 1  0 )
    | -φ   0 |          | θ |
```

Look at what R is doing: **two states, one source of randomness.** This is the fewer-shocks-than-states situation the selection matrix exists for — the bookkeeping matrix earning its keep. statsmodels' `SARIMAX` is precisely this construction industrialized: it builds the representation and hands it to the same Kalman machinery we validated on the Nile. That is why it lives in the `statespace` module, why it produces exact likelihoods rather than the approximations of older ARIMA routines — and why it can digest series with holes in them, which classical Box–Jenkins never could.

We fit the most famous specification in the literature — the **airline model**, SARIMA(0,1,1)(0,1,1)₁₂, *named after this very dataset* (episode 3 alumni will feel at home):

> **[FIGURE 8 — notebook cell 47]** The SARIMAX airline-model forecast: clearly seasonal, tracking the sawtooth, sitting a touch low. RMSE ≈ 43.

```
SARIMAX airline model: {'MAE': 39.45, 'RMSE': 43.19}
```

The fit-forecast-evaluate code is *identical* to the structural model's — same `get_forecast`, same helpers — which is the unification made tangible: one workflow, different matrix fillings. The score is the interesting part: respectable, clearly seasonal, and a notch behind the basic structural model (33) *on the dataset this ARIMA specification was named after*. Differencing twice — regular plus seasonal — is a heavier-handed way to remove structure than modeling it, and a two-year horizon feels the difference.

> **[FIGURE 9 — notebook cell 49]** The four-panel diagnostics: standardized innovations over time (no drift), histogram + KDE vs N(0,1) (close), Q–Q plot (hugging the line, mild tails), correlogram (no significant spikes).

The diagnostics pass — this is *not* a misspecified model, just one that generalizes slightly worse here. That distinction (specification failure vs genuine out-of-sample difference between adequate models) is exactly the kind of verdict only a held-out future can deliver, which is why we always keep one.

### Exponential smoothing is a state space model — and α is a Kalman gain

The second reveal is sneakier, and it is my favorite result in this entire corner of statistics. Simple exponential smoothing (episode 2) began life as a heuristic — a sensible-looking update rule with a hand-tuned α, no probability model in sight:

```
forecast:   ŷ_t = l_{t-1}
update:     l_t = α·y_t + (1-α)·l_{t-1}
```

Now define the one-step error ε_t = y_t − l_{t-1} and substitute. The recursion rearranges *exactly* into:

```
y_t = l_{t-1} + ε_t              (observation equation)
l_t = l_{t-1} + α·ε_t            (state equation)
```

A state space model with T = 1, Z = 1... and the selection matrix is **the smoothing parameter itself: R = α.** But notice the structural twist: the *same* shock ε_t appears in both equations. Where the local level model had two independent noises (measurement and level), here one innovation drives everything — a **single source of error (SSOE)** model, versus the multiple-source (MSOE) structural models above. That is the *entire* mathematical difference between the two feuding traditions.

And the reinterpretation lands beautifully: the smoothing parameter is a **Kalman gain frozen at its steady state**. The local level's signal-to-noise ratio q and the smoother's α are the same dial in different vocabularies. Choosing α by minimizing squared errors, as the 1950s heuristics did, was maximum likelihood all along — nobody had written down the likelihood yet.

One practical consequence before the results: statsmodels ships *two* exponential smoothing implementations, and the difference is exactly this section. `sm.tsa.ExponentialSmoothing` runs the classical recursions — fast, heuristic, limited intervals. `ETSModel` implements the state space formulation: real MLE, real prediction intervals, AICs you may legitimately compare against SARIMAX and structural models. Unification being the theme, the notebook uses the latter. Fitting ETS(A,A,A) — Holt-Winters with a likelihood — on the log series:

```
smoothing parameters (a.k.a. steady-state Kalman gains):
  smoothing_level        0.9999
  smoothing_trend        0.0001
  smoothing_seasonal     0.0001

ETS(A,A,A): {'MAE': 31.77, 'RMSE': 36.95}
```

RMSE ≈ 37 — between the structural model and the ARIMA. But look at those parameters with SSOE eyes: α ≈ 1 (the level absorbs each innovation whole — on this smooth series, the "level" simply tracks the data), while β and γ ≈ 0.0001 (the trend and seasonal states barely update — effectively a fixed growth rate and a stable seasonal pattern). Which is, translated across frameworks, **the same verdict the structural model reached** when its trend variance went to zero and its seasonal variance came out tiny. Two parameterizations, one conclusion about the data. Unification working exactly as advertised.

## Power-user corner: your own model from raw matrices

The high-level classes cover the standard cases; the framework's real power is that you are not limited to them. Subclass `MLEModel`, write down your own five matrices, and inherit everything else — filter, likelihood, optimizer, standard errors, forecasting, diagnostics. The notebook rebuilds the local linear trend this way as a teaching exam: declare two states, fill Z, T and R in `__init__` (they never change), write the three variances into H and Q inside `update()` (called at every optimizer step), and — the step everyone forgets — supply a `transform_params` pair that squares parameters on the way in and square-roots on the way out, so the optimizer can roam freely while variances stay positive. *Forget that transform and `fit()` will eventually wander into negative variances and explode; this is the single most common bug in hand-rolled state space models.*

The exam result: our hand-specified model and `UnobservedComponents` produce **identical variance estimates to six decimals**. The black box and the white box are the same box. (Their reported log-likelihoods differ, though — exact versus approximate diffuse initialization, the same bookkeeping gotcha the Nile validation taught. Parameters agree; likelihood accounting differs. If you ever compare AICs across implementations, make sure the initialization conventions match.)

One more structural whisper from the estimates: on the log airline data, the *measurement* variance gets squeezed to zero — the series is so smooth that the model deems each month's reading an essentially noiseless glimpse of a wandering level. Boundary estimates are messages, not bugs.

## The party trick: missing data, natively

Time to cash the cheque the introduction wrote. Real archives have holes — instruments fail, wars interrupt record-keeping. Classical Box–Jenkins has no honest answer (impute first, then pretend you didn't). The Kalman filter's answer is built into its own recursion: **when y_t is missing, skip the update step.** The prediction step still runs — the state coasts forward through T, uncertainty widening — the gain is simply never applied, and the likelihood skips a term. Then the smoother, running backward, uses post-gap data to pin down what happened inside the gap. Interpolation is not preprocessing; it is a posterior estimate from the same model.

The notebook tests this the honest way: erase 1911–1925 from the Nile — fifteen consecutive years, 15% of the sample — refit *directly on the series with NaNs in it* (no error, no imputation, no special arguments), and compare the reconstruction against the hidden truth.

```
variances fitted on gapped data : [12813, 1870]
variances fitted on full data   : [15078, 1479]

RMSE inside the gap - Kalman smoother    : 172.25
RMSE inside the gap - linear interpolation: 185.15
```

> **[FIGURE 10 — notebook cell 62]** The Nile with the 1911–1925 block shaded grey: full data in faint blue (the hidden truth), gapped observations in green, and the smoothed level in red bridging the hole with a gently sloping line between the level estimates at the gap's edges.

Two readings, one modest and one important. The modest one: the Kalman reconstruction beats naive linear interpolation, 172 vs 185 — real but not dramatic, because for a local level model the optimal bridge genuinely *is* close to a sloping line. The important one is *what* gets connected: naive interpolation joins the two noisy observations that happen to sit at the gap's edges; the smoother joins noise-filtered *level estimates*, each already an average over many surrounding years. And the deeper advantages don't show in the point-accuracy number at all: the model reports **uncertainty inside the gap** (the state variance visibly inflates where data is absent — linear interpolation is silently overconfident); the parameters came out close to the full-data values because the likelihood correctly *knew* those years were absent rather than being fed invented values masquerading as evidence; and the same mechanism scales unchanged to seasonal and multivariate settings, where straight-line imputation becomes actively destructive.

## Head-to-head: three traditions, one holdout

Same training window, same 24-month holdout, same metric; the seasonality-free local linear trend joins as the baseline everyone should beat.

```
model                            RMSE    delta_vs_baseline   improved
BasicStructural                 32.58        -68.10            True
ETS(A,A,A)                      36.95        -63.73            True
SARIMAX airline                 43.19        -57.49            True
LocalLinearTrend (no seasonal) 100.68          0.00              -
```

> **[FIGURE 11 — notebook cell 67]** The two-year holdout in thick black, with all three seasonal forecasts dashed: everyone locks onto the seasonal shape; everyone undershoots the strongest summer peaks; the structural model rides highest and closest, ETS a shade below, SARIMAX consistently a notch low.

All three seasonal models demolish the baseline — cutting its error by 55–70% — which quantifies what the eye already knew: on this series, *seasonality is most of the forecastable signal*. Among the three, the structural model leads, ETS follows a few passengers behind, and SARIMAX trails while remaining entirely respectable; its double differencing anchors the forecast to the most recent observed level, and a holdout that keeps accelerating steadily pulls away from that anchor.

Resist the ranking's gravity, though. This is one series and one horizon, and the honest headline is that three models from historically *rival* traditions land within about ten passengers RMSE of each other — because under the hood they are one framework making three sets of assumptions about the same matrices. The M-competitions' old verdict survives its state space reformulation intact: no family dominates everywhere. ;-)

## Closing time

We climbed from a forty-line hand-built Kalman filter — validated to machine precision against statsmodels on a real river — up the structural ladder (level → +slope → +seasonality), watched ARIMA and exponential smoothing collapse into the same two-equation framework (with α unmasked as a frozen Kalman gain), built a model from raw matrices that matched the library to six decimals, and put the machinery to work on the problem classical methods dodge: fifteen missing years, handled by the filter's own logic rather than by preprocessing.

The single mental model to keep: **a state space model is two equations and five matrices.** Everything else this episode did was a consequence — the Kalman filter as the recursive estimator, the prediction error decomposition as the likelihood, filtering versus smoothing as real-time versus hindsight, diffuse initialization as honest ignorance, NaN-tolerance as a skipped update. And beyond this episode the same two equations keep going: multivariate systems and dynamic factor models, time-varying regression coefficients, and — via simulation smoothing and particle filters — the non-Gaussian, nonlinear territory where Kalman recursions become building blocks inside Bayesian samplers. Different destinations, same architecture.

## Key Takeaways

1. **Separate the state from the measurement.** One equation for how the hidden truth evolves, one for how you noisily observe it. Every classical model is a special case of this split.
2. **The Kalman filter is a likelihood machine.** The prediction error decomposition turns filtering into maximum likelihood — that is what `fit()` maximizes, and why structural models have real standard errors and AICs.
3. **Filtered = real time; smoothed = hindsight.** Forecast and monitor with the filter; decompose and analyze history with the smoother. Confusing them is an episode-8 sin — the smoother has seen the future.
4. **Unification has cash value.** ARIMA and ETS inside one framework means comparable likelihoods, proper intervals, missing-data handling — and cross-framework confirmation, like a zero trend variance and a zero β telling the same story in two dialects.
5. **Variances on the boundary are messages.** σ²_trend → 0 means "the trend is deterministic"; σ²_irregular → 0 means "the measurements are clean". The model selects structure from the inside; listen to it.
6. **Missing data is a skipped update, not a preprocessing crisis.** The filter predicts through gaps, the smoother interpolates them, and the uncertainty bands tell you honestly how little you know in between. No imputation ritual required.

---

*United States of Banan is a reader-supported publication. To receive new posts and support my work, consider becoming a free or paid subscriber.*

[ ✓ Subscribed ]
