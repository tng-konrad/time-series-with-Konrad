# Time Series with Konrad: episode 9

### Causal methods: Granger, cointegration, Toda-Yamamoto — who's driving whom?

**KONRAD BANACHEWICZ**
AUG 03, 2026

Every episode of this series so far has asked some version of the same question: *what happens next?* Smoothers, the ARIMA family, hierarchical reconciliation, conformal wrappers — all of it in service of a better forecast. This episode asks a different question, and a more dangerous one: *who is driving whom?*

Dangerous, because time series are where correlation-versus-causation goes to get worse. With cross-sectional data, a spurious correlation at least has to work for it. With time series, two variables that have *nothing to do with each other* can produce a regression with an R² of 0.95 and t-statistics in the double digits, purely because both drift upward for fifty years. We will produce exactly such a regression below, on purpose, with real data.

And yet time also hands us something cross-sectional data never can: **a cause must precede its effect**. The arrow of time (episode 8 readers know it well) is not just a constraint on validation — it is an identification device. If the past of X helps predict Y beyond what Y's own past provides, X carries *some* kind of information about Y's future. An entire family of econometric methods has been built on that one axiom, and this episode climbs it as a ladder: **VAR → Granger causality → cointegration → VECM → Toda-Yamamoto → structural VAR**.

↪ *That same arrow of time is why ordinary cross-validation quietly breaks on temporal data — episode 8 turned it into a full toolkit of leakage-proof validation, and it's the honest foundation for every forecast comparison below.* → **<LINK TO EPISODE 8 HERE>**

One disclaimer before we start, and it applies to every rung: these methods detect **predictive precedence**, not manipulative causality. "X Granger-causes Y" means X's past improves forecasts of Y — a statement about information, not about what happens if you *intervene* on X. Granger himself, tired of the philosophical crossfire, preferred "temporally related". Keep that asterisk taped to your monitor for the next five thousand words.

Our arena is real throughout: **fifty years of the US economy** — quarterly GDP, consumption, investment, money, prices, and interest rates, 1959–2009, straight from FRED (bundled with statsmodels, so the notebook runs offline). No simulations this time. Real data means real mess, and the mess, as you'll see, is where the lessons live. The goal, as always, is practical understanding over mathematical rigor.

📓 All code lives in the companion notebook:
https://github.com/tng-konrad/time-series-with-Konrad/blob/main/m09-fable-version.ipynb

## Groundwork: one dataset, two guises

The whole episode runs on three headline series — real GDP, real consumption, real investment — used in two forms, and the split between those forms *is* the organizing idea of everything below:

- **Log levels**: trending, non-stationary. This is where the cointegration story lives.
- **Growth rates** (first differences of logs, ×100 → percent per quarter): stationary. This is where VAR, Granger, and SVAR live.

Why logs? Two reasons that never change: logs turn exponential growth into roughly linear trends, and they make differences interpretable as percentage changes — `g_gdp = 0.5` reads as "GDP grew half a percent this quarter".

> **[FIGURE 1 — notebook cell 22]** Log levels of US real GDP, consumption, and investment, 1959–2009: three lines climbing together for five decades, with investment (red) visibly jagged — deep plunges in 1975, 1982, and 2008 — while consumption is nearly smooth.

Three things to register in this plot, because each one becomes a section later. The lines *trend* — the mean changes over time, the textbook signature of non-stationarity. The lines *climb together* — bound by accounting (consumption and investment are components of GDP) and by the shared business cycle; "moving together" is what cointegration will later try to formalize. And investment is the drama queen of the national accounts, plunging in every recession while consumption barely flinches — remember that when we ask which variable does the *adjusting* in the error-correction model.

> **[FIGURE 2 — notebook cell 24]** The same three series as quarterly growth rates: trends gone, all three oscillating around a stable mean, with investment swinging ±10% per quarter while consumption rarely leaves the ±2% band.

Eyeballs are not a test, so we formalize with the two-test battery from episode 3, run through one helper: **ADF** (null: unit root) and **KPSS** (null: stationary). Opposite nulls is the point — when they agree, the verdict is solid; when they disagree, the series is telling you it lives near the boundary.

↪ *New to ADF, KPSS, and the stationarity-testing ritual? The ARIMA episode built them from scratch — they're the prerequisite for every method in this one.* → **<LINK TO EPISODE 3 HERE>**

> **[TABLE 1 — notebook cell 26]** Stationarity verdicts, levels vs growth rates side by side.

```
levels:  ADF p = 0.38-0.57 (unit root not rejected),  KPSS p = 0.01 (stationarity rejected)
growth:  ADF p = 0.000     (unit root rejected),      KPSS p = 0.10 (no objection)
```

Unanimous, both ways: the levels are **I(1)** — integrated of order one, accumulating shocks, wandering — and one difference makes them **I(0)**, stationary. This confirms the fork in the road. Standard VAR and Granger machinery *requires* stationary inputs; estimated on I(1) levels its standard errors and test distributions are simply wrong. So: difference and work with growth rates (statistically safe, but differencing destroys all information about *levels* — about long-run relationships), or keep the levels and reach for cointegration. We walk the first path first.

## Vector Autoregression: everything depends on everything

We start one rung below the headline: an **AR(1)** for GDP growth alone, the model this series has known since episode 2:

```
g_gdp(t) = c + φ·g_gdp(t−1) + ε(t)
```

Fitted: `c ≈ 0.53`, `φ ≈ 0.30`. Growth has mild momentum — a strong quarter tends to be followed by an above-average one — dying out geometrically within a year. Fine. But this model believes GDP growth is explained *only by its own past*: consumption and investment, two-thirds and a sixth of GDP respectively, are not consulted at all.

↪ *The AR(1) recurrence — this quarter leaning on the last — is the same momentum idea we first met among the smoothers and autoregressive models earlier in the series.* → **<LINK TO EPISODE 2 HERE>**

The **VAR(p)** — vector autoregression — fixes exactly this, by brute-force generosity. Replace the scalar with a vector: each of the *k* variables is regressed on *p* lags of **all k variables**, its own past and everyone else's:

```
X(t) = c + A₁·X(t−1) + ... + Aₚ·X(t−p) + ε(t)
```

where `X(t)` stacks the three growth rates and each `Aᵢ` is a 3×3 coefficient matrix. This was Sims' original manifesto: instead of deciding by fiat which variables are exogenous (as the old simultaneous-equation systems did), let *everything* depend on *everything's* past and let the data sort out the coefficients. A pleasant technical bonus: because every equation shares the identical regressor set, each can be estimated independently by plain OLS — no system estimator required.

The one modeling decision is the lag order *p*, and it is a genuine bias/variance trade-off. Too few lags leave dynamics unmodeled — the leftovers show up as serially correlated residuals and biased tests. Too many burn parameters fast (each extra lag costs k² = 9 coefficients here), inflating standard errors and wrecking forecasts. The referees are the information criteria, and their personalities matter:

- **AIC**: mild penalty. Inconsistent (can overshoot the true order forever) but avoids underfitting and tends to minimize forecast error.
- **BIC**: harsh penalty. Consistent — finds the true order asymptotically — but notoriously prone to *underestimating* the lag length in small samples.
- **HQIC**: the compromise candidate.

> **[TABLE 2 — notebook cell 32]** Lag-order selection for the growth-rate VAR: all four criteria agree on p = 1.

Here, no drama: everyone votes p = 1 (quarterly growth carries short memory). File the AIC-vs-BIC personality profiles away, though — later in this episode the criteria *will* disagree, and the tiebreak will matter. *(Small gotcha for statsmodels users: use `VAR(df).select_order()`, not the similarly-named `select_order` in the vecm module — that one selects difference-lags for a VECM on level data, a different question entirely.)*

The fitted VAR(1) summary rewards a slow read:

- In the **GDP equation**, the strongest regressor is not GDP's own lag — that one is actually *negative* and marginal (−0.34, t ≈ −2.0). It is **lagged consumption growth**: coefficient ≈ 0.75, t ≈ 5.7. Yesterday's consumption tells you far more about today's GDP than yesterday's GDP does. Hold that thought.
- **Consumption** listens mostly to itself (own lag ≈ 0.33) — smooth, self-propelled, hard for others to predict.
- **Investment** hangs on consumption's every word: lagged consumption enters its equation with a coefficient of ≈ 4.6 (t ≈ 6.7). Consumers pulling back is very bad news for next quarter's capex.
- And at the bottom of the summary, the **residual correlation matrix**: GDP–consumption residuals correlated at 0.61, GDP–investment at 0.76. These are *same-quarter* co-movements the lag structure cannot touch — parked in the residuals, unexplained. The final section of this episode is entirely about this parked correlation.

Does joint modeling actually buy forecasting power? We test it the only honest way (episode 8 supplied the sermon): a chronological hold-out of the last 20 quarters — 2004Q4 through 2009Q3, which means the validation window contains the financial crisis. Forecasting into 2008 is nobody's idea of a fair fight; that's what makes it a good test.

> **[TABLE 3 — notebook cell 38]** RMSE, single-equation AR(1) per series vs joint VAR forecast:

```
series   single RMSE   joint RMSE   joint better?
g_gdp       0.9272       0.9275         False
g_cons      0.7459       0.7451         True
g_inv       5.2876       5.2830         True
```

The verdict is honest rather than triumphant: the VAR edges out the univariate baseline on consumption and investment and loses a hair on GDP, every margin thin. Two lessons worth more than a fake landslide. First, with a crisis in the window, *all* linear forecasts miss badly — 2008 is exactly the break none of these models can see coming. Second, quarterly growth is mostly fresh news (the equation R²s sit at 5–15%), so cross-series lags can only do so much. **The VAR's value here is not forecasting power; it is that the fitted system is the substrate every causal question below is asked on.**

## Granger causality: precedence, formalized

A fitted VAR contains all the cross-lag coefficients; **Granger causality** is the formal question of whether a specific block of them is zero. The definition is operational, not philosophical. Compare two regressions:

```
restricted:    Y(t) = own lags of Y                      + ε(t)
unrestricted:  Y(t) = own lags of Y + lags of X          + ε(t)
```

and F-test whether X's lags jointly earn their keep. If they do, *X Granger-causes Y*: the past of X contains information about the future of Y beyond what Y's own past provides.

### The bivariate test — and a famous asymmetry

We ask the most classical pairwise question in empirical macro: between GDP and consumption, who leads whom? Testing at every lag from 1 to 4 (reporting the whole profile, not one cherry-picked lag — you'll see why shortly):

```
g_cons -> g_gdp :  p = 0.000, 0.000, 0.000, 0.000     (lags 1..4)
g_gdp  -> g_cons:  p = 0.091, 0.784, 0.369, 0.264
```

A strikingly clean asymmetry. **Consumption growth leads GDP growth** — unanimous across lags, p ≈ 0 — while GDP growth tells you essentially nothing extra about future consumption. The economics reads beautifully: households are *forward-looking*. Consumption responds to expected lifetime income (the permanent-income hypothesis), so shifts in consumer behavior front-run measured output rather than trail it. Consumption is an early sentiment reading on the whole economy.

Before celebrating: note what a bivariate test **cannot** rule out. A latent third variable — expectations, financial conditions, policy — driving both series with different delays would produce exactly this pattern with no direct link at all. This omitted-variable trap is the standing objection to every two-variable Granger result. Let's watch it bite on a different pair:

```
g_inv -> g_cons :  p = 0.015, 0.359, 0.853, 0.847     (lags 1..4)
```

Taken at face value, lag 1 says investment Granger-causes consumption (p = 0.015), and a story obligingly suggests itself — investment booms, jobs, spending. But look at the profile: the significance exists *only* at lag 1 and evaporates completely afterwards. A result this fragile to an arbitrary modeling choice should already raise an eyebrow — and there's an obvious confounder standing right outside the bivariate frame: the business cycle itself moves both variables. The cure is to test *inside* the full system.

### Conditional Granger causality — and the multiple-testing haircut

Instead of pairwise isolation, test each link within the fitted three-variable VAR: does X's past improve the prediction of Y **after controlling for everyone else's past**?

> **[FIGURE 3 — notebook cell 48]** Heatmap of conditional Granger p-values, rows = effect, columns = cause, dark green = strong link. Clear links: cons → gdp (0.000) and cons → inv (0.000); pale borderline cells: gdp → inv (0.014) and inv → cons (0.049); nothing predicts consumption.

The suspicious `inv → cons` link survived conditioning only barely — attenuated from p = 0.015 to p = 0.049, a three-fold jump. On simulated data with a known chain structure, such spurious links vanish outright; on real data, where everything is a little connected to everything, this is the typical outcome: the link weakens to the boundary, and your confidence should weaken with it.

But now count what we just did: **six hypothesis tests**, and we're cheerfully reading the cells that came out near p ≈ 0.05 as discoveries. At a 5% threshold, six tests carry roughly a one-in-four chance of at least one false positive even if no links exist at all. The bluntest fix is **Bonferroni**: divide the threshold by the number of tests.

```
tests run: 6    Bonferroni-adjusted threshold: 0.0083

links surviving the correction:
   g_cons -> g_gdp   (p = 0.0000)
   g_cons -> g_inv   (p = 0.0000)
```

The pruning is instructive: the two borderline links — precisely the ones we had already flagged as fragile — are exactly the ones that fall. What remains is a defensible little causal graph with consumption at the top of the food chain. That convergence (fragile across lags → attenuated by conditioning → killed by correction) versus the robustness of the consumption links is what an honest Granger analysis looks like. In genuinely high-dimensional systems — dozens of series, hundreds of ordered pairs — Bonferroni becomes too blunt an axe, and the modern toolkit switches to sparse VARs with false-discovery-rate control; the principle stays the same: **never read a matrix of p-values as if each cell were the only test you ran.**

## Cointegration: when trending series share a destiny

Sections above took the differencing fork: safe, stationary, and *blind to levels*. A growth-rate VAR can tell you consumption growth leads GDP growth; it cannot say anything about the *ratio* of consumption to GDP, because that information was destroyed by the first `diff()`. Long-run questions live in the levels — and the levels are a minefield. Let's step on a mine deliberately.

### The spurious regression trap

We regress log *real* GDP on the log *price level* (CPI) — real output "explained" by how expensive things are, a relationship with no serious economic content at this frequency.

> **[FIGURE 4 — notebook cell 53]** Two scatter plots. Left: log real GDP vs log CPI in levels — a tight, snaking, near-perfect relationship, R² = 0.958. Right: the identical regression on growth rates — a shapeless cloud, R² = 0.004.

```
R² in levels: 0.958        R² in growth rates: 0.004
```

Not a typo. Two variables whose main shared property is *drifting upward for fifty years* produce an R² of 0.96, complete with enormous t-statistics — and the moment you remove the trends, there is nothing there. This is the **spurious regression** phenomenon: when both series are I(1), the usual OLS distribution theory fails, and any two stochastic trends look profoundly related. It is the single best argument for the stationarity discipline of the earlier sections.

But notice the trap this leaves us in. Level regressions are radioactive; differencing destroys level information. How do we ever test a *genuine* long-run level relationship — like "consumption is a stable share of GDP"?

**Cointegration** is the escape hatch. Two or more I(1) series are cointegrated if some linear combination of them is stationary:

```
β′·X(t) ~ I(0)     even though each component of X(t) is I(1)
```

Individually they wander; *together* they are tethered by an equilibrium, and the combination measures the deviation from it. When cointegration holds, a levels relationship is meaningful after all — Granger's representation theorem even guarantees an error-correction structure (next section). The question is how to test for it.

### Engle–Granger: the transparent two-step

Step one: regress one level series on the other by OLS, keep the residuals. Step two: ADF-test the residuals. If the residuals are stationary, the regression line is a genuine long-run anchor — the two series never drift far from it. One technicality is baked into statsmodels' `coint()`: because OLS *chose* the combination that makes the residuals smallest, ordinary ADF critical values are too easy to beat; the corrected **Phillips–Ouliaris** distributions are used instead.

We run a matched pair of tests — one where theory predicts cointegration, one where it doesn't:

```
consumption vs GDP   (theory: stable consumption share):   p = 0.029   -> cointegrated
investment  vs M1    (no theory ties these levels):        p = 0.215   -> nothing
```

Exactly as ordered: the consumption–GDP pair rejects "no cointegration" (one of the classic "great ratios" of macroeconomics), while investment–money — both I(1), both trending — correctly comes up empty. Unlike the spurious OLS above, this test knows the difference between *trending together* and *tethered together*.

Engle–Granger's virtues are simplicity and interpretability. Its limits are structural: it handles only **one** cointegrating relation, it is **asymmetric** (swap which variable goes on the left and the residuals — and possibly the verdict — change), and first-stage estimation error contaminates the second stage. For a *system* of three variables, which can share up to two independent equilibria, we need the industrial-strength tool.

### Johansen: system-wide, and honest about it

The **Johansen test** treats all variables symmetrically in one maximum-likelihood framework and estimates the **cointegration rank** r: how many independent stationary combinations exist. Rank 0 → no cointegration (difference-VAR is correct); full rank → everything was stationary all along; in between → r genuine long-run relations. The **trace test** proceeds sequentially — test "r ≤ 0", if rejected test "r ≤ 1", and so on; the first non-rejection pins down the rank.

> **[TABLE 4 — notebook cell 57]** Johansen trace test on the GDP/consumption/investment log levels:

```
null           trace stat    5% critical    reject?
rank <= 0         30.64         29.80         True
rank <= 1         10.55         15.49         False
rank <= 2          2.96          3.84         False
```

Verdict: **rank 1** — a single long-run equilibrium binding the three series. But look at *how* the first rejection happened: 30.64 against 29.80, a margin under 3%. Knife-edge results deserve a robustness check, and the Johansen test is notoriously sensitive to two choices we made silently: the deterministic-term specification and the short-run lag count. So we do what the practitioner literature (the Pantula principle, formalized) says to do — rerun the verdict across specifications:

> **[TABLE 5 — notebook cell 59]** Selected cointegration rank across a 2×4 grid of specifications:

```
                k_ar_diff=1   k_ar_diff=2   k_ar_diff=3   k_ar_diff=4
det_order=0          0             1             1             1
det_order=1          0             0             0             1
```

The grid splits down the middle: allow a linear trend in the data (`det_order=1`) and the equilibrium mostly disappears. **The existence of a long-run relation among these three series depends on an auxiliary assumption about trends.** :-/

This is not a failed demo — it *is* the lesson, and it's exactly what fifty years of structurally shifting data should produce. The "great ratios" have genuinely drifted: consumption's share of GDP rose from ~63% to ~71% across our sample. Real-world macro cointegration is *borderline*, and an honest analyst reports it as such instead of quietly picking the specification that flatters the hypothesis. We proceed with the rank-1 reading — defensible, and we need a VECM to dissect — but we carry the fragility forward. It is about to become the best possible motivation for the section after next.

## VECM: dynamics tethered to equilibrium

Conditional on cointegration, the right model is the **Vector Error Correction Model** — a VAR in differences plus one crucial extra term per equation, the lagged *deviation from equilibrium*:

```
ΔX(t) = α·[β′·X(t−1)] + Γ·ΔX(t−1) + ... + ε(t)
```

Read it aloud: this quarter's changes respond to short-run dynamics (the Γ terms) **and** to how far the system strayed from its long-run relation last quarter. The equilibrium recipe is **β** — the cointegrating vector. Each variable's responsiveness to disequilibrium is its **α** loading, the *speed of adjustment*: the share of the gap closed per quarter, and — more interestingly — the answer to *who does the correcting*.

> **[TABLE 6 — notebook cell 62]** Estimated β and α for the rank-1 VECM:

```
            beta (equilibrium recipe)    alpha (adjustment speed)
log_gdp            1.000                        0.0003
log_cons          -6.448                        0.0025
log_inv            4.328                       -0.0145
```

Two stories in one table, one clean and one confessional.

The clean one is **α**: GDP essentially ignores the disequilibrium (0.0003 ≈ zero), consumption barely reacts, and **investment carries the adjustment** — a loading an order of magnitude larger. That is exactly the economics the very first plot promised: the volatile component absorbs the shocks; the smooth components set the anchor.

The confessional one is **β**. A textbook great-ratio equilibrium would carry weights near the component shares — something like `log_gdp − 0.7·log_cons − 0.2·log_inv`. Coefficients of −6.4 and +4.3 do not look like that; they are what maximum likelihood extracts when the underlying equilibrium is weak and drifting, as our sensitivity grid warned. Borderline rank evidence buys you imprecisely identified equilibrium coefficients. And there is a final exam any claimed cointegrating relation must sit: compute the combination `β′·X(t)` and *look at it*.

> **[FIGURE 5 — notebook cell 64]** The estimated error-correction term over 1959–2009, with its long-run mean as a dashed red line. Instead of busy oscillation around the mean, the series moves in slow multi-year swells — above the mean through most of the 1960s, sagging through the late 1980s and early 1990s — and exits the sample with a violent 2008–09 plunge that looks like anything but mean-reversion.

```
ADF p-value of the spread: 0.256
```

A healthy error-correction term is a coiled spring. Ours is a lazy river with a waterfall at the end, and the ADF test agrees with the eyeball: we cannot reject a unit root *in the very combination the model treats as stationary*. Verdict, stated plainly: over this half-century, the GDP–consumption–investment "equilibrium" is statistically fragile — borderline rank, specification-dependent, sluggish spread. A pairs trader would walk away (those decade-long swells are exactly the mean-reversion trades that ruin you). A macro-econometrician shrugs, notes that great-ratio drift is a known feature of the era, and reaches for a method that doesn't *require* resolving the question.

Which is the perfect setup, because that method exists, and it is criminally underused.

## Toda–Yamamoto: causality without the pre-test minefield

Recap our predicament, because it is the *usual* predicament. Granger tests need stationarity. Our levels are I(1). Differencing fixes that but discards the long run — and if the series *are* cointegrated, a difference-VAR is misspecified (it omits the error-correction term we just met). But whether they are cointegrated is, as we just spent two sections discovering, genuinely uncertain: unit-root tests have low power, and rank verdicts flip with the deterministic spec. Every route to a causality test requires first winning a pre-test we might lose — and losing silently invalidates everything built on top. Running standard Granger tests directly on I(1) levels is no escape either: the Wald statistic loses its chi-squared distribution (a non-stationary level term distorts the critical values) and manufactures spurious causality at well-documented rates.

**Toda and Yamamoto (1995)** cut the knot with a trick that is almost insultingly simple:

```
1. find d_max = the highest integration order suspected in the system (here: 1)
2. pick the lag order p for a VAR in LEVELS (criteria + residual checks)
3. fit an intentionally over-parameterized VAR(p + d_max) in levels
4. Wald-test ONLY the first p lags of the candidate cause; ignore the extra lags
```

The unrestricted augmenting lags soak up the non-stationarity; the tested block then behaves asymptotically as if everything were stationary — standard chi-squared inference, valid whether the series are I(0), I(1), or cointegrated in any combination. No pre-test to lose. The price is a mildly over-parameterized model. Cheap.

We deploy it on the most fought-over causal question in postwar macroeconomics: **does money cause income?** Friedman's monetarists read money-supply movements as driving output; the Keynesian rebuttal said money passively accommodates activity. Sims' 1972 paper on this exact question is how Granger causality entered macroeconomics in the first place. Our arena: log M1 versus log real GDP.

Step one, `d_max`: ADF says both levels are deep in unit-root territory (p = 0.82 and 0.38) and both first differences are clean (p ≤ 0.002). Both I(1), so `d_max = 1`. Note the role reversal: unit-root tests still run, but they now only size the buffer — get `d_max` generously wrong and you merely waste a lag; the test stays valid.

Step two, the lag order — and here the information criteria finally have the fight we foreshadowed: **BIC says 2, HQIC says 3, AIC says 6.** The tiebreaker is not taste; it is the residuals. A Granger-type test assumes the VAR has absorbed all serial dependence, so we run a whiteness (Portmanteau) test at each candidate:

```
p = 2:  residual whiteness p = 0.0004    <- BIC's choice leaves glaring autocorrelation
p = 3:  residual whiteness p = 0.0394    <- still failing at 5%
p = 6:  residual whiteness p = 0.5498    <- clean; AIC wins this one
```

BIC's small-sample underfitting bias, caught in the act — exactly the personality flaw the textbooks warn about. We take p = 6, so the fitted system is a levels VAR(7) with the seventh lag as the untested buffer.

Now the benchmark to beat: the *invalid* test, run deliberately. Standard Granger on the raw I(1) levels:

```
naive Granger on levels (do not trust):
   m1 -> gdp :  p = 0.091, 0.292, 0.079, 0.115     <- flirting with significance
   gdp -> m1 :  p = 0.673, 0.152, 0.209, 0.145     <- nothing
```

A monetarist with flexible standards could report "money causes income, significant at the 10% level". Remember both verdicts. Now the valid test — run at p = 6, and, because this episode has taught us to distrust single specifications, at p = 4 and 5 too:

> **[TABLE 7 — notebook cell 73]** Toda–Yamamoto results across lag choices:

```
lags   cause      effect     wald      p       causes?
 4     log_m1  -> log_gdp     3.6    0.459      no
 4     log_gdp -> log_m1     32.4    0.000      YES
 5     log_m1  -> log_gdp     3.3    0.651      no
 5     log_gdp -> log_m1    119.6    0.000      YES
 6     log_m1  -> log_gdp     9.3    0.160      no
 6     log_gdp -> log_m1    192.8    0.000      YES
```

Both verdicts **flip** relative to the naive test. Money → income, the direction the invalid test flirted with, evaporates (p = 0.16–0.65 across every lag choice) — that flirtation was precisely the spurious causality broken distribution theory produces. And income → money, which the naive test *missed entirely*, comes back overwhelming (Wald statistics up to 193, p ≈ 0). Non-stationarity doesn't just create false positives; it hides true positives.

The surviving direction tells a coherent story: money doesn't front-run output — economic activity drives money holdings. Income rises, households and firms demand more transaction balances, the banking system supplies them. In this sample, money is the passenger, not the driver. (One dataset, M1, bivariate — the literature has fought over each of those choices — but the methodological point stands regardless: the valid and invalid tests disagree *completely*, and only one has a defensible distribution theory.)

*Pro tip, or rather pro caveat: TY has a known blind spot. When causality flows primarily through the cointegrating relation itself — the long-run level channel — the augmenting lag absorbs it, and the test can hemorrhage power (simulations show rejection rates collapsing from 80% to 10% in strongly cointegrated systems). Read our m1 → gdp "no" as "no short-run predictive channel"; a purely long-run channel would be much harder for this test to see.*

## Structural VAR: the missing quarter

Everything so far shares one silent restriction: causes take *at least one quarter* to act. Every Granger-type statement is about lagged influence — effects inside the sampling interval are invisible to it. Worse: they are not absent from the data. Remember the residual correlations parked in the VAR summary (GDP–investment at 0.76)? Within a quarter, things clearly move together; the reduced-form VAR just refuses to say who moves whom.

The **Structural VAR** confronts this by positing contemporaneous structure:

```
B₀·X(t) = c + B₁·X(t−1) + ... + u(t)        u(t): orthogonal structural shocks
```

The matrix `B₀` encodes same-quarter causal links — and the data alone *cannot* estimate it, because "A moves B within the quarter" and "B moves A within the quarter" produce the identical covariance matrix. Extra assumptions must be purchased. The classic one is the **recursive (Cholesky) ordering**: arrange the variables from "slowest-moving" to "fastest-moving" and allow same-quarter influence to flow only downward. The ordering *is* the causal assumption — untestable from this data, imposed by the analyst, and consequential.

We demonstrate on the canonical monetary system: **output growth → inflation → the T-bill rate**, ordered exactly so. The logic: production decisions are sticky within a quarter (output reacts to nothing contemporaneously); prices respond to current activity but not to current policy; and the policy rate — set by a committee watching everything in real time — responds to both within the quarter. Reasonable, standard... and an assumption.

(One honest wrinkle first: the T-bill rate fails ADF (p = 0.28) while KPSS just barely tolerates it — our two-test battery flags a boundary case. A nominal rate can't literally be a random walk, but over Volcker-to-zero it is certainly *persistent*. The monetary-VAR literature keeps the rate in levels anyway, precisely because differencing would discard the policy signal; we follow the convention and log the assumption. SVAR work is an exercise in stacking defensible assumptions and knowing you did.)

AIC picks 6 lags. We compute **orthogonalized impulse responses** — the Cholesky factorization in our chosen order converts correlated reduced-form residuals into uncorrelated "structural" shocks, and we trace the economy's response to a one-standard-deviation surprise in the T-bill rate: the closest a 1959–2009 observational dataset gets to "what happens when the Fed unexpectedly tightens?"

> **[FIGURE 6 — notebook cell 78]** Impulse responses to a T-bill rate shock, 16 quarters, dashed 95% bands. Top: output growth — a brief positive blip, then negative from about two quarters out, staying mostly below zero for two years. Middle: inflation — *rising* for the first several quarters after the tightening. Bottom: the rate itself, decaying slowly back.

The output panel is the textbook story: tighter money bites with the famous "long and variable" lag, and the wide bands remind you macro effects are estimated, not etched. The middle panel is the gift that keeps on giving: inflation *goes up* after a contractionary shock. This is not a bug in the notebook — it is the celebrated **price puzzle**, one of the most replicated artifacts in empirical macro. The standard diagnosis: the Fed tightens when it *sees inflation coming* that our little three-variable system does not; the model then misreads anticipatory tightening as causing the inflation that follows. It is the omitted-variable trap from the Granger section, resurfacing at the structural level — richer systems add commodity prices or Fed forecasts to make it fade.

I could not have scripted a better closing exhibit for identification-by-assumption: the ordering was plausible, the code is correct, and the result still contains a well-understood distortion because *the system is too small for the assumption to be true*. Structural conclusions inherit the quality of their identifying assumptions. Always.

> **[FIGURE 7 — notebook cell 80]** Forecast error variance decomposition, three panels. Output-growth uncertainty stays ~86% own-shock at the four-year horizon. The T-bill rate panel is the interesting one: its own shock explains only ~31% of its long-horizon variance — roughly 50% is output shocks, ~18% inflation.

The FEVD asks the complementary question — at each horizon, what *share* of each variable's forecast uncertainty does each shock explain? The punchline sits in the bottom panel: the "policy instrument" is mostly *endogenous*. The Fed spends the bulk of its variance reacting to the economy rather than surprising it — which is what a central bank is supposed to look like in the data, and a fitting last word for an episode about disentangling who-drives-whom: **the variable everyone calls "the cause" is itself mostly an effect.**

## Closing time

We climbed six rungs on one real dataset. **VAR** modeled the system jointly (forecast gains honest but thin — the value was the substrate, not the RMSE). **Granger causality** formalized predictive precedence and delivered a genuinely robust finding — consumption leads GDP — while the conditional test and the Bonferroni haircut stripped away the links that were artifacts of isolation and multiplicity. **Spurious regression** showed why I(1) levels are radioactive (R² = 0.96 between real output and the price level!), **Engle–Granger** and **Johansen** offered the disciplined alternative and — this being real data — returned a borderline, specification-sensitive verdict, which the **VECM** dutifully converted into a clean adjustment story (investment does the correcting) wrapped around a confessed-fragile equilibrium. That very uncertainty was the launchpad for **Toda–Yamamoto**, which needs no pre-test victories and promptly flipped both verdicts of the invalid levels test on the money-income question. And the **SVAR** bought contemporaneous answers at the price of an ordering assumption — a price the price puzzle itemized in public.

The throughline, one sentence: *causal conclusions in time series are only as strong as the stationarity facts and identifying assumptions underneath them — establish the first, state the second, and stress-test both.* Beyond this ladder lies the modern causal-discovery frontier — sparse Granger networks with FDR control, identification via non-Gaussianity or nonlinearity, graph-discovery algorithms like PCMCI built for dozens of autocorrelated series — different machinery, same two commandments.

## Key Takeaways

1. **Predictive precedence ≠ manipulative causality.** Every method here answers "does X's past inform Y's future?" — not "what happens if we intervene on X?". Granger preferred "temporally related"; so should you.
2. **Stationarity structure dictates the toolkit.** I(0) → VAR/Granger/SVAR. I(1) → cointegration or Toda–Yamamoto. Test with ADF *and* KPSS — their opposite nulls make disagreement informative.
3. **Bivariate Granger results are guilty until conditioned.** Test inside the full system, report the whole lag profile, and correct for multiple testing — our fragile links failed all three checks; the real one passed all three.
4. **Cointegration verdicts on real data are often borderline.** Ours flipped with the deterministic specification, and the "stationary" spread failed its own ADF exam. Report the sensitivity grid, not just the flattering cell.
5. **When pre-tests are uncertain, use Toda–Yamamoto.** Extra untested lags buy valid inference across I(0)/I(1)/cointegrated — and on money vs income, the valid test reversed *both* naive verdicts. Mind the long-run blind spot.
6. **SVAR conclusions inherit their ordering assumption.** The price puzzle is what it looks like when a plausible identification meets a too-small system. No free lunch — least of all a structural one. ;-)

---

*United States of Banan is a reader-supported publication. To receive new posts and support my work, consider becoming a free or paid subscriber.*

[ ✓ Subscribed ]
