# Time Series with Konrad: episode 6

### Intermittent demand: from Croston to hierarchical Bayes and beyond

**KONRAD BANACHEWICZ**
FEB 23, 2026

Most of the forecasting we have done so far quietly assumed that *something happens every period*: passengers fly every month, savings change every quarter, electricity gets consumed every day. The series wiggle, they trend, they cycle - but they are never just flat zero for weeks at a time.

Walk into a real retail catalog and that assumption falls apart immediately. The best-selling items sell every day, sure. But the long tail - spare parts, niche SKUs, slow movers, anything seasonal or specialized - sells in a pattern that looks like this: zero, zero, zero, three, zero, zero, zero, zero, one, zero, zero… Mostly nothing, punctuated by the occasional spike. This is **intermittent demand**, and it dominates: spare parts, hospital supplies, and the bulk of any product catalog live here.

The trouble is that almost everything in the standard toolkit chokes on it. An ARIMA model fit to a series of mostly zeros learns to predict… mostly zeros, with confidence intervals that make no sense. Exponential smoothing flatlines. RMSE rewards a forecast of pure zero. We need a different way of thinking, and it turns out the right one has a single organizing idea running through it: **when a series is too sparse to trust on its own, borrow strength** - from its own past, from other items, from similar products. Everything below is a variation on that theme.

In this episode we climb a ladder of methods, each rung adding exactly one ingredient: Croston's classic decomposition, the TSB fix for obsolescence, a modern hierarchical-Bayesian extension (TSB-HB), then two practical applications - forecasting a whole catalog at once with gradient boosting, and the extreme cold-start problem of brand-new products with no history at all.

📓 *All code lives in the companion notebook:*
https://github.com/tng-konrad/time-series-with-Konrad/blob/main/m06-sales-demand-forecasting.ipynb

## When the data is mostly zeros

Before reaching for a method, it pays to name what we are looking at. Not all "spiky" series are alike, and the standard vocabulary is the two-dimensional classification of Syntetos, Boylan & Croston (2005), built from two numbers computed per series:

- **ADI** (Average Demand Interval): periods divided by periods *with* demand. ADI = 1 means a sale every day; ADI = 5 means, on average, one sale every five days. This measures intermittency **in time**.
- **CV²** (squared coefficient of variation of the *positive* demand sizes): variance of the nonzero demands over their squared mean. This measures volatility **in size**, ignoring the zeros entirely.

Two thresholds (ADI = 1.32, CV² = 0.49) carve the plane into four boxes:

| | CV² < 0.49 | CV² ≥ 0.49 |
|---|---|---|
| **ADI < 1.32** | smooth | erratic |
| **ADI ≥ 1.32** | intermittent | lumpy |

"Smooth" is the territory of episodes 2 and 3 - classical smoothing and ARIMA. The methods in this post target the bottom row: **intermittent** (regular sizes, irregular timing) and **lumpy** (irregular in both). When you run this classifier across a Walmart store from the M5 dataset, the overwhelming majority of items land in those two cells. This is not a niche problem you can wave away - it is most of the catalog.

We will use the M5 competition data (daily Walmart unit sales, California stores) throughout, starting with a single grumpy item - a hobby product that sells every few days - and scaling up from there.

> **[FIGURE 1 — raw intermittent series.** Daily sales of item `HOBBIES_1_288` in store `CA_1`, 2012–2015. Companion notebook, cell that plots `df.set_index('date').sales`. Shows the mostly-zero, occasionally-spiky pattern the whole episode is about.**]**

## Croston: occurrence × size

The foundational insight is due to Croston (1972): an intermittent series is really two signals braided together, and you should estimate them separately -

- the average **size** of demand *when it occurs*, and
- the average **interval** between occurrences,

then forecast demand-per-period as size divided by interval. Both are tracked with plain exponential smoothing, but - and this is the trick - **updated only on periods with positive demand**. On a zero day, nothing moves. With demand $X_t$, level $a_t$, smoothed interval $p_t$, and smoothing constant $\alpha$, on a demand day:

$$a_{t+1} = \alpha X_t + (1-\alpha)a_t, \qquad p_{t+1} = \alpha q + (1-\alpha)p_t, \qquad f_{t+1} = \frac{a_{t+1}}{p_{t+1}}$$

where $q$ counts periods since the last sale. On a zero day all three states freeze and $q$ ticks up.

Notice what the forecast *is*: a constant. Croston does not try to predict *which* day the next spike lands on - that is genuinely unpredictable - it aims at the correct **average demand per period**, which is exactly the quantity inventory planning needs. Against a spiky holdout the RMSE of a flat line looks unimpressive, but it is hitting the right target.

> **[FIGURE 2 — basic Croston, 28-day holdout.** Observed series (solid) vs. the flat Croston forecast (red dashed) over the final 28 days. Companion notebook, `plot_forecast(... 'Croston (α=0.4)' ...)`. Makes the "constant forecast aimed at the mean" point visually.**]**

The freezing-on-zeros behavior is the model's defining feature, and also its defining bug. Which brings us to the fix.

## TSB: fixing the obsolescence blind spot

Here is the problem with Croston: if an item stops selling, the forecast never reacts. A product that has gone dead for six months still carries the same cheerful demand estimate it had on its last good day, because the smoothers only update when something sells. Symmetrically, a sudden surge after a long quiet stretch should pull the forecast *up*, and Croston shrugs.

In 2011, Teunter, Syntetos and Babai proposed the repair now universally called **TSB**: replace the smoothed interval with a smoothed **probability of demand occurrence** $p_t$, and update it **every period** - including the zeros. The level still updates on demand days only, but the probability moves toward 1 on a sale and decays toward 0 otherwise, with its own constant $\beta$:

$$p_{t+1} = \beta + (1-\beta)p_t \;\text{ if } X_t > 0, \qquad p_{t+1} = (1-\beta)p_t \;\text{ otherwise.}$$

The forecast becomes a **product**, not a ratio: $f_{t+1} = a_{t+1}\,p_{t+1}$. *Caveat emptor* if you implement this yourself: it is probability times level (not divided, as in the original), and the forecast reads the states at $t+1$, not $t$. Get those backwards and the thing silently produces nonsense.

That single decaying-probability line is the whole improvement: stand TSB next to Croston after a long dry spell and the TSB forecast sits visibly lower, because its occurrence probability bled out through the zeros.

> **[FIGURE 3 — Croston vs. TSB, 28-day holdout.** Observed (solid) with both flat forecasts overlaid: Croston (red dashed) and TSB (green dashed). Companion notebook, the second `plot_forecast` call. The gap between the two lines is the obsolescence fix.**]**

But look at what *both* methods still share. Each series is fit in **complete isolation**. The output is a single point forecast - no distribution to set safety stock from. And everything hinges on hand-picked smoothing constants. A brand-new item with three weeks of history gets the exact same machinery as one with five years, and gets a terrible, noise-driven estimate out of it. The TSB-HB paper even notes how sensitive TSB is to the $(\alpha, \beta)$ pair - their benchmark grid-searches it. All of which sets up the appeal of a method that tunes itself, shares information across items, and emits a full distribution.

## TSB-HB: hierarchical Bayes does the borrowing for you

[Bai & Chu (2025)](https://arxiv.org/abs/2511.12749) keep TSB's multiplicative heart - for each item $i$, $\hat{Y}_i = \hat{\pi}_i \cdot \hat{S}_i$ (occurrence probability × expected size) - but rip out the per-series smoothers and replace them with a **fully generative hierarchical model**, estimated by empirical Bayes across the entire panel of items at once. Two layers.

**Occurrence — Beta–Binomial.** Each item's daily sale probability is a draw $\pi_i \sim \text{Beta}(\alpha, \beta)$ from a panel-level prior, and its count of demand days is $m_i \mid \pi_i \sim \text{Binomial}(n_i, \pi_i)$. The posterior mean has a lovely closed form:

$$\hat{\pi}_i = \frac{\alpha + m_i}{\alpha + \beta + n_i} = \lambda_i \frac{m_i}{n_i} + (1-\lambda_i)\,\mu_\pi, \qquad \lambda_i = \frac{n_i}{n_i + \phi}$$

That is a **shrinkage rule**: a blend of the item's own frequency $m_i/n_i$ and the panel mean $\mu_\pi$, weighted by how much evidence $n_i$ the item brings. Lots of history → trust the item. Little history → lean on the crowd.

**Size — Log-Normal with a random intercept.** Positive sizes are right-skewed, so they live on the log scale: $\log S_{i,t} \sim \mathcal{N}(\mu_i, \sigma^2)$, with item means themselves drawn from a panel prior $\mu_i \sim \mathcal{N}(\mu_0, \tau^2)$. Estimating the variance components by REML yields a second shrinkage rule with credibility weight $w_i = m_i\tau^2/(m_i\tau^2 + \sigma^2)$. Actuaries will recognize this instantly - it is Bühlmann credibility, the century-old idea of pricing a small portfolio by blending its own experience with the book average. The same mathematics, rediscovered for spare parts.

The hyperparameters $(\alpha, \beta, \mu_0, \tau^2, \sigma^2)$ are fit *once*, by maximizing marginal likelihood over all items - a couple of tiny L-BFGS-B runs. After that, every item's forecast is a closed-form one-liner. Why this is worth the trouble:

- **Sparse items borrow strength automatically.** An item with three sales gets pulled toward the panel average instead of trusting its own noise - and the pull scales with the evidence, no thresholds, no tuning.
- **Cold start is built in.** Set $m_i = 0$ and the formulas gracefully return the panel-level prediction. An item with no history is forecast entirely from the prior, with the full prior uncertainty attached.
- **No smoothing constants.** The model tunes itself from the data.
- **It is generative**, so it produces full predictive *distributions* - quantiles, prediction intervals - not just a point.

Fit all three methods on a 200-item panel and compare on a common 28-day holdout (per-series, because aggregate numbers can hide a model winning big on a handful of series while quietly losing everywhere else):

| method | mean RMSE | median RMSE |
|---|---|---|
| Croston | baseline | baseline |
| TSB | ≈ Croston | ≈ Croston |
| TSB-HB | **lower** | **lower** |

The honest summary: TSB-HB does **not** demolish the classical methods. On data-rich series, shrinkage barely moves the estimates - by design. What it does is win *consistently*, with the margin concentrated exactly on the sparse, lumpy items where per-item estimates are noisiest. On the paper's M5 subset that consistency adds up to roughly a **6.1% MAE and 10.5% RMSE** improvement over tuned TSB - and remember TSB here got hand-set constants while TSB-HB tuned itself.

## What shrinkage actually does

The single most instructive picture in the paper (Figure 1, reproduced in the notebook) plots each item's raw maximum-likelihood estimate on the x-axis against its hierarchical posterior on the y-axis, with the diagonal marking "no shrinkage." Two panels, and they look completely different.

> **[FIGURE 4 — what shrinkage does (two panels).** Left: per-item occurrence frequency (MLE) vs. hierarchical posterior probability. Right: per-item mean size (MLE) vs. shrunken posterior. Red dashed diagonal = no shrinkage. Companion notebook, the side-by-side scatter cell. Left panel hugs the diagonal; right panel bends toward the center.**]**

For **occurrence probability**, every item contributes all ~1,800 training days of evidence, so the credibility weight is essentially 1, the points hug the diagonal, and shrinkage is barely visible (the paper reports correlation 0.9987, variance down only 11%). For **size**, the evidence is only the *demand days* - for a lumpy item, maybe a dozen observations - so shrinkage bites hard: extreme points bend visibly toward the global mean, correlation drops, and cross-item variance shrinks substantially (0.9194, −18.6%).

That contrast *is* the explanation for why TSB-HB wins on lumpy series. Not magic - just a model that refuses to trust five noisy observations more than they deserve, applying exactly as much regularization as the data shortage warrants.

## Read the band, not the line

The capability the classical methods simply cannot offer: a **distributional** forecast. Because TSB-HB is generative, getting quantiles is trivial - simulate from it. Draw a Bernoulli occurrence with probability $\hat\pi_i$, draw a log-size, exponentiate, multiply, repeat a few thousand times, read off the empirical quantiles. The distribution is **zero-inflated by construction**: a fraction $1-\hat\pi_i$ of the draws is exactly 0.

Plot the observed holdout against the median and the 10–90% band for a typical intermittent item and the lesson jumps out: **read the band, not the line**. The *median* often sits at exactly zero - on most days, most likely, nothing sells, and the model says so without flinching. The operationally useful number is the upper quantile: q90 answers "how much stock covers demand on 90% of days," which feeds straight into a service-level policy. The paper's calibration results show these intervals hit their nominal coverage at roughly a **third** of AutoARIMA's interval width - sharp *and* honest. A point forecast would have mumbled "0.4 units/day" and left the safety-stock question entirely unanswered.

> **[FIGURE 5 — probabilistic forecast.** Observed holdout (solid) vs. TSB-HB median forecast (green dashed with markers) and the 10–90% band (coral bounds with shaded fill), for the series where TSB-HB beat TSB most. Companion notebook, the quantile-forecast plot. The median pinned at zero with a meaningful q90 is the whole point.**]**

So where are we on the ladder? Croston gave us occurrence × size; TSB made it robust to obsolescence; TSB-HB made it self-tuning, panel-aware, and probabilistic, all in closed form. Its one remaining limitation is structural: the forecast is *constant over the horizon*, with no calendar awareness - no weekday effects, no promotions, no prices. Capturing those needs features. Which is the next section.

## Forecasting the whole catalog at once

The Croston family - TSB-HB included - produces a flat per-series forecast and ignores the calendar. For a full catalog there are also two brute practical facts: there is a *lot* of store × product combinations, and the series share structure (weekly cycles, holiday effects, category dynamics) that per-series models cannot exploit beyond the intercept-level pooling TSB-HB does.

The industrial-strength alternative - the one that actually won M5 - is to stop thinking "time series" and start thinking **supervised regression**: stack every series into one big table, engineer features that encode each series' recent past and the calendar, and fit a single gradient-boosting model on everything jointly. Two feature families do most of the work:

- **Lag features**: yesterday-shifted-by-28 demand, grouped by series so the shift never leaks across boundaries. We use lag 28 - exactly the horizon - so the feature is available for every day we predict, with no recursive feeding of predictions into features.
- **Rolling means of those lags** over 7- and 28-day windows: a stable recent-demand-level signal. This is conceptually the same job Croston's smoothed level performed, now learned jointly with everything else.

Add calendar features (weekday, week, month, quarter, day-of-month - the structure the smoothers were blind to) and integer-encoded item/store/category identifiers, and a *single global model* recovers item-specific behavior through those IDs - the regression analogue of per-item parameters.

The split is - once again, and always - **chronological**. Shuffle a time series at your peril: it leaks future information through the lag features and turns your validation score into a comforting lie.

On the loss function: look at a kernel density of daily sales and you see a giant spike at zero with a long right tail - intermittency, at the panel level. That shape disqualifies plain squared error immediately. Two count-aware objectives fit better:

> **[FIGURE 6 — target distribution.** Kernel density estimate of daily sales across the training panel: a tall spike at zero with a long right tail. Companion notebook, `ytrain.plot.density()`. This is the picture that rules out plain squared error and motivates Poisson/Tweedie.**]**

- **Poisson** - a natural first choice for "units per day," predicting nonnegative rates.
- **Tweedie** - and this one is beautiful. For power $1 < p < 2$ the Tweedie distribution is a *compound Poisson–Gamma*: a Poisson number of events, each with a Gamma-distributed size. That is, almost verbatim, the occurrence × size decomposition the entire Croston family is built on - now expressed as a loss function. It puts an exact point mass at zero plus a skewed continuous part, matching the data better than Poisson, and it was the loss of choice among top M5 solutions.

Feature importance, as always on this kind of data, is topped by the **rolling means of lagged sales** - recent demand level, echoing what every smoother in this post estimates - followed by item identity and the weekly cycle. (Importance plots double as a debugging tool: a feature you expected to matter sitting at zero usually means a construction bug, and an implausibly dominant one is the classic fingerprint of target leakage.)

> **[FIGURE 7 — LightGBM feature importance.** Split-count importance from the Tweedie model. Companion notebook, `lgb.plot_importance(m_tweedie)`. Rolling means of lagged sales on top, then item id and calendar features.**]**

The price of all this power is history: the model needs a past to build its lag features from. Which raises the last question of the episode.

## New launches: cold start with no history at all

What about a product that has *never sold* - launching next season, no rows in the panel at all? TSB-HB handled an item with zero sales *inside an existing panel* (forecast the prior), but a brand-new fashion item has no panel membership to lean on. All we have are its **attributes** - category, color, fabric, release date - and the early-sales curves of *previous* launches.

So we learn the mapping *attributes → 12-week sales curve* from past launches and apply it to new ones. (The split here is **between products**, the cold-start analogue of our chronological split.) Two approaches:

- **Multi-output ridge regression**: one-hot the attributes, fit twelve independent ridge regressions (one per week of the curve). The baseline any fancier method must beat. It recovers the *shape and rough level* of the launch curve - the early peak and decay typical of fashion - but two products with identical attributes get identical predictions. For pre-launch inventory commitment, shape and level are exactly what is needed.

> **[FIGURE 8 — attribute-only cold-start forecast.** Three example test products: real 12-week curve (solid) vs. multi-output ridge prediction (red dashed). Companion notebook, the first new-launch plotting loop over products `[0, 12, 80]`. Shape and level recovered, product-specific deviations not.**]**

- **Embeddings + clustering**: learn dense vectors for the categorical attributes with *cat2vec* (treat each product's attribute list as a sentence, run Word2Vec, so co-occurring attributes land nearby), cluster past products in that space, and forecast each new product's curve as the **average curve of its cluster**.

Squint at that second method and you will recognize the thesis of the whole episode one last time: it is **pooling** again. A new product borrows the history of similar past products, just as TSB-HB's cold-start item borrowed the panel prior. The only difference is that "similar" is now defined by learned embeddings rather than membership in a global pool - the borrowing has gone *local*. Both methods agree on the broad shape and differ mostly in smoothness; neither predicts product-level surprises, and both deliver a defensible pre-launch curve from attributes alone.

> **[FIGURE 9 — cold-start methods head-to-head.** Same three test products: real curve (solid), multi-output ridge (red dashed), cluster-average (green dashed). Companion notebook, the final new-launch plotting loop. The two methods agree on shape; the cluster average is smoother.**]**

## Closing time

We climbed a ladder, and the rungs were a single idea getting steadily more sophisticated. **Croston** decomposes intermittent demand into occurrence and size and smooths each. **TSB** repairs its obsolescence blind spot by smoothing a per-period occurrence probability instead of an interval. **TSB-HB** keeps that multiplicative heart and rebuilds it as a generative hierarchical Bayesian model - Beta–Binomial occurrence, Log-Normal sizes, empirical-Bayes shrinkage across the panel - buying self-tuning estimates that are robust where data is sparse, principled cold-start behavior, calibrated probabilistic forecasts, and closed-form $O(N)$ computation. **LightGBM** with a Tweedie objective handles thousands of series jointly once you engineer lag, rolling, and calendar features (and split chronologically, always). And for **new launches** with no history, you predict the curve from attributes - ridge as the baseline, embeddings-plus-clustering as the pooling-flavored alternative.

One idea stitched all of it together, and it is worth saying plainly because it generalizes far beyond demand: **when per-series data is weak, borrow strength** - across time through smoothing, across items through hierarchical shrinkage, or across similar products through embeddings and clusters. The zeros in the data are not the enemy. The enemy is pretending each sparse series must stand entirely on its own.

In the next episode we will stay in the world of many related series, but turn to models that learn shared structure directly across them - global deep-learning forecasters. Stay tuned.

---

*United States of Banan is a reader-supported publication. To receive new posts and support my work, consider becoming a free or paid subscriber.*
