# Time Series with Konrad: episode 8

### Validation methods: leakage, purging, and other ways to stop predicting the past

**KONRAD BANACHEWICZ**
JUL 20, 2026

Every episode in this series has ended the same way: fit a model, hold out some data, report an error metric. This episode is about the step we've been taking for granted — because the dangerous assumptions are the ones you don't notice, and the most dangerous one in all of forecasting hides inside the innocent phrase *"hold out some data"*.

For ordinary tabular problems, holding out data is routine: shuffle, slice off a random 25%, or run K-Fold, and read off the score. Those recipes rest on the assumption that your rows are independent draws from one static distribution — exchangeable tiles pulled from a bag (episode 7 readers have met this bag before). Time series are the opposite of a bag. The order *is* the information: yesterday tells you about today, trends carry over, and the world at time `t` was built from everything before it and nothing after. Physics calls this the **arrow of time**; econometrics translates it into one non-negotiable rule — *cause precedes effect, so a forecast made at time `t` may only use information available at or before `t`.*

↪ *That "bag" of interchangeable draws is the i.i.d. assumption conformal prediction leaned on in episode 7 to build honest prediction intervals — this episode is what happens when the bag turns out to be a lie.* → **<LINK TO EPISODE 7 HERE>**

Violate that rule during validation — even indirectly, through a shuffled row, an overlapping label, or a rolling feature computed across a split boundary — and you are no longer measuring forecasting skill. You are measuring your model's ability to **predict the past**. It will look magnificent at it. The failure mode has a name, **data leakage**, and a signature: a beautiful validation score, followed by a collapse in production, followed by a meeting you don't want to attend.

**A validation score is a claim about the future made from inside the past — and every shortcut in the splitting scheme is a loan taken out against deployment.**

So this episode does what we always do: climb a ladder, one rung at a time, each rung fixing exactly one leak the previous rung missed. Random split and K-Fold (the broken baselines), walk-forward validation (**TimeSeriesSplit**), **GroupTimeSeriesSplit**, **PurgedGroupTimeSeriesSplit**, and finally **Combinatorial Purged Cross-Validation (CPCV)**. And because "trust me, it leaks" is not an argument, we'll run every rung on real data with a sealed-off future, so we can *measure* how honest each method's claim turns out to be. The goal, as always, is not mathematical rigor but practical understanding.

📓 All code lives in the companion notebook:
https://github.com/tng-konrad/time-series-with-Konrad/blob/main/m08-fable-version.ipynb

## Groundwork: manufacturing a future

To compare validation methods we need one luxury that practitioners never have at decision time: access to the future. We manufacture it the only honest way — by cutting off the final 90 days of a real dataset and sealing them away. Everything before the cut is "the present", where the validation methods operate; everything after is "the future", touched exactly once per method to reveal how good its claim was.

The dataset is the classic **store-item demand** set from Kaggle: five years (2013–2017) of genuine daily unit sales for 50 items in one retail store — the same flavor of retail panel as the M5 data from episode 6, just small enough to run a few dozen model fits without leaving your desk. About 91,000 rows: one row per (date, item).

↪ *This panel is a bite-sized cousin of the M5 retail-demand data we forecasted end-to-end in episode 6 — start there for the full sales-forecasting workflow this one stress-tests.* → **<LINK TO EPISODE 6 HERE>**

The *structure* matters as much as the content. This is **panel data**: at every timestamp we observe many entities at once — exactly like the Ubiquant market-prediction dataset (one `time_id`, many `investment_id`s) that motivates much of the financial validation literature. Items in the same store share a common environment: the same promotions, the same weather, the same seasonal shopping rhythm. Their sales on any given day are *contemporaneously correlated*. Hold that thought; it will break something later.

> **[FIGURE 1 — notebook cell 20]** Two stacked panels. Top: total daily sales across all 50 items, 2013–2017, with a dashed green line marking the hold-out boundary near the end of 2017. Bottom: three individual items (1, 15, 28) plotted over the same span.

Three features jump out of the top panel, each with a consequence for validation. First, an **upward trend** — 2017 runs visibly higher than 2013. The mean changes over time, which is the textbook definition of **non-stationarity** (episode 3 veterans will remember the Dickey–Fuller ritual). A model trained on early years faces a future sitting at levels it has never seen — so "how hard is the future?" genuinely depends on *which* future, something a scheme that scrambles time can never respect. Second, a **strong yearly cycle**: summer peaks, winter troughs, meaning performance depends on *where in the calendar* a validation block happens to fall — a first hint of why a single train/test path can mislead. Third, the fine weekly sawtooth. And in the bottom panel: three items at different levels, all dancing to the same rhythm. That's the contemporaneous correlation, visible to the naked eye.

↪ *The machinery behind "non-stationarity" — ADF, KPSS, differencing, the whole Dickey–Fuller ritual — got its thorough treatment in the ARIMA episode, and it's the prerequisite for seeing why this data leaks.* → **<LINK TO EPISODE 3 HERE>**

> **[FIGURE 2 — notebook cell 22]** Bar plot of the autocorrelation of total daily sales at lags 0 through 30. High across the board, with pronounced ripples at lags 7, 14, 21, 28.

The autocorrelation function says the rest (reminder: episode 1). Correlations stay high across all 30 lags — knowing today's sales tells you a lot about next week and even next month — so adjacent observations are **highly redundant**. This is exactly the property that breaks the i.i.d. assumption underlying shuffled splits: put today in training and tomorrow in validation, and you've handed the model most of the answer. The ripples at multiples of 7 confirm the weekly cycle. Strongly serially correlated *and* non-stationary: this is the profile of data on which naive validation fails hardest.

Now, the supervised problem. Features are the natural ones: day-of-week, month, a trend index, today's sales, sales 7 and 14 days ago, and a 28-day rolling mean of past sales. The target is the **average daily sales over the next 7 days** — a realistic demand-planning quantity. Two numbers in that setup deserve to be pulled out and framed, because the entire second half of this post is about them:

- The **label looks 7 days ahead.** The target attached to day `t` is computed from sales on days `t+1 … t+7`. It has a *look-ahead footprint*: it physically contains a week of the future. Two rows one day apart share 6 of their 7 label days — their targets are almost the same number.
- The **rolling feature looks 28 days back.** It has a *look-back footprint*: it summarizes a month of history into one number.

Every leak we hunt below is one of these two footprints, or the raw serial correlation, crossing a split boundary.

Finally, the ground truth. We train one gradient-boosted tree model (a fixed, deliberately un-tuned `HistGradientBoostingRegressor` — we're comparing validation schemes, not chasing leaderboard points) on the development period and score it once on the sealed 90 days:

```
True generalisation RMSE: 5.4454
```

In units of daily sales per item — that's the number every validation method below is trying to estimate *without ever seeing the hold-out*. One subtlety in how we fit this reference, because it foreshadows everything: the labels of the last 7 development days are computed from hold-out-period sales, which means that at deployment time *those labels would not exist yet*. Training on them would be a small act of time travel. So the honest reference fit drops them. Seven days out of 1,700 — numerically negligible here, but it's the same principle that returns at full scale later under the name *purging*.

Our scorecard for each method is a single number, **optimism**:

```
optimism = validation claim − true test error
```

Negative optimism is the dangerous direction — the method claimed things were better than they are, and the size of the gap is a direct measurement of the leaked information. Positive is conservative: wasteful, but safe. Near zero is the goal.

## The wrong way: shuffled validation

We start at the bottom of the ladder, with the two defaults of the tabular-ML world.

### Random split

Hold out a random 25% of rows, train on the rest. The verdict:

```
Random-split validation RMSE: 3.275
True generalisation RMSE    : 5.4454
Optimism                    : -2.1704
```

The random split claims an error of 3.3 when the truth is 5.4 — it underestimates by roughly **40%**. If this were a real project, you would ship this model believing it nearly twice as accurate as it is. :-/

> **[FIGURE 3 — notebook cell 34]** Zoomed timeline strip of the first ten days (500 rows, white lines separating consecutive days): blue training ticks and orange validation ticks interleaved *inside every single day*. The pattern repeats across all five years.

Where does the fantasy come from? Two leaks, both flagged in the groundwork, both now cashed in. First, **serial correlation**: for almost every validation row, rows from the same item just days away sit in the training set — and with lag and rolling-mean features, those neighbors are nearly duplicate questions with known answers. Second, and worse, the **overlapping labels**: train on Monday's row and you have effectively seen ~86% of Tuesday's label. The model doesn't need to forecast anything; pattern-matching its own training set is enough.

The gap between 3.3 and 5.4 is not noise, and it is not model instability. It is the **market price of leaked future information** — the performance boost purchased by violating causality. And the damage runs deeper than one bad number: any hyperparameter sweep or feature selection run against this score will systematically favor the models that are best at *exploiting the leak*, not the ones best at forecasting. The validation scheme doesn't just misjudge your model; it steers you toward the wrong one.

### K-Fold

K-Fold is the respectable cousin: partition into 5 folds, each takes a turn as validation, average the scores. Averaging genuinely reduces the *variance* of the estimate. Unfortunately, variance was never the problem:

```
K-Fold (shuffled) mean validation RMSE: 3.2498
Optimism                              : -2.1956
```

The same fantasy number, now delivered with five-fold confidence — a badly biased estimate with a reassuringly small standard error. The tight agreement between folds is not robustness; it's five folds sharing one leak.

> **[FIGURE 4 — notebook cell 39]** Fold-anatomy heatmap, one row per fold, one column per day. Blue = day used purely for training, orange = purely validation, violet = *the day itself is split between the two*. All five rows are wall-to-wall violet.

Not one of the ~1,700 days, in any fold, sits cleanly on one side of the boundary: a shuffled draw takes about ten of each day's fifty rows for validation and trains on the other forty. There is no arrangement of time here at all — only a uniform mist of contamination. Keep this picture in mind; the rest of the post is, visually, the process of separating that violet into clean blue and clean orange.

*One honest footnote before we move on:* there is a legitimate theoretical result (Bergmeir and co-authors) that standard K-Fold *can* be valid for time series — for a **stationary** series, with a purely autoregressive model whose errors are uncorrelated. The exception is real. Our data flunks the entry requirement on every count — trend, seasonality, serial correlation, and labels that mechanically overlap — and so does essentially every financial and retail panel in the wild. Know the exception; don't bet the backtest on it.

Can we do better? (You know the drill.)

## Walk-forward validation: respecting the arrow of time

Everything from here on obeys one rule: **in every split, all training rows come strictly before all validation rows.** The foundational implementation is scikit-learn's **TimeSeriesSplit** — walk-forward validation, a.k.a. rolling-origin evaluation. Slice the timeline into consecutive segments and roll the forecasting origin forward: train on segment one, validate on segment two; train on one-and-two, validate on three; and so on. Every fold is a miniature dress rehearsal of deployment.

```
TimeSeriesSplit (expanding) mean validation RMSE: 5.6934
Optimism                                        : +0.2480
```

Night and day. The estimate jumps from the shuffled fantasy of 3.3 to 5.7 — within a few percent of the truth, and on the *conservative* side rather than the optimistic one. **No other single change in this post buys as much honesty as simply refusing to train on the future.** The residual conservatism has honest causes: early folds train on only a fraction of the history, and on trending data every fold genuinely faces the deployment problem of predicting a level slightly above anything it has seen. That difficulty is real — the sealed future poses it too.

> **[FIGURE 5 — notebook cell 45]** The same fold-anatomy heatmap, now a clean staircase: in every fold a solid blue training block sits entirely to the left of a solid orange validation block, with the unused future in grey to the right. Each successive training block is longer than the last.

The staircase is the expanding-window variant: each fold trains on *all* history accumulated so far. `TimeSeriesSplit` offers a second flavor via `max_train_size` — a **sliding window** that keeps only the most recent observations and drops the oldest as it rolls forward. The choice between them is not a technicality; it is an implicit hypothesis about your data:

- **Expanding window:** uses maximum data; bets that the data-generating process is stable and old history still teaches. Risk: archaic regimes polluting the fit when the world has moved on.
- **Sliding window:** adapts to drift by forgetting; bets that recent history is what matters (the usual bet in financial markets, where relationships decay). Risk: starving the model of long-cycle structure.

On our data, the bet resolves cleanly. A one-year sliding window gives:

```
TimeSeriesSplit (sliding, 1y) mean validation RMSE: 6.7044
Optimism                                          : +1.2590
```

Distinctly pessimistic — a one-year window has seen each calendar season exactly once, and a model asked about next summer benefits from having seen four summers, not one. Retail demand is stable enough that history helps; your mileage on tick data will differ. **Test both windows: the data will tell you which hypothesis it rewards.**

So are we done? Walk-forward respects chronology, the estimate is nearly honest… time to check the fine print. `TimeSeriesSplit` is honest about *rows* — but our data has 50 rows per day, and scikit-learn cuts the row sequence wherever the index arithmetic dictates. A quick check of whether any single day ends up on both sides of a fold boundary:

```
Days appearing in BOTH train and validation (per fold):
  fold 0: [283]
  fold 1: [566]
  fold 2: [850]
  fold 3: [1133]
  fold 4: [1416]
```

Every fold slices exactly one day in half — some of that day's items in training, the rest in validation.

## GroupTimeSeriesSplit: a day is never cut in two

Why care about a single day? Because of the contemporaneous correlation from Figure 1. Items in the same store on the same day share demand shocks — the same weather, the same holiday, the same footfall. Training on 30 items from a given day tells the model a great deal about the other 20 items of *that same day* sitting in validation. The model gets graded partly on questions whose context it has already seen. This is **intra-group contamination**: not forecasting, but cross-sectional interpolation wearing a forecasting costume.

On our data the damage is one day per fold — small. But scale the structure up: in the Ubiquant dataset there are thousands of investments per `time_id`, and a row-level cut through a timestamp leaks an entire market snapshot into the training set. The model happily learns that "when investment A moved up at `time_id` 100, investment B did too" — a genuine correlation with zero *predictive* content, since both observations describe the same instant.

The fix is a splitter that combines the chronology of `TimeSeriesSplit` with the group discipline of `GroupKFold`: all rows sharing a timestamp travel together, wholly in training or wholly in validation. Scikit-learn has no built-in that does both, so the notebook rolls its own **GroupTimeSeriesSplit** (about twenty lines — the group label is just "which day does this row belong to"). The boundary check now returns empty arrays at every fold, and:

```
GroupTimeSeriesSplit mean validation RMSE: 5.5291
Optimism                                 : +0.0837
```

The closest single-number claim of any method in this post — under a tenth of an RMSE unit from the truth.

> **[FIGURE 6 — notebook cell 56]** The fold-anatomy heatmap again: visually the same staircase as Figure 5, but every blue/orange boundary is now a clean one-pixel transition — no violet column of a split day at any edge.

The picture is almost identical to the previous one, *which is the point*: the fix is microscopic in the image and decisive in principle. For panel data, the report's framing is exact and worth engraving somewhere visible: **group-aware splitting is not an upgrade, it is the entry requirement.**

One more thing before moving on. This time we kept the per-fold scores instead of just the mean:

```
Per-fold RMSE: [5.79, 4.51, 4.39, 6.24, 6.71]
```

A 50% swing between the easiest and hardest fold, depending on which stretch of the calendar each one validates. File that away — it becomes the central topic two sections from now.

## PurgedGroupTimeSeriesSplit: minding the footprints

Chronology: enforced. Group integrity: enforced. Still not enough — because our *label* has a temporal footprint, and footprints don't respect boundaries.

Recall: the target at day `t` is the average of sales over days `t+1 … t+7`. Now look at any train/validation boundary. The last seven training days' labels are computed from sales that occur *inside the validation block*. The model is being trained on numbers that are, in part, summaries of the very period it is about to be graded on. This is precisely the leak we quietly removed from the reference fit back in the groundwork — and the general cure is **purging** (popularized by Marcos López de Prado in the context of financial labels like the triple-barrier method, where every label depends on a future price path): *drop every training observation whose label footprint overlaps the validation period.*

Purging has a sibling, the **embargo**, guarding the opposite direction: features with a *look-back* footprint (our 28-day rolling mean) computed for days just *after* a test block would smuggle test-period sales into any subsequent training set. In a forward-only walk that direction never occurs — training data never follows validation — so the embargo stays dormant for now. It wakes up one rung from here.

Implementation-wise this is one parameter: a `group_gap` of days dropped from the end of training, immediately before each validation block. *Pro tip: the gap is not a knob to fiddle by feel — it must be at least the label's look-ahead footprint (here, 7 days). Shorter leaves residual overlap; much longer throws away clean data. If you cannot state your label's footprint, you do not yet know your leak.*

```
PurgedGroupTimeSeriesSplit (gap=7d) mean validation RMSE: 6.2276
Optimism                                                : +0.7822

Per-fold RMSE, unpurged: [5.79, 4.51, 4.39, 6.24,  6.71]
Per-fold RMSE, purged  : [5.78, 4.78, 4.60, 5.95, 10.03]
```

> **[FIGURE 7 — notebook cell 63]** The staircase heatmap once more, now with a thin grey seam — seven days wide against a 1,700-day axis, so look closely — between the end of each blue training block and the start of each orange validation block.

Two lessons here, one honest and one cautionary — and the honest one first, because this is the section where a less scrupulous post would oversell. On *this* dataset, purging changes the verdict only modestly, and the leaked rows it removes (7 days × 50 items per fold, out of tens of thousands) were too few to have inflated the unpurged scores much. The comprehensive report behind this episode says the same: versus plain group splitting, the improvement is often marginal. But the contamination fraction scales with `horizon / block_length`. Give your labels a 30-day footprint and your validation blocks a few weeks — routine numbers in finance — and the leaked share of training data stops being a rounding error. Purging is cheap insurance whose payout grows with exactly the problems you can't see coming.

The cautionary lesson hides in fold 4. Folds 0–3 barely move under purging, but fold 4 jumps from 6.7 to 10.0 — on a *seven-day* change in training data. What happened? Fold 4's validation block is the long final stretch of the development period, the part sitting highest on the five-year trend, and a tree ensemble extrapolates a trend essentially by carrying its most recent mapping forward. That makes the freshest training week disproportionately load-bearing: withhold it, and the model's systematic under-prediction of the trending 2017 levels roughly doubles. (Carve the same seven days out of the *middle* of the training set instead, and the score barely moves — it's recency fold 4 misses, not quantity.)

Whatever the mechanism, notice what it implies: **a single-path estimate is hostage to where its boundaries happen to fall on the calendar.** No amount of purging fixes that, because it isn't a leak — it's a sampling problem. Which brings us to the top of the ladder.

## CPCV: beyond a single path

Every method so far — however careful about chronology, groups, and footprints — evaluates the model along **one** historical path: train on the past, test on what follows, boundaries at the same dates every time. The resulting number is conditional on that one arrangement. We've now seen the symptom twice: per-fold scores ranging from 4.4 to 6.7 depending on the season under test, and one fold swinging by three RMSE units when its boundary shifted a week. If your one path happens to test on benign stretches, you overestimate the model; on brutal ones, you underestimate it. And if you *tune* against that single path, you are quietly overfitting the backtest itself — the strategy starts fitting the one history you happened to record.

**Combinatorial Purged Cross-Validation** breaks the single-path dependence with a beautifully blunt idea: stop treating history as one sequence, and treat it as material for many *alternate histories*. The mechanism:

1. Cut the timeline into `N` sequential blocks of whole days.
2. Choose every possible combination of `k` blocks as the test set; train on the rest.
3. Around every test block, defend both boundaries: **purge before** it (sized by the label's 7-day look-ahead) and **embargo after** it (sized by the feature's 28-day look-back — awake at last, because training data can now legitimately sit *after* a test block).

The bookkeeping bonus is that the test blocks chain together into complete backtest paths, and their number has a closed form:

```
φ(N, k) = (k / N) · C(N, k)
```

With `N = 6` and `k = 2`: `C(6,2) = 15` folds to fit (a threefold increase in compute over our 5-fold walks — CPCV's reputation for expense is earned), stitching into `φ = 5` complete alternate backtests. Every previous method gave us exactly one.

> **[FIGURE 8 — notebook cell 72]** The fold-anatomy heatmap for all 15 CPCV folds. Orange test blocks appear at every position along the timeline, not just the trailing edge; many folds train on blue blocks sitting *after* their test blocks; and each orange block is flanked by grey seams — the narrow 7-day purge on its left, the wider 28-day embargo on its right.

This picture is the whole method in one glance: test sets everywhere, training on both sides, and every boundary defended in the direction its footprint points. Running all 15 folds:

```
CPCV per-fold RMSEs : [3.544, 3.965, 3.973, 4.305, 6.220, 3.742, 3.795, 4.068,
                       7.623, 3.833, 4.082, 8.365, 4.202, 7.656, 5.460]
CPCV mean RMSE      : 4.9889  | std: 1.6007
True generalisation : 5.4454
```

> **[FIGURE 9 — notebook cell 74]** Histogram of the 15 CPCV fold scores: a main mass around 4, a heavy right tail reaching past 8. A solid orange line marks the CPCV mean (4.99); a dashed green line marks the true error (5.45), sitting comfortably inside the distribution.

This histogram is the payoff of the entire episode. Where every earlier method handed us a single bar — one number, take it or leave it — CPCV hands us a *shape*, and the shape is legible. The low scores come from folds whose test blocks sit in the interior of the timeline with training data on both sides: those folds **interpolate**, and interpolating a trending series is genuinely easier than extrapolating it. The upper tail belongs to the folds whose test sets include the final, highest-level block with no later data to lean on — the folds that most resemble actual deployment. That's also why the CPCV *mean* lands a shade optimistic: honesty compels the observation that averaging in easy alternate histories flatters the number. A practitioner reads the distribution accordingly — the upper tail as the deployment stress case, and the standard deviation as a direct measurement of **regime sensitivity**, the very quantity every single-path method silently averaged away.

And that's the real upgrade. The output stops being an estimate and becomes a risk assessment. Not just *"what error should I expect?"* but *"what's the plausible range? how bad is the bad case? if each fold's score were a strategy's Sharpe ratio, what's the probability the observed performance was luck?"* For high-stakes model selection — the quant use case that motivated the method — that last question is the one that matters, and no single path can answer it.

## Putting it together

One dataset, seven claims, one sealed future. The full scoreboard:

> **[TABLE 1 — notebook cell 77]** The `results_df` comparison table, reproduced here:

| method | cv_val_rmse | true_test_rmse | optimism |
|---|---|---|---|
| Random split | 3.2750 | 5.4454 | **−2.1704** |
| K-Fold (shuffled) | 3.2498 | 5.4454 | **−2.1956** |
| TimeSeriesSplit (expanding) | 5.6934 | 5.4454 | +0.2480 |
| TimeSeriesSplit (sliding) | 6.7044 | 5.4454 | +1.2590 |
| GroupTimeSeriesSplit | 5.5291 | 5.4454 | +0.0837 |
| PurgedGroupTimeSeriesSplit | 6.2276 | 5.4454 | +0.7822 |
| CPCV (mean of 15 folds) | 4.9889 | 5.4454 | −0.4565 |

> **[FIGURE 10 — notebook cell 79]** Bar chart of the optimism column: two deep red bars for the shuffled methods, green bars near zero for the honest ones (expanding TimeSeriesSplit, GroupTimeSeriesSplit, CPCV), orange bars on the conservative side for the sliding window and the purged split.

Read the optimism column top to bottom and the whole argument is laid bare. The two shuffled methods under-report the true error by ~2.2 units — 40% — purchased entirely with leaked future information. The moment chronology is enforced, optimism collapses to a fraction of a unit. The group split turns in the single most accurate point claim; purging trades a little point accuracy for a guarantee about contamination, landing conservative — and when a validation method errs, conservative is the side you want to err on; the sliding window is honest but handicapped by its short memory on this stable process; and CPCV's mean sits slightly below truth for the structural reason above, while its distribution — invisible in a one-number column — is where its actual value lives.

Zooming out from our one experiment to the general properties:

| | Random / K-Fold | TimeSeriesSplit | GroupTSS | PurgedGroupTSS | CPCV |
|---|---|---|---|---|---|
| Respects temporal order | No | Yes | Yes | Yes | Yes (within paths) |
| Handles panel/grouped data | No | No | Yes | Yes | Yes |
| Handles label/feature footprints | No | No | No | Yes (purge) | Yes (purge + embargo) |
| Output | Point | Point | Point | Point | **Distribution** |
| Robustness of estimate | None (overfit) | Low (single path) | Low (single path) | Medium (single path) | High (distributional) |
| Compute | Cheap | Medium | Medium | Medium | `C(N,k)` fits |

And the decision guide, which is *not* "always pick the last column" — the right method is dictated by the structure of your data and the stakes of your decision:

- **Simple univariate series, simple features, next-step labels** → **TimeSeriesSplit** is a valid baseline. Test both windows; the expanding-vs-sliding choice encodes your stability-vs-drift hypothesis, and the data will vote.
- **Panel data — multiple entities per timestamp** (stocks per `time_id`, items per day, sensors per reading) → **group-aware splitting is the mandatory minimum.** A timestamp must never be divided.
- **Rolling-window features or forward-looking labels** (rolling statistics, multi-day targets, triple-barrier labels) → add **purging** sized to the label footprint, and an **embargo** sized to the feature footprint wherever training can follow testing.
- **High-stakes selection, where the backtest drives the decision** → **CPCV.** Pay the `C(N,k)` compute; in exchange, the fragile single number becomes a distribution, the worst case becomes visible, and "was it luck?" becomes an answerable question.
- And in every case, at every rung: **keep one final chronological hold-out that no validation method, however sophisticated, is allowed to touch.** It's the only estimate that shares the one property reality insists on.

## Closing time

We took one real retail panel, sealed away its final three months, and asked seven validation methods the same question: *how well will this model do on data it has never seen?* The shuffled methods answered with a confident fantasy, off by 40%, built entirely from leaked future information. Enforcing chronology recovered honesty at a stroke; group integrity closed the same-day leak that row-based splitting can't even see; purging closed the subtler leak carried by forward-looking labels; and the combinatorial method replaced the single fragile number with a distribution that finally showed how much the answer depends on *which* future you get.

The throughline is uncomfortable and worth sitting with: a validation score is a claim about the future made from inside the past. The methods at the top of this ladder are not more clever than the ones at the bottom — they are more *honest about what could not have been known, and when*. In forecasting, that honesty is the whole game. Everything else in this series — the smoothers, the ARIMA family, the hierarchies, the conformal wrappers — is only as good as the validation scheme that grades it.

## Key Takeaways

1. **Standard cross-validation is unsuitable for time series.** Random splits and shuffled K-Fold violate the arrow of time; on our data they under-reported the true error by 40%. The tight fold agreement of K-Fold is not robustness — it is five folds sharing one leak.
2. **Chronological order is the non-negotiable minimum.** All training data strictly before all validation data, always. This one change bought more honesty than everything else combined.
3. **The data's structure dictates the method.** Panel data makes group-aware splitting mandatory, not optional — a timestamp must never be split across the boundary.
4. **Know your footprints.** A label that looks `h` days ahead demands a purge of at least `h`; a feature that looks `w` days back demands an embargo of at least `w` wherever training data can follow test data. If you can't state the footprints, you can't know the leak.
5. **Robustness requires more than one path.** Every walk-forward scheme is hostage to where its boundaries fall. CPCV's distribution of scores turns validation from estimation into risk assessment — at `C(N,k)` times the compute. No free lunch, as usual. ;-)
6. **Seal a final hold-out and touch it once.** However sophisticated the cross-validation, the last word belongs to a stretch of genuine, untouched future.

---

*United States of Banan is a reader-supported publication. To receive new posts and support my work, consider becoming a free or paid subscriber.*

[ ✓ Subscribed ]
