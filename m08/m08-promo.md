# Episode 8 — promo materials

## LinkedIn post

Our cross-validation reported an RMSE of 3.3. The real error was 5.4. Nothing was broken — the validation was doing *exactly* what we asked it to. That 40% gap wasn't noise. It was the market price of leaked future information, and it's hiding in more time series pipelines than anyone wants to admit.

That's the new episode of Time Series with Konrad: **validation methods for time series** — why the K-Fold and train_test_split you trust everywhere else quietly lie to you on temporal data, and the ladder of methods that fix it, one leak at a time. Built on a real retail panel (5 years of daily item-level sales), with the final 90 days sealed away as a "future" — so we can *measure* how honest each method's claim actually is, instead of arguing about it.

The ladder (each rung closes exactly one leak the last one missed):

🎲 **Random split & shuffled K-Fold** — the tabular-ML defaults. Both under-report the true error by ~40% on our data. K-Fold's tight fold agreement isn't robustness; it's five folds sharing one leak
➡️ **TimeSeriesSplit (walk-forward)** — new ingredient: *chronology*. Train strictly on the past, validate on what follows. This single change buys more honesty than everything else combined (optimism collapses from −2.2 to +0.25). Expanding vs sliding window is a testable hypothesis about stability vs drift — and the data votes
👥 **GroupTimeSeriesSplit** — new ingredient: *a day is never cut in two*. On panel data, row-based splitting leaks same-day cross-sectional info. Group-aware splitting isn't an upgrade — it's the entry requirement
✂️ **PurgedGroupTimeSeriesSplit** — new ingredient: *a gap*. Your label looks 7 days ahead; your rolling feature looks 28 days back. Purge + embargo those footprints at the boundary (the López de Prado machinery from quant finance)
🎰 **Combinatorial Purged CV (CPCV)** — new ingredient: *many alternate histories*. Stop evaluating on one path. 15 folds stitch into 5 full backtests, turning a single fragile number into a distribution — and turning "what error should I expect?" into "how bad is the bad case, and was this performance just luck?"

The honest payoff, measured against a sealed future nobody was allowed to touch: everything left of walk-forward is fantasy; everything right of it is a defensible, increasingly careful answer to the same question. And the rule that survives every rung — **keep one final chronological hold-out you never, ever touch.** It's the only estimate that shares the one property reality insists on: the future comes strictly after everything you knew.

A validation score is a claim about the future made from inside the past. Every shortcut in the split is a loan taken out against deployment.

Full post + reproducible Jupyter notebook (Python, scikit-learn, custom purged/combinatorial splitters you can lift straight into your own pipeline) 👇
[link to post]

What's the worst validation-leak surprise you've shipped to production — and which rung of this ladder would have caught it?

#TimeSeries #CrossValidation #DataLeakage #Backtesting #Forecasting #MachineLearning #DataScience #Python #ScikitLearn #CPCV #ModelValidation #QuantFinance #MLOps #DemandForecasting #PanelData #FeatureEngineering

---

## SEO description (154 characters)

Why K-Fold lies on time series: walk-forward, group, purged and combinatorial (CPCV) validation benchmarked on real data — with a 40% leakage gap exposed.

### Alternatives

- (138 chars) Time series cross-validation done right: walk-forward, GroupTimeSeriesSplit, purging, embargo and CPCV — benchmarked honestly on real retail data.
- (156 chars) Standard K-Fold under-reports time series error by 40%. See the leaks — and the purged, group-aware, combinatorial fixes — measured against a sealed future.
- (119 chars) Validation methods for time series: data leakage, purging, embargo and CPCV explained and benchmarked in Python.