# Episode 9 — promo materials

## LinkedIn post

Two time series. Fifty years of data. R² = 0.958.

The relationship between them? None whatsoever.

That regression — real GDP "explained" by the price level — is the opening trap of the new episode of Time Series with Konrad, and it's why causal inference on time series needs its own toolkit. Correlation is cheap when everything trends.

Episode 9 climbs the full ladder of causal methods for time series, every step on 50 years of real US macroeconomic data (no toy simulations):

📈 Vector Autoregression (VAR) — model the system jointly, choose lags with AIC/BIC and know their personalities
🔁 Granger causality — a famous asymmetry: consumption growth leads GDP, not the other way around. And why bivariate results are guilty until conditioned + Bonferroni-corrected
⚖️ Cointegration (Engle-Granger, Johansen) — spurious regression vs genuine long-run equilibrium, and what a knife-edge trace test really tells you
🧲 Vector Error Correction (VECM) — who does the adjusting when the system drifts from equilibrium? (Spoiler: the volatile one)
🛡️ Toda-Yamamoto — causality testing that survives unit roots and uncertain cointegration. On the classic "does money cause income?" question, it flips BOTH verdicts of the naive test
🏛️ Structural VAR — impulse responses, variance decompositions, and the price puzzle: what happens when a plausible identifying assumption meets a too-small system

The honest thread throughout: real macro cointegration is borderline, specification-sensitive, and exactly the situation half these methods were invented for. We show the sensitivity grids instead of hiding them.

Full post + companion Jupyter notebook (Python, statsmodels, fully reproducible) 👇
[link to post]

If you work in forecasting, quant research, or economics — which of these methods do you actually use in production? Curious about the split.

#TimeSeries #CausalInference #Econometrics #DataScience #Forecasting #GrangerCausality #Cointegration #MachineLearning #QuantitativeFinance #Python #Statistics #Macroeconomics

---

## SEO description (150 characters)

Granger causality, cointegration, VECM, Toda-Yamamoto & structural VARs — hands-on causal inference for time series on 50 years of real US macro data.

### Alternatives

- (114 chars) Causal methods for time series: Granger, cointegration, VECM, Toda-Yamamoto & SVAR — hands-on, on real macro data.
- (156 chars) Who drives whom? A practical tour of causal time series methods — Granger tests, cointegration, VECM, Toda-Yamamoto and SVAR — with Python code and real data.
