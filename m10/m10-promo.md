# Episode 10 — promo materials

## LinkedIn post

I wrote a Kalman filter in 40 lines of NumPy and checked it against statsmodels on a real dataset.

Maximum difference in the estimates: 0.000000000006.

That validation opens the new episode of Time Series with Konrad — on **state space models**, the framework that quietly ended the oldest feud in forecasting. For decades the Box-Jenkins (ARIMA) camp and the exponential smoothing camp went to different conferences. It turns out both were fitting the same model all along: two equations, five matrices, one Kalman filter underneath.

What's inside (all on real data — the Nile river and the classic airline passengers series):

🔧 A hand-built Kalman filter, explained line by line and validated to machine precision
🌊 The local level model on the Nile — watching the filter detect the 1899 Aswan dam level shift, and why "filtered vs smoothed" is the real-time vs hindsight distinction every practitioner needs
📈 The structural ladder: level → trend → seasonality, each component with its own uncertainty band
🔁 The unification, part 1: ARIMA rewritten as a state space model (that's literally why it's called SARIMAX)
🎭 The unification, part 2: exponential smoothing's α parameter unmasked as a frozen Kalman gain — the 1950s heuristics were doing maximum likelihood without knowing it
🛠️ A custom model from raw matrices via MLEModel, matching the library to 6 decimals
🕳️ The party trick: delete 15 years from the Nile record and watch the filter handle the gap natively — no imputation, honest uncertainty bands included
🏁 A three-way forecast shootout: structural model vs ETS vs SARIMAX on a two-year holdout

Honest results included: the famous "airline model" ARIMA finishes third on the dataset it was named after — and the notebook explains why.

Full post + reproducible Jupyter notebook (Python, statsmodels) 👇
[link to post]

What's your go-to: structural models, ETS, or ARIMA? And did you know they were the same thing underneath?

#TimeSeries #KalmanFilter #StateSpaceModels #DataScience #Forecasting #Python #Statsmodels #ARIMA #ExponentialSmoothing #MachineLearning #Econometrics #Statistics

---

## SEO description (140 characters)

State space models & the Kalman filter in Python: local level to SARIMAX and ETS, unified — with a hand-built filter validated on real data.

### Alternatives

- (136 chars) Kalman filters, hidden states, and the framework that unifies ARIMA and exponential smoothing — hands-on state space modeling in Python.
- (125 chars) State space methods for time series: build a Kalman filter from scratch, unify ARIMA & ETS, and handle missing data natively.
