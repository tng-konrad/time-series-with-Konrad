# Episode 11 — promo materials

## LinkedIn post

I trained five neural network architectures on the same forecasting problem. The spread between them: 0.05 °C. The gap between the worst of them and the naive baseline: ten times that.

That scoreboard opens the new episode of Time Series with Konrad — on **deep learning for time series**: RNN, GRU, LSTM, and the fine art of not forgetting. After ten episodes of classical models, the series switches paradigms — and reports honestly on what neural networks actually buy you (hint: it's not what the tutorials promise).

What's inside (all on real data — ten years of Melbourne daily temperatures, trained in minutes on a laptop GPU):

🧠 The windowing trick that turns forecasting into supervised learning — and the scaler-leakage trap in half the deep learning forecasting code on the internet
📉 The vanishing gradient, made visible: a 15-line NumPy experiment showing why plain RNNs forget beyond a few dozen steps (spectral radius 0.9 vs 1.1 — one line vanishes, one explodes)
🚪 GRU and LSTM demystified: both are the same trick — an additive, gate-protected shortcut through time. One equation each, no hand-waving
🔀 Multi-step forecasting three ways: recursive vs direct vs encoder-decoder (seq2seq) — with a per-horizon error curve that contradicts the textbook story
🧭 Why the "error-compounding" recursive strategy actually WON at 14 days (mean reversion is forgiving), yet lost the day-1 crown it should have kept
🌡️ The noise floor nobody talks about: past day 5, every architecture converged to "forecast the climate, accept the weather"
↔️ Bidirectional LSTMs: when they're legitimate, and why they amplify pipeline leaks into beautiful, fake validation scores
⚙️ Modern Keras 3 on the PyTorch backend — same code runs on CUDA, Apple silicon (MPS), or CPU

Honest results included: the stacked RNN lost to the single-layer one, GRU and LSTM tied to the third decimal, and the 12,705-parameter LSTM beat tomorrow-equals-today by just 12%.

Full post + reproducible Jupyter notebook (Python, Keras 3 / PyTorch) 👇
[link to post]

What's your experience — has an LSTM ever beaten your gradient-boosting or statistical baseline by enough to justify the complexity?

#TimeSeries #DeepLearning #LSTM #GRU #RNN #Forecasting #MachineLearning #DataScience #Python #Keras #PyTorch #NeuralNetworks #SequenceModels #MLOps

---

## SEO description (149 characters)

Deep learning for time series in Keras 3: RNN, GRU, LSTM, seq2seq — vanishing gradients explained, honest benchmarks vs naive baselines on real data.

### Alternatives

- (136 chars) RNN, GRU & LSTM for forecasting: windowing, vanishing gradients, multi-step strategies, and honest results on real data in modern Keras.
- (115 chars) Hands-on deep learning for time series: RNN to encoder-decoder in Keras 3 / PyTorch, with honest real-data results.
- (147 chars) LSTM vs GRU vs plain RNN for time series forecasting — gates, gradient flow, recursive vs direct multi-step, and the baselines most tutorials skip.
