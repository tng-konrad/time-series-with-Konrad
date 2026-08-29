# Episode 14 — promo materials

## LinkedIn post

A model that never saw my data just beat one trained on it. By 17%. In under two seconds, on a laptop.

That's the new episode of Time Series with Konrad: **foundation models for time series forecasting** — the zero-shot paradigm tested honestly. Download pretrained weights, hand over raw history, get a calibrated probabilistic forecast back. No windowing, no scaling, no fit(). Sounds like a press release — so we ran the experiment: three foundation models vs hand-trained specialists vs the seasonal naive baseline, on real retail demand data, 120 walk-forward forecasts, point AND probabilistic scoring.

What's inside (everything reproducible, runs on a laptop GPU):

🧩 The tokenization problem solved by hand: how do you feed a continuous float stream into a transformer? Patching (32-day shapes as tokens, attention matrix shrunk to <0.1%) and quantization (your sales history literally rewritten as a 4,096-word language) — both rebuilt in a few lines of NumPy
🚀 **TimesFM 2.5** (Google, 200M params): the patched decoder — 120 two-week forecasts in 6.5s, zero training, with uncertainty bands that fan out like a textbook
🦖 **TiRex** (NX-AI, 35M params): a foundation model with NO attention at all — an xLSTM built for streaming and edge, one-sixth the size, hanging with the giants. The architecture war ends in a draw once everyone gets the same pre-training food
🌐 **Chronos-2** (Amazon, 120M params): universal forecasting + in-context learning across series — and the honest finding that feeding it 50 sibling series changed *nothing*, because redundant context isn't information
📊 The scoreboard: Chronos-2 7.74 RMSE vs GRU 9.37 vs seasonal naive 11.55 — every zero-shot model beat every trained specialist. And the quiet star: 80% prediction intervals that covered 79–80% of 1,680 test days, out of the box
🥶 The killer app, measured: **cold start**. With only 60 days of history, TiRex zero-shot beat a GRU trained on five years — while a scratch-trained model fell behind copy-last-week
⚠️ The part the press releases skip: data leakage (is your benchmark inside their training corpus?), the evaluation crisis, decontaminated checkpoints, and why seasonal naive never, ever retires

Honest results included: the specialist that won the single-window close-up lost the 120-forecast aggregate to everything. Single windows lie; walk-forward decides.

Full post + reproducible Jupyter notebook (Python, Keras 3 / PyTorch backend, TimesFM + TiRex + Chronos-2 running locally on Apple silicon MPS) 👇
[link to post]

Have you put a time series foundation model into production yet — and did it survive contact with your seasonal naive baseline?

#TimeSeries #FoundationModels #Forecasting #MachineLearning #DataScience #ZeroShot #TimesFM #Chronos #TiRex #xLSTM #Transformers #DeepLearning #Python #PyTorch #DemandForecasting #ProbabilisticForecasting #MLOps #GenAI #LLM #MLEngineering

---

## SEO description (136 characters)

Time series foundation models tested honestly: TimesFM 2.5, TiRex and Chronos-2 zero-shot vs trained baselines on real data — with code.

### Alternatives

- (140 chars) Do time series foundation models beat models trained on your data? TimesFM, TiRex and Chronos-2 benchmarked zero-shot on real retail demand.
- (151 chars) Zero-shot forecasting with TimesFM 2.5, TiRex and Chronos-2: calibrated intervals, cold-start wins, data-leakage caveats — honest benchmarks in Python.
- (122 chars) Foundation models for time series forecasting: zero-shot TimesFM, TiRex and Chronos-2 vs GRU and seasonal naive baselines.
