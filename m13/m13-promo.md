# Episode 13 — promo materials

## LinkedIn post

A transformer with 115,708 parameters lost to a GRU from 2014. A linear model with 1,190 parameters then beat them both. And the architecture that finally won changed exactly one line of code — it flipped an axis.

That's the new episode of Time Series with Konrad: **transformers for time series forecasting** — the full arc from "attention is all you need" to "are transformers even effective for forecasting?" (spoiler: a one-layer linear model famously said no) to the two repairs that made them earn their place: PatchTST and iTransformer. All built by hand in Keras 3, all benchmarked on real data with parameter counts and training times printed next to every RMSE.

What's inside (real data throughout — 50 daily retail demand series + Melbourne temperatures, trained in minutes on a laptop GPU):

🧮 Scaled dot-product attention in 4 lines of NumPy — and a one-line proof of its fatal flaw: shuffle the past, and the output shuffles along *exactly*. Attention is permutation-equivariant; a time series IS its ordering
📉 The vanilla transformer (timestamps as tokens): loses to a GRU. Even with a 3.8M-parameter head, it only manages a tie at 56× the size
⚖️ The DLinear moment, replicated honestly: a 1,190-parameter linear model beats the GRU, the LSTM, and both vanilla transformers on real retail data
🧩 PatchTST: make tokens out of fortnights, not days — meaningful shapes instead of noisy points, 196× smaller attention matrices, and channel independence that multiplies your training set by 50
🔄 iTransformer: one Permute layer — tokens become whole series, attention runs ACROSS channels — and it takes the crown, beating the GRU by 7% and finally clearing the linear bar. Bonus: positional encodings become unnecessary as a side effect
🗺️ The attention map you can actually read: the model elects a few "reference items" as shared seasonal oracles — visible structure that a correlation matrix can't express
📈 The scaling argument nobody shows you: sweep the lookback from 4 weeks to a full year — the iTransformer converts long context into accuracy at flat ~6s training time, while the GRU gains nothing and its training time grows 8×
🌡️ The rematch transformers lose: univariate temperatures, short memory, one channel — a 3,591-parameter GRU beats every transformer at 1/22nd the size. Overkill, measured

Honest results included: the episode's best architecture on one dataset didn't crack the top three on the other. No free lunch — only better-matched tools.

Full post + reproducible Jupyter notebook (Python, Keras 3 / PyTorch backend, runs on Apple silicon MPS or CUDA) 👇
[link to post]

Which side have you seen in production — transformers earning their parameter count on multivariate data, or a small GRU / linear model quietly refusing to be replaced?

#TimeSeries #Transformers #DeepLearning #Forecasting #MachineLearning #DataScience #Python #Keras #PyTorch #AttentionMechanism #PatchTST #iTransformer #DemandForecasting #NeuralNetworks #FoundationModels #MLEngineering

---

## SEO description (148 characters)

Transformers for time series forecasting: attention from scratch, PatchTST and iTransformer vs LSTM/GRU — honest benchmarks on real data in Keras 3.

### Alternatives

- (134 chars) When do transformers beat LSTMs for forecasting? Attention, PatchTST and iTransformer benchmarked honestly on real retail demand data.
- (159 chars) From attention basics to iTransformer: why vanilla transformers lose to GRUs, when a linear model beats both, and the axis flip that finally wins — in Keras 3.
- (118 chars) Transformers vs LSTM/GRU for time series: real benchmarks, attention maps, and the cases where each one actually wins.
