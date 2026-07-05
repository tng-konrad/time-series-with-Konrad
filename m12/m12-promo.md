# Episode 12 — promo materials

## LinkedIn post

A neural network trained on 4 months of sales data lost to "copy last week" — by a full 2 RMSE points. The same network, pretrained on 50 *other* series and never shown a single observation from the target, beat that baseline without any training at all.

That experiment opens the new episode of Time Series with Konrad — on **transfer learning for time series**: what to do when the series you must forecast is too young to learn from. The answer is the two-stage paradigm behind every foundation model — pretrain where data is plentiful, fine-tune where it isn't — built by hand, so you can see exactly where it pays off and exactly where it backfires.

What's inside (all on real data — the Kaggle store-item demand dataset, 500 daily retail series, trained in minutes on a laptop GPU):

🏪 The setup nobody escapes: a newly opened store, 120 days of history, and a stakeholder who wants weekly forecasts now
📉 The humbling baseline: 13,319 parameters vs 69 training windows — and why from-scratch deep learning loses to a one-line seasonal naive
🎯 Zero-shot transfer: a GRU pretrained on 50 sibling series beats local training while using ZERO target data — the foundation-model pitch, in miniature
🔧 Fine-tuning as a trust dial: freeze the encoder (455 trainable parameters can't overfit) → unfreeze everything at 1/10th the learning rate — the canonical scratch → zero-shot → head-only → full-FT ladder, with honest numbers at every rung
🎭 Self-supervised pretraining, both families: masked autoencoding (Ti-MAE style) and contrastive learning (SimCLR's NT-Xent loss in ~15 lines of keras.ops) — and why BOTH lost to supervised pretraining here
🔍 The reconstruction plot that explains everything: the masked autoencoder learned to encode window *levels* — true, and useless, because standardization already removed them. Inspect what your pretext task actually learned before transplanting it
📈 The sample-efficiency curve: at 60 days of history, transfer beats scratch by 12 RMSE points; at 480 days, by 0.4. Transfer pays the most exactly when you have the least
⚠️ Negative transfer, live on stage: pretraining on Melbourne *weather* to forecast shop demand — worse than scratch, and invisible without a baseline (plus the DTW-based source selection fix)

Honest results included: seasonal naive beat every method that didn't transfer from a matched source — including both self-supervised approaches. No free lunch, only borrowed ones.

Full post + reproducible Jupyter notebook (Python, Keras 3 / PyTorch, runs on Apple silicon or CUDA) 👇
[link to post]

What's your experience — have you ever caught negative transfer in production, or did the pipeline look healthy all the way down?

#TimeSeries #TransferLearning #DeepLearning #Forecasting #MachineLearning #DataScience #Python #Keras #PyTorch #FoundationModels #SelfSupervisedLearning #ContrastiveLearning #PretrainedModels #DemandForecasting #MLOps

---

## SEO description (156 characters)

Transfer learning for time series forecasting: pretraining, fine-tuning, zero-shot and negative transfer — honest benchmarks on real retail data in Keras 3.

### Alternatives

- (136 chars) Pretrain, fine-tune, forecast: hands-on transfer learning for time series with honest results — when it wins, and when it quietly hurts.
- (155 chars) How to forecast a series with 4 months of history: supervised pretraining, masked autoencoders, contrastive learning and fine-tuning, benchmarked honestly.
- (113 chars) Transfer learning for time series in Keras 3: zero-shot beats local training — and weather pretraining backfires.
