# Time Series with Konrad: episode 14

### Foundation models for time series: forecasting without fitting

**KONRAD BANACHEWICZ**
DEC 7, 2026

Every model in this series — all thirteen episodes of it — has shared one assumption so basic we never said it out loud: *you train the model yourself, on your own data, for your own task*. One dataset, one model, one training run. That is how forecasting has worked for fifty years, from exponential smoothing (episode 2) through ARIMA (episode 3) to last episode's transformers. Even the transfer learning of episode 12 only *softened* the assumption — you still fine-tuned, you still fit.

This episode examines the claim that the assumption is now optional. **Time series foundation models (TSFMs)** are large networks pre-trained once, by someone else, on billions to trillions of time points scraped from wildly unrelated domains — web traffic, electricity load, weather, retail, synthetic sine-and-noise cocktails. The bet behind them is that time series, like language, has a *universal grammar*: trends, seasonalities, mean reversion, and noise textures recur across domains. You download the weights, hand the model your raw history, and get a full probabilistic forecast back in under a second. No windowing, no scaling, no `fit()`.

The core principle — stated, as always, as a hypothesis to be tested rather than a fact to be admired:

**A model that has seen a million weekly rhythms should recognize yours without ever training on it. Whether it actually does — and whether it beats a model that *did* train on your data — is an empirical question, and we run the experiment.**

The ladder we climb: the honest baselines and two hand-trained **specialists** (a linear model and a GRU, our incumbents from episodes 11 and 13) → the tokenization problem, solved by hand in a few lines of NumPy (**patching** and **quantization**) → three genuinely different foundation models run zero-shot on a laptop: **TimesFM 2.5** (Google, a patched decoder), **TiRex** (NX-AI, an xLSTM with no attention at all), and **Chronos-2** (Amazon, a universal forecaster with in-context learning) → a 120-forecast head-to-head with probabilistic scoring → the **cold-start** stress test, forecasting from 60 days of history → and the uncomfortable part nobody puts in the press release: data leakage and the evaluation crisis. Real data throughout, honest errors throughout. Practical understanding over rigor, as always.

📓 All code lives in the companion notebook:
https://github.com/tng-konrad/time-series-with-Konrad/blob/main/m14-fable-version.ipynb

## The arena, and the bar to clear

The dataset is an old friend: the Kaggle store-item demand data from episodes 12 and 13 — five years (2013–2017) of daily unit sales. We take store 1, pivot it into a 1,826 × 50 panel (one row per day, one column per item), and pick item 1 as the close-up series: about 20 units a day, and five years of it show three structures stacked on top of each other.

> **[FIGURE 1 — notebook cell 20: full 2013–2017 daily sales of store 1, item 1]**

A slow upward trend, an annual cycle peaking every summer, and a furry texture of week-scale noise. Nothing exotic — which is the point. Keep in mind, though, that fifty well-behaved daily series is *friendly* data; that friendliness will matter when we interpret the results.

> **[FIGURE 2 — notebook cell 22: the last 120 days of the same series]**

Zooming into the final 120 days reveals what the full plot smears out: a stubborn **weekly seesaw** — weekend peaks, midweek troughs — riding the autumn downslope of the annual cycle. Our forecast horizon is 14 days: exactly two full periods of this rhythm. Any competent forecaster must reproduce the seesaw, not just the level.

The protocol: everything before October 1st, 2017 is training territory; the final quarter becomes a walk-forward test with **twelve weekly forecast origins × ten items = 120 two-week forecasts** per model. Chronological splits only — never shuffle a time series (the leakage liturgy from episode 8 applies in full). And before anything with parameters, the baselines every forecasting claim must survive:

```
naive:           {'MAE': 11.07, 'RMSE': 12.03}
seasonal naive:  {'MAE': 5.36,  'RMSE': 6.53}
```

Copy-yesterday versus copy-last-week, and last week wins by nearly a factor of two — the weekly cycle carries most of the predictable signal. **Seasonal naive is the bar: it costs nothing, trains nothing, and knows nothing except "weeks repeat."**

> **[FIGURE 3 — notebook cell 28: seasonal naive forecast vs actuals, first origin]**

It nails the *shape* — weekend peaks land on weekends — but copies last week's level and noise verbatim, wiggles and all. Those two failure modes (level drift and noise-copying) are precisely the headroom any learned model has to work with.

↪ *Every model here is judged on a strict chronological walk-forward — the leakage-proof validation episode 8 is devoted to, and the reason these zero-shot claims are falsifiable rather than flattering.* → **<LINK TO EPISODE 8 HERE>**

## The specialist paradigm: train your own

First, the incumbent workflow this whole episode interrogates. Two specialists from previous episodes, trained on the pooled windows of all ten arena items (84 days in, 14 days out, per-item z-scoring — 16,370 windows in total): a **linear model**, the embarrassingly effective baseline that episode 13 showed beating far fancier architectures, and a **GRU**, the recurrent champion of episode 11. What is new is that we now account for their *cost*:

```
linear:  1,190 parameters,  4.5s to train
GRU:    13,774 parameters, 23.5s to train
```

> **[FIGURE 4 — notebook cell 38: linear specialist forecast vs actuals, first origin]**

The linear model scores RMSE **5.04** on the close-up window — a clear improvement over seasonal naive's 6.53, and the plot shows why: it keeps the weekly seesaw but *smooths away* last week's idiosyncratic wiggles (it has averaged over sixteen thousand training windows, not copied one week), and it sits slightly below last week's level, having learned the autumn downslope. The GRU, on the same window, scores **6.64** — worse than the linear model, echoing episode 13. Hold that ranking loosely: one forecast is fourteen noisy days of one series, and single-window verdicts are close to coin flips. The 120-forecast scoreboard will have the final word (and a surprise).

↪ *Meet the two specialists properly: the linear model that humbled the transformers in episode 13 (**<LINK TO EPISODE 13 HERE>**) and the GRU built from scratch in episode 11 (**<LINK TO EPISODE 11 HERE>**).*

The seconds look trivial. Now multiply by a realistic catalogue: a retailer with 50,000 series, retrained weekly, is running a permanent training pipeline with monitoring, scheduling, and failure handling. **That operational bill — not the 24 seconds — is what the zero-shot paradigm proposes to eliminate.**

## How do you feed a float into a transformer?

Before running the foundation models, the one problem every single one of them had to solve first. A transformer is a machine for sequences of **tokens** — discrete symbols from a finite vocabulary. Language arrives pre-tokenized; a time series does not. It is a stream of continuous floats with no vocabulary, arbitrary scale (sales of 20 a day here, server requests in the millions elsewhere), and a noise level that makes any single observation nearly meaningless. Feeding one float per token is also computationally hostile: attention cost grows with the *square* of sequence length — episode 13 territory.

The field converged on two answers. We build both, in a few lines of NumPy, on our own series.

**Patching** — episode 13's big idea, industrialized. One `reshape` cuts a 512-day context into 16 non-overlapping blocks of 32 days; inside the real model each block is pushed through a small MLP to become one embedding vector, and *those* are the tokens.

> **[FIGURE 5 — notebook cell 47: the last six patches, each 32-day block one colored segment]**

Sixteen tokens instead of 512 shrinks the attention matrix to **under 0.1%** of its one-token-per-day size — this is what makes 16k-point contexts feasible at all. And the semantic argument matters as much as the economics: like a word compared to a letter, a patch is a unit *worth attending to*. A single day is mostly noise; a 32-day shape — four weekly cycles and a stretch of seasonal slope — is a recognizable pattern.

**Quantization** — the opposite philosophy, introduced by the original Amazon **Chronos**: if transformers are language machines, turn the series *into language*. Divide by the mean absolute value (scale invariance), chop the range into 4,096 bins (the vocabulary), assign each day its bin id:

```
Vocabulary size: 4096
Tokens actually used by this series: 37
First ten tokens: [1638 1911 1183 2184 1456 1274 1911 2275 3094  728]
Round-trip error (RMSE, sales units): 0.002
```

The series is now literally a sentence of integers, and a T5 language model can be trained on such sentences with ordinary cross-entropy, predicting a *distribution over the vocabulary* for the next token — probabilistic forecasts by construction. Note two things in the printout. The discretization loss is trivial: 0.002 sales units, invisible next to day-to-day noise. And our series uses only **37 of the 4,096 words** — the vocabulary is sized for the diversity of a whole pre-training universe, not one series.

> **[FIGURE 6 — notebook cell 52: the last 120 days as a step plot of token ids]**

The same 120 days as before, now written in a 4,096-symbol alphabet. The weekly seesaw is perfectly recognizable — and this picture is the entire Chronos hypothesis in one image: *if the structure survives tokenization, a language model can learn its grammar.*

(A third route — building tokens from *lagged values* at seasonal offsets, taken by the early open model **Lag-Llama** — preserves exact values and encodes periodicity explicitly, but has lost ground to the two above.)

Who uses what? The current generation, in one table:

| Model | Maker | Backbone | Params | Tokenization | Probabilistic output |
|---|---|---|---|---|---|
| **TimesFM 2.5** | Google | decoder-only transformer | 200M | patches (float) | continuous quantile head |
| **TiRex** | NX-AI | stacked xLSTM (no attention) | 35M | patches + masking | 9-level quantile grid |
| **Chronos-2** | Amazon | transformer w/ group attention | 120M | patches (float) | direct multi-quantile |

Spot the pattern: **patching won.** Even the quantization pioneer's own successor abandoned the vocabulary — we will get to that story. The wider field agrees on something else too: Salesforce's **Moirai**, the other major "universal forecaster," scrapped its masked-encoder design for a decoder-only one in version 2.0. When rivals converge, pay attention. ;-)

## Scoring a distribution, not a point

One new tool before the ladder. Foundation models advertise *probabilistic* forecasts — we ask each for its 10th, 50th, and 90th percentiles, giving a median point forecast plus an **80% prediction interval**. Two questions need scoring. Is the interval *honest*? Coverage: the fraction of actuals that land inside the band, which should be about 80%. And is each quantile *sharp*? The **pinball loss** for quantile level τ:

L_τ(y, q) = max( τ·(y − q), (τ − 1)·(y − q) )

In plain English: for the 90th percentile (τ = 0.9), undershooting the actual costs nine times more than overshooting it — so the loss is minimized by a value the actual exceeds only 10% of the time, which is exactly what a 90th percentile *should* be. The honest quantile is the optimal prediction; the score cannot be gamed. If this smells like episode 7's conformal prediction concerns, good nose — calibration is the same question, and we are about to ask it of 200-million-parameter strangers.

↪ *Coverage, prediction intervals, and calibration got their full workout in the conformal-prediction episode — the yardstick we're about to hold these foundation models to.* → **<LINK TO EPISODE 7 HERE>**

## TimesFM 2.5: the patched decoder

**TimesFM** is Google Research's flagship: a decoder-only transformer over patch tokens, pre-trained on roughly 100 billion real-world time points — much of it Google Trends and Wikipedia pageview series, chosen precisely because web attention encodes the human weekly and holiday rhythms that transfer to domains like retail. Version 2.5, the current one, is leaner and stronger than its predecessors: 200M parameters (down from 500M), a context window stretched to 16,384 points, and no more manual frequency flag.

Its architectural signature is the **asymmetric patch**: input patches of 32 points, but *output* patches of 128. A conventional autoregressive model forecasting 256 steps needs 256 generation steps, each feeding on its own possibly-wrong output; TimesFM emits 128-point blocks, so long horizons take a couple of forward passes instead of hundreds — faster, and with far less error compounding. An optional 30M-parameter **quantile head** bolts probabilistic output onto the point forecaster.

The workflow of the new paradigm, in its entirety: hand over raw history, receive forecasts.

```
All 120 two-week forecasts in 6.3s — zero training

TimesFM 2.5: {'MAE': 4.95, 'RMSE': 5.55}
80% interval coverage: 0.64
```

Read that first line again. The GRU needed 23.5 seconds of training before it could forecast anything — and pays again for every new dataset. TimesFM produced all 120 forecasts, for series it had never seen, in six seconds flat.

> **[FIGURE 7 — notebook cell 60: TimesFM 2.5 zero-shot forecast with 80% band, first origin]**

A genuinely learned forecast, not a copied week: smooth weekly seesaw, correctly depressed level, and — new for this series — a coral uncertainty band that visibly *widens with horizon*, day 14 admitting more doubt than day 1. That fan is textbook forecasting behavior, produced zero-shot. The 64% coverage on this single window is under the nominal 80%, but fourteen days decide nothing; the verdict on calibration waits for the full arena.

## TiRex: recurrence strikes back

Can attention be skipped entirely? The ladder's contrarian rung says yes. **TiRex**, from NX-AI (the Linz lab of LSTM inventor Sepp Hochreiter), is a foundation model with **no attention at all**: its backbone is a stack of **xLSTM** blocks — the 2024 revival of the LSTM with exponential gating and better parallelism.

That changes the economics, not just the aesthetics. A transformer must re-attend over the whole context for every new observation, at cost quadratic in context length; a recurrent model *carries its history* in a fixed-size hidden state and absorbs each new patch at constant cost. TiRex's natural habitat is therefore streaming and edge deployment — its makers benchmark it on industrial PLC hardware with a few gigabytes of RAM — and the whole model is only **35M parameters**, a sixth of TimesFM, yet competitive with the largest entries on zero-shot leaderboards like GIFT-Eval.

One training trick worth knowing, because episode 12 taught its ancestor: **contiguous patch masking** — long masked spans the model must reconstruct *in parallel* during pre-training. At inference, the forecast horizon is just one more masked span, decoded in a single forward pass: no token-by-token generation, no compounding one-step errors.

```
All 120 forecasts in 2.3s

TiRex: {'MAE': 5.04, 'RMSE': 5.75}
80% interval coverage: 0.64
```

> **[FIGURE 8 — notebook cell 65: TiRex zero-shot forecast with 80% band, first origin]**

Statistically indistinguishable from TimesFM on the close-up, from a model one-sixth the size with a completely different computational core. The deeper lesson of this rung: **at today's scale of pre-training data, the backbone — attention versus recurrence — matters less than the recipe** of patches, masking, and massive multi-domain exposure. Episode 13's architecture war, at foundation scale, ends in a draw once everyone gets the same food. (A successor, **TiRex-2**, extends the recipe to multivariate inputs and known-future covariates; its checkpoint is gated on Hugging Face, so we run the original.)

## Chronos-2: universal forecasting

The top rung, and a family with a story arc. Chronos began as the purest "time series is a language" bet — the T5-with-4,096-token-vocabulary design whose tokenizer we rebuilt above. Accurate, but slow: sampling forecasts token by token. **Chronos-Bolt** swapped in patches and direct quantile prediction for a ~250× speedup. And **Chronos-2**, the current generation (120M parameters), is a near-complete redesign that keeps only the name: continuous patch embeddings, a single pass predicting all quantile levels directly, and — its true novelty — **universality**: one checkpoint handles univariate series, multivariate groups, and known-future covariates, none of which the tokenized original could represent at all. The family's own migration away from quantization is the field's verdict on that debate.

```
All 120 forecasts in 0.3s

Chronos-2: {'MAE': 4.9, 'RMSE': 5.62}
80% interval coverage: 0.57
```

> **[FIGURE 9 — notebook cell 70: Chronos-2 zero-shot forecast with 80% band, first origin]**

The fastest of the three — from a model that would have been the slowest in its first incarnation — and a close-up RMSE between its two ladder-mates. All three foundation models have now landed in the same neighborhood on this window (5.55–5.75, versus 5.04 for the linear specialist), which is itself the finding: **three alien architectures, trained by three companies on three different data piles, agree with each other more than they differ from the locals.**

Chronos-2's headline feature deserves its own experiment. Reshape alone changes the semantics: the same fifty series that entered as fifty independent tasks can enter as *one* task with fifty variates. Inside the model, a **group attention** layer lets every series' representation peek at every other's — cross-series learning at inference time, with no weight updates. This is the time series analogue of prompting an LLM with related examples, and where an LLM's in-context learning *emerged* from scale, here it is deliberately engineered into the architecture.

```
Close-up item, forecast alone:      {'MAE': 4.9,  'RMSE': 5.62}
Close-up item, forecast in a group: {'MAE': 5.08, 'RMSE': 5.71}
```

The result is a finding, not a triumph: the group forecast is essentially identical to the solo one. And it makes sense. These fifty items are siblings — same store, same weekly rhythm, same summer peak — so item 1's own history already contains everything its siblings could tell you. The group's information is *redundant*, not complementary. **Context sharing pays when the neighbors know something the target doesn't**: a recent study on 200 low-voltage electricity feeders found Chronos-2's group mode genuinely useful when weather covariates were missing — because neighboring feeders *were* the weather signal, by proxy. Remember episode 12's lesson that transfer only works when there is something to transfer? Same theorem, new clothes.

## The ones we won't run

For completeness, the rest of the zoo — each with one idea worth stealing even if you never load the checkpoint. **Moirai 2.0** (Salesforce) pursues universality hardest: any-variate attention that flattens an arbitrary number of variables into one sequence, and a Mixture-of-Experts variant that matched dense rivals with up to 65× fewer *active* parameters. **Toto 2.0** (Datadog) specializes in high-cardinality observability telemetry — a scaling family up to 2.5B parameters trained on over a trillion points, with *causal* patch normalization (statistics computed only over past patches, so normalization itself cannot leak the future — a leakage mode I bet you hadn't considered) and an arcsinh transform taming metrics that span orders of magnitude. **TimeGPT** (Nixtla) is the API-first option: no weights, conformal intervals (episode 7 says hello), zero infrastructure — at the price of sending your data out. **MOMENT** (CMU) is the multi-tasker, as comfortable with classification and anomaly detection as with forecasting; **Time-MoE** scaled sparse mixture-of-experts to 2.4B parameters on 300 billion points to show scaling laws hold for time series too.

## The verdict: 120 forecasts, seven forecasters

Close-ups shown, anecdotes collected. The full scoreboard — ten items, twelve origins, point accuracy for everyone, probabilistic scores for those who offer quantiles, and the compute column the whole paradigm is about:

```
                    zero_shot   rmse  pinball  coverage_80  compute_s
Seasonal naive (7)          -  11.55      -         -            0.0
Linear (trained)           no   9.46      -         -            4.5
GRU (trained)              no   9.37      -         -           23.5
TimesFM 2.5               yes   8.30    1.98       0.79          6.3
TiRex                     yes   8.76    2.10       0.79          2.3
Chronos-2                 yes   7.74    1.86       0.80          0.3
Chronos-2 (group)         yes   8.01    1.91       0.79          0.8
```

> **[FIGURE 10 — notebook cell 79: horizontal bar chart of arena RMSE, zero-shot models in green, trained specialists in blue]**

Read it slowly; there is a lot in seven rows.

- **All three foundation models beat both trained specialists.** Chronos-2 wins the arena outright at **7.74** — roughly 17% better than the GRU and 33% better than seasonal naive — with TimesFM second and TiRex, at a sixth of the parameters, third yet still ahead of everything trained on this very panel.
- **The close-up verdict flipped.** On one window the linear model looked like the winner and the GRU looked broken; over 120 forecasts the GRU edges the linear model, and both lose to all the strangers. **Single-window rankings deceive; walk-forward aggregates decide.** If you take one methodological sentence from this episode, take that one.
- **Calibration is the quiet star.** Coverage of the 80% bands: 0.79, 0.79, 0.80 across 1,680 forecast days. All three models are within a point of nominal — uncertainty estimates you could hand to an inventory planner as-is, produced for series the models had never seen. Episode 7 needed the whole conformal apparatus to manufacture that honesty; here it fell out of a download.
- **The group mode does not help** (8.01 vs 7.74) — redundant context, as diagnosed. It is at least cheap: per series, the fifty-variate group calls cost less than the separate batches.
- **The compute column is the paradigm in one glance.** The specialists *spent* their seconds before they could forecast at all, and must spend them again for every new dataset. The foundation models' seconds are one-off inference, amortizing a pre-training bill someone else already paid.

In the bar chart, every green bar is shorter than every blue bar — and the spread *within* the greens is smaller than the gap between greens and blues. On this kind of data, **which foundation model you pick matters less than whether you pick one.**

Can we do better than a draw between paradigms, then? No — but we can find the corner where the gap becomes a chasm.

## Cold start: the killer app

The arena gave every model five years of history — the setting where "train your own" is strongest. The advertised sweet spot of foundation models is the opposite corner: a series too *young* to train on. A product launched two months ago, a sensor installed last quarter, a store opened in spring — episode 12's cold-start problem, revisited with new weapons. Simulation: every approach gets only the **last 60 days** of history before the same test fortnight. The specialists shrink to 28-day windows (with 60 days you cannot fill an 84-day lookback even once), leaving them **19 training windows** where the arena versions had sixteen thousand.

↪ *Cold start was the whole problem episode 12 attacked with transfer learning — worth comparing how borrowed weights and a pre-trained foundation model each handle a data-starved series.* → **<LINK TO EPISODE 12 HERE>**

```
Chronos-2 (512d ctx)         5.62
TiRex (60d ctx)              5.91
Chronos-2 (60d ctx)          6.11
Seasonal naive (7)           6.53
GRU (trained, full history)  6.64
TimesFM 2.5 (60d ctx)        6.68
GRU (trained on 60d)         6.74
Linear (trained on 60d)      8.47
```

The ranking tells the story cleanly. **The zero-shot models degrade gracefully**: TiRex and Chronos-2 lead the 60-day field — and TiRex on 60 days *beats the GRU trained on five years*. Sixty days is two months of weekly cycles: enough for a model that has seen a million weekly rhythms to lock on. **The starved specialists don't degrade — they collapse**: the 60-day linear model finishes behind seasonal naive (nineteen windows cannot estimate an honest 28×14 map), and the 60-day GRU essentially ties the free baseline. Training on a cold series buys you complexity risk, not accuracy. And one more reading, easy to miss: **history still helps the history-lovers** — Chronos-2 with its full 512-day context remains the best number in the table. A foundation model is not a reason to throw history away; it is a way to survive not having any.

> **[FIGURE 11 — notebook cell 86: TiRex forecasting from only 60 days of history, with 80% band — everything the model was given is on screen]**

Sixty days in, a calibrated two-week forecast out — seesaw in place, level tracked, honest band around it. Rewind to episode 2 and consider what this plot would have taken: identify the seasonality, difference the trend, pick orders, fit, diagnose, iterate. Whatever reservations the next section raises, *this* — competent probabilistic forecasts on data-starved series, for free — is the genuinely new capability. (Standard caveat: one origin, one item; a demonstration of the mechanism, not a benchmark. The mechanism replicates at scale in the literature.)

## The evaluation crisis

Time to spoil the party, carefully. Our arena result mirrors the headline claims of every foundation-model paper — and the fine print of those claims has become its own research topic. Three issues belong in every practitioner's head.

**Data leakage.** Zero-shot means *these weights never saw this series* — but who checks? The pre-training corpora are scraped from the same public archives (Monash, GluonTS, Kaggle, M-competitions) that everyone benchmarks on. Our own arena is not above suspicion: the store-item dataset is a popular Kaggle competition from 2018, public for years before any of these models were trained, and none of the three publishes a manifest that would rule out its presence in their corpora. When a foundation model aces a famous public benchmark, some of that performance may be memory, not generalization. The field's countermeasures: benchmarks with enforced cutoffs (GIFT-Eval, fev-bench), evaluation on data published *after* a model's training cutoff, and — most directly — decontaminated checkpoints like TiRex-2's variants, retrained with the benchmark datasets explicitly excluded. **Your own private data remains the only benchmark you can fully trust.**

**The baseline problem.** A recurring, embarrassing literature finding: on stable, low-frequency series — monthly sales with clean seasonality, say — statistical workhorses (ETS, Theta, seasonal naive) still match or beat billion-parameter models at a millionth of the cost. Our arena is favorable terrain for the big models: daily frequency, long history, subtle trend interactions. Move to twenty-four points of monthly data and the picture can invert. No leaderboard exempts you from running the free baseline on *your* data first.

**Benchmarks are not deployments.** Pooled RMSE over 1,680 forecast days is a fine scorecard and a poor business metric. That electricity-feeder study made the point sharply by scoring peak-load errors against the physical damage curve of a grid fuse — the forecast errors that *cost money* are not the ones RMSE emphasizes. Before promoting any model, foundation or local, translate its errors into the units your decisions are made in.

## Closing time

We climbed from a copied week to a downloaded forecaster: seasonal naive set the bar; a linear model and a GRU cleared it the classical way, at the price of a training pipeline; patching and quantization showed how a continuous series becomes tokens at all; and three foundation models — a patched decoder, an attention-free xLSTM, and a universal in-context learner — cleared everything, zero-shot, with calibrated uncertainty thrown in for free. The cold-start test found the paradigm's true home (sixty days of history, no time to train), and the evaluation crisis reminded us to hold the trophy loosely: friendly data, possible leakage, and a free baseline that never stops being mandatory.

A practical routing rule to leave with. **Central business forecasting** over many series with decent history: a patch transformer like TimesFM 2.5 or Chronos-2, with covariates when you have them. **Streaming and edge**, where memory and latency rule: a recurrent model like TiRex. **High-cardinality machine telemetry**: the observability specialists like Toto 2.0. **No infrastructure at all**: a managed API like TimeGPT — if your data may leave the house. And in every cell of that grid: seasonal naive first, always, because the day a fifty-year-old heuristic beats your foundation model, you want to be the one who finds out. :-)

The throughline: **pre-training moved the cost of forecasting from your training pipeline to someone else's — but it did not move the burden of proof.** In the next installment we take the middle road this episode deliberately skipped: *fine-tuning* a foundation model on your own panel — episode 12's transfer playbook, replayed with 200-million-parameter borrowed experience.

## Key Takeaways

1. **Zero-shot forecasting works:** three foundation models with nothing but raw history beat specialists trained on the very panel being forecast — 7.74/8.30/8.76 RMSE versus 9.37 for the GRU and 11.55 for seasonal naive.
2. **Calibration came free:** 80% intervals covered 79–80% of 1,680 forecast days, zero-shot. What episode 7 built with conformal machinery, these models ship in the download.
3. **Patching won the tokenization war:** even Chronos, the quantization pioneer, abandoned its own vocabulary. Tokens should be shapes, not samples — episode 13's lesson, industrialized.
4. **The backbone matters less than the recipe:** a 35M-parameter xLSTM hangs with 200M-parameter transformers. Pick by deployment shape (streaming? edge? batch?), not by ideology.
5. **In-context learning needs complementary context:** feeding fifty sibling series to Chronos-2 changed nothing — the neighbors must know something the target doesn't, or the group is just redundancy.
6. **Cold start is the killer app:** 60 days of history was enough for zero-shot models to beat a fully-trained GRU, and nowhere near enough to train anything. If your catalogue is full of young series, this paradigm is for you.
7. **Single windows lie:** the model that won the close-up finished behind every foundation model over 120 forecasts. Walk-forward aggregates or it didn't happen.
8. **Trust, but decontaminate:** public benchmarks may live inside the pre-training corpora. Evaluate on private data, demand post-cutoff tests, and never retire the seasonal naive baseline. No free lunch — not even a pre-trained one.

United States of Banan is a reader-supported publication. To receive new posts
and support my work, consider becoming a free or paid subscriber.

[ ✓ Subscribed ]
