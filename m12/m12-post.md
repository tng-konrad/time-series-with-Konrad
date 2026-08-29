# Time Series with Konrad: episode 12

### Transfer learning for time series: forecasting with borrowed experience

**KONRAD BANACHEWICZ**
OCT 12, 2026

Every model in this series so far — from exponential smoothing (episode 2) to last episode's LSTMs — has shared a quiet assumption: the series you want to forecast comes with *enough history to learn from*. Real deployments violate it constantly: a newly opened store, a freshly installed sensor, a product launched last quarter. You have four months of data and a stakeholder who wants a weekly forecast anyway.

↪ *That quiet "enough history to learn from" assumption goes all the way back to the classical smoothers of episode 2 — this episode is what to do when it fails.* → **<LINK TO EPISODE 2 HERE>**

The classical answer is to shrug and fit something tiny. The deep learning answer — the subject of this episode — is **transfer learning**: pretrain a network on a *source* domain where data is plentiful, then hand the learned weights to your data-starved *target* as a starting point. The network arrives already knowing what demand curves look like; the scarce local data only has to teach it what makes *this one* different. Last episode ended by noting that deep learning earns its complexity budget through *global training* — one model across many related series. This episode is that idea, weaponized.

The core principle:

**A data-scarce series cannot speak for itself — so you let it borrow weights from series that can. The entire craft is in choosing what to borrow, and how hard to overwrite it.**

The ladder we climb: training from scratch on 120 days of data (the humbling baseline) → supervised pretraining on 50 related series and **zero-shot transfer** → fine-tuning the head only → fine-tuning everything → two flavors of **self-supervised pretraining** (masked autoencoding and contrastive learning) → a sample-efficiency curve → and a deliberate act of **negative transfer**, where we pretrain on Melbourne weather to forecast shop demand, just to watch it fail. Real data throughout — the Kaggle store-item demand dataset, 500 daily retail series — and honest errors throughout, including the rungs that lose to a one-line baseline. Practical understanding over rigor, as always.

📓 All code lives in the companion notebook:
https://github.com/tng-konrad/time-series-with-Konrad/blob/main/m12/m12-fable-version.ipynb

## Two domains, one task

Let's make the scenario concrete. A retail chain has an established store with five years of daily sales history across 50 items — our **source domain**. It also has a recently opened store, where item 15 has been selling for about four months — our **target domain**. Head office wants weekly demand forecasts for the new store *now*. The task is the same everywhere: given 28 days of history, predict the next 7. The data budgets are anything but. A **window**, in the sense used all through this episode, is one training example sliced from a series — 28 consecutive days as the input, the following 7 days as the label, the slice then slid forward to produce the next example. Cut up this way, the source yields roughly 40,000 training windows. The target yields 86.

> **[FIGURE 1 — graphs/graph12-01.png — notebook cell 26: two-panel plot, three source series (top) and the target series (bottom)]**

Look at the two panels and you can see why this is a *friendly* transfer setting. Every series — source and target alike — speaks the same shape grammar: an annual cycle, a weekly rhythm visible as the thickness of the band, a mild upward trend. What differs is the *level*: one item sells 20 units a day, another 100. The dynamics generalize; the scales don't.

That observation drives the single most load-bearing preprocessing decision of the episode: **per-series standardization**. Each series is z-scored by its own mean and standard deviation, so the network only ever sees "standard deviations around my own typical level". This is what makes weights portable at all — a network trained on raw store-1 quantities would be lost on a target with a different baseline, but a network trained on shapes can read any series that has been translated into shape language. (The scaling statistics come from training data only, and for the target that means the scarce 120 days — the usual leakage discipline from episode 8, doing double duty.)

↪ *Why the scaler may only ever see training data is the crux of episode 8 on validation and leakage — the discipline that keeps every number in this post honest.* → **<LINK TO EPISODE 8 HERE>**

> **[FIGURE 2 — graphs/graph12-02.png — notebook cell 28: the target series with the ignored history in grey, the 120 available days highlighted, the 84-day test period in green]**

The split is chronological, as always: the final 84 days are the test period — twelve non-overlapping one-week forecasts, walk-forward style, where the model receives the actual observed previous 28 days at each origin (the way it would in production) but is never retrained. One detail in that plot deserves a pause: the test window covers the year-end demand slide, and a model trained only on the 120 summer-and-autumn days *has never seen this part of the annual cycle*. A test like that may seem harsh, but it mirrors reality — every newly opened store eventually meets its first winter. Whether a forecaster can anticipate a season it never observed locally is where borrowed knowledge has to earn its keep.

Before any neuron trains, the goalposts:

```
persistence:     {'MAE': 24.14, 'RMSE': 29.04}
seasonal naive:  {'MAE': 13.61, 'RMSE': 17.97}
```

Both numbers are errors in units sold per day: MAE is the typical daily miss, RMSE the version that punishes large misses harder, and lower is better for both. Persistence (next week = flat line at today's value) is poor — it can't follow the weekly rhythm. But **seasonal naive** (next week = last week) lands at RMSE 17.97, and that number deserves respect: retail demand is dominated by exactly the weekly pattern that copy-last-week reproduces for free. Hold onto 17.97. Not every rung of this ladder will clear it. ;-)

## Rung 1: from scratch, and a dose of humility

The architecture for the whole episode is deliberately modest: a single GRU layer (64 units — see episode 11 for what the gates are doing) reads the 28-day window, and a `Dense(7)` head maps the final hidden state to a week of forecasts. Two named parts — **encoder** and **head** — and that split is *the* central abstraction of transfer learning: the encoder learns a general-purpose representation of "what the last four weeks looked like", the head is a task-specific readout. Transferring a model means keeping the first and re-training (or replacing) the second.

↪ *If "GRU", "encoder" and "hidden state" need unpacking, episode 11 built these recurrent networks from the ground up — the gates especially.* → **<LINK TO EPISODE 11 HERE>**

First, the no-transfer reference point: random initialization, 120 days of history, hope. The windowing arithmetic is brutal — 120 days yield 86 windows, split chronologically into 69 for training and 17 for validation. (A caveat on that split: consecutive windows overlap, so the last training window and the first validation window share 34 of their 35 days, which makes the validation loss mildly optimistic. A purged gap à la episode 8 would fix it — but at 86 windows total we can't afford to burn a third of the training set on a gap, so we accept the optimism and let the untouched 84-day test period deliver the real verdict.) Against those 69 windows stand 13,319 parameters. This is the textbook overfitting configuration, and early stopping is the only thing standing between the network and pure memorization.

```
from scratch:    {'MAE': 15.82, 'RMSE': 20.05}
```

Read that next to the baseline table. The deep network, trained from scratch on four months of data, **loses to copy-last-week** (20.05 vs 17.97). This deserves to be stated bluntly, because it is the single most common failure mode of deep learning on short series — and it is the standard argument for either staying classical or importing knowledge from elsewhere.

> **[FIGURE 3 — graphs/graph12-03.png — notebook cell 40: observed test period vs the from-scratch forecast and the seasonal naive]**

The picture explains the number. The scratch network has learned the weekly wiggle — walk-forward evaluation feeds it fresh true history every week, so the oscillations sit in roughly the right places — but its level tracking is loose, and through the December decline it drifts, because nothing in its 120 training days told it that demand falls in winter. Seasonal naive tracks the level embarrassingly well for a method with zero parameters; its weakness is being exactly one week behind every change. Keep both failure patterns in mind. The transfer rungs will fix the network's; nothing will fix the baseline's.

## Rung 2: supervised pretraining, and a zero-shot surprise

New ingredient: **a source domain**. We train the *identical* architecture on all 50 store-1 series — each z-scored by its own statistics, windowed, and pooled into one training set of ~40,000 windows. Because the source task (predict next week from the last four) is the same as the target task, and because a series' future values come free with the data, this is **supervised pretraining** — no labeling budget required, which is a luxury forecasting enjoys and classification does not.

There is nothing exotic about the pretraining itself: same MSE loss, bigger batches, eight epochs, under a minute on a laptop GPU. The interesting part is one line at the end, `get_weights()`, which extracts the learned parameters as a list of arrays. That list is the distilled experience of five years times fifty series, and every remaining rung of the ladder is a different opinion about what to do with it.

The cheapest opinion: do nothing. Apply the source model to the target with **zero adaptation** — what the foundation-model literature calls **zero-shot forecasting**:

```
zero-shot transfer:  {'MAE': 12.65, 'RMSE': 15.63}
```

A model that has *never seen a single observation from the target store* beats the from-scratch network by more than four RMSE points, and clears the seasonal-naive bar that scratch could not. Why does this work? Two reasons, worth separating. Mechanically, per-series standardization means the target's inputs arrive in the language the network already speaks. Statistically, the dynamics of item-level demand — the weekly rhythm, the way peaks decay, the autumn slide that the target's own 120 days never showed — are shared across stores, and 40,000 windows estimate them far better than 69 ever could.

This is, in miniature, the pitch of time series foundation models (more on those in Closing time). It also sets up the obvious question: if *ignoring* the target's data already wins, can its 69 windows buy anything more?

## Rungs 3 and 4: fine-tuning, two flavors

They can. The mechanism is **fine-tuning** — resuming training from the pretrained weights on the target's own data — and the literature splits it into two configurations along one axis: how much of the network the scarce labels are allowed to touch.

**Head-only fine-tuning** (a.k.a. *partial-parameter* fine-tuning): transplant all the pretrained weights, freeze the encoder, retrain only the head. With 455 trainable parameters against 69 windows, overfitting is nearly impossible — this is the safest way to spend scarce target labels, and it makes **catastrophic forgetting** (the destruction of pretrained features by aggressive updates on a tiny dataset) structurally impossible. One Keras gotcha that costs people real debugging hours: `trainable = False` only takes effect at *compile* time. Flip the flag without recompiling and the encoder happily trains anyway.

```
head-only fine-tune:  {'MAE': 12.30, 'RMSE': 15.36}
```

A modest step past zero-shot: the head calibrates the readout to this particular series — its noise level, its slightly different weekly profile — while the representation stays pure store-1 knowledge.

**Full fine-tuning**: unfreeze everything and let the target data reshape the entire network. Most expressive, most dangerous — with 69 windows, a network given free rein could shred five years of borrowed knowledge chasing one summer's noise. Hence the two safety rails, both boring, both essential:

- **A gentle learning rate.** We drop from the Adam default `1e-3` to `1e-4`. Starting from good weights, large steps can only destroy information before rebuilding it from data that cannot support the reconstruction. Small steps *adjust*.
- **Early stopping**, watching a chronological validation slice, halting the moment the encoder starts memorizing.

```
full fine-tune:  {'MAE': 11.58, 'RMSE': 14.85}
```

Best score of the episode. And the ordering that has emerged — scratch 20.05 → zero-shot 15.63 → head-only 15.36 → full 14.85 — is the canonical transfer learning result, worth internalizing as a hierarchy of *how much you trust the target data*: not at all, only to calibrate the readout, or to gently reshape the representation itself.

> **[FIGURE 4 — graphs/graph12-04.png — notebook cell 55: observed test period vs from-scratch and full fine-tune forecasts]**

If the episode had to be a single frame, this would be it. The fine-tuned forecast holds the weekly amplitude steadily and — the decisive difference — *follows the year-end decline down*, because its encoder watched four winters happen in fifty other series. The scratch model, whose entire worldview is one summer, has no reason to expect December to differ from October. Transfer learning's contribution is not sharper wiggles; it is **imported seasonal context** that the target's own history could never provide.

## Rung 5: self-supervised pretraining — the masked autoencoder

So far, pretraining was supervised — possible because forecasting labels come free. But suppose your source corpus supports no such task (mixed horizons, no clean targets, or a downstream task that isn't forecasting at all). The modern answer is a **pretext task**: an artificial objective that forces a network to learn structure from unlabeled sequences. The pretraining literature sorts these into two great families, and we'll build the essential version of each.

Family one is **generative / reconstructive**: corrupt the input, train the model to restore it. The lineage runs from TimeNet (a GRU autoencoder trained across many datasets at once) to Ti-MAE and the masked-patch transformers; our recurrent version masks 40% of the timesteps in each source window — zeroing them, which after standardization means "an unremarkable day" — and trains an encoder–decoder to reconstruct the original from the corrupted version. Filling a hole mid-week requires knowing where in the weekly cycle it sits; filling several consecutive holes requires a sense of the local trend. In principle, exactly the skills a forecaster needs.

> **[FIGURE 5 — graphs/graph12-05.png — notebook cell 58: one source window, original vs the masked input with 40% of days zeroed]**

The encoder is *identical in shape* to the forecaster's encoder — that's what makes the coming transplant a one-liner — and the decoder is the seq2seq skeleton from episode 11 (`RepeatVector`, GRU, `TimeDistributed(Dense(1))`). The fit call is where self-supervision becomes visible: input `X_masked`, target `X_original`, labels nowhere in sight.

So it trains, the loss goes down, and then we do the thing this series always does: *look at what was actually learned*.

> **[FIGURE 6 — graphs/graph12-06.png — notebook cell 62: original window, masked input, and the autoencoder's reconstruction — which is a nearly flat line at the window's mean level]**

Look closely at that reconstruction, because it is doing much less than the dropping loss suggested. The autoencoder nails the window's *level* and slow drift — and essentially nothing else; the day-to-day structure is smoothed into a near-flat line. No training bug here: when the fine structure of noisy demand is hard to pin down, predicting each window's own average is a safe, low-loss strategy, and minimizing MSE through a 64-number bottleneck settles on it rationally. Still, it should worry us, because our standardization already removed the level. An encoder whose code mostly says "this window sits 1.2 standard deviations above its mean" has learned something *true* and possibly something *useless*.

The transplant confirms it. (One craft note: because the forecasting head starts random, fine-tuning must go in two stages — warm up the head with the encoder frozen, *then* unfreeze at the gentle rate. Skip the warm-up and the random head's large early gradients shred the pretrained encoder before the head is even calibrated; we measured the one-stage shortcut at close to 3 RMSE points worse.)

```
masked AE + fine-tune:  {'MAE': 15.85, 'RMSE': 20.15}
```

Roughly level with training from scratch. Miles behind supervised pretraining. Still short of copy-last-week. :-/

None of this says "autoencoders bad". What it does say: **the pretext task must be aligned with the downstream task, and its benefits must be measured, not assumed**. Reconstruction rewards summarizing what happened, while forecasting demands committing to what happens next — and on a modest, homogeneous corpus, the two share less than one might hope. When labels for your actual task are free — as they always are in forecasting — supervised pretraining is very hard to beat.

## Rung 6: contrastive pretraining

Family two abandons reconstruction entirely. **Contrastive learning** creates two randomly *augmented* views of each window and trains the encoder to place views of the same window close together in embedding space — the space of the encoder's outputs, where every window becomes a vector of 64 numbers — while pushing apart views of different windows. The literature calls this pretext **instance discrimination**: the encoder learns what makes each window *itself*, invariant to nuisance perturbations. This is the engine inside TS-TCC, TS2Vec and friends; we implement its clean core. (Clean also means simplified: we draw the two augmented views once, up front, and reuse them for all eight epochs — full SimCLR-style training re-samples fresh augmentations every batch, which strengthens the invariance signal at the cost of a custom data pipeline.)

The augmentations are the design decision, because **choosing augmentations is choosing your invariances**. Ours: random amplitude scaling (each window multiplied by a factor in 0.8–1.2 — "overall size is a nuisance") and jitter (small Gaussian noise — "day-level noise is a nuisance"). What survives both is the window's *shape*, so shape is what the embedding encodes.

> **[FIGURE 7 — graphs/graph12-07.png — notebook cell 67: one source window and its two augmented views — a positive pair]**

The loss — **NT-Xent**, inherited from SimCLR — reads as a classification problem in disguise:

```
loss_i = −log [ exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ) ]
```

In plain English: normalize all the embeddings in the batch, then ask each one a multiple-choice question — *out of everyone here, which embedding is your partner view?* — scored by softmax over cosine similarities. The temperature `τ` sharpens the question: low τ makes the loss obsess over the hardest, most confusable negatives. With batch size 256 the question has 511 candidate answers, so random guessing scores ln(511) ≈ 6.2; our training loss drops from ~3.6 to ~2.1, meaning the encoder gets much better than chance at re-identifying windows through their distortions; not perfect though, nor should it be, since retail demand windows genuinely resemble one another.

For the adaptation step we go maximally frugal, straight from the transfer learning playbook for low-data regimes: freeze the encoder, extract its 64-number embedding for each of the target's 86 windows, and fit a **ridge regression** as the head. Not even a neural layer — a linear map with an L2 penalty, fitting in milliseconds, with nothing left to overfit.

```
contrastive + ridge head:  {'MAE': 15.06, 'RMSE': 18.97}
```

A full point better than scratch and the masked autoencoder, so instance discrimination did produce genuinely transferable shape-features — but still a point behind seasonal naive, and four behind supervised transfer. A fair reading: on a small, homogeneous source corpus, and for a downstream task whose labels were free anyway, the discriminative pretext cannot match pretraining on the task itself. (One caveat from the literature worth passing along: an encoder trained to ignore amplitude can become *amplitude-agnostic*, a liability when size matters downstream, and the reason production frameworks bolt on mechanisms to reintroduce it.) Where contrastive encoders earn their reputation is a different setting: one encoder pretrained once, serving *many* downstream tasks (classification, anomaly detection, forecasting), each with its own cheap head.

## When does transfer help?

That completes the ladder. Before the final scoreboard, two experiments zoom out and map its boundaries — how much target data it takes for transfer to stop mattering, and what happens when the source is chosen badly.

First, the data question. The real claim behind transfer learning is about sample efficiency: pretrained models should need **less target data** to get good. That's a claim about a curve, so we trace it — for target histories of 60, 120, 240 and 480 days, train a from-scratch model and a fully fine-tuned model under identical protocols:

```
target history   from scratch   full fine-tune
60 days             27.23           15.22
120 days            20.05           14.85
240 days            26.20           14.65
480 days            14.22           13.82
```

> **[FIGURE 8 — graphs/graph12-08.png — notebook cell 79: test RMSE vs days of target history, both methods, with the seasonal-naive bar as a horizontal reference line]**

Read it as a practitioner's decision chart. The fine-tuned curve is **low and steady** — even 60 days (a mere 26 windows!) lands within a point and a half of the best result. The scratch curve is **high and erratic**, and not even monotonic: 240 days scores *worse* than 120, because at these sample sizes training is so seed- and split-sensitive that luck dominates. That instability is itself a finding — below a few hundred days, a from-scratch deep model is not merely worse on average, it is *unreliable*. The vertical gap between the curves is what pretraining is worth at each data size, and it is largest where forecasting is hardest. **Transfer learning buys the most precisely when you have the least.**

Second, the source question. Everything so far pretrained on demand data to forecast demand. But the promise "pretrain on anything big, fine-tune on your problem" is most tempting when even related data is scarce — so what does an *unrelated* source buy? We pretrain the same architecture on ten years of daily Melbourne minimum temperatures (yes, the very series from episode 11 — a perfectly good dataset with real daily dynamics, and nothing whatsoever to do with retail):

```
weather zero-shot:   {'MAE': 16.78, 'RMSE': 20.78}
weather fine-tune:   {'MAE': 16.84, 'RMSE': 20.66}
```

**Negative transfer**, live on stage. Weather zero-shot is worse than training from scratch — the encoder confidently applies temperature physics to shop demand: smooth mean-reverting drifts, no weekly rhythm, because weather does not know about weekends. And fine-tuning claws back only to scratch level: 69 windows are enough to *calibrate* a right representation, not to *rebuild* a wrong one. Same recipe on the matched source: 14.85.

The practical sting of negative transfer is that it is invisible without a baseline — the pipeline runs, the losses decrease, the forecasts look plausible. There are two standard defenses. Always train the scratch baseline; it is cheap insurance. And choose sources by *measured similarity* rather than availability — the classification-transfer literature ranks candidate source datasets by their Dynamic Time Warping distance to the target (a shape-similarity measure that stretches the time axis to line two series up) and pretrains on the nearest, which reliably predicts which transfers will help. "Pretrain on something big" is half the recipe. The other half is "…that resembles your target."

## The verdict

The full scoreboard, sorted best-first — every method scored by the identical walk-forward protocol on the same twelve test weeks:

```
                            MAE    RMSE
full fine-tune            11.58   14.85
head-only fine-tune       12.30   15.36
zero-shot transfer        12.65   15.63
seasonal naive            13.61   17.97
contrastive + ridge head  15.06   18.97
from scratch              15.82   20.05
masked AE + fine-tune     15.85   20.15
weather fine-tune         16.84   20.66
weather zero-shot         16.78   20.78
persistence               24.14   29.04
```

> **[FIGURE 9 — graphs/graph12-09.png — notebook cell 87: horizontal bar chart of test RMSE, color-coded by pretraining source (green = supervised on the matched demand source, turquoise = self-supervised on the same source, blue = no pretraining, orange = mismatched weather source), with the seasonal-naive bar as a vertical reference line]**

Two readings of that board. First, the podium belongs entirely to supervised transfer, and even zero-shot, which used **no target data at all**, beat everything trained on the target alone. Second, look where the naive baseline sits: above the entire self-supervised contingent, above scratch, above both weather transfers. In the chart, the only bars left of the red line are the green ones — supervised pretraining on relevant data, *on the task itself*. The turquoise bars used the very same source and still fell short: relevant data alone was not enough; the pretext had to match the downstream task too. Matched-source pretraining plus gentle fine-tuning cut the from-scratch error by a quarter and cleared the strong baseline by three RMSE points — on a series with four months of history. Everything else on the board is a way of spending a complexity budget without earning it back.

## Closing time

We climbed from a data-starved series to a borrowed brain. From-scratch training established that 69 windows cannot feed 13,319 parameters. Supervised pretraining on fifty sibling series produced a model that beat local training *zero-shot*, and fine-tuning added a hierarchy of trust on top: freeze the encoder when you trust the target least, unfreeze gently as evidence accumulates. The two self-supervised families — reconstruct the corrupted, discriminate the augmented — showed that pretext tasks are hypotheses, not guarantees; ours learned levels our scaling had already removed, and one look at the reconstruction plot told us more than any amount of tuning would have. The sample-efficiency curve located transfer's payoff right where data is scarcest, and the weather experiment demonstrated that the *choice* of source matters more than the *act* of pretraining.

The throughline: **pretraining is a hypothesis about your data, and hypotheses get tested** — against a scratch baseline, against a naive baseline, on a chronological split. Every genuinely bad outcome in this episode (masked AE, weather transfer) would have looked like a success to a pipeline that skipped its baselines.

Where does this go next? Everything we hand-rolled — pretrain on a corpus, transfer, adapt cheaply — is exactly what **time series foundation models** (Chronos, TimesFM, MOMENT, MOIRAI, Lag-Llama, TinyTimeMixer) productize at scale: transformer-family architectures pretrained on millions of heterogeneous series, applied zero-shot or lightly fine-tuned. Our zero-shot rung is a working miniature of that paradigm, and our negative-transfer rung is the caveat that ships with it, because a matched small source plus fine-tuning can still beat a generic giant. There is also a rich middle ground we only waved at: **domain adaptation**, for when the target has *no* labels at all (adversarial feature alignment, source-free adaptation, and a whole taxonomy of what you're allowed to access while adapting). Both are natural sequels.

## Key Takeaways

1. **Scale is the enemy, shape is the asset:** per-series standardization is what makes weights portable — the network reasons in "standard deviations around my own level", a language every series speaks.
2. **Zero-shot is the cheapest experiment you're not running:** a model pretrained on fifty related series beat local training without seeing the target at all. Try it before you train anything.
3. **Fine-tuning is a trust dial, not a switch:** freeze the encoder when target data is scarce (455 parameters can't overfit); unfreeze everything only with a small learning rate and early stopping riding shotgun.
4. **Pretext tasks are hypotheses:** masked reconstruction learned window levels — true, and useless after standardization. Inspect what your pretraining actually learned before transplanting it.
5. **Negative transfer is invisible without baselines:** weather pretraining underperformed scratch while looking perfectly healthy in training. The scratch baseline is cheap insurance; DTW-based source selection is the principled fix.
6. **Transfer pays where data is short:** at 60 days of history the fine-tuned model beat scratch by 12 RMSE points; at 480 days the gap was 0.4. Borrow when you're poor.
7. **The naive baseline remains undefeated as a referee:** seasonal naive outscored every method that didn't transfer from a matched source. Check it first, always.

United States of Banan is a reader-supported publication. To receive new posts
and support my work, consider becoming a free or paid subscriber.

[ ✓ Subscribed ]
