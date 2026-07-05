# Time Series with Konrad: episode 11

### Deep learning for time series: RNN, GRU, LSTM, and the fine art of not forgetting

**KONRAD BANACHEWICZ**
SEP 14, 2026

Every model in this series so far has shared one worldview: write down a small parametric story about how the series evolves — a smoothed level (episode 2), an ARMA polynomial (episode 3), a state space recursion (episode 10) — and estimate a handful of parameters, one series at a time. This episode switches paradigms. A **recurrent neural network (RNN)** makes almost no assumptions about the shape of the dynamics. Instead it maintains a **hidden state** — a vector of numbers acting as a running memory — and *learns* how to update that memory from the data itself. No stationarity tests, no order selection, no seasonal differencing.

You might expect this to be the episode where the neural networks sweep the board. It isn't (spoiler: the final scoreboard is a photo finish, and the naive baseline is closer than anyone would like to admit). What the networks buy you is different, and more interesting than raw accuracy — but that freedom has a price, and the price is the real subject of this episode. Plain RNNs *forget*: the mathematics of backpropagation through time makes gradients decay or explode exponentially with sequence length, so a vanilla recurrent cell cannot learn dependencies more than a few dozen steps back. The whole modern zoo — GRU, LSTM, encoder–decoders, bidirectional wrappers — exists to route around that single mathematical fact.

The core principle of the episode:

**A recurrent network's memory is a vector, and keeping the gradient flowing through that memory is the whole game — the architectures are just increasingly clever plumbing for it.**

The ladder we climb: turning a series into supervised tensors (and dodging a leakage trap on the way) → a NumPy experiment that shows the vanishing gradient with your own eyes → SimpleRNN → stacked RNN → GRU → LSTM → three competing strategies for multi-step forecasting → covariates and a bidirectional finale. Real data throughout: ten years of daily minimum temperatures in Melbourne, 3,650 observations, trained in minutes on a laptop (on an Apple silicon Mac, the notebook uses the built-in GPU — Keras 3 on the PyTorch backend, one environment variable). Practical understanding over rigor, as always.

📓 All code lives in the companion notebook:
https://github.com/tng-konrad/time-series-with-Konrad/blob/main/m11-fable-version.ipynb

## From a series to supervised tensors

A feedforward network assumes its inputs are unordered and independent — feed it the same thirty numbers shuffled and it can't tell the difference. Time series violate that assumption by construction. The recurrent fix: process the sequence *one step at a time*, carrying a hidden state `h_t` that summarizes everything seen so far:

```
h_t = tanh(W_x·x_t + W_h·h_{t-1} + b)
```

In plain English: new memory = a squashed blend of the new observation and the old memory. The two weight matrices are *shared across all time steps* — the network learns one update rule and applies it recursively, which is what lets the same layer handle a window of length 30 or 300. The `tanh` keeps the state bounded in (−1, 1); remember that detail, it comes back to haunt us shortly.

But a neural network doesn't consume a time series — it consumes (input, target) pairs. The bridge is **windowing**: slide a 30-day window along the series; its contents become one training input, the next day becomes the target, shift by one, repeat. A series of n observations yields roughly n overlapping examples, stacked into the 3D tensor every Keras recurrent layer demands:

```
[batch, timesteps, features]
```

How many windows, how long each window is, how many variables are measured per step. Univariate forecasting means `features = 1`; the multivariate finale will just make that number bigger. **Forecasting has become supervised learning**, and everything we know about training and validating supervised models applies — including the ways it can go quietly wrong.

> **[FIGURE 1 — notebook cell 18]** Daily minimum temperature in Melbourne, 1981–1990. A strong, obvious annual cycle (Southern Hemisphere summers peaking around New Year), no trend, and a thick band of day-to-day weather noise around the seasonal signal.

The seasonal part is easy; the noise band is irreducible; the interesting question is how much of the in-between structure — cold snaps, warm spells, multi-day persistence — each architecture can capture.

Two pipeline decisions before any model, and the *order* between them is the lesson:

1. **Split chronologically first.** Eight years (1981–1988) for training, the last two for validation. Never shuffle a time series — episode 8 was an entire post about why.
2. **Then scale, fitting on the training segment only.** Networks want inputs in a small range (large values push `tanh` and sigmoids into their flat, gradient-free zones), so we MinMax-scale to roughly [0, 1]. The trap — present in a distressing share of published deep learning forecasting code — is calling `fit_transform` on the *full* series before splitting. Do that and the scaler's min and max have quietly absorbed information from the validation years into every training value.

*Pro tip: after the honest version, the scaled validation segment is not guaranteed to stay inside [0, 1] — if 1989 had brought a record heat wave, it would scale above 1. That slight untidiness is what no-leakage looks like. If your scaled test set fits suspiciously perfectly, ask why.*

And because windows overlap, one more subtlety: a window is assigned to train or validation by *where its target falls*, not where it starts. The first validation window predicts 1 January 1989 from the last thirty days of December 1988 — inputs from the past are fair game, because by the time you forecast January 1989 for real, December 1988 is known history.

Before a single neuron trains, we fix the goalposts with two naive baselines — tomorrow equals today (**persistence**), and tomorrow equals the same day last year (**seasonal naive**):

```
{'Persistence (t-1)':      {'MAE': 1.953, 'RMSE': 2.481}}
{'Seasonal naive (t-365)': {'MAE': 2.965, 'RMSE': 3.728}}
```

Persistence lands at 2.48 °C — day-to-day weather is sticky. The seasonal naive is much worse, because any *particular* day last year carries a full dose of last year's noise. Every network below is judged against the 2.48 mark; the empirical literature is littered with published architectures that never checked. ;-)

## Why plain RNNs forget

Before fitting anything, let's watch the disease the fancier architectures cure — no Keras, no training, just repeated matrix multiplication.

Training a recurrent network means **backpropagation through time (BPTT)**: unroll the network across the sequence, push the loss gradient backwards through every step. Each step backwards multiplies the gradient by the Jacobian of the state transition — essentially by the recurrent weight matrix `W_h`, damped further by the activation's derivative. Over a gap of k steps the gradient contains a *product of k such matrices*. The scalar caricature says it all:

```
∂h_T / ∂h_0 ≈ w^T
```

Powers of a number: collapse to zero for |w| < 1, blow up for |w| > 1. The multivariate version is governed by the **spectral radius** ρ of `W_h` — the magnitude of its largest eigenvalue. ρ < 1 is *sufficient* for gradients across long gaps to vanish (and the `tanh` derivative, always below 1, only makes it worse); ρ > 1 is *necessary* for them to explode. There is no comfortable middle.

The experiment: draw a random 32×32 matrix (the size of our future hidden layers), rescale its spectral radius to exactly 0.9, 1.0, or 1.1, multiply it by itself sixty times, and record the norm.

> **[FIGURE 2 — notebook cell 29]** Gradient magnitude vs. distance backpropagated, log scale. Three straight-ish lines: ρ = 0.9 decays steadily to a couple of percent of its starting value by step 60; ρ = 1.1 amplifies by a factor of hundreds; ρ = 1.0 holds roughly level.

The picture is the whole vanishing-gradient literature in one chart. At ρ = 0.9 — a matrix only *slightly* contractive — whatever happened sixty days ago contributes a couple of percent of its original signal to the weight update; push the horizon to a few hundred steps and it is effectively zero. At ρ = 1.1 the same sixty steps head toward numerical overflow. And ρ = 1.0 is exactly what it looks like: a knife-edge that nothing in gradient descent keeps a weight matrix balanced on.

The practical consequences are asymmetric. **Exploding** gradients announce themselves loudly (the loss becomes `NaN`) and have a crude fix: *gradient clipping* — rescale any gradient whose norm exceeds a threshold. **Vanishing** gradients fail silently: training proceeds, loss decreases, and the network simply never learns that what happened forty days ago matters, because no usable error signal survives the trip back. That silent failure is what gates were invented for.

## SimpleRNN: the first rung

The simplest recurrent forecaster: one `SimpleRNN` layer implementing exactly the `tanh` update above, and one `Dense` layer projecting the final hidden state onto a single number — tomorrow's scaled temperature. Total parameter count: 1,121. Compare that with the *two* parameters of single exponential smoothing back in episode 2, and you see immediately why neural networks need more data and a guardrail against overfitting.

The guardrail is **early stopping**: watch the validation loss after every epoch, and when it stops improving for five epochs running, halt and *rewind to the best weights*. It plays the role AIC played for ARIMA — the brake on fitting the noise. (The gotcha flag is `restore_best_weights=True`; without it you keep the weights from five epochs *past* the peak, already sliding downhill.)

> **[FIGURE 3 — notebook cell 36]** SimpleRNN training history: training and validation loss per epoch. A steep drop in the first two epochs as the network learns the seasonal shape, then a long shallow tail until early stopping calls it around the halfway mark.

One quirk in that chart worth demystifying: the validation loss sits *below* the training loss, which looks backwards. Two boring reasons suffice — the training loss is averaged over the whole epoch (including its clumsy early batches) while validation is measured once, with the improved end-of-epoch weights; and these two particular validation years simply contain slightly tamer weather than the eight training ones. Neither is a red flag. A validation loss creeping *upwards* would be.

The score, after un-scaling back to degrees Celsius (skip that inverse transform and you'll report impressively tiny errors on the [0,1] scale, meaning nothing):

```
{'MAE': 1.759, 'RMSE': 2.227}
```

Beats persistence (2.48). The thirty-day window lets the network average out weather noise and lean on the seasonal position — something a one-day-memory forecast structurally cannot do.

> **[FIGURE 4 — notebook cell 40]** SimpleRNN one-step-ahead forecast vs. actuals, first 200 validation days. The prediction tracks the seasonal descent into winter and the multi-day swings, but visibly smooths the series — the sharpest one-day spikes are undershot.

That smoothing is not a flaw to fix. It is what minimizing squared error looks like when part of the variation is genuinely unpredictable: the model forecasts the conditional mean and lets the irreducible noise go. Hold that thought — it returns with a vengeance in the multi-step section.

Can we do better by stacking a second recurrent layer on top? The code change is one flag — `return_sequences=True` on the first layer, so it hands the *whole* sequence of hidden states to the layer above instead of just the final one (forget the flag and you meet the most famous error message in Keras RNN work: *expected ndim=3, found ndim=2*). The honest result:

```
{'MAE': 1.768, 'RMSE': 2.246}
```

A hair *worse* than the single layer. A small dataset with one strong seasonal signal offers the extra layer nothing new to find, so its extra parameters buy noise. **More depth does not mean better forecasts** — the recurring lesson of this series, now in neural form.

## GRU: gates as learned valves

Now the first cure for the vanishing gradient. The **Gated Recurrent Unit** keeps the single hidden state but wraps its update in two learned valves — each a sigmoid, so each is a vector of numbers between 0 and 1, one per hidden unit. The **reset gate** `r_t` decides how much old state to consult when proposing new content; the **update gate** `z_t` decides how much of the state to actually overwrite:

```
h_t = (1 − z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

That equation is the cure, and it rewards a stare. The plain RNN *replaces* its state wholesale through a `tanh` at every step — sixty steps means sixty matrix multiplications between an early input and a late gradient, the geometric decay from our NumPy detour. The GRU update is instead a **weighted average**: wherever the update gate sits near 0, the state — and, crucially, the gradient flowing back through it — passes essentially untouched. An *additive shortcut through time* instead of a multiplicative gauntlet. And the gates are learned, so the network itself decides, dimension by dimension and step by step, what to preserve and what to rewrite.

In Keras the diff is literally one word — `SimpleRNN` becomes `GRU` — though each layer now computes three transformations where the plain RNN computed one, so the parameter count roughly triples (9,729 vs. 3,201 for the stacked pair). The score:

```
{'MAE': 1.741, 'RMSE': 2.219}
```

Ahead of both plain RNNs — but only by a nose. Honest reading: gating is about *long-range* credit assignment, and with a 30-step window over a strongly seasonal series, there is only so much long range to assign. The gap grows with sequence length and problem complexity; the point of the ladder is that when your problem does need 200 steps of memory, the fix is a one-word diff.

## LSTM: the constant error carousel

The second cure is older, bigger, and the most famous: the **Long Short-Term Memory** cell (Hochreiter & Schmidhuber, 1997). Where the GRU has one state and two gates, the LSTM splits memory into two parallel pathways: the **cell state** `c_t` — a protected "memory highway" updated only by element-wise scaling and addition, never squashed through a `tanh` between steps — and the **hidden state** `h_t`, the working output. Three sigmoid gates manage the traffic (forget `f_t`, input `i_t`, output `o_t`):

```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
h_t = o_t ⊙ tanh(c_t)
```

The gradient of `c_t` with respect to `c_{t-1}` is, to leading order, just `f_t` — no weight matrix, no activation derivative. Where the network learns to hold the forget gate near 1, gradients flow back across long gaps essentially undamped. The original papers call this the **constant error carousel**, and it is the GRU's additive-shortcut idea in its purest form. The price: the heaviest parameter bill of the family. The whole zoo in one line —

```
RNN: 1 transformation per layer  |  GRU: 3  |  LSTM: 4
```

— everything else is bookkeeping. The score, same scaffolding, one-word diff, 12,705 parameters:

```
{'MAE': 1.741, 'RMSE': 2.219}
```

Identical to the GRU to the third decimal on this run, and the two trade places freely under different seeds. Practical folklore worth recording: try the GRU first (fewer parameters, faster), reach for the LSTM when sequences are very long. What you should *not* expect from either is a large win over well-configured simpler models on a short-window univariate task. The gates solve a memory problem; you only get paid where the problem exists.

## Multi-step forecasting, three ways

Everything so far predicts one day ahead. Real planning questions rarely stop there — and *how* you extend a model to a horizon is a genuine modeling decision with its own failure modes. We forecast 14 days from each 30-day window, three ways:

1. **Recursive:** keep the one-step LSTM, feed each prediction back in as if it were an observation, repeat 14 times. Exactly how ARIMA produced its multi-step forecasts in episode 3. No new training — but every step consumes the previous step's *error* as input, so mistakes can compound. How badly turns out to depend on the series; we will measure it rather than assume it.
2. **Direct multi-output:** change the read-out to `Dense(14)` and train the network to emit the whole horizon in one forward pass. Every output is anchored to real observed inputs; nothing is fed back, so there is nothing to compound. This one-shot mapping of history onto a whole future path is something classical recursive forecasters structurally cannot do.
3. **Encoder–decoder (seq2seq):** the structured version of direct forecasting. An encoder LSTM reads the 30-day window and compresses it into a 32-number **context vector**; `RepeatVector` copies that vector once per future day; a decoder LSTM unrolls its own dynamics across the horizon; and a shared `TimeDistributed(Dense(1))` read-out converts each decoder state to a temperature. Unlike the flat `Dense(14)`, the decoder's day-9 state is computed from its day-8 state — the *shape* of the forecast trajectory is itself modeled. This is the template that scales: swap the repeated context vector for a learned attention over encoder states and you have rebuilt the architecture behind modern sequence models.

The overall 14-day scores:

```
Recursive LSTM   {'MAE': 2.135, 'RMSE': 2.754}
Direct LSTM      {'MAE': 2.165, 'RMSE': 2.792}
Seq2seq LSTM     {'MAE': 2.148, 'RMSE': 2.771}
```

Wait. The *recursive* strategy — the one we set up as the error-compounding villain — wins? Averages over a horizon hide exactly the thing this section is about, so let's split the error by lead time.

> **[FIGURE 5 — notebook cell 61]** RMSE by days-ahead for the three strategies, with 1-step persistence as a dotted reference line. The recursive curve starts far below the others (2.23 °C at day 1 vs. ~2.6 for direct and seq2seq), climbs steeply for two days, joins the pack by day 3; from there all three drift together up to about 2.9 °C by day 14.

This chart does *not* show the cartoon version of the story, and it rewards a careful read. Three observations, in order of how much they should surprise you:

**First, the recursive curve starts far below the others.** It *is* the one-step LSTM, scored on its specialty. The direct and seq2seq models pay a hidden price for multi-output training: the loss averages all fourteen days, so they trade away day-1 sharpness to serve the whole horizon. If you mostly care about tomorrow, the specialist wins — by 0.4 °C, the largest gap anywhere in this episode.

**Second, the compounding is real but self-limiting here.** Why no runaway? Because temperature is strongly **mean-reverting**: fed its own outputs, the LSTM's forecast relaxes toward the seasonal average — and the seasonal average happens to be a perfectly sensible 14-day forecast. Errors compound viciously when a series *trends*, each fabricated input walking the forecast further off the path. On a series that pulls everything back to a seasonal level, recursion is forgiven. This is exactly the regime the forecasting-competition literature maps out: recursive statistical methods rule short horizons, multi-output deep models pull ahead as horizons stretch and dynamics get less forgiving — up to high single-digit percentage gains at long leads.

**Third, the ceiling.** By day 5, every strategy's error exceeds what naive persistence achieves at day 1, and the curves flatten near 2.9 °C — essentially the error of forecasting the seasonal climate and accepting the weather. Two weeks out, *what day it is* carries more information than anything in the last thirty days of observations. No architecture in this notebook can beat that; it is the irreducible noise floor of the problem, and **recognizing such a floor is as much a forecasting skill as lowering it**.

> **[FIGURE 6 — notebook cell 63]** One validation window's 14-day forecasts against reality: all three model paths sit in a calm band around 13 °C while the actual temperature takes a cold detour down to 8 °C.

One concrete trajectory, to keep the aggregate honest — and window 100 is instructive precisely because the models *miss*. None of them foresees the cold snap, and none should be expected to: a cold snap five days out is weather, not climate, and a squared-error-minimizing forecast correctly declines to gamble on it. This is the conditional-mean effect stretched over a horizon — smooth, central, unbothered — and it is a preview of why the natural next step after point forecasts is *probabilistic* forecasting: the interesting object in that chart is not the line but the uncertainty band that should have surrounded it (episode 7 readers already know where this is going).

## Covariates, and a bidirectional word of warning

Two ingredients remain. First, **covariates**: nothing about `[batch, timesteps, features]` says features must equal 1. We hand the network four extra signals per day — 7- and 30-day rolling means of the recent past, and the day-of-year encoded as a sine/cosine pair. Two details carry the craft:

- Every rolling feature is computed on `shift(1)` data, so the value available on day t is built from days t−7 … t−1, *never including day t itself*. A feature must be computable at prediction time — apply that rule mechanically to everything you engineer.
- The sine/cosine pair encodes the calendar as coordinates on a circle, fixing two problems at once: the raw day number has a fake cliff (December 31 = 365 sits numerically far from January 1 = 1, though they are neighbors), and a single sine cannot tell spring from autumn — the cosine breaks the tie. It tells the network *where in the season it is* directly, instead of making it infer that from the window's shape.

Second, the **Bidirectional** wrapper: hand it any recurrent layer and it builds two copies — one reading the window forwards, one backwards — and concatenates their outputs, doubling both the representation and the parameter bill.

This one comes with the sharpest safety warning of the episode. A bidirectional layer's backward pass consumes the *end* of its input first — so if the sequence extends into the future relative to the prediction target, the model is reading tomorrow to predict today. In sequence *classification* (is this heartbeat arrhythmic? when will this machine fail, given its full sensor log?) the whole sequence is legitimately available and bidirectionality shines. In *forecasting* it is safe only when the input window lies entirely in the past — ours does; reading thirty historical days backwards breaks no laws of physics. The trap is subtle pipeline errors — windows built before splitting, targets inside the window's span — where a bidirectional model will happily exploit the leak and reward you with beautiful validation scores that evaporate in production. Empirical audits of leaky setups have measured performance overstated by double-digit percentages. **Rule of thumb: bidirectional layers amplify whatever your pipeline does — including its mistakes.**

The result:

```
{'MAE': 1.730, 'RMSE': 2.195}
```

Best score in the episode. The margin over the plain LSTM is real but modest, and its anatomy is worth stating: the rolling means largely duplicate information the thirty-day window already carries, so the genuinely new signal is the day-of-year encoding. Covariates earn their keep in proportion to how much *outside* information they inject; on richer problems — prices, promotions, holidays, sensor arrays — this is where the large gains live.

## The verdict

The full one-step scoreboard, sorted best-first:

```
                        MAE    RMSE
BiLSTM + covariates     1.730  2.195
GRU                     1.741  2.219
LSTM                    1.741  2.219
SimpleRNN               1.759  2.227
Stacked RNN             1.768  2.246
Persistence (t-1)       1.953  2.481
Seasonal naive (t-365)  2.965  3.728
```

Two honest readings. Every network cleared the persistence bar — not a given, and the first thing to check in any deep learning forecasting exercise. And the spread across architectures is modest: five neural models span 2.20 to 2.25 °C, a two-percent band, while the step from persistence down to *any* of them is more than ten times that. On a small univariate series with a dominant seasonal cycle, the window already contains most of what is knowable, and the architectures differ mainly in how gracefully they extract it.

The decisive contrasts in this episode were never between cell types. They were between *task framings* (the one-step specialist beat the multi-output models by 0.4 °C on day-1 error, while the horizon's noise floor swallowed everyone by day 5) and between *pipelines* (a leaked scaler would have flattered every number here).

So when does deep learning actually earn its complexity budget? When the levers this small demo couldn't pull exist: **global training** (one network across thousands of related series, sharing patterns the way one store's demand informs another's — the cross-learning that local models like ARIMA and ETS structurally cannot do), **raw covariates** classical models cannot ingest (text, order books, sensor arrays), genuinely **long-range dependencies**, and **direct multi-horizon output** on series where recursion is not forgiven. Classical statistical baselines keep winning where series are short, sparse, and numerous — and they remain the honest first thing to fit. No free lunch, as usual.

## Closing time

We climbed from a flat temperature file to an encoder–decoder: windowing turned forecasting into supervised learning on `[batch, timesteps, features]` tensors; a NumPy experiment showed why products of Jacobians doom plain RNNs beyond a few dozen steps; GRU and LSTM fixed it with the same trick wearing different plumbing — an additive, gate-protected path through time that lets both information and gradients travel far; three multi-step strategies revealed that the choice of *framing* can matter more than the choice of *cell*; and a bidirectional multivariate finale took the top spot while illustrating exactly the kind of power tool that amplifies pipeline mistakes.

The throughline: **the architecture ladder is the glamorous part, but the wins and losses in this episode came from data discipline** — split before you scale, assign windows by target date, engineer features that exist at prediction time, and draw the per-step error curve before believing any claim about forecasting strategies, including mine.

In the next installment: the encoder–decoder we built is one attention layer away from the Transformer — and the modern forecasting architectures (N-BEATS, TFT and friends) plus the global-model libraries that wrap them all build on exactly the primitives assembled here.

## Key Takeaways

1. **Windowing is the bridge:** deep learning forecasting is supervised learning on sliding windows; once you can produce the 3D tensor (and split it by target date, scaler fitted on the past only), every architecture is a drop-in.
2. **Memory is a gradient-flow problem:** spectral radius below 1 → vanishing, above 1 → exploding; gates cure it with an additive shortcut through time. That is the entire conceptual content of GRU and LSTM.
3. **Match the strategy to the horizon — empirically:** recursive forecasting keeps day-1 sharpness and compounds errors; multi-output spreads the budget evenly. On mean-reverting data the specialist held its own; on trending data it won't. Draw the per-step curve and let it choose.
4. **Know your noise floor:** past day 5, every model converged to the error of "forecast the climate, accept the weather". Recognizing an irreducible floor beats chasing it with bigger networks.
5. **Bidirectionality is a scalpel:** legitimate when the whole input sequence truly exists at prediction time; a leak amplifier when it doesn't.
6. **Baselines before backprop:** persistence at 2.48 °C was within 12% of the best 12,705-parameter network. Always check.

United States of Banan is a reader-supported publication. To receive new posts
and support my work, consider becoming a free or paid subscriber.

[ ✓ Subscribed ]
