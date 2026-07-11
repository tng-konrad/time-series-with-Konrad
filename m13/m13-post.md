# Time Series with Konrad: episode 13

### Transformers for time series: attention, patches, and one glorious axis flip

**KONRAD BANACHEWICZ**
NOV 9, 2026

Since 2017, the transformer has swallowed natural language processing, then computer vision, then audio, then protein folding. You might reasonably expect that time series forecasting fell somewhere in that list. It did not — and the story of *why not* is one of the most instructive episodes in modern machine learning. In 2023, a widely cited AAAI paper asked "Are Transformers Effective for Time Series Forecasting?", answered with a one-layer **linear model**, and beat a whole generation of sophisticated attention architectures with it. Publicly. By a lot.

The field's response was not to add more layers. It was to go back and question something so basic it had never been questioned: *what should a token be?* The architectures that emerged from that reckoning — patch transformers and inverted transformers — are the ones that finally made attention earn its keep on forecasting problems. This episode builds the whole arc by hand: the mechanism, the failure, the humiliation, and the two repairs.

The core principle:

**A time series carries its information in the ordering — trend, seasonality, momentum are properties of the sequence, not of the bag of values. Attention, by design, processes bags. Every transformer that forecasts well is a repair job on that mismatch, and every successful repair changes the same thing: what a token is.**

The ladder we climb: the GRU and LSTM incumbents from episode 11, promoted to a genuinely multivariate task (fifty series forecast jointly) → scaled dot-product attention built in four lines of NumPy, including a live demonstration of its blind spot → a **vanilla transformer** that loses to the GRU → a 1,190-parameter **linear model** that beats both (the DLinear moment) → **PatchTST**, which makes tokens out of fortnights instead of days → **iTransformer**, which makes tokens out of *series* instead of timestamps and takes the crown → a lookback-length sweep showing exactly where attention's advantage lives → and a rematch on univariate temperature data, where the whole transformer family gets beaten by a GRU one-twentieth its size. Real data throughout, honest errors throughout, parameter counts and training times printed next to every RMSE. Practical understanding over rigor, as always.

📓 All code lives in the companion notebook:
https://github.com/tng-konrad/time-series-with-Konrad/blob/main/m13-fable-version.ipynb

## One store, fifty channels, one task

The dataset is an old friend: the Kaggle store-item demand data from episode 12 — five years (2013–2017) of daily unit sales, 50 items, 10 stores. Last episode we treated its series one at a time. This time the *panel structure itself* is the point: we take store 1 and pivot it into a single matrix with one row per day and one column per item — a 1,826 × 50 array where each column is what this episode calls a **channel**. The task: given twelve weeks of all fifty channels, predict the next two weeks of all fifty channels, simultaneously, in one network call.

↪ *Same store-item dataset, different angle: episode 12 used it for transfer learning, forecasting a brand-new store from borrowed weights.* → **<LINK TO EPISODE 12 HERE>**

> **[FIGURE 1 — graphs/graph13-01.png — notebook cell 24: three stacked line plots, items 1 / 15 / 28 daily sales over 2013–2017]**

Three items are enough to see everything that matters. First, the **shared shape**: every panel shows the same summer-peaked annual cycle, the same gentle multi-year growth, the same weekly rhythm. Second, the **different scales** — items differ several-fold in volume, which is why each channel gets z-scored by its own training-era mean and standard deviation (fitted on training data only; the leakage liturgy from episode 8 applies as always). Third, the noise: item-level demand is noisy in a way store-level aggregates are not, and no model below will be forgiven for pretending otherwise.

↪ *"Fitted on training data only" is the entire moral of episode 8 — the validation discipline that decides whether any RMSE in this post can be trusted.* → **<LINK TO EPISODE 8 HERE>**

The shared shape deserves a number:

> **[FIGURE 2 — graphs/graph13-02.png — notebook cell 26: 50×50 correlation heatmap of the item series]**

Every pair of the fifty series is substantially positively correlated — mean around 0.7, and even the weakest pair sits near 0.5. No block structure, no cluster of loners: one store, one demand climate, fifty noisy views of it. **Keep this matrix in mind.** If a model could look *across* channels, it would have fifty noisy measurements of the same seasonal signal to average over. Whether any architecture in this episode manages to cash that in is, frankly, the plot.

The split is chronological (all of 2017 held out — 51 weekly forecast origins covering every season; the 26 weeks before that for early stopping), and before any neuron trains, the goalposts:

```
seasonal naive (lag 7):    {'MAE': 8.466, 'RMSE': 11.019}
seasonal naive (lag 364):  {'MAE': 8.18,  'RMSE': 10.601}
```

Copy-last-week versus copy-last-year, and the *year* wins — 364 rather than 365 so the weekdays align, a trick worth stealing. The annual cycle in this data is strong enough that last year's fortnight (right season, right weekdays) beats last week's fortnight (right weekdays, last week's noise). Every learned model below must clear 10.601 to justify its electricity bill.

## The incumbents

The recurrent recipe from episode 11, scaled up: a GRU reads the 84 days step by step — each step now seeing a 50-dimensional vector — and its final 64-number hidden state feeds a linear head that emits all 14 × 50 = 700 outputs at once. Direct multi-step forecasting; no recursive feedback loop, no error accumulation (we measured that pathology two episodes ago).

Pause on what this architecture *asks* of the hidden state: 64 numbers must summarize everything relevant about 4,200 input values, and the summary is built strictly left to right. By the time the encoder reaches yesterday, week one survives only as whatever trace the gates chose to keep. That compression bottleneck — and the sequential walk that builds it — is precisely what attention was invented to dismantle.

↪ *The GRU and LSTM this section promotes to a multivariate task were built and dissected in episode 11 — start there if the gates and hidden states are unfamiliar.* → **<LINK TO EPISODE 11 HERE>**

```
GRU:   {'MAE': 7.022, 'RMSE': 9.349}    67,772 params, ~20s to train
LSTM:  {'MAE': 6.973, 'RMSE': 9.328}    74,940 params
```

Both clear the naive bar by a healthy margin — the incumbents are not strawmen — and they are statistically indistinguishable from each other, replicating episode 11's finding and roughly forty papers' worth of literature. The GRU carries the recurrent flag from here.

> **[FIGURE 3 — graphs/graph13-03.png — notebook cell 41: observed 2017 test year for item 15 with the GRU's stitched weekly forecasts overlaid]**

The forecasts track the level, the annual swell, the weekly oscillation — a smoothed shadow of the truth. What no model will do is predict the day-to-day noise, and none should try: item-level demand has a large irreducible component, which is why the interesting differences below are measured in tenths of a unit, not leaps. The eyeball test is officially retired for this episode; the scoreboard decides.

## Attention in four lines of NumPy

Before building transformer forecasters, we build the mechanism — it fits in one cell. Take a 28-day window of item 1 and let each day be a token with the most embarrassing embedding imaginable: its own value.

```
scores   = Q @ K.T / sqrt(d)     # how relevant is day j to day i?
weights  = softmax(scores)       # each row: a probability distribution
attended = weights @ V           # each day becomes a weighted mix of all days
```

Three roles, all derived from the same tokens in self-attention: the **query** is what each position asks, the **key** is what each position advertises, the **value** is what it hands over if selected. The scaling by √d keeps the softmax out of its saturated zone. That's it — that is the mechanism that ate machine learning:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

> **[FIGURE 4 — graphs/graph13-04.png — notebook cell 46, first plot: the 28-day standardized demand window]**

> **[FIGURE 5 — graphs/graph13-05.png — notebook cell 46, second plot: the 28×28 attention-weight heatmap, "rows ask, columns answer"]**

Reading the heatmap: row *i* shows how day *i* distributes one unit of attention over all 28 days. With raw values as embeddings, high-demand days attend to high-demand days and low to low — the weekly rhythm shows up as a plaid of bands, and the model discovers "days like me" without ever being told what a weekday is.

Two properties, and they are the yin and yang of this entire episode:

- **The superpower:** everything is one matrix multiplication from everything else. Day 1 influences day 28 as directly as day 27 does — a *constant path length* to the entire past. A GRU relates those days through 27 applications of its gates, each one an opportunity to forget.
- **The blind spot:** nothing in those four lines asks *where* a day sits. Relevance is decided purely by content.

The blind spot deserves its own demonstration, because it is the crux of the whole "are transformers effective?" controversy. Scramble the 28 days into a random order, run attention on the scrambled window, and compare:

```
attention(shuffled) == shuffle(attention(original))  -->  True
```

Exactly true, to machine precision. Attention is **permutation-equivariant**: reorder the inputs and the outputs reorder along, nothing else changes. No statistic it computes knows whether demand was rising or falling, whether the spike came before the dip, whether Monday exists. For language this is a feature — meaning survives reordering, mostly. For a numerical series it is close to disqualifying, because **a time series is its ordering**. The 2023 AAAI paper called raw self-attention *anti-order*, and this one-liner is their argument compressed.

The standard patch is to stamp the position onto the token itself, so content-based lookup can see position *as content* — the **sinusoidal positional encoding**, a fingerprint of sines and cosines at geometrically spaced wavelengths added to every token.

> **[FIGURE 6 — graphs/graph13-06.png — notebook cell 50: the positional-encoding matrix, embedding dimensions × 84 positions]**

Two honest caveats before we trust the patch. Positional encoding makes order *recoverable*, not *respected* — the model must spend capacity learning to use the stamps, where a GRU gets order for free from its architecture. And the research record says this repair is only partly effective; the fixes that actually worked changed what a token is instead. Which brings us to the ladder.

## Rung 1: timestamps as tokens

The direct translation of the NLP recipe, which is exactly how the first generation of time series transformers worked. Embed **each timestamp's 50-channel snapshot** as one token (a Dense layer maps 50 numbers to 64), add the positional stamps, and run the sequence of 84 tokens through two standard encoder blocks — the *attention → add & norm → feed-forward → add & norm* sandwich, essentially unchanged since 2017. Attention operates across time: an 84 × 84 matrix connecting days to days.

Two design smells, both flagged by the later literature. A temporal token mixes all fifty items at one instant, assuming that whatever matters about a timestamp is the joint state of everything at that exact tick. And a single day of noisy demand is a nearly semantic-free atom — attention is being asked to find meaningful relationships between things that individually mean almost nothing.

```
vanilla transformer:  {'MAE': 7.516, 'RMSE': 9.997}    115,708 params
```

The first upset, delivered quietly: **worse than the GRU** (9.997 vs 9.349), only modestly better than copy-last-year. All that parallel machinery, the constant path length, four heads and two blocks — and the sequential architecture from 2014 wins.

Maybe the pooling head threw away the signal? We re-run with a `Flatten` head that hands the forecast layer all 84 processed tokens — ballooning the model to 3,834,108 parameters:

```
vanilla transformer (flat head):  {'MAE': 7.122, 'RMSE': 9.365}    3,834,108 params
```

Fifty-six times the GRU's parameters to achieve a dead heat with it. The problem is not the head.

> **[FIGURE 7 — graphs/graph13-07.png — notebook cell 62: head-averaged 84×84 attention map from the trained vanilla model]**

Pulling the attention map out of the trained model (a two-line sub-model idiom, in the notebook) shows what it chose to do with its freedom: soft vertical bands — a few reference days that every query consults — plus faint weekly texture. No crisp diagonal, no clean lag-7 or lag-364 stripes. Point-wise attention over noisy single-day tokens learned to be an expensive smoother. The diagnosis from our NumPy section stands: anti-order mechanism, semantically thin tokens, 84 high-resolution opportunities to overfit noise.

## Rung 2: the linear reality check

Before repairing the transformer, we administer the exam that shook the field. The **DLinear / LTSF-Linear** argument: if the transformer's edge is real, it should at minimum beat a *single linear layer* mapping the last 84 days of a channel to its next 14 — one 84 × 14 weight matrix, shared by all fifty items, each channel regressed on its own past. 1,190 parameters. No attention, no gates, no activation function. (In the notebook this is a `Permute → Dense(14) → Permute` sandwich, which is the entire architecture.)

Why take such a model seriously? Because a linear map on a standardized window can express more forecasting wisdom than intuition suggests: weighted seasonal averaging, trend extrapolation, exponential decay of recency — all linear in the history. It is the perfect control experiment for "does the fancy machinery earn its keep?"

```
linear:  {'MAE': 6.859, 'RMSE': 8.977}    1,190 params, 4s to train
```

**Better than the GRU. Better than the LSTM. Better than both vanilla transformers.** From a model whose parameters would fit on a single sheet of paper. This is the honest replication of the 2023 result, at teaching scale, and if you take one deployable lesson from this episode: for tasks dominated by trend and stable seasonality, a per-channel linear model on a standardized window is a devastating baseline, and any architecture you propose should be required to beat it.

Note also what the linear model did *not* need: cross-channel information. Its channel independence turned out to be a regularizer, not a handicap. The two rungs that remain are the architectures the field built after absorbing exactly this lesson — and neither of them responds by piling on capacity. Both change the tokenization.

## Rung 3: a time series is worth six words (PatchTST)

**PatchTST** (ICLR 2023 — the paper's title is literally *"A Time Series is Worth 64 Words"*) makes two moves.

**Patching.** Instead of 84 one-day tokens, cut the window into six *fortnights* and make each segment one token. A token is no longer a semantic-free number; it is a **shape** — fourteen days containing two full weekly cycles, a level, a slope. Attention between such tokens compares local patterns ("does the fortnight two months ago look like the latest one?"), which is a question with actual forecasting content.

> **[FIGURE 8 — graphs/graph13-08.png — notebook cell 72: one 84-day window drawn as six color-coded 14-day patches]**

The benefits compound nicely:

- **Meaningful tokens:** local trends and rhythms live *inside* the embedding, instead of having to be reassembled by attention from single points.
- **Quadratic relief:** the attention matrix shrinks from 84² to 6² — a 196-fold reduction, which is what lets published PatchTST models afford year-long lookbacks.
- **Built-in denoising:** day-to-day noise is averaged away inside the patch embedding rather than being available to overfit.

**Channel independence.** Every channel becomes its own univariate training sample flowing through one shared network. Counterintuitive — surely cross-channel information helps? — but the published ablations are consistent: with many channels and few windows, a channel-mixing model has ample freedom to hallucinate spurious correlations, and independence wins on both accuracy and robustness. There is also a quiet gift for our data-starved regime: folding channels into the batch turns 1,183 multivariate windows into 59,150 univariate ones. This is the only model in the episode that trains on fifty-nine thousand samples.

```
PatchTST:  {'MAE': 6.948, 'RMSE': 9.137}    73,294 params
```

Two comparisons locate it. Against the vanilla transformer, patching claws back nearly a full RMSE unit — the tokenization fixes turn a transformer that loses to the GRU into one that beats it. Against the linear model, it comes up a touch short (9.137 vs 8.977). That also faithfully miniaturizes the literature: PatchTST's published wins over DLinear are real but measured in single-digit percentages, earned mostly on datasets richer in nonlinear structure than a demand panel. Attention across *time* is now respectable — but it still hasn't shown us anything a lag-weighted average couldn't. What we haven't tried is attention across *channels*.

## Rung 4: flip the axes (iTransformer)

Here is the move this episode has been building toward, and it is almost insultingly simple. The **iTransformer** (ICLR 2024) keeps every component of the vanilla transformer — same attention, same encoder blocks — and changes only the axis they run on. One `Permute` before the embedding: instead of 84 tokens of shape "all channels at one instant", we get **fifty tokens of shape "one channel's entire 84-day history"**. Attention now connects *series* to *series*: the 50 × 50 attention matrix asks "when forecasting item 12, which other items' histories are informative?" — exactly the question our correlation heatmap has been begging someone to ask since Figure 2.

And notice what the inversion quietly dissolves. The permutation problem? *Gone, without positional encodings* — time now lives inside each token, read by an order-aware linear embedding (one weight per lag, like our linear model), while the new sequence axis — channels — genuinely *is* an unordered set. For the first time in this episode, attention's set-processing nature matches its input. The feed-forward layers, which act per token, become per-channel temporal processors; the forecast head is a per-token linear readout of a representation that attention has enriched with information from 49 sibling series.

```
iTransformer:  {'MAE': 6.602, 'RMSE': 8.696}    277,646 params, ~6s to train
```

**Best score of the episode** — the first transformer to beat not just the GRU (by 7%) but the linear reality check. The margin over the linear model, about 0.28 RMSE, is the honest measured value of cross-channel attention on this panel: fifty noisy views of one demand climate, finally allowed to denoise each other. And mind the cost column: the biggest accurate model of the episode trained in a *third* of the GRU's wall-clock time, because fifty parallel tokens on a GPU beat eighty-four sequential steps every day of the week.

> **[FIGURE 9 — graphs/graph13-10.png — notebook cell 83, second plot: the trained iTransformer's 50×50 channel-to-channel attention map (the notebook shows it right after a re-plot of the correlation matrix, for comparison)]**

The promised diagnostic. The learned attention map is *not* a copy of the correlation matrix — it is sparser and directional, with a handful of bright **columns**. Columns are keys: a bright column is an item whose history many other items consult when forecasting themselves. The model has elected a few high-signal reference items as shared seasonal oracles — more economical than attending to everything, and something symmetric pairwise correlation cannot even express. This asymmetry is the fingerprint of learned, task-driven structure rather than mere co-movement.

## When transformers win: the lookback sweep

The ladder compared architectures at one window length. But the deepest difference between recurrence and attention is *how they scale with history* — so we sweep the lookback from four weeks to a full year, everything else fixed, same 51 test forecasts throughout:

```
lookback      GRU RMSE    iTransformer RMSE
 28 days        9.410          8.668
 84 days        9.349          8.696
182 days        9.426          8.537
364 days        9.257          8.388
```

> **[FIGURE 10 — graphs/graph13-11.png — notebook cell 88, first plot: test RMSE vs lookback window, GRU vs iTransformer]**

> **[FIGURE 11 — graphs/graph13-12.png — notebook cell 88, second plot: training wall-clock seconds vs lookback window, GRU vs iTransformer]**

This pair of plots is the strongest pro-transformer evidence in the episode, and it is a *scaling* argument, not a single-number one.

**Accuracy.** The GRU is flat — offer it a full year, long enough to contain last year's version of every day it must predict, and it cannot cash that in: 364 sequential gate applications stand between January's signal and the forecast, and 64 numbers are a narrow pipe for a year of fifty channels. The iTransformer *improves* steadily with lookback (8.67 → 8.39), because lag 364 is exactly one embedding weight away, same as lag 1.

**Cost.** The GRU's training time grows linearly with the window — ~10 seconds at four weeks, ~76 at a year; the sequential walk cannot be parallelized. The iTransformer's stays flat at ~5–6 seconds throughout: its attention runs over 50 channel tokens *regardless of window length*. **Longer memory is free for the inverted transformer and linearly expensive for recurrence.** That single sentence is most of what "transformers beat RNNs" legitimately means in forecasting.

## When they are overkill: the Melbourne rematch

Every result so far came from transformer home turf — many correlated channels, long useful context. Honest evaluation requires the away game. So: the Melbourne daily minimum temperatures from episode 11. One channel, so cross-variate attention has nothing to attend across. Short useful memory — tomorrow's temperature depends on the last few days plus the annual cycle. Heavy weather noise. 2,886 training windows. Everyone rebuilds small (28 days in, 7 out) and fights again.

The baselines already tip you off:

```
persistence:               {'MAE': 2.679, 'RMSE': 3.405}
seasonal naive (lag 365):  {'MAE': 2.882, 'RMSE': 3.668}
```

Persistence *beats* the seasonal copy here — the exact opposite of the demand panel. Temperature is mean-reverting and weather-dominated: the most informative fact about next week is the current anomaly, not the calendar. **The structure of the best naive forecast is a free diagnostic of what a series rewards** — and this one rewards short memory, which should already lower your expectations for architectures whose specialty is long context. ;-)

```
                     MAE     RMSE    params
GRU                  2.022   2.623    3,591
vanilla transformer  2.025   2.649   79,623
LSTM                 2.032   2.651    4,583
linear               2.047   2.681      203
PatchTST             2.069   2.691   69,255
```

The recurrent incumbent takes the rematch: best accuracy, three orders of magnitude fewer weights than the transformers it beats. The 203-parameter linear model lands mid-pack, between the two transformers. Everything clears both naive baselines comfortably, so all the models are doing real work — the transformers simply are not doing *more* work than the cheap alternatives, on a task that has nothing for them: no long-range dependencies to shortcut, no channels to attend across (an iTransformer on one channel is a one-token sequence — attention over a set of size one is a very expensive identity function), and small noisy data that rewards strong inductive biases. Reaching for a transformer here buys 22× the parameters for a slightly worse forecast.

> **[FIGURE 12 — graphs/graph13-13.png — notebook cell 99: scatter of test RMSE vs parameter count (log scale), Melbourne rematch — GRU alone in the bottom-left]**

Accuracy against size, log axis, down-left is good. On the demand panel this plot would flatter the iTransformer; here the GRU owns the corner. **The same architectures, competently tuned, reverse their ranking when the data stops playing to attention's strengths.** Match the tool to the signal, not to the publication year.

## The verdict

The demand-panel scoreboard, one protocol, one test year:

```
                                  MAE     RMSE     params   train_s
iTransformer                     6.602    8.696    277,646     6.2
linear (channel-independent)     6.859    8.977      1,190     4.3
PatchTST                         6.948    9.137     73,294    34.5
LSTM                             6.973    9.328     74,940    22.5
GRU                              7.022    9.349     67,772    19.9
vanilla transformer (flat head)  7.122    9.365  3,834,108     4.4
vanilla transformer              7.516    9.997    115,708    11.8
seasonal naive (lag 364)         8.180   10.601          —       —
seasonal naive (lag 7)           8.466   11.019          —       —
```

> **[FIGURE 13 — graphs/graph13-14.png — notebook cell 104: horizontal bar chart of the scoreboard, bars annotated with parameter counts, color-coded by family]**

Read it bottom to top and you get the whole episode in nine lines: the naive copies; a vanilla transformer that beats them but loses to 2014-vintage recurrence; the incumbents, solid and interchangeable; a 1,190-parameter linear model above them all — the result that forced the field to rethink; the patch transformer restoring attention to respectability; and the inverted transformer alone at the top, the only model that monetized the panel's cross-channel structure. Note how weakly bar length correlates with parameter count. In the small-data regime that most real forecasting lives in, that non-correlation is the deepest fact on the board.

## Closing time

We climbed from a mechanism to a verdict: four lines of NumPy revealed both attention's superpower (constant path length to the whole past) and its congenital defect (permutation equivariance — it processes bags, and a time series is not a bag); the vanilla transformer inherited the defect and lost to a GRU; a linear model beat them both and recalibrated everyone's dignity; patching fixed the tokens (shapes, not points) and channel independence fixed the sample count; and the iTransformer fixed the axis — tokens as series, attention across channels, order handled where order lives. The lookback sweep then showed the payoff is structural, not incidental: attention converts long context into accuracy at flat cost, recurrence converts it into training time. And Melbourne showed the whole edifice gracefully declining to matter on a univariate, short-memory signal.

The throughline: **transformers won on this data only when their tokenization respected the structure of the signal — and the winning change was never more capacity, always a better question.** Where does it go next? The tokenization ideas you now own are load-bearing everywhere: exogenous covariates enter iTransformer as just more variate tokens (TimeXer industrializes this); hybrids run GRUs along the channel axis (iGRU) or bolt LSTM decoders onto transformers to stabilize long horizons; and masked-*patch* pretraining is precisely what turns PatchTST into the backbone of the time series foundation models we met in episode 12 — patching is what made series look enough like sentences for the whole transfer playbook to industrialize. When you next meet a foundation model, you know exactly what is inside.

## Key Takeaways

1. **Attention is anti-order:** shuffle the past and the output shuffles along, exactly. Positional encodings make order recoverable, not respected — the successful fixes changed the tokenization instead.
2. **The linear baseline is a moral obligation:** 1,190 parameters beat two RNNs and two transformers on real retail data. If your architecture can't beat a standardized linear map, it has no business in production.
3. **Patch, don't point:** tokens should be shapes (a fortnight) rather than samples (a day) — meaningful semantics, 196× smaller attention, built-in denoising.
4. **Channel independence is a regularizer:** treating each series as its own sample multiplied the training set by 50 and beat channel-mixing — spurious cross-channel correlations are a bigger danger than lost information.
5. **Invert when channels correlate:** iTransformer's axis flip — tokens as series, attention across channels — was the only change that beat the linear model, and it made positional encodings unnecessary as a side effect.
6. **Attention buys long memory at flat cost:** a year of lookback improved the iTransformer for free while the GRU gained nothing and paid 8× the training time. That scaling, not any single RMSE, is the honest pro-transformer argument.
7. **Read your naive baselines as a diagnostic:** seasonal-naive-wins ⇒ long memory matters ⇒ transformer country. Persistence-wins ⇒ short memory ⇒ a small GRU (or a linear model) is probably all you need.
8. **No free lunch, as always:** the best model on the panel (iTransformer) wasn't in the top three on the temperatures, and the winner there (GRU) lost the panel by 7%. Match the tool to the signal, not to the publication year.

United States of Banan is a reader-supported publication. To receive new posts
and support my work, consider becoming a free or paid subscriber.

[ ✓ Subscribed ]
