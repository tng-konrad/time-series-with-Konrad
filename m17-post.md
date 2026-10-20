# Time Series with Konrad: episode 17

### Anomaly detection for time series: when the usual stops being usual

**KONRAD BANACHEWICZ**  
OCT 18, 2026

For most of this series, we have asked what happens next. Anomaly detection asks a different question: how surprised should we be by what just happened? A quiet taxi rank at four in the morning is ordinary. The same activity during the morning rush deserves a closer look. The number alone cannot tell us which situation we are in.

There is another complication. Christmas can be unusual without being a problem. A long demand slump can reflect an incident, a new operating regime, or a reference model that has become stale. The red dot on a chart is where the investigation begins.

**A useful anomaly detector measures surprise against an appropriate reference, then turns that evidence into a manageable decision.** We will build that reference gradually: robust thresholds, seasonal context, forecast residuals, EWMA, Isolation Forest, PCA, and a small neural autoencoder. Then we will examine change detection and the practical business of deciding when to notify somebody.

The goal is practical understanding, with enough math to explain the machinery. We will use real New York taxi data throughout, keep the future out of training, and report what happens when the more sophisticated method loses. The GPU has been warned. ;-)

📓 All code lives in the companion notebook:  
[m17-anomaly.ipynb](https://github.com/tng-konrad/time-series-with-Konrad/blob/main/m17-anomaly.ipynb)

## First, decide what unusual means

Our dataset contains **10,320 half-hourly NYC taxi passenger counts**, from July 2014 through January 2015. It comes from the Numenta Anomaly Benchmark, or **NAB**, whose [dataset notes](https://github.com/numenta/NAB/blob/ea702d75cc2258d9d7dd35ca8e5e2539d71f3140/data/README.md) trace the underlying records to the NYC Taxi and Limousine Commission. These are passenger counts aggregated into half-hour buckets; they are not individual trip records.

> **[INSERT GRAPH 1 HERE — `graphs/graph17-1.png` — notebook cell 22: raw NYC taxi passenger counts.]**

The repeated daily swings dominate the picture. Underneath them, activity changes across the months, and several unusually quiet stretches interrupt the rhythm. A global upper-and-lower limit will have to accommodate all those normal troughs and peaks before it can recognize anything unusual.

Three kinds of anomaly matter here. A **point anomaly** is an isolated surprising reading. A **contextual anomaly** is surprising given its time of day, weekday, or recent history. A **collective anomaly** is a surprising sequence: perhaps an otherwise busy daytime pattern disappears for several days. One episode can fit more than one description.

NAB provides five event windows associated with the marathon, Thanksgiving, Christmas, New Year, and a snowstorm. These are broad evaluation windows around known events. We can ask whether a detector hits them, but treating every half-hour inside as physically abnormal would overinterpret the labels. Likewise, an unlabeled observation is not certified normal.

> **[INSERT TABLE 3 HERE — `graphs/table17-3.png` — notebook cell 26: Official NAB event windows.]**

Before modeling, we check timestamps, duplicates, missing values, and nonnegative counts. All observations lie on the expected half-hour grid. That makes a 48-row lag genuinely one day. Forward-filling a long missing stretch would manufacture a flat line, which a sequence detector might then proudly rediscover as an anomaly of our own making.

## Give the experiment a clock

We use four consecutive blocks:

> **[INSERT TABLE 4 HERE — `graphs/table17-4.png` — notebook cell 28: Chronological training, validation, calibration, and test blocks.]**

The ordering follows the discipline from [episode 8 on validation](https://github.com/tng-konrad/time-series-with-Konrad/blob/main/m08-validation-methods.ipynb). Models fit on training. Their thresholds come from calibration. Test labels describe performance after those choices have been made. We never shuffle overlapping time windows across the four roles.

Every detector produces a continuous **anomaly score**, with larger values meaning more unusual. For the main comparison, we choose an upper-tail cutoff from its calibration scores, using a finite-sample order statistic at a nominal 1% tail setting. Each method gets its own numerical cutoff because their score units differ.

That setting is a calibration choice, not a promise of 1% future false alarms. Demand changes, residuals are dependent, and our reference contains real-world messiness. Keeping those qualifications visible will matter when we reach the scoreboard.

## Rung 1: start with a robust distance

The simplest reference is one typical level and one typical spread. **Median absolute deviation**, or **MAD**, makes that reference less sensitive to extreme observations than the familiar mean and standard deviation:

$$
s_t = \frac{|X_t-m|}{1.4826\,\operatorname{median}(|X_i-m|)},
\qquad m=\operatorname{median}(X_i).
$$

Here, the indexed observations used to estimate the median and spread belong to training. In plain English, we measure how far the current count sits from a typical historical count, in robust units of spread. The factor 1.4826 makes MAD comparable to standard deviation for a large Gaussian sample; it does not turn taxi demand into Gaussian noise.

> **[INSERT GRAPH 4 HERE — `graphs/graph17-4.png` — notebook cell 38: Raw MAD flags and calibrated scores on the test period.]**

The detector catches some large peaks, but many quiet event readings remain within the ordinary global range. It hits all five official windows while flagging only about **4.1% of their labeled duration**. That is our first warning that “found every event” can conceal rather weak coverage.

Raising the cutoff suppresses flags and can miss more events; lowering it catches more departures and adds workload. Neither adjustment supplies the missing calendar context. For that, we need a better reference.

## Rung 2: give normal a calendar

We group training observations by weekday and half-hour, then plot the median profile for each day.

> **[INSERT GRAPH 2 HERE — `graphs/graph17-2.png` — notebook cell 30: training-only median daily profiles by weekday.]**

The curves separate familiar overnight troughs, daytime activity, and weekday–weekend differences. There are 48 positions per day and **336 per week**. That weekly rhythm explains why the same count can be routine at one timestamp and surprising at another.

We also inspect **STL decomposition**, which separates trend, seasonality, and a remainder, using the weekly period and robust fitting.

> **[INSERT GRAPH 3 HERE — `graphs/graph17-3.png` — notebook cell 32: robust STL decomposition of training data only.]**

The panels separate the recurring pattern from slower movement and the variation left over. They also expose a tempting shortcut: decompose the entire record, then detect outliers in its remainder. Standard STL smoothers use observations on both sides of a timestamp. That would be retrospective analysis with access to future test data. Here STL is a training-only diagnostic.

For the live-style detector, we freeze a median count for each weekly slot:

$$
e_t=X_t-\widehat X_{\mathrm{weekday,\ half\!hour}}.
$$

We subtract the expected count for this time of week, then apply the same robust distance rule to the residual. A count of 5,000 now means something different overnight and during a weekday commute.

> **[INSERT GRAPH 5 HERE — `graphs/graph17-5.png` — notebook cell 43: Seasonal MAD with the fixed weekly expected level.]**

The red line shows the ordinary-week reference; the score below measures departures from it. More quiet holiday readings stand out, although the frozen summer reference also disagrees with some ordinary later activity. Context helps, but it can become stale.

> **[INSERT GRAPH 6 HERE — `graphs/graph17-6.png` — notebook cell 45: Seasonal MAD zoom around Christmas, December 20–28.]**

The close-up makes the mechanism clearer. Overnight troughs can remain perfectly ordinary while Christmas daytime counts fall far below their expected rise. The problem is a changed daily pattern, even though many individual counts are unremarkable globally. Seasonal MAD's F1 rises from **0.0750 to 0.3719** on this experiment, using the same calibration rule.

The profile still assumes stable timing and a reasonably stable residual scale. Moving peaks and changing variance can create mistakes. Known holiday calendars could improve a business forecast; here we deliberately measure departures from an ordinary week.

## Rung 3: let recent observations inform the expectation

A frozen weekly profile ignores what happened five minutes ago—or, here, half an hour ago. **Ridge regression** gives us a modest forecast model using the previous one and two observations, counts one and two days ago, last week's count, and the weekly expected level.

Ridge fits a weighted combination of those inputs while discouraging very large coefficients. We standardize features using training estimates and choose the penalty on September forecasting error. Among four candidates, the selected penalty is `0.1`, with validation RMSE of **1,298.47 passengers per half-hour**. Stronger penalties constrain the coefficients more; weaker penalties let the fit follow the training relationships more closely.

> **[INSERT TABLE 10 HERE — `graphs/table17-10.png` — notebook cell 50: Ridge penalty selection on chronological validation.]**

The anomaly score becomes:

$$
s_t=|X_t-\widehat X_{t\mid t-1}|.
$$

We compare the arriving count with a prediction made from earlier observations. Every forecasting lag is shifted: the current target cannot appear among its own inputs. As the test period advances, earlier test measurements become legitimate history for the next prediction. This is sequential one-step forecasting.

> **[INSERT GRAPH 7 HERE — `graphs/graph17-7.png` — notebook cell 54: validation residual autocorrelation, seasonal reference versus ridge.]**

The ridge residuals show less persistence than the seasonal residuals, but substantial autocorrelation remains. Predictable structure has been reduced rather than eliminated. That matters for any claim about independent errors or textbook control-chart behavior.

> **[INSERT GRAPH 8 HERE — `graphs/graph17-8.png` — notebook cell 56: ridge one-step forecasts, residual scores, and test flags.]**

The green forecasts follow recent activity, including parts of the unusual periods. That is useful for prediction but can weaken sustained anomaly coverage: after demand collapses, yesterday's quiet readings help predict another quiet reading. A fixed model can follow an incident through changing inputs without being retrained.

Here ridge reaches **0.3566 F1**, slightly below Seasonal MAD, despite a shorter median first-hit delay. Better next-step forecasting and better incident detection are related objectives, but they can pull in different directions.

## Rung 4: accumulate small disagreements with EWMA

An isolated moderate residual may be uninteresting. Several moderately low readings in a row suggest something more persistent. **Exponentially weighted moving average**, or **EWMA**, adds memory to the signed seasonal residuals:

$$
S_t=\alpha z_t+(1-\alpha)S_{t-1},\qquad s_t=|S_t|.
$$

Here, `z_t` is the signed, robustly standardized seasonal residual. With `alpha=0.10`, the new smoothed value combines 10% of the latest residual and 90% of its previous state. Smaller alpha means longer memory and slower response; larger alpha reacts faster and preserves more noise. This is the smoothing idea from [episode 2](https://github.com/tng-konrad/time-series-with-Konrad/blob/main/m02-smoothing-methods.ipynb), applied to evidence of departure.

The order matters: we smooth **signed** residuals, then take the absolute value. Repeated low readings reinforce one another. Smoothing absolute errors instead would track persistent error magnitude, allowing high and low surprises to reinforce each other too.

> **[INSERT GRAPH 9 HERE — `graphs/graph17-9.png` — notebook cell 61: EWMA of contextual residuals, alpha 0.10.]**

The score builds through sustained departures and decays after recovery. F1 rises to **0.4337**, with fewer out-of-window flag runs. The price is slower response and lingering memory. The baseline itself remains fixed: smoothing evidence does not authorize us to declare a new normal.

## Rung 5: Isolation Forest needs temporal features

So far we have inspected one residual or its smoothed history. A disruption can also change variability, recent direction, or the relationship between a short stretch and a whole day. **Isolation Forest** lets us consider these combinations together.

Its intuition is pleasantly simple: unusual feature vectors tend to be isolated by fewer random tree splits. But the estimator has no calendar or memory of its own. We supply the current contextual residual, a three-hour rolling mean and standard deviation, a daily rolling mean, and the latest residual change. All windows trail the current timestamp.

We fit 200 trees on training features and calibrate their continuous scores separately. Scikit-learn returns lower `score_samples` values for more abnormal observations, so the notebook negates them. Also, `contamination="auto"` is not an oracle for the real incident rate; we bypass its built-in label threshold.

> **[INSERT GRAPH 10 HERE — `graphs/graph17-10.png` — notebook cell 67: Isolation Forest on trailing temporal features.]**

The model responds to sustained quiet stretches and unusual combinations outside the gold windows. It produces our highest average precision, **0.6074**, and highest F1, **0.4681**, at the shared operating rule. The input representation is doing substantial work here.

A disturbance also remains inside rolling summaries after its original timestamp. Some extended scores therefore come from retained history. And five derived columns still represent one measured taxi series; they do not establish that we can detect relationships breaking between five physical sensors.

## Rung 6: reconstruct the whole daily pattern

Handcrafted summaries can miss shape. A different strategy learns what ordinary daily windows look like and scores how poorly it can reconstruct a new one. We begin with **principal component analysis**, or **PCA**, as a linear reference before adding neural machinery.

We make 48-point windows and compress each into eight coordinates, then reconstruct its 48 values. Both reconstruction methods use the same score:

$$
s_t=\frac{1}{48}\sum_{j=0}^{47}(X_{t-j}-\widetilde X_{t-j})^2.
$$

Here, the values have been standardized using one training-based center and scale; the tilde denotes reconstruction. We average the squared reconstruction errors and assign the result to the window's final timestamp, when all its inputs have arrived. Backdating it to the start would give us a day's worth of fictional foresight.

*Pro tip:* we keep a global training scale. Standardizing each window separately would remove its level and amplitude, potentially hiding the demand collapse we want to recognize. An invariance that helps another task can erase the signal for this one.

> **[INSERT GRAPH 11 HERE — `graphs/graph17-11.png` — notebook cell 74: PCA reconstruction scores with eight components.]**

PCA reacts strongly around the first event and New Year, but its error remains low through some quiet periods. A demand change can still lie inside the learned linear patterns. It hits only **three of five windows**, with average precision **0.2365**. Compressing and reconstructing the data well is no guarantee of detecting the events we care about.

More components preserve more training variation but can also reconstruct unusual patterns more easily. Fewer components force stronger compression and can penalize ordinary complexity. Eight is our fixed teaching comparison, not a test-tuned optimum.

## Rung 7: make reconstruction nonlinear

The **autoencoder** keeps that bottleneck idea but learns nonlinear transformations. Our small dense network follows `48 → 24 → 8 → 24 → 48`. It receives a daily window and tries to reproduce it. No event labels enter training.

> **[INSERT TABLE 20 HERE — `graphs/table17-20.png` — notebook cell 77: Keras autoencoder architecture and parameter counts.]**

The model uses **Keras with the PyTorch backend**, running on Apple MPS in the saved experiment. The output layer is linear because globally standardized counts can be negative. We use Adam optimization, batches of 128, and at most 40 epochs. Early stopping watches September windows that do not overlap training at their beginning; calibration remains separate.

> **[INSERT GRAPH 12 HERE — `graphs/graph17-12.png` — notebook cell 79: training and chronological validation reconstruction loss.]**

Both losses fall, with validation remaining above training. The best validation epoch reaches the 40-epoch cap, so early stopping does not fire. This shows improving reconstruction, which is the objective we optimized. It has yet to establish useful anomaly discrimination.

> **[INSERT GRAPH 13 HERE — `graphs/graph17-13.png` — notebook cell 81: Keras autoencoder reconstruction scores and flags.]**

The autoencoder hits all five windows and improves on PCA, but average precision is **0.3492**, behind the seasonal, forecasting, EWMA, and Isolation Forest alternatives. F1 is **0.2763**. A neural network has joined the experiment; it has not won it.

One reason reconstruction deserves caution is capacity. A flexible model can reproduce repeated unusual patterns, especially when they were present in training. We have not run a controlled contamination experiment here, so that is a failure mode to investigate rather than a proven diagnosis of this result. Longer windows also dilute isolated errors and retain old disturbances. Architecture cannot undo the meaning of the representation we chose.

## The scoreboard needs more than one number

All seven detectors score the same 4,992 test observations. Their cutoffs come from the same earlier calibration block, and their point flags remain untouched.

> **[INSERT TABLE 22 HERE — `graphs/table17-22.png` — notebook cell 84: Detector comparison on the shared test period.]**

**Precision** measures how many flags fall inside official windows. **Recall** measures how much labeled duration gets flagged. **Average precision** summarizes the precision–recall ranking across score thresholds; higher is better. It is distinct from the performance of our particular calibrated cutoff.

The windows cover **20.7%** of the test period, so these are window-membership metrics on a relatively generous label definition. Five events on one series cannot establish which detector will dominate another application.

Workload changes the interpretation again. EWMA produces about **4.37 out-of-window flag runs per 30 non-event days**, versus **16.74** for Isolation Forest. Its median first-hit delay is longer: **44.5 hours**, versus **30 hours**. Those delays begin at annotation-window boundaries, not verified physical onset times, and include detected windows only. PCA's misses must remain visible alongside its delay.

Isolation Forest offers the strongest ranking here. EWMA offers fewer separate stretches of out-of-window flags. An operator has a reason to care about both.

## The shortcut that manufactures a perfect F1

The companion notebook includes a deliberately label-informed audit: place one prediction at the start of each official window. That gives five real flags across 1,035 labeled half-hours.

> **[INSERT TABLE 23 HERE — `graphs/table17-23.png` — notebook cell 86: Point adjustment: actual flags versus credited coverage.]**

**Point adjustment** expands a single hit inside an event to credit its entire duration. The predictions have acquired no information, yet F1 becomes perfect. This audit is not a detector; it uses labels to expose the scoring shortcut.

We already have an honest way to credit an event hit: report event recall alongside raw point coverage and delay. Formal range-aware metrics, affiliation measures, and VUS can examine temporal overlap or tolerance more deeply. They need explicit definitions too. A tidy benchmark number earns trust through the evaluation protocol behind it.

## A threshold cannot promise away drift

Our nominal tail setting is 1%, but future out-of-window flag fractions differ from it. For Seasonal MAD, the test fraction is **2.50%**; for Isolation Forest, **2.70%**. These are retrospective label-based error proxies, not verified false incident rates.

The notebook's sensitivity analysis also tries a 0.1% calibration tail. With only **912 calibration observations**, the finite-sample rank lies beyond the sample, so the conservative threshold becomes infinity. There is no finite cutoff at that requested rank. Adding decimal places to a configuration setting does not add observations.

> **[INSERT TABLE 24 HERE — `graphs/table17-24.png` — notebook cell 88: Seasonal MAD: calibration-threshold sensitivity.]**

The rank rule has a conformal interpretation under suitable exchangeability assumptions. Overlapping windows, dependence, and changing demand complicate that interpretation, as readers of [episode 7](https://github.com/tng-konrad/time-series-with-Konrad/blob/main/m07-conformal-prediction.ipynb) will recognize. We use it here as transparent empirical calibration.

Extreme-value methods such as POT and SPOT offer another way to model score tails, provided there are enough representative tail observations and suitable assumptions. Adaptive calibration also needs a policy for which new data to trust. Neither automatically cures a bad reference, and we do not implement or claim their guarantees in this experiment.

## When the reference itself changes

A continuing disagreement raises another question: has something changed in the process? A **change point** marks a boundary between statistical regimes. It can concern level, variance, or other relationships, and a later regime can return to an earlier state. Gradual drift spreads that evolution across time.

For online evidence, we apply a two-sided **CUSUM** chart to signed, standardized forecast residuals:

$$
C_t^+=\max(0,C_{t-1}^++z_t-k),\qquad
C_t^-=\max(0,C_{t-1}^--z_t-k).
$$

Repeated positive or negative residuals build the corresponding accumulator; the slack `k` discounts small deviations. We use `k=0.5` and an illustrative decision limit of eight. A larger limit or slack resists alarms but can delay detection; smaller values react more readily. Residual dependence prevents us from assigning textbook independent-noise run lengths to these settings.

> **[INSERT GRAPH 14 HERE — `graphs/graph17-14.png` — notebook cell 92: online residual CUSUM with accumulator resets.]**

The accumulators repeatedly build and cross the limit. Without resetting them, the chart is above its limit at **all 4,992 test timestamps**; with resets, it triggers at **682**. Those counts are alarm points, not incident counts. The reference and decision rule differ from the seven-detector comparison, so these figures do not belong in its ranking.

> **[INSERT TABLE 25 HERE — `graphs/table17-25.png` — notebook cell 92: CUSUM: alarm points with and without accumulator resets.]**

Resetting an accumulator does not update the baseline. A sustained disagreement can trigger again. Accepting a new regime requires a separate operational judgment; automatically learning every high-score period as normal can absorb an unresolved failure.

## Look backward to understand the regimes

For retrospective analysis, we summarize each complete day by its seasonal-residual mean and log standard deviation. Those features capture changes in level and scale. We standardize them against training days, then use **optimal partitioning** to divide the history into internally similar segments:

$$
\text{objective}=\text{within-segment squared error}
+\lambda\times\text{number of internal boundaries}.
$$

The algorithm trades a better fit against additional boundaries. Larger penalties produce fewer segments; smaller penalties allow more. Every segment must last at least seven days. The notebook's three penalty settings yield 11, nine, and four segments, illustrating how much that choice matters.

> **[INSERT TABLE 27 HERE — `graphs/table17-27.png` — notebook cell 97: Offline segmentation: sensitivity to the boundary penalty.]**

> **[INSERT GRAPH 15 HERE — `graphs/graph17-15.png` — notebook cell 99: offline segmentation of daily residual summaries.]**

The red fitted levels and vertical boundaries summarize sustained historical differences, including quiet holiday stretches. The optimization also sees scale, so a boundary need not coincide with the most obvious mean jump. These are estimated partitions, not verified causal change dates.

The implementation is exact dynamic programming with quadratic time complexity. PELT adds pruning and can achieve linear expected cost under appropriate conditions; we do not relabel our implementation as PELT. More fundamentally, this fit uses the full history. Its dates explain past structure and cannot be presented as live detection times.

## From 358 red dots to a review log

We finish with a small operational policy on Seasonal MAD, the method selected in configuration before inspecting the test comparison. Its original cutoff produces **358 flagged half-hours**. We require at least two flags among the last three observations, then allow a notification record only when six hours have passed since the previous one.

That produces **54 notification records**. The persistence rule reduces isolated warnings; the cooldown limits repeated records during a prolonged disagreement. A long event can still generate reminders, and a short event can be delayed or missed.

> **[INSERT TABLE 29 HERE — `graphs/table17-29.png` — notebook cell 102: Persistence and cooldown: first ten notification records.]**

For Christmas, the first raw flag appears **9.5 hours after the official window begins**; the first notification appears after **10.5 hours**. That one-hour difference is a visible cost of the policy. It is not a claim about warning time before Christmas or before a documented failure onset.

> **[INSERT TABLE 30 HERE — `graphs/table17-30.png` — notebook cell 104: Event review: raw coverage and notification delays.]**

The notebook writes a local table of timestamps, counts, scores, and retrospective window membership. Nothing is sent externally. We retain the original point metrics, because reducing notifications does not retroactively improve detection accuracy. The next operational step would be to review which records deserved attention, who should own them, and how much delay the application can tolerate.

## Closing time

We started with distance from a typical count, added the calendar, let recent history shape a forecast, accumulated residual evidence, learned boundaries around temporal features, and reconstructed complete daily windows. Each step changed what the detector could recognize. The most elaborate reconstruction model still finished behind simpler contextual methods on this dataset.

> **[INSERT TABLE 32 HERE — `graphs/table17-32.png` — notebook cell 106: Method selection: useful questions and principal cautions.]**

The throughline is the reference: what should this observation look like, what history is available when we judge it, and when should we accept that the process has changed? Useful monitoring also needs honest labels, separate calibration, and an alert policy whose costs are visible. The model supplies evidence; the system must make that evidence worth somebody's attention.

## Key Takeaways

1. **Make normal contextual.** A quiet overnight reading and a quiet weekday commute can have the same value and different meanings.
2. **Keep the future in its place.** Fit, select, calibrate, and evaluate chronologically; align window scores to their endpoints.
3. **Compare against simple methods.** EWMA and a seasonal reference remain serious contenders beside learned models.
4. **Measure hits, coverage, delay, and burden.** Event recall alone rewards detectors that barely touch broad windows; point adjustment can erase that weakness.
5. **Review changes before adapting.** A new regime can be legitimate, but an unresolved failure can also persist long enough to look normal.

