# Recipe: Writing a Teaching Notebook in the m01–m05 Style

A reconstruction of the methodology used in `m01-the-basics` through `m05-survival-analysis`, written so a new notebook on a different topic can be produced at the same level of technical detail, for the same audience.

---

## 1. Audience and voice

The implied reader is comfortable reading Python but **new to the domain** being taught. Calibrate accordingly:

- Explain library mechanics even when standard: what `pd.to_datetime` does, why `.copy()` is used, why `inplace=True` matters, what an alias like `np` is. Nothing is assumed beyond basic Python literacy.
- Weave domain theory **into the code commentary**, not into separate theory sections. The concept of stationarity is taught while explaining the `adfuller` call; additive vs. multiplicative decomposition is taught at the moment `model='multiplicative'` appears; hazard ratios are explained while describing `print_summary` columns. Theory always arrives attached to a concrete line of code.
- Flag the "gotcha" parameters explicitly and explain *why* they are set: `optimized=False` ("This is important! It tells the model *not* to..."), the minus sign on Cox risk scores, `dropna()` before an ADF test because the test cannot handle NaNs, skipping the first rows of a decomposition trend because of edge NaNs.
- Tone: instructive, patient, slightly informal. Sentences like "Let's break it down step-by-step" and "This is crucial" are characteristic. No academic citations, no equations beyond inline pseudo-formulas (`Original Series = Trend + Seasonality + Residuals`).

## 2. The core structural rule: code cell → explanation cell

This is the single most important convention. The notebook is a strict alternation:

1. A code cell (small — one logical action: load data, define one function, fit one model, make one plot).
2. A markdown cell immediately after it, explaining **that cell and only that cell**.

Explanation cells follow one of two registers (both appear in the series; pick one and stay consistent within a notebook):

- **Bulleted line-by-line** (m01, m02 style): each line or argument of the code gets a bullet, with nested bullets for individual keyword arguments. Bold the API names and statistical terms on first use.
- **Flowing paragraphs** (m03–m05 style): 2–5 short paragraphs that walk through the cell in execution order — "First, ... Next, ... Finally, ..." — naming each function, each argument, and what it returns.

Either way, an explanation covers three layers for every non-trivial cell:
1. **What** each call does mechanically (function, arguments, return value, where the result is stored).
2. **Why** this step exists in the workflow (e.g., "splitting chronologically simulates a real forecasting scenario").
3. **How to interpret** the output (what a small p-value means here, what a steeper slope on the cumulative hazard plot indicates, what "lower IBS" implies).

**Permitted omissions:** once a pattern has been explained in full, later repetitions of the *identical* pattern (e.g., the per-method evaluation block repeated for MiddleOut/TopDown/MinTrace, or the third and fourth `seasonal_decompose` call) may appear as bare code cells with no markdown. Explain a pattern once, thoroughly; then let repetition stand on its own.

## 3. Notebook skeleton

Use `#` headers for the major phases and `##` headers for individual methods/variants. The canonical skeleton:

```
# Setup
# Utils            (a.k.a. "Functions")
# <Concept groundwork section>      e.g. "Decomposition", "Groundwork", "Basic toolkit"
# <Method family section>
## <Simplest variant>
## <Next variant>
## <Most complex variant>
# <Application or model-building section>   e.g. "Application: anomaly detection", "Building a model"
```

### 3.1 Setup section (always first)

One imports cell, grouped with comment headers, in this order:

```python
# Standard library imports
import warnings
import os

# Third-party imports          (core data → modeling → visualization, or grouped by package)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from <domain_library> import <SpecificClasses>

# Configuration & Settings
warnings.simplefilter(action='ignore', category=FutureWarning)
```

Then a configuration cell with the `CFG` class — this exact idiom recurs in every notebook:

```python
# general settings
class CFG:
    data_folder = './data/'
    graph_folder = './graphs/'
    img_dim1 = 20
    img_dim2 = 10
    SEED = 42
    metric = 'rmse'        # add notebook-specific constants here

# display style
plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (CFG.img_dim1, CFG.img_dim2)

np.random.seed(CFG.SEED)
```

Both cells get full explanations (yes, including what `import numpy as np` means and why a seed of 42 ensures reproducibility).

### 3.2 Utils section

3–6 small reusable helper functions, each in its own cell with its own explanation. The recurring helper types:

- **A metrics function** returning a rounded dict or scalar (`forecast_metrics` → MAE/RMSE; `compute_cindex` for survival). Round to 2–4 decimals; convert to `float`.
- **A pretty-printer for a statistical test** (`print_adf_result`), handling NaNs internally and formatting output with f-strings.
- **A split function** that respects the data's structure (chronological `train_valid_split` on a date index; `.tail(horizon)` per group for panel data).
- **A combined plotting wrapper** when the same two-panel or overlay plot is needed repeatedly (`plot_acf_pacf`).
- **Comparison/merge helpers** when the notebook compares methods (`merge_forecasts`, `compute_error_per_level`, `compare_baseline_vs_reconciled` returning a tidy DataFrame with a delta column and an `improved` boolean).

Design principle: anything done more than twice in the body becomes a helper *here*, parameterized with sensible defaults (`value_col="value"`, `metric='rmse'`), so the body cells read as short declarative calls.

### 3.3 Groundwork / basic-toolkit section

Before the headline methods, establish the data and the elementary tools:

- Load a real dataset from `CFG.data_folder` (or an API like FRED via `pandas_datareader`); convert the date column with `pd.to_datetime`; set it as the index; **immediately plot the raw series** (`.plot(xlabel="")`). A plot of the raw data always precedes any modeling.
- Show `df.head(3)` after any non-trivial preparation as a sanity check.
- If a concept is best seen on controlled data first, **simulate it** (white noise via `np.random.normal`, AR/MA processes via `ArmaProcess.generate_sample`) and inspect it with the standard diagnostic plots before touching real data.
- Run the elementary descriptive/diagnostic tools (decomposition, ACF/PACF, a stationarity test, Kaplan-Meier curve — whatever the domain's "first look" tools are) and interpret the output in the explanation cells.

### 3.4 Methods sections — the pedagogical ladder

Order method variants from simplest to most complex, one `##` subsection each, with each step adding exactly one new ingredient:

- m02: Single ES → Double (adds trend) → Triple (adds seasonality)
- m03: AR → MA → ARMA → ARIMA (adds differencing) → SARIMA (adds seasonal terms) → SARIMAX (adds exogenous regressors)
- m04: Baseline → BottomUp → MiddleOut → TopDown → MinTrace
- m05: KM curve → group comparison + log-rank test → Nelson-Aalen → Cox regression → Random Survival Forest → head-to-head comparison

Within each subsection, the cells follow a fixed micro-template:

1. **Constants cell** (optional): SCREAMING_SNAKE_CASE values defined just above first use — `HORIZON = 12`, `ALPHAS = [0.2, 0.5, 0.9]`, `WINDOW_SIZE = 25`. Never magic numbers inline.
2. **Fit cell**: instantiate model, fit. Show both an "automatic" route (auto_arima, `optimized=True`) and a "manual" route (explicit orders, fixed alphas) where the library offers both — the manual route teaches what the parameters mean, the automatic route is what you'd use in practice.
3. **Inspect cell**: `model.summary()` / `params_formatted` / `print_summary(...)` — and explain how to read each column.
4. **Predict + evaluate cells**: forecast over the holdout, merge with actuals into a tidy results DataFrame (`actual`, `predicted`, CI bounds), call the metrics helper.
5. **Diagnose cell**: residual ACF/PACF, `plot_diagnostics()`, or the domain's equivalent — with an explanation of what "good" looks like (no significant residual autocorrelation, normal Q-Q, etc.).
6. **Plot cell**: overlay observed vs. fitted vs. forecast.

For the repeated variants (steps 2–6 re-run with a new method), reuse the *identical* cell sequence verbatim so the reader can diff methods, not code.

### 3.5 Application / model-building section

End (or punctuate) the notebook with at least one of:

- **A practical application** that connects the method to a real problem the reader recognizes (rolling z-score anomaly detection on sensor data in m02).
- **A full mini-workflow** on a fresh dataset: load → plot → decompose/diagnose → split chronologically → fit → evaluate residuals → plot predictions vs. actuals → report metrics ("Building a model" in m02).
- **A head-to-head model comparison** with a shared metric table or overlaid diagnostic curves (RSF vs. Cox C-index/IBS/Brier in m05; baseline vs. reconciled per level in m04).

## 4. Code style conventions

- Pandas-first: `Series.plot()` / `DataFrame.plot()` over raw matplotlib where possible; `plt.figure()` + explicit `plt.plot`/`plt.scatter` only when overlaying heterogeneous elements.
- Plot semantics are consistent across the whole series: **observed = default solid line, linewidth=2; smoothed/fitted = red; forecast = green dashed with `marker="o"`; CI bounds = dashed coral/red shades; anomalies = red scatter with `zorder=3`**. Always `plt.legend()`; titles name the method ("Holt-Winters Additive Smoothing"); dynamic labels embed parameter values via f-strings (`f"Smoothed (α={alpha})"`).
- Splits are always **chronological** — a cutoff date, `iloc[:-24]`, or per-group `.tail(horizon)` — never random shuffles. Explain why in the markdown (simulating real forecasting on unseen future data).
- Datasets: small, classic, recognizable (airline passengers, sunspots, FRED macro series, M5 sales, Telco churn). Local CSVs under `./data/`; if preprocessing was needed, do it in a separate helper notebook and link to it in a comment.
- Light inline comments in code cells marking the role of each block (`# Original series`, `# Highlight anomalies`, `# fit the model`); the heavy lifting of explanation lives in the markdown cells, not the comments.
- Bare expression as the last line of a cell (`results_df`, `df.head(3)`, `fit1.params_formatted`) to display output; occasional `print()` after a plotting call to suppress the repr.

## 5. Length and density targets

Per notebook: roughly **25–45 code cells**, of which 60–80% carry an explanation cell. Explanations run ~100–350 words for new material, with the longest reserved for the Setup imports, the first appearance of each model class, and interpretation-heavy outputs (test summaries, diagnostics). Total prose on the order of 5,000–9,000 words.

## 6. Checklist for a new notebook

1. Pick the topic's method family and order its variants into a ladder where each rung adds one concept.
2. Choose one small real dataset (plus synthetic data if a concept benefits from a controlled demo) and one evaluation metric set.
3. Write Setup (grouped imports + CFG + style + seed) and Utils (metrics, splitter, printers, plot wrappers).
4. Groundwork: load, plot raw data, run the domain's first-look diagnostics, explain how to read them.
5. For each ladder rung: constants → fit (manual and/or auto) → inspect → predict → evaluate → diagnose → plot, reusing the identical cell sequence.
6. Close with an application or a head-to-head comparison producing a tidy comparison table.
7. Pass through and write the explanation cells: three layers (what / why / interpretation), full detail on first occurrence, silence on exact repetitions.
8. Verify conventions: chronological splits only, no magic numbers, consistent plot colors, seeds set, `head(3)` sanity checks after data prep.
