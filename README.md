# Time Series with Konrad

Companion Python notebooks for **Time Series with Konrad**, a series of practical tutorials published on [United States of Banan on Substack](https://konradb.substack.com/).

The posts explain the intuition, assumptions, and tradeoffs behind time series methods. These notebooks accompany them with the code: worked examples, data preparation, model fitting, diagnostics, visualizations, and comparisons you can run and explore yourself.

The series progresses from the basics and classical statistical methods to deep learning, transformers, foundation models, and agentic workflows. The emphasis is on understanding what each method adds, when it helps, and where a simpler approach remains hard to beat.

**[Read the tutorials on Substack →](https://konradb.substack.com/)**

## Notebooks

Start with episode 1 and follow the progression, or jump to the topic you need. The episode numbers in the filenames correspond to the tutorial series.

| Episode | Topic and notebook |
| :---: | :--- |
| 01 | [The basics: decomposition and dependence](m01-the-basics.ipynb) |
| 02 | [Exponential smoothing](m02-smoothing-methods.ipynb) |
| 03 | [ARMA, ARIMA, SARIMA, and SARIMAX](m03-arima-family.ipynb) |
| 04 | [Hierarchical forecasting](m04-hierarchical-forecasting.ipynb) |
| 05 | [Survival analysis](m05-survival-analysis.ipynb) |
| 06 | [Sales forecasting and intermittent demand](m06-sales-and-demand.ipynb) |
| 07 | [Conformal prediction](m07-conformal-prediction.ipynb) |
| 08 | [Time series validation](m08-validation-methods.ipynb) |
| 09 | [Causal methods](m09-causal-methods.ipynb) |
| 10 | [State space methods](m10-state-space.ipynb) |
| 11 | [Deep learning: RNN, GRU, and LSTM](m11-dl-for-ts.ipynb) |
| 12 | [Transfer learning](m12-transfer-learning.ipynb) |
| 13 | [Transformers](m13-transformers.ipynb) |
| 14 | [Time series foundation models](m14-tsfm.ipynb) |
| 15 | [Self-supervised learning](m15-ssl.ipynb) |
| 16 | [Agentic methods](m16-agents.ipynb) |

## Agentic forecasting series

| Module | Notebook | Article |
| :---: | :--- | :--- |
| A01 | [From a forecasting notebook to an agentic workflow](a01-forecasting-workflow.ipynb) | [Post draft](a01-post.md) |

A01 uses 16 Australian retail series and compares a deterministic report, one local LLM call, and a two-call prompt chain over the same forecasts. It defaults to **offline replay**: numerical results are recomputed and genuine archived model responses are used only when their inputs match. Both archived LLM reports contain documented semantic defects; see the [report audit](examples/a01/report_audit.md).

Each run exports all three graphs and nine tables to `graphs/`, numbered separately in notebook order: `atsf01-graph1.png`–`atsf01-graph3.png` and `atsf01-table1.png`–`atsf01-table9.png`. The post includes exact filename placeholders and a complete asset index.

After the environment setup below, run the notebook interactively or execute it in a fresh kernel:

```sh
.venv/bin/python -m scripts.execute_a01
.venv/bin/python -m unittest discover -s tests -p 'test_a01*.py' -v
```

For live narration, start Ollama with the installed `qwen3-coder-next:q4_K_M` model and change `CFG.run_mode` to `"live"` in the notebook's Setup section. `CFG.llm` contains the local endpoint, model, timeout, and generation settings. Paths and display settings also live in `CFG`; export and diagram helpers are grouped under Utils. No model or data download occurs during default notebook execution. Live calls have one attempt per stage; they never overwrite the bundled capture.

To save another live attempt from the archived numerical experiment, use a new output path:

```sh
.venv/bin/python -m scripts.capture_a01 --output examples/a01/another_run.json
```

The source panel is already bundled. To reproduce its extraction, download the pinned [ABS Table 11 workbook](https://www.abs.gov.au/statistics/industry/retail-and-wholesale-trade/retail-trade-australia/jun-2025/8501011.xlsx) and run the standard-library-only preparation script, supplying the actual retrieval date:

```sh
.venv/bin/python scripts/prepare_a01_data.py /path/to/8501011.xlsx --retrieved-on YYYY-MM-DD
```

The [implementation record](docs/module-01-implementation.md) documents provenance, validation, results, and limits. The [plan](docs/module-01-plan.md) and its [critique history](docs/module-01-critic-review.md) explain the design.

## Running the notebooks

The project uses Python 3.12. Dependencies are defined in [`pyproject.toml`](pyproject.toml), with a resolved environment in [`uv.lock`](uv.lock).

With [uv](https://docs.astral.sh/uv/), set up the environment from the repository root:

```sh
git clone https://github.com/tng-konrad/time-series-with-Konrad.git
cd time-series-with-Konrad
uv sync --locked
```

Open a notebook in your preferred Jupyter-compatible editor, select the Python environment in `.venv`, and run the cells in order. Use the repository root as the working directory so paths such as `./data/` resolve correctly.

Some later notebooks download pretrained models or use local services. Their setup cells describe the additional requirements and device settings.

## Repository layout

- `mNN-*.ipynb` — the companion notebooks, organized by episode.
- `data/` — datasets and supporting data files used in the examples.
- `graphs/` — exported figures used in the tutorial posts.
- `scripts/` — supporting utilities.

For the explanations and the story behind the code, head to [United States of Banan](https://konradb.substack.com/). New notebooks are added as the series grows.

