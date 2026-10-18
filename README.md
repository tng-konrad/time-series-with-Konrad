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

