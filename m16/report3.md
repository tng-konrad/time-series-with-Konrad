# Agentic Solutions for Time Series Analysis: Forecasting and Anomaly Detection / Root Cause Analysis

## TL;DR
- **Agentic LLM systems add the most value not as raw numerical predictors but as orchestrators**: they plan, call classical/foundation-model tools, run code in sandboxes, integrate textual context (news/events), and produce auditable rationales. For pure numeric accuracy on clean series, a well-tuned foundation model (Chronos, TimesFM, Moirai, Toto) or even a simple linear/statistical baseline usually matches or beats an LLM at far lower cost — a result rigorously shown by "Are Language Models Actually Useful for Time Series Forecasting?" (Tan et al., NeurIPS 2024), whose ablations found that removing the LLM component "does not degrade forecasting performance—in most cases, the results even improve," beating Time-LLM in 19 of 26 cases across 13 datasets.
- **The strongest current use cases are context-aware forecasting, explainable anomaly detection, and observability/AIOps root-cause analysis** — tasks that genuinely require reasoning over heterogeneous evidence. Frameworks like TimeCopilot (open source), TimeSeriesScientist, Nexus, TimeXL/TimeCAP, SigLLM (MIT), LLMAD, and RCAgent/OpenRCA exemplify this, but benchmarks (OpenRCA, CiK, TimeSeriesGym) show current agents still fail most of the hardest tasks — on OpenRCA "even the best-performing LLM can only solve 11.34% of failure cases."
- **Recommended learning path**: start with a foundation-model baseline + backtesting discipline, then a single tool-calling forecasting agent (TimeCopilot or an AutoGluon-wrapping agent), then add retrieval/context, then multi-agent planner–forecaster–critic loops with evaluation gates. Treat cost, latency, non-determinism, hallucinated numbers, and data leakage as first-class engineering risks.

## Key Findings

**1. There are five recurring design patterns.** (a) *LLM-as-forecaster* — serialize numbers to text and let the model extrapolate (LLMTime; PromptCast) or reprogram/align a frozen LLM (Time-LLM). (b) *Tool-calling / code-generating agents* that orchestrate classical libraries (StatsForecast, AutoGluon, Darts, sktime). (c) *Multi-agent pipelines* with planner/analyst/forecaster/critic-reflection roles (TimeSeriesScientist, Nexus, TS-Agent, TimeXL). (d) *Retrieval-augmented / context-aware forecasting* that ingests news and textual side-information (Context-is-Key benchmark, TimeCAP, TimeXL, TS-RAG). (e) *Agentic RCA / AIOps* that reasons across metrics, logs and traces (RCAgent, D-Bot, OpenRCA, Flow-of-Action).

**2. Agentic ≠ foundation model, but they combine.** Time-series foundation models (TSFMs) — Chronos/Chronos-2 (Amazon), TimesFM (Google), Moirai/Moirai-2 (Salesforce), Lag-Llama, MOMENT (CMU), Toto (Datadog), Time-MoE — are single-pass numeric predictors. Agents wrap them: they diagnose the series, pick/ensemble among TSFMs and statistical models, add context, and explain. TimeCopilot and TS-Reasoner explicitly use TSFMs as tools/encoders inside an LLM reasoning loop.

**3. A skeptical evidence base is essential.** Tan et al. (NeurIPS 2024) showed removing the LLM from three popular LLM-forecasters didn't hurt (often helped) accuracy; a simple "PAttn" attention baseline beat them, and pretrained LLMs "do no better than models trained from scratch, do not represent the sequential dependencies in time series, and do not assist in few-shot settings." OpenRCA shows frontier LLMs solve only ~11% of real cloud RCA cases. These are the load-bearing caveats.

## Details

### A) Approaches and architectures

**LLM-as-forecaster (the foundational lineage).**
- **LLMTime** — "Large Language Models Are Zero-Shot Time Series Forecasters," Gruver, Finzi, Qiu & Wilson (NYU), NeurIPS 2023, arXiv:2310.07820. Encodes numbers as digit strings; GPT-3/LLaMA-2 zero-shot extrapolate at or above purpose-built models. Notably GPT-4 was *worse* than GPT-3 (alignment/RLHF hurts calibration). Code: github.com/ngruver/llmtime.
- **PromptCast** — Xue & Salim, IEEE TKDE 2023: forecasting reframed as prompt-based text-to-text.
- **Time-LLM** — Jin et al., ICLR 2024, "reprogramming" a frozen LLM with text-prototype patches + Prompt-as-Prefix. Code: github.com/KimMeen/Time-LLM; also in Nixtla's NeuralForecast.
- Related alignment methods: GPT4TS/FPT (Zhou et al. 2023), TEMPO, UniTime, AutoTimes, CALF, TEST.

**Time-series foundation models (the tools agents call).**
- **Chronos / Chronos-Bolt / Chronos-2** (Amazon; Ansari et al. 2024) — tokenize + T5-style; Chronos-2 unifies univariate/multivariate/covariate forecasting. github.com/amazon-science/chronos-forecasting.
- **TimesFM** (Google; Das et al., ICML 2024) — decoder-only. github.com/google-research/timesfm.
- **Moirai / Moirai-MoE / Moirai-2** (Salesforce; Woo et al. 2024) — any-variate encoder; github.com/SalesforceAIResearch/uni2ts.
- **Lag-Llama** (Rasul et al. 2023) — probabilistic decoder-only.
- **MOMENT** (Goswami et al., ICML 2024, CMU) — masked encoder pretrained on the "Time Series Pile."
- **Toto** (Datadog) — "Time Series Optimized Transformer for Observability," trained on observability telemetry (Toto 2.0 reported at >1 trillion time-series points, ~75% real telemetry); powers Watchdog/Bits AI research. **TimeGPT** (Nixtla; Garza & Mergenthaler-Canseco) — first commercial TSFM, forecasting + anomaly detection API.
- Community benchmarks: **GIFT-Eval** and **fev-bench** standardize TSFM evaluation.

**Multi-agent forecasting pipelines.**
- **TimeSeriesScientist (TSci)** — "TimeSeriesScientist: A General-Purpose AI Agent for Time Series Analysis," Zhao et al., arXiv:2510.01538 (Oct 2025). Four agents: Curator (diagnostics/preprocessing), Planner (model choice), Forecaster (fit/validate/ensemble), Reporter (transparent report). **Public repo (MIT license, LangGraph-based): github.com/Y-Research-SBU/TimeSeriesScientist** (project site Y-Research-SBU.github.io/TimeSeriesScientist). On the ETT benchmark it reported average MAE ≈ 4.87 vs the best LLM baseline (Claude-3.7) at 6.94 and GPT-4o at 9.94 — roughly a 30% lower MAE than the strongest LLM baseline.
- **Nexus** — "Nexus: An Agentic Framework for Time Series Forecasting," Das et al. (Google + Penn State), arXiv:2605.14389 (May 2026). Multi-agent decomposition into a historical-context agent, macro- and micro-level temporal-reasoning forecaster agents, and a synthesizer/calibration agent; argues forecasting is "an agentic reasoning problem extending well beyond only sequence modeling." Evaluated on Zillow and stock data post-dating LLM knowledge cutoffs; reported up to ~86.6% MAPE reduction vs a Chain-of-Thought baseline in a best-case Claude-4.5-Sonnet Zillow configuration (typical per-backbone gains 1–15%; some long-horizon stock cases were slightly worse than CoT). **No official code repo found** — a third-party reproduction exists at github.com/NiharJani2002/nexus-forecasting (community, not the authors').
- **TS-Agent** (arXiv:2510.07432) — iterative "insight gathering" over raw series.
- **Structured Agentic Workflows for Financial Time-Series** (arXiv:2508.13915) — planner + reflective feedback; reports RMSE reductions of up to ~30% over DS-Agent and 15–40% over ResearchAgent.
- **DCATS** — "Empowering Time Series Forecasting with LLM-Agents," Yeh et al. (Visa Research), arXiv:2508.04231. A *data-centric* agent that cleans data using metadata; reports an average 6% error reduction across models and horizons.

**Context-aware / retrieval-augmented forecasting.**
- **Context is Key (CiK)** — Williams et al. (ServiceNow/Mila), NeurIPS 2024 workshop / ICML 2025, arXiv:2410.18959. "A collection of 71 manually designed forecasting tasks spanning seven real-world domains" (climatology, economics, energy, public safety, transportation, retail, mechanics), drawn from 2,644 time series; introduces Region-of-Interest CRPS (RCRPS); the best baseline is Llama-3.1-405B-Instruct using a simple "Direct Prompt" method, which outperforms statistical models and numeric-only TSFMs. Code: github.com/ServiceNow/context-is-key-forecasting.
- **TimeCAP** — "TimeCAP: Learning to Contextualize, Augment, and Predict Time Series Events with Large Language Model Agents," Lee et al., AAAI 2025 (Vol. 39 No. 17, pp. 18082–18090), arXiv:2502.11418. A summary/contextualizer agent captures textual context; a prediction agent forecasts; plus a multimodal encoder. **Code + 7 datasets: github.com/geon0325/TimeCAP.**
- **TimeXL** — "TimeXL: Explainable Multi-modal Time Series Prediction with LLM-in-the-Loop," Jiang et al., NeurIPS 2025, arXiv:2503.01013. Prototype-based multimodal encoder + prediction/reflection/refinement LLM agents in a closed loop; reports up to 8.9% AUC improvement over baselines across four real-world datasets. (No official public code repo confirmed.)
- **TS-RAG** (arXiv:2503.07649) and **TimeRAF** (arXiv:2412.20810) — retrieval-augmented TSFMs.

**Agentic AutoML / MLE for time series.**
- **TimeSeriesGym** — Cai et al. (CMU/MOMENT team), NeurIPS 2025, arXiv:2505.13291. Benchmark of 33 challenges from 23 data sources across 8 problem types (data handling, HPO, code migration, feature engineering, research-code utilization) for AI ML-engineering agents, combining quantitative metrics with LLM-based evaluation. Code: github.com/moment-timeseries-foundation-model/TimeSeriesGym.
- **MLZero** (arXiv:2505.13941, multi-agent end-to-end ML automation) and **AutoGluon-TimeSeries** (Shchur et al., AutoML 2023) as the tool backbone.

**Anomaly detection.**
- **SigLLM** — Alnegheimish et al. (MIT), arXiv:2405.14755, "Large language models can be zero-shot anomaly detectors for time series?" Extends MIT's **Orion** library; two pipelines: *Prompter* (directly prompt the LLM for anomaly indices) and *Detector* (forecast-and-compare residuals). Across 11 datasets the Detector pipeline reached avg F1 ≈ 0.525 and beat the Prompter by ~135% in F1, but "state-of-the-art deep learning models still surpassed its performance by an average of 30%." Code: github.com/sintel-dev/sigllm.
- **LLMAD** — Liu et al., "Large Language Models can Deliver Accurate and Interpretable Time Series Anomaly Detection," KDD 2025, arXiv:2405.15370. In-context retrieval of similar normal/abnormal segments + Anomaly Detection Chain-of-Thought (AnoCoT); performance "comparable to state-of-the-art deep learning methods" while adding interpretable explanations. Code: github.com/LJunius/LLMAD.
- **AnomalyLLM** (knowledge distillation), **TriP-LLM**, **CALM** (arXiv:2508.21273, streaming/continuous), **AD-LLM** (Yang et al., benchmark, arXiv:2412.11142), **ChatAD** (arXiv:2601.13546).

**Agentic RCA / AIOps.**
- **RCAgent** — Wang et al., CIKM 2024: tool-augmented autonomous agents for cloud RCA.
- **D-Bot / LLM-as-DBA** — Zhou et al. (Tsinghua), VLDB 2024, arXiv:2312.01454 & arXiv:2308.05481. Tree-of-thought RCA + multi-agent collaboration for database diagnosis; "verified on real benchmarks (including 539 anomalies of six typical applications)," producing a diagnosis report "within acceptable time (e.g., under 10 minutes compared to hours by a DBA)."
- **OpenRCA** — Xu et al. (Microsoft), ICLR 2025. Benchmark of 335 real failure cases + 68 GB telemetry from three enterprise systems + an RCA-agent baseline; "even the best-performing LLM can only solve 11.34% of failure cases." Code: github.com/microsoft/OpenRCA. Follow-ups: OpenRCA 2.0 (causal process supervision; ~20.7% exact root-cause-set recovery averaged over 11 frontier LLMs), RCAEval (735 cases across 11 fault types), and a process-level failure analysis ("Why Do AI Agents Systematically Fail at Cloud RCA?", arXiv:2602.09937, 1,675 runs, 12 pitfall types).
- Others (from the awesome-LLM-AIOps list): mABC (EMNLP 2024 Findings), Flow-of-Action (WWW 2025), Cloud Atlas, eARCO, ThinkFL, TN-AutoRCA (telecom).

**Key architectural components.** Across the surveys ("From Prompts to Agents: A Comprehensive Survey of LLM-Driven Time Series Analysis," 150+ studies, repo github.com/CoderPowerBeyond/Agent-Prompt-TS-Survey; "LLM Agents for Time-Series: A Survey," arXiv:2608.26226, 47 papers; IJCAI 2024 "Empowering Time Series Analysis with LLMs"), agents are decomposed into: **perception** (time-series-to-text or patch encoders, TSFM embeddings), **planning** (model/tool selection, hypothesis narrowing), **tool use** (statistical/ML libraries, code sandboxes, DB/observability APIs), **memory** (historical series, past incidents), and **reflection/critique** (self-evaluation, backtesting loops, human-in-the-loop).

### B) Notable research and benchmarks

- **Reasoning benchmarks:** TimeSeriesExam (Cai et al. 2024, 763 multiple-choice questions from 104 templates, calibrated via item-response theory), the Time-Series Reasoning benchmark (Merrill et al. 2024), MTBench (finance/weather), TSRBench, Time-MQA/TSQA (Kong et al. 2025, ~200k QA pairs across 12 domains, LoRA-tuned Mistral/Llama-3/Qwen backbones).
- **TS-Reasoner** — two distinct papers share the name: (i) arXiv:2410.04047, a compositional inference agent orchestrating expert tools, benchmarked on TimeSeriesExam; (ii) arXiv:2510.03519, "Aligning Time Series Foundation Models with LLM Reasoning" (uses TimesFM as frozen encoder; TS-Reasoner-7B ≈ 54.83% on TimeSeriesExam, a 17.5% relative gain over Qwen2.5-7B-Instruct, using <half the training data of ChatTS).
- **Forecasting-with-context:** CiK (above); ChatTime; TimeXL; a chain-of-thought-via-RL approach (COUNTS, arXiv:2510.01116).
- **Position/critique papers:** Tan, Merrill, Gupta, Althoff & Hartvigsen, "Are Language Models Actually Useful for Time Series Forecasting?" NeurIPS 2024 spotlight, arXiv:2406.16964 (code github.com/BennyTMT/LLMsForTimeSeries; the "PAttn" baseline); "Position: Beyond Model-Centric Prediction — Agentic Time Series Forecasting" (arXiv:2602.01776); "Position: Universal Time Series Foundation Models Rest on a Category Error" (arXiv:2602.05287, argues LLMs model semantic correlation not causal/generative temporal structure).
- **Evaluation methodology:** proper backtesting/rolling-origin cross-validation; probabilistic metrics (CRPS, RCRPS, quantile loss, MASE, sMAPE); success-rate + execution-error tracking for agents; OpenRCA's all-or-nothing exact-match scoring (1 point only if all root-cause elements match); process-level (not just outcome) evaluation is an emerging demand (OpenRCA 2.0).

### C) Hands-on implementations, tutorials, repos

**Agentic forecasting frameworks (open source).**
- **TimeCopilot** — github.com/TimeCopilot/timecopilot; site timecopilot.dev; paper arXiv:2509.00616. "GenAI forecasting agent": LLM reasoning orchestrating 30+ TSFMs (Chronos, Moirai, TimesFM, TimeGPT) plus statistical/ML/deep models under one API; three-step workflow (feature analysis → model selection/evaluation via cross-validation → final forecast + explanation); natural-language querying; anomaly detection; sktime support; one-line `uvx timecopilot forecast <url>`. Accepted at a NeurIPS 2025 workshop; claims #1 on GIFT-Eval (self-reported). Also runs a live drift benchmark, "Impermanent" (github.com/TimeCopilot/impermanent).
- **TimeSeriesScientist** — public LangGraph-based repo above (`python main.py --data_path … --horizon 96 --llm_model gpt-4o`).

**Classical / foundation-model libraries agents wrap.**
- **Nixtla ecosystem**: StatsForecast, MLForecast, NeuralForecast, TimeGPT; Nixtla now advertises MCP + agentic capabilities (enterprise waitlist).
- **AutoGluon-TimeSeries** — 3-lines-of-code probabilistic AutoML forecasting with ensembling; AWS blog tutorial "Easy and accurate forecasting with AutoGluon-TimeSeries"; docs at auto.gluon.ai; SageMaker integration.
- **Darts** (unit8) — now with a unified FoundationModel API integrating four TSFMs (arXiv:2606.27438). **sktime**, **aeon**, **skforecast**, **PyTorch Forecasting**, **GluonTS**.
- **Anomaly detection**: **Orion** (MIT sintel-dev) + SigLLM; **Merlion** (Salesforce); **PyOD / TODS**; **TimeEval** (benchmarking); Rob Hyndman's ported detectors (STRAY, DOBIN) in sktime.

**Agent frameworks + observability MCP servers.**
- LangChain/LangGraph Pandas DataFrame agents for temporal EDA (Data Science Dojo tutorial, datasciencedojo.com/blog/langchain-agents-for-time-series-analysis).
- **MCP servers for time series/observability**: official **Grafana MCP** (mcp-grafana: dashboards, Prometheus/Loki/Tempo, alerts, OnCall, Sift investigations, Pyroscope); **Prometheus MCP** (AWS Managed Service for Prometheus — natural-language PromQL via Amazon Q/Cline/Cursor; AWS Cloud Operations blog); **Datadog MCP Server** (telemetry to Codex/Claude Code/Cursor for incident investigation); community unified **observability-mcp** (github.com/thotischner/observability-mcp — Prometheus + Loki cross-signal anomaly detection).

**Runnable tutorials (2024-2026).**
- MarkTechPost (20 June 2026): "How to Build a Forecasting Pipeline with TimeCopilot Using Foundation Models and Automated Anomaly Detection" — builds an end-to-end pipeline on real airline-passenger data plus a synthetic seasonal series with injected anomalies: rolling cross-validation across statistical/foundation/GPU models with multiple error metrics, probabilistic forecasts with prediction intervals, anomaly detection, and an optional LLM agent that auto-selects a model and returns a natural-language analysis. URL: marktechpost.com/2026/06/20/how-to-build-a-forecasting-pipeline-with-timecopilot-using-foundation-models-and-automated-anomaly-detection/
- Nixtla NeuralForecast getting-started (M4 Hourly; fit/predict/cross_validation).
- AutoGluon-TimeSeries quickstart (AWS blog + docs).
- SigLLM tutorials/data.csv notebook (github.com/sintel-dev/sigllm).

**Industry / production (marketing vs documented).**
- **Datadog**: Toto TSFM is *research* and, per its introduction blog, "is still early in its development and isn't currently deployed in any production systems"; it powers R&D behind Watchdog/Bits AI. The Bits AI agent suite (SRE/Dev/Security) does RCA→remediation with GitHub/Jira integration — a shipped product but vendor-described. Third-party efficiency claims (e.g., a "34% reduction in mean-time-to-investigate" attributed to a Block case study) should be treated as vendor/case-study figures.
- **Dynatrace Davis AI**: emphasizes *causal* AI (deterministic, topology-based RCA) over "correlation and chatbots," now adding predictive + generative layers (natural-language problem summaries, remediation-artifact generation such as Kubernetes resource limits). Feb 2025 "preventive operations" release. Largely documented, but marketing-framed.
- **AWS**: AutoGluon-TimeSeries, Chronos on SageMaker, Prometheus MCP — technically documented and open.
- **Nixtla, Salesforce (Merlion/Moirai), Google (TimesFM), Microsoft (OpenRCA)** — code-backed.
- Vendors such as New Relic, Grafana, Splunk, Anodot make AIOps/anomaly claims that are mostly product marketing; verify against docs.

### D) Practical guidance

**When agentic adds value:**
- The task needs **exogenous/textual context** (upcoming events, promotions, planned outages, news) that pure numeric models can't see — CiK and TimeCAP show this is where LLMs shine (the CiK Direct-Prompt LLM outperforms numeric-only TSFMs on context-dependent tasks).
- You need **explanations, auditability, or a report** for non-expert stakeholders (TimeSeriesScientist, LLMAD).
- **Heterogeneous, multi-step diagnosis** across metrics/logs/traces (AIOps RCA).
- **Automating the workflow** (model selection, cleaning, HPO) across many series where hiring analysts doesn't scale (DCATS, AutoGluon-wrapping agents).

**When it's overkill:**
- Clean, stationary, high-frequency univariate/multivariate series where a tuned statistical model or a zero-shot TSFM already wins — the Tan et al. result is decisive here. Don't pay LLM cost/latency for numeric extrapolation the LLM does worse.

**Failure modes / engineering considerations:**
- **Hallucinated numbers** and poor uncertainty calibration (GPT-4 worse than GPT-3 in LLMTime; LLMs "do not represent the sequential dependencies in time series").
- **Non-determinism** — same prompt, different forecast; pin seeds/temperature, log everything.
- **Cost & latency** — multi-agent loops multiply token spend; D-Bot's "under 10 minutes" is fine for RCA but too slow for high-frequency forecasting.
- **Data leakage** — LLMs may have seen your public benchmark; evaluate on data strictly after knowledge cutoff (as Nexus does with post-cutoff Zillow/stocks).
- **Backtesting discipline** — rolling-origin CV, probabilistic metrics, no peeking; agents that "improve" on a leaky split are illusory.
- **Reproducibility** — version prompts, tools, model checkpoints; prefer process-level eval (OpenRCA 2.0) over outcome-only, since "even correct answers often arise from misaligned reasoning."

## Recommendations

**Stage 0 — Baseline first (before any agent).** Run AutoGluon-TimeSeries or Nixtla StatsForecast, plus a zero-shot TSFM (Chronos-2 or TimesFM). Establish rolling-origin backtests with MASE/CRPS. *Threshold to change decision:* if a TSFM already hits your accuracy target, stop — don't add an agent.

**Stage 1 — Single tool-calling forecasting agent.** Install TimeCopilot; reproduce the MarkTechPost tutorial. Let the LLM select/ensemble among your baselines and explain its choices. *Threshold:* adopt only if it improves accuracy OR materially cuts analyst time without accuracy loss.

**Stage 2 — Add context / retrieval.** If your series is event-driven, add textual context using the CiK "Direct Prompt" pattern or TimeCAP-style summary+prediction agents. Measure with RCRPS or region-of-interest error on context-sensitive windows.

**Stage 3 — Multi-agent + evaluation loops.** Move to a planner–forecaster–critic architecture (TimeSeriesScientist as reference implementation) with a reflection/backtesting gate that rejects forecasts failing validation. Add human-in-the-loop approval for high-stakes actions.

**Stage 4 — Anomaly detection & RCA.** For monitoring, start with Orion/SigLLM (forecast-and-compare) or a TSFM residual detector, then add LLMAD-style explanations. For incident RCA, wire an agent to Grafana/Prometheus/Datadog MCP servers, but keep humans gating remediation and expect OpenRCA-level accuracy ceilings (~11% exact-match today).

**Reading order:** (1) Tan et al. NeurIPS 2024 (the skeptic's baseline); (2) LLMTime (intuition for why serialized numbers work); (3) CiK (why context matters); (4) a survey (arXiv:2608.26226 or "From Prompts to Agents"); (5) TimeCopilot + TimeSeriesScientist papers/repos (buildable references); (6) OpenRCA + SigLLM for the anomaly-detection/RCA side.

## Caveats
- **Field is fast-moving and pre-print heavy.** Many cited works (Nexus, OpenRCA 2.0, and several 2026 arXiv IDs) are recent preprints not yet peer-reviewed; treat quantitative claims as provisional.
- **Self-reported benchmarks.** TimeCopilot's "#1 on GIFT-Eval" and vendor MTTR/MTTI reduction figures are self- or vendor-reported; independent replication is limited.
- **Negative results are real.** LLMs frequently underperform simple baselines on pure numeric forecasting (Tan et al.) and fail most hard RCA cases (OpenRCA); the value is in orchestration, context integration, and explanation — not raw numeric prediction. SigLLM likewise trailed SOTA deep-learning detectors by ~30% F1.
- **Marketing vs documented:** Datadog Toto (research, not in production per its launch blog), Dynatrace Davis (causal AI, product-documented but marketing-framed), and most APM vendors' "agentic AIOps" claims should be verified against technical docs before you rely on them.
- **Some arXiv identifiers surfaced in search carry unusually high numbers** (e.g., 2602.x, 2605.x, 2606.x, 2608.x), consistent with 2026 submissions given the current date (August 2026); where a repo or venue could not be independently confirmed (notably TimeXL and the official Nexus code), that is explicitly flagged above.