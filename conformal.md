# **The Conformalization of Time Series Forecasting: Theoretical Frameworks, Algorithmic Paradigms, and Implementation Strategies**

## **The Epistemology of Uncertainty in Machine Learning**

The pervasive deployment of machine learning algorithms in high-stakes forecasting domains—ranging from algorithmic trading and medical prognostics to supply chain optimization and power grid management—has precipitated an urgent need for rigorous uncertainty quantification. Historically, machine learning models have been formulated to generate precise point predictions, optimizing expected loss metrics such as Mean Squared Error (MSE) or Mean Absolute Error (MAE).1 However, point predictions inherently fail to communicate the confidence or potential error margin of the forecast. While these metrics give an impression of the accuracy of the model on average across a distribution, they do not provide a reliable representation of the error that can be expected for a single, individual new prediction.2  
The scientific community has traditionally approached this uncertainty through two dominant epistemological lenses: the Bayesian framework and the Frequentist framework. Bayesian methodologies offer a principled approach to uncertainty via the formulation of posterior distributions over model parameters and predictions. However, Bayesian inference typically requires stringent, often intractable assumptions regarding the underlying data-generating priors and model architectures, making exact inference computationally prohibitive for modern deep learning models.3 Frequentist approaches, conversely, rely heavily on asymptotic properties, assuming that probability represents the frequency of repeated events over infinite trials.3 In the context of machine learning, frequentist confidence intervals often assume that the parameters are fixed and that the residuals follow a specific parametric distribution—typically a Gaussian distribution.3 When confronted with the complex, heavy-tailed, and asymmetric realities of empirical data, these distributional assumptions frequently collapse, leading to intervals that are either perilously narrow or uninformatively wide.4  
Conformal prediction (CP) has emerged as a revolutionary, distribution-free framework designed to circumvent these limitations.4 Originally formalized by Vladimir Vovk, Glenn Shafer, and colleagues, conformal prediction acts as a lightweight algorithmic wrapper that can be applied to any arbitrary predictive model—often referred to as the underlying "heuristic" or "base estimator".7 Instead of outputting a single scalar estimate, conformal prediction generates a set or interval of possible values that is mathematically guaranteed to contain the true outcome with a user-specified probability, without making any assumptions about the underlying distribution of the data.4  
The theoretical elegance of conformal prediction lies in its absolute mathematical validity under finite samples, completely independent of the underlying predictive algorithm's intrinsic accuracy.10 The framework employs a game-theoretic approach to probability, heavily reliant on backward-looking betting protocols. In this framework, the law of large numbers does not require independence or exact levels of probability; the only requirement is whether the probabilities specified for successive events are rates at which a bettor can place successive bets against predictive errors.7 If the underlying predictive model is highly accurate, the resulting conformal prediction intervals will be tightly constrained, offering highly specific, actionable ranges. Conversely, if the model is poor and produces highly variant errors, the conformal prediction framework will maintain the target coverage guarantee by outputting correspondingly wide prediction intervals. This fundamental property ensures that the user is never misled by an overconfident, underperforming model.2

## **The Mathematics of Nonconformity and Exchangeability**

To understand the operational mechanics of conformal prediction, it is necessary to examine the foundational concept of the nonconformity measure.11 The nonconformity measure evaluates how "unusual" or divergent a new data point is relative to previously observed data.11 In the context of regression and continuous forecasting, the most common nonconformity score is the absolute residual of the prediction. For a given data point ![][image1], the nonconformity score ![][image2] is defined as the absolute difference between the true target ![][image3] and the model's point prediction ![][image4]:  
![][image5]  
Given a calibration set consisting of ![][image6] observations, the standard split-conformal prediction framework computes these nonconformity scores for all points in the calibration sequence.11 To construct a prediction interval for a novel, unseen observation ![][image7], the method calculates the empirical quantile of these calibration scores.8 Specifically, the framework defines a threshold ![][image8] as the ![][image9]\-th smallest value among the calibration scores, incorporating a finite-sample correction factor.11  
The resulting prediction interval ![][image10] is formulated symmetrically around the point prediction:  
![][image11]  
This construction provides the marginal coverage guarantee, which can be expressed mathematically as the probability that the true value ![][image12] falls within the conformal set is at least ![][image13].8  
However, the foundational proof of this validity relies on one critical, uncompromising assumption: exchangeability. Exchangeability posits that the joint probability distribution of a data sequence remains invariant under any permutation of its indices.7 Conceptually, this is often illustrated by the analogy of drawing tiles blindly from a bag; exchangeability dictates that the order in which data points are observed conveys no additional information about their underlying distribution, because the conditional probabilities of ordering the tiles are identical to drawing them successively at random without replacement.7 Standard conformal prediction methods fundamentally require that the historical calibration data and the future test data are perfectly exchangeable.14

## **The Temporal Conundrum: Violations of Exchangeability in Time Series**

The straightforward application of standard conformal prediction to time series forecasting represents a fundamental theoretical violation. Time series data is inherently non-exchangeable.15 The temporal ordering of observations is the defining characteristic of the data; the value at time ![][image14] is frequently highly dependent on the value at time ![][image15] (autocorrelation), features cyclical periodicities (seasonality), and is subject to structural breaks, regime changes, and volatility clustering (heteroscedasticity).16  
Because the data-generating process in a time series evolves sequentially, the residuals generated by a forecasting model on historical calibration data are not drawn from the same distributional state as the residuals that will be encountered in the future.16 A model trained on a period of macroeconomic stability, for instance, will yield narrow nonconformity scores. If standard split-conformal prediction is applied to extrapolate this static interval into a period of high volatility, the algorithm will drastically under-cover the true outcomes, falling far below the nominal ![][image13] target coverage level.  
This failure mode is starkly illustrated by examining structural level shifts in autoregressive data. Consider a ridge regression model trained on 500 time steps of a stationary series that subsequently undergoes a sudden level shift of magnitude ![][image16].12 Following this shift, the model's predictive residuals are no longer centered at zero; they exhibit a severe mean bias (e.g., ![][image17]) and a specific standard deviation (e.g., ![][image18]).12 Standard conformal prediction, assuming exchangeability, will either fail to cover the shifted data entirely, or if forcefully recalibrated using the new raw absolute residuals, it will achieve correct coverage only by drastically inflating its quantile threshold. In this scenario, it produces intervals of average width ![][image19], centered symmetrically around the biased point prediction.12 An oracle that knew the bias could shift the interval center by the bias amount ![][image20] and achieve the exact same coverage with an interval width of approximately ![][image21]—a massive 60% reduction in interval inefficiency.12  
To address the breakdown of exchangeability and the inefficiencies of static intervals, the theoretical machine learning community has developed specialized algorithmic paradigms. These advanced methodologies abandon the assumption of static data ordering and instead frame uncertainty quantification as a sequential, adaptive control process that continuously updates its understanding of the model's error distribution as time progresses.11

## **Algorithmic Paradigms for Time Series Conformal Prediction**

The mitigation of non-exchangeability in time series forecasting has catalyzed the development of several distinct classes of conformal prediction algorithms. These methods utilize strategies ranging from ensemble block-resampling and adaptive online learning to explicit conditional residual modeling using advanced deep learning architectures.11

### **Ensemble Batch Prediction Intervals (EnbPI)**

The Ensemble Batch Prediction Intervals (EnbPI) algorithm represents a significant theoretical leap in adapting conformal prediction to dynamic time series without requiring computationally prohibitive data splitting or model refitting.14 Developed to scale sequentially producing prediction intervals, EnbPI leverages ensemble predictors and specialized bootstrapping techniques heavily inspired by the Jackknife+aB procedures.14  
Rather than relying on a static, isolated calibration set, EnbPI utilizes a resampling scheme known as block bootstrapping.16 Because traditional random bootstrapping destroys the autocorrelation of time series data, the BlockBootstrap mechanism samples data in contiguous temporal blocks.16 The algorithm trains an ensemble of base models (such as multiple Random Forests or XGBoost estimators) on these block-resampled subsets of the historical data.16 During the prediction phase, EnbPI generates out-of-sample point predictions by aggregating the outputs of the bootstrap ensemble, typically calculating the mean or median of the predictions to form a robust point estimate.20  
The critical innovation of the EnbPI framework lies in its handling of sequential updates during inference. As new test observations become available at time ![][image22], the algorithm strictly avoids the computationally expensive process of retraining the underlying base estimators on the new data.19 Instead, it updates the empirical distribution of past residuals using a sliding window of recent observations of size ![][image23].19 By dynamically shifting the window of nonconformity scores, EnbPI calculates the prediction intervals based on the most recent error dynamics. This allows the interval width to adjust rapidly to recent deteriorations in model performance or sudden increases in systemic noise, drastically tightening the intervals compared to static conformal methods.16

### **Adaptive Conformal Inference (ACI)**

While EnbPI updates the empirical distribution of residuals through a sliding window, Adaptive Conformal Inference (ACI) takes a rigorous control-theoretic approach to interval calibration to combat non-exchangeability.12 Proposed by Isaac Gibbs and Emmanuel Candès, ACI directly addresses the challenge of achieving valid long-run marginal coverage under arbitrary and potentially adversarial distribution shifts.23  
ACI acknowledges that the empirical coverage at any specific finite time step might fail due to the violation of exchangeability. To compensate, it maintains an online, dynamically adjusted miscoverage rate, denoted as ![][image24], which is updated sequentially at every time step based on the empirical success or failure of the previous interval.12 This is governed by a stochastic approximation update rule:  
![][image25]  
Here, ![][image26] is the target error rate (e.g., 0.10 for 90% coverage), ![][image27] is an indicator function that evaluates to 1 if the true value ![][image28] falls outside the predicted interval ![][image29] and 0 if it is successfully covered, and ![][image30] acts as the learning rate or step size governing the responsiveness of the adaptation.12  
The mechanism operates analogously to a proportional control loop. If the model is under-covering (the true value frequently falls outside the interval), the indicator function evaluates to 1, causing the internal ![][image31] parameter to decrease. A smaller ![][image26] translates to a demand for higher statistical confidence, forcing the algorithm to select a higher quantile from the residual distribution, thereby widening the subsequent interval.12 Conversely, if the interval is excessively conservative and over-covers the data, the indicator evaluates to 0, which increases ![][image31] and progressively shrinks the interval, driving the long-run coverage rate precisely toward the target ![][image26].12 This guarantees that the long-run average miscoverage converges to the target level almost surely, provided the sequence of ![][image24] remains bounded within ![][image32].12  
Recent advancements have expanded upon this foundation to address specific failure modes of standard ACI.

* **AgACI (Aggregated ACI):** Addresses the sensitivity of ACI to the choice of the step size ![][image30] by aggregating multiple ACI instances running with different learning rates, creating a more robust adaptive ensemble.12  
* **DtACI (Dynamically-tuned ACI):** Introduces an additional optimization step in which the step-size parameter of ACI's gradient descent update is tuned dynamically over time, allowing the procedure to be adaptive to both the size and type of the distribution shift without requiring prior knowledge of the shift's velocity.23  
* **BC-ACI (Bias-Corrected ACI):** Solves the inherent architectural constraint of standard ACI, which always constructs intervals centered exactly at the point prediction ![][image33].12 BC-ACI utilizes an Exponentially Weighted Moving Average (EWM) to track the directional bias of recent residuals. It explicitly re-centers the prediction interval by shifting it by the magnitude of this tracked bias, simultaneously adjusting the quantile threshold based on the Mean Absolute Deviation (MAD) of the residuals.12 This dual mechanism maintains rigorous coverage while drastically reducing interval width during structural level shifts.12

### **Conformalized Quantile Regression (CQR)**

Traditional conformal prediction applies a uniform, homoscedastic adjustment to predictions—adding and subtracting a fixed scalar ![][image8] from the point estimate to form the upper and lower bounds.10 This assumes that the predictive uncertainty is uniform across the entire input space and throughout time. In real-world time series, however, uncertainty is distinctly heteroscedastic; long periods of low volatility are frequently interrupted by sudden bursts of high volatility.10  
Conformalized Quantile Regression (CQR) elegantly addresses this inefficiency by merging the distribution-free guarantees of conformal prediction with the localized, heteroscedastic modeling capabilities of classical quantile regression algorithms.9 Rather than training a base heuristic model to predict the conditional mean of the target variable, CQR explicitly requires a base estimator capable of predicting specific conditional quantiles, such as the 5th and 95th percentiles (to construct a preliminary 90% interval).25  
To train these underlying quantile models, the optimization process abandons standard metrics like Mean Squared Error and instead minimizes the pinball loss function, defined as:  
![][image34]  
where ![][image35] represents the residual distance between the true value and the predicted quantile.17 By minimizing the pinball loss across different values of ![][image26], the model learns to map the asymmetric boundaries of the conditional distribution, outputting a preliminary interval ![][image36].9  
Because standard quantile regression models are asymptotically valid but typically uncalibrated in finite samples (often under-covering the true distribution), CQR applies a rigorous conformal correction over these heuristic bounds.9 CQR calculates a nonconformity score based on the signed distance between the true value and the predicted quantiles in a split-conformal calibration set.25 A scalar threshold is then derived from these calibration scores and applied additively to the predicted bounds, systematically expanding or contracting the preliminary interval to achieve exact mathematical coverage.9 This methodology allows the final prediction intervals to dynamically widen during high-volatility periods and narrow during stable regimes, vastly improving the efficiency and tightness of the intervals compared to standard homoscedastic approaches.10

### **Sequential Predictive Conformal Inference (SPCI)**

Moving beyond mere adaptive tracking and homoscedastic corrections, Sequential Predictive Conformal Inference (SPCI) approaches the non-exchangeable time series problem by treating the nonconformity scores themselves as a predictable stochastic process.17 Because time series residuals often exhibit strong temporal autocorrelation (a large predictive error today is highly predictive of a large error tomorrow), SPCI actively exploits this serial dependence to tighten prediction bounds.17  
Instead of constructing intervals using the historical empirical distribution of past residuals, SPCI frames conformal prediction as a secondary forecasting task. Given an arbitrary, user-specified point prediction algorithm that generates a sequence of historical residuals, SPCI trains an explicit conditional quantile estimator to forecast the specific quantile of the *future* nonconformity score.17 In its classical formulation, SPCI fits a Quantile Random Forest (QRF) on the historical sequence of these residuals, optimizing the pinball loss to map the trajectory of model errors.17  
During the online prediction phase, the QRF ingests recent residuals and outputs the precise quantile required to maintain coverage for the upcoming time step. This methodology directly tackles the conditional coverage gap—the phenomenon where a model achieves 90% coverage on average over a year, but systematically fails to cover 90% of the data during a specific, highly volatile month. By adaptively re-estimating the conditional quantile of non-conformity scores based on their temporal dependence, SPCI ensures that the interval is dynamically calibrated to the specific volatility state of the time series at any given micro-moment, yielding asymptotically valid conditional coverage.17

### **SPCI-T: Transformer Conformal Prediction for Time Series**

As the dimensionality and complexity of time series datasets have grown, the forecasting community has increasingly turned to deep learning foundation models. Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks have largely been superseded by Transformer architectures (e.g., PatchTST, Informer, Autoformer), which excel at capturing long-range dependencies and complex multivariate interactions through multi-head self-attention mechanisms.1  
However, standard Transformer models suffer from the same calibration deficiencies as classical statistical algorithms; they produce point estimates devoid of probabilistic context, or they output softmax distributions that are notoriously overconfident and poorly calibrated.1 To rectify this, researchers have successfully integrated Transformer architectures directly into the core of the conformal prediction framework.  
The most sophisticated synthesis of deep learning and temporal conformal prediction is observed in the SPCI-T (Sequential Predictive Conformal Inference with Transformers) architecture, developed by Junghwan Lee, Chen Xu, and Yao Xie.15 Building upon the foundational logic of SPCI—which models future residual quantiles based on past residuals—SPCI-T completely replaces the classical Quantile Random Forest with a highly parameterized Transformer decoder.1  
The hypothesis underpinning SPCI-T is that the evolution of forecasting errors in complex systems (such as high-frequency financial markets or grid-level energy demand) contains latent, non-linear dependencies that traditional machine learning cannot isolate.1 The Transformer decoder acts as the conditional quantile estimator. It ingests a sequence of past prediction residuals alongside exogenous temporal features. The masked self-attention mechanisms within the decoder allow the model to dynamically weight the importance of specific historical errors, learning complex temporal dependencies and identifying latent volatility regimes across prolonged historical sequences.1  
By acting as a deeply contextualized conditional quantile estimator, the Transformer decoder provides highly reactive bounds for the prediction intervals. Extensive empirical evaluations on simulated and real-world datasets confirm that SPCI-T achieves superior interval efficiency. It successfully maintains the strict finite-sample coverage guarantees of the conformal prediction framework while outputting narrower, more precise intervals than state-of-the-art baselines like EnbPI, standard ACI, or standard SPCI.1

## **Empirical Evaluation and Benchmarking Metrics**

The transition from theoretical formulation to practical deployment necessitates rigorous evaluation frameworks. To benchmark the performance of competing conformal prediction algorithms on time series data, researchers rely on a specific triad of evaluation metrics that quantify both the validity and the efficiency of the generated intervals.11

| Evaluation Metric | Mathematical Focus | Practical Interpretation |
| :---- | :---- | :---- |
| **Marginal Coverage Rate** | Calculates the empirical percentage of true target values ![][image28] that fall within the predicted intervals ![][image29] over the entire test set. | Measures absolute validity. If the target risk level ![][image26] is 0.10, a valid algorithm must achieve an empirical marginal coverage of approximately 90%. |
| **Mean Interval Width** | Computes the average distance between the upper and lower bounds of the prediction intervals across all test samples. | Measures efficiency. Among algorithms that successfully achieve the target coverage rate, the algorithm with the smallest mean interval width is superior, as it provides more precise, actionable forecasts.35 |
| **Winkler Interval Score** | A composite proper scoring rule that jointly penalizes the width of the interval and heavily penalizes instances where the true value falls outside the interval. | The ultimate determinant of interval quality. It distinctly separates algorithms that "cheat" by creating needlessly wide intervals from those that provide tight, highly calibrated bounds that still successfully capture the true values.11 |

## **Open-Source Ecosystems and Repositories**

The rapid theoretical advancement of conformal prediction has been mirrored by the aggressive development of robust, production-ready open-source libraries. For practitioners utilizing Python, two dominant ecosystems facilitate the seamless integration of conformal methodologies into existing machine learning and deep learning pipelines: MAPIE and the Nixtlaverse. Furthermore, a rich ecosystem of academic GitHub repositories provides reference implementations for bleeding-edge algorithms.

### **Academic and Reference Repositories**

To aid in the theoretical understanding and practical replication of leading methodologies, the academic community maintains several highly curated repositories containing the foundational code for modern conformal algorithms.

| Repository | Lead Authors / Origin | Primary Focus & Notable Features | Source |
| :---- | :---- | :---- | :---- |
| aangelopoulos/conformal-prediction | Anastasios Angelopoulos | The definitive starting point for general conformal frameworks. Contains extensive Jupyter notebooks demonstrating CP on Imagenet, medical expenditure data, and weather time series without requiring model retraining. | 36 |
| aangelopoulos/conformal-time-series | Anastasios Angelopoulos | Focuses on non-exchangeable sequences, taking a control systems outlook. Implements Conformal PID Control, adaptive conformal tracking, and online quantile regression. | 37 |
| Jayaos/TCPTS | Junghwan Lee, Chen Xu, Yao Xie | The official implementation of SPCI-T and SPCI. Provides the PyTorch code for conditioning Transformer decoders on prediction residuals to forecast conditional quantiles. | 15 |
| yromano/cqr | Yaniv Romano, Evan Patterson, E. Candès | The original Python implementation of Conformalized Quantile Regression (CQR) and equalized coverage algorithms. Wraps deep neural networks and random forests for heteroscedastic intervals. | 9 |
| Rose-STL-Lab/CPTC | NeurIPS 2025 Submission | Focuses on time series with structural change points. Implements standard CP, ACI, AgACI, DtACI, Multi-Valid Prediction (MVP), and novel state-switching dynamical systems. | 38 |
| andreacini/corel | Andrea Cini, et al. (ICML 2025\) | Official repository for Relational Conformal Prediction. Focuses on correlated, multi-dimensional time series using relational and graph-based machine learning approaches. | 39 |

### **MAPIE: Model Agnostic Prediction Interval Estimator**

MAPIE operates as a highly mature extension of the Scikit-learn ecosystem, originally part of the scikit-learn-contrib project.40 Its primary architectural advantage is its strict adherence to standard Scikit-learn API conventions (fit, predict, partial\_fit), making it trivial to inject uncertainty quantification into pipelines utilizing standard models like Random Forests, Support Vector Machines, or Gradient Boosting frameworks such as XGBoost and LightGBM.40  
For time series applications, MAPIE exposes two critical objects: the MapieTimeSeriesRegressor and the MapieQuantileRegressor.25

* **MapieTimeSeriesRegressor:** This class is the core engine for sequential data, offering native implementations of both EnbPI (utilizing the BlockBootstrap class) and ACI (Adaptive Conformal Inference).16 It explicitly supports the partial\_fit method, allowing the conformal thresholds to ingest new observations continuously during an online deployment. Furthermore, it supports the .adapt\_conformal\_inference method to dynamically update the inference quantile via a learning rate parameter ![][image30], thereby preventing interval decay during rapid distribution shifts.21  
* **MapieQuantileRegressor:** This class supports Conformalized Quantile Regression (CQR), operating as a split-conformal wrapper around underlying quantile regressors (e.g., LightGBM configured with a quantile objective).25

### **The Nixtlaverse: StatsForecast, MLForecast, and NeuralForecast**

Nixtla provides an end-to-end Python framework specialized strictly for large-scale time series forecasting, partitioned into three primary libraries based on the underlying algorithmic philosophy.28 Notably, the high-performance capabilities of the Nixtlaverse have also inspired ports to memory-safe languages, such as the neuro-divergent crate in Rust, which provides 100% API compatibility with Python's NeuralForecast while delivering massive reductions in memory usage.45

| Nixtla Library | Base Architectures | Conformal Implementation Strategy | Source |
| :---- | :---- | :---- | :---- |
| **StatsForecast** | Classical statistical methods: ARIMA, Exponential Smoothing, ADIDA, Theta family. | Injected effortlessly via the ConformalIntervals class. Enables distribution-free intervals for models that historically relied heavily on strict Gaussian assumptions, utilizing cross-validation windows. | 4 |
| **MLForecast** | Machine learning models: XGBoost, LightGBM, linear models, KNN. | Utilizes the PredictionIntervals class. Manages complex feature engineering (lags, rolling means, differences) and calculates probabilistic prediction intervals using cached out-of-sample cross-validation conformal windows. | 43 |
| **NeuralForecast** | Deep learning models: NBEATS, NHITS, and Transformer architectures (TFT, Informer, PatchTST). | Integrates conformal prediction natively by allowing models to be optimized with point loss functions or specific distribution losses. Utilizes post-training cross-validation refitting to calibrate precise prediction intervals without requiring complete neural network retraining. | 28 |

## **Implementation Strategy I: Building Conformalized Intervals for XGBoost**

Gradient boosting frameworks such as XGBoost are ubiquitous in tabular forecasting due to their computational speed and predictive precision.52 However, XGBoost natively outputs only point predictions, leaving practitioners blind to predictive uncertainty.54 Utilizing MAPIE and MLForecast allows practitioners to map these point predictions into rigorously calibrated intervals using the EnbPI, CQR, and cross-validation paradigms.

### **XGBoost via MAPIE (The EnbPI Paradigm)**

The EnbPI approach within MAPIE relies on sequential data passing. Because standard K-Fold cross-validation would induce severe data leakage (training on the future to predict the past, which breaks the theoretical guarantees of Jackknife+ and CV+ methods), the BlockBootstrap resampling module must be explicitly utilized to maintain temporal block structures during the calibration phase.16

Python  
import numpy as np  
import pandas as pd  
from xgboost import XGBRegressor  
from mapie.regression import MapieTimeSeriesRegressor  
from mapie.subsample import BlockBootstrap  
from sklearn.model\_selection import TimeSeriesSplit, RandomizedSearchCV

\# 1\. Instantiate and Optimize the Base XGBoost Estimator  
\# Hyperparameters are typically optimized using TimeSeriesSplit to avoid leakage  
xgb\_model \= XGBRegressor(n\_estimators=100, max\_depth=5, learning\_rate=0.1, random\_state=42)

\# 2\. Configure the Block Bootstrap Resampling  
\# This ensures temporal dependencies and autocorrelation are not destroyed   
\# during the conformal calibration sampling phase  
cv\_strategy \= BlockBootstrap(  
    n\_resamplings=20,   
    n\_blocks=10,   
    overlapping=False,   
    random\_state=42  
)

\# 3\. Initialize the MAPIE Time Series Regressor with EnbPI  
\# Setting method="enbpi" activates the Ensemble Batch Prediction Intervals algorithm  
mapie\_ts \= MapieTimeSeriesRegressor(  
    estimator=xgb\_model,  
    method="enbpi",  
    cv=cv\_strategy,  
    agg\_function="mean", \# Aggregates the base ensemble point predictions  
    n\_jobs=-1  
)

\# 4\. Fit the framework on the historical training data  
\# X\_train and y\_train must maintain their chronological sequential ordering  
mapie\_ts.fit(X\_train, y\_train)

\# 5\. Online Prediction and Dynamic Updating Loop  
\# In time series, base models deteriorate. We must update the residuals sequentially.  
alpha \= 0.10 \# Target risk level representing a 90% confidence interval  
gap \= 1 \# Step size for the online sequential forecasting loop

y\_pred \= np.zeros(len(X\_test))  
y\_pis \= np.zeros((len(X\_test), 2, 1))

\# Generate the initial prediction bounds for the first step  
y\_pred\[:gap\], y\_pis\[:gap, :, :\] \= mapie\_ts.predict(  
    X\_test.iloc\[:gap, :\],   
    alpha=alpha,   
    ensemble=True,   
    optimize\_beta=True  
)

\# Execute the sequential forecasting loop  
for step in range(gap, len(X\_test), gap):  
    \# CRITICAL UPDATE STEP: Feed the newly observed actual targets back into   
    \# the MAPIE model to update the empirical distribution of residuals dynamically  
    mapie\_ts.partial\_fit(  
        X\_test.iloc\[(step \- gap):step, :\],  
        y\_test.iloc\[(step \- gap):step\]  
    )  
      
    \# Predict the next time step utilizing the newly updated conformal thresholds  
    y\_pred\[step:step+gap\], y\_pis\[step:step+gap, :, :\] \= mapie\_ts.predict(  
        X\_test.iloc\[step:step+gap, :\],  
        alpha=alpha,  
        ensemble=True,  
        optimize\_beta=True  
    )

\# The multidimensional array y\_pis now contains the rigorously calibrated lower and upper bounds

In this implementation, the partial\_fit method is the operational cornerstone of the EnbPI paradigm.16 If the pipeline bypasses partial\_fit, it executes a static conformal prediction based exclusively on the training set residuals, leading to rapidly degrading coverage. By injecting newly observed data at each step, EnbPI calculates the most recent nonconformity scores, effectively sliding the calibration window forward. This dynamic adaptation causes the prediction intervals to expand organically in response to suddenly volatile market conditions or structural shifts, minimizing interval width while maintaining perfect mathematical coverage.16

### **Heteroscedastic Conformalization for Tree Models via MAPIE (The CQR Paradigm)**

While EnbPI is potent, it generally applies homoscedastic corrections unless modified heavily. For datasets exhibiting intense heteroscedasticity, Conformalized Quantile Regression (CQR) is mathematically superior.25  
CQR requires a base model capable of minimizing the pinball loss to output raw quantiles. While XGBoost can be adapted, LightGBM exposes a highly native objective="quantile" parameter, making it the preferred estimator for integration with the MapieQuantileRegressor.25

Python  
from lightgbm import LGBMRegressor  
from mapie.regression import MapieQuantileRegressor  
from sklearn.model\_selection import train\_test\_split

\# 1\. Execute Split-Conformal Data Partitioning  
\# CQR relies heavily on a dedicated calibration subset strictly isolated from the training data  
X\_train\_temp, X\_test, y\_train\_temp, y\_test \= train\_test\_split(X, y, shuffle=False)  
X\_train, X\_calib, y\_train, y\_calib \= train\_test\_split(X\_train\_temp, y\_train\_temp, shuffle=False)

\# 2\. Instantiate the Base Quantile Estimator  
\# The model MUST optimize for quantile loss to produce the preliminary heteroscedastic bounds  
quantile\_estimator \= LGBMRegressor(  
    objective='quantile',   
    alpha=0.5,   
    random\_state=42,  
    verbose=-1  
)

\# 3\. Configure the MAPIE Quantile Regressor  
\# alpha here defines the overall target miscoverage rate (1 \- 0.20 \= 80% target coverage)  
cqr\_params \= {  
    "method": "quantile",  
    "cv": "split", \# Specifies the split-conformal methodology  
    "alpha": 0.20    
}

mapie\_cqr \= MapieQuantileRegressor(quantile\_estimator, \*\*cqr\_params)

\# 4\. Fit the Model and Calibrate the Intervals  
\# The framework trains the base LightGBM model on X\_train and calibrates   
\# the conformal corrections using the independent X\_calib set  
mapie\_cqr.fit(  
    X\_train, y\_train,  
    X\_calib=X\_calib, y\_calib=y\_calib  
)

\# 5\. Generate Predictions and Conformal Bounds  
y\_pred, y\_pis \= mapie\_cqr.predict(X\_test)

\# Evaluation metrics such as mapie.metrics.regression\_coverage\_score   
\# and regression\_mean\_width\_score can now be applied to y\_pis to verify finite-sample validity.

### **Scaling XGBoost with MLForecast**

For scaling tree-based models across multiple concurrent time series efficiently, the Nixtlaverse provides a streamlined interface via MLForecast.43 The MLForecast library abstracts away the manual loops of sequential residual updating and handles the conformal intervals natively via the PredictionIntervals object, leveraging out-of-sample cross-validation windows.49

Python  
import pandas as pd  
from mlforecast import MLForecast  
from mlforecast.target\_transforms import Differences  
from mlforecast.utils import PredictionIntervals  
from xgboost import XGBRegressor

\# 1\. Define the base forecasting models  
models \=

\# 2\. Initialize the MLForecast object  
\# This module automatically handles the complex feature engineering required for time series  
mlf \= MLForecast(  
    models=models,  
    freq='D', \# Specifies a Daily frequency  
    target\_transforms=)\], \# Applies first-order differencing to stationarize the series  
    lags= \# Automatically generates autoregressive lag features  
)

\# 3\. Fit the Pipeline with Conformal Prediction Windows  
\# n\_windows specifies the number of cross-validation slices used to   
\# collect the out-of-sample residuals for the conformal calibration  
mlf.fit(  
    train\_df,  
    id\_col='unique\_id',  
    time\_col='ds',  
    target\_col='y',  
    prediction\_intervals=PredictionIntervals(n\_windows=5, h=30)  
)

\# 4\. Forecast into the future with specified confidence levels  
\# level= requests the specific bounds that provide 90% and 95% coverage probabilities  
forecasts \= mlf.predict(30, level=)

By specifying n\_windows, the MLForecast library splits the historical data into sequential temporal folds. The XGBoost model generates out-of-sample predictions for each fold, creating an empirical distribution of nonconformity scores that perfectly mimics the testing environment's structural forecasting characteristics.44 The prediction step then applies these rigorously calibrated thresholds to generate the final bounds alongside the point forecasts.

## **Implementation Strategy II: Deep Learning and Transformer Architectures**

When dealing with massive multivariate datasets requiring deep architectural representations—such as high-frequency sensor telemetry or global macroeconomic indicators—NeuralForecast provides native support for modern deep learning architectures. This includes models like NBEATS, NHITS, and advanced Transformer variations such as PatchTST, Temporal Fusion Transformers (TFT), and Informer.28  
NeuralForecast achieves conformalization through cross-validation on a model trained using either a standard point loss function or a dedicated probabilistic loss function (e.g., DistributionLoss).44 The conformal process requires zero additional retraining of the base parameters, making the extraction of uncertainty quantification computationally negligible once the base network is fully optimized.44

Python  
from neuralforecast import NeuralForecast  
from neuralforecast.models import PatchTST, NHITS, TFT  
from neuralforecast.utils import PredictionIntervals  
from neuralforecast.losses.pytorch import MAE, DistributionLoss

horizon \= 24  
input\_window \= 96 \# The historical look-back window fed into the neural network

\# 1\. Define the Deep Learning Models  
\# PatchTST is a leading Transformer-based forecasting architecture utilizing channel independence  
\# NHITS is an advanced MLP-based model specialized in hierarchical interpolation  
models \=),   
        scaler\_type="robust"  
    )  
\]

\# 2\. Configure the Conformal Prediction Object  
\# The framework will construct the distribution-free bounds post-training   
\# using cross-validation over the specified number of windows  
prediction\_intervals \= PredictionIntervals(n\_windows=3, h=horizon)

\# 3\. Initialize and Fit the NeuralForecast Pipeline  
nf \= NeuralForecast(models=models, freq='H') \# Specifies an Hourly frequency

\# To construct the precise prediction intervals, the fit process will   
\# perform cross-validation internally based on the n\_windows parameter,  
\# extracting the necessary non-conformity scores from the validation splits.  
nf.fit(train\_panel\_df, prediction\_intervals=prediction\_intervals)

\# 4\. Generate Future Forecasts with Mathematical Confidence Levels  
forecast\_df \= nf.predict(level=)

In this sophisticated deep learning architecture, the PatchTST model utilizes its multi-head attention mechanism to map input tokens into highly accurate point forecasts.29 During the internal cross-validation phase managed by the PredictionIntervals object, the framework systematically compiles the prediction residuals of the Transformer across the validation splits and calibrates the distribution-free conformal boundaries. Because deep learning models often require substantial computational overhead—frequently relying on GPU acceleration for optimization—obtaining robust prediction bounds via conformal prediction without retraining the entire neural network multiple times from random initialization presents an immense scalability advantage.44

## **Synthesis and Strategic Outlook**

The integration of conformal prediction into time series forecasting represents a fundamental paradigm shift in the field of quantitative modeling and operations research. By explicitly divorcing the validity of prediction intervals from the correctness of the underlying model and shedding the fragile assumption of Gaussian error distributions, conformal methods provide an impenetrable mathematical backstop against systemic uncertainty.4 They transform arbitrary, uncalibrated heuristics into rigorous, reliable probabilistic forecasting engines.  
However, as demonstrated by the theoretical divergence between standard marginal conformalization and the specialized methods tailored for sequential data, the violation of exchangeability inherent in time series remains a formidable challenge.16 Frameworks such as the Ensemble Batch Prediction Intervals (EnbPI) algorithm and Adaptive Conformal Inference (ACI) have successfully resolved this by shifting from static calibration to dynamic, reactive control paradigms. They operate on the fundamental premise that uncertainty in time series is not a static constant to be measured once in a historical vacuum, but a continuously evolving target that must be tracked, modeled, and corrected sequentially as new data arrives.12  
The cutting edge of the discipline now sits at the highly complex intersection of conditional residual modeling and advanced deep learning.27 Architectures such as SPCI-T (Transformer Conformal Prediction) signal a transition toward treating the *volatility and error processes themselves* with the same architectural rigor as the primary signal generation.1 By deploying localized Transformer decoders specifically designed to attend over past forecasting errors, the ecosystem is evolving beyond simple control-theoretic adjustments. It is moving toward fully autonomous, context-aware uncertainty quantification engines capable of anticipating heteroscedastic outbursts and structural shifts before they manifest catastrophically.1  
Practically, the proliferation of specialized open-source frameworks such as MAPIE and the Nixtlaverse has democratized access to these highly advanced mathematical constructs. By encapsulating complex block-bootstrapping, cross-validation calibration windows, and pinball loss optimization into user-friendly abstractions that perfectly mimic familiar API conventions, these libraries allow enterprise practitioners to conformalize immense pipelines spanning millions of independent time series—from classical ARIMA architectures up through massive modern Transformers.4 As the adoption of highly parameterized, black-box deep learning models accelerates in critical sectors such as finance, healthcare, and infrastructure management, the rigorous, distribution-free guarantees provided by time-series conformal prediction will complete their transition from being an optional analytical enhancement to an absolute systemic necessity.

#### **Works cited**

1. Transformer Conformal Prediction for Time Series \- OpenReview, accessed May 21, 2026, [https://openreview.net/pdf?id=3dDDKaSrye](https://openreview.net/pdf?id=3dDDKaSrye)  
2. Prediction intervals for any machine learning model \- The Valence Kjell, accessed May 21, 2026, [https://www.valencekjell.com/posts/2022-09-14-prediction-intervals/](https://www.valencekjell.com/posts/2022-09-14-prediction-intervals/)  
3. Conformal prediction tutorial \- David Stutz, accessed May 21, 2026, [https://davidstutz.de/wordpress/wp-content/uploads/2023/11/ucl2023-slides.pdf](https://davidstutz.de/wordpress/wp-content/uploads/2023/11/ucl2023-slides.pdf)  
4. Conformal Prediction \- Nixtla \- Nixtlaverse, accessed May 21, 2026, [https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/conformalprediction.html](https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/conformalprediction.html)  
5. Evaluation of conformal-based probabilistic forecasting methods for short-term wind speed forecasting \- Proceedings of Machine Learning Research, accessed May 21, 2026, [https://proceedings.mlr.press/v204/althoff23a.html](https://proceedings.mlr.press/v204/althoff23a.html)  
6. \[2107.07511\] A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification \- arXiv, accessed May 21, 2026, [https://arxiv.org/abs/2107.07511](https://arxiv.org/abs/2107.07511)  
7. A Tutorial on Conformal Prediction \- Journal of Machine Learning Research, accessed May 21, 2026, [https://jmlr.csail.mit.edu/papers/volume9/shafer08a/shafer08a.pdf](https://jmlr.csail.mit.edu/papers/volume9/shafer08a/shafer08a.pdf)  
8. Conformal Prediction \- stat.berkeley.edu, accessed May 21, 2026, [https://www.stat.berkeley.edu/\~ryantibs/statlearn-s23/lectures/conformal.pdf](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf)  
9. yromano/cqr: Conformalized Quantile Regression \- GitHub, accessed May 21, 2026, [https://github.com/yromano/cqr](https://github.com/yromano/cqr)  
10. Conformalized Quantile Regression, accessed May 21, 2026, [https://papers.neurips.cc/paper/8613-conformalized-quantile-regression.pdf](https://papers.neurips.cc/paper/8613-conformalized-quantile-regression.pdf)  
11. Conformal Prediction Algorithms for Time Series Forecasting: Methods and Benchmarking, accessed May 21, 2026, [https://arxiv.org/html/2601.18509v2](https://arxiv.org/html/2601.18509v2)  
12. Bias-Corrected Adaptive Conformal Inference for Multi-Horizon Time Series Forecasting, accessed May 21, 2026, [https://arxiv.org/html/2604.13253v1](https://arxiv.org/html/2604.13253v1)  
13. Boosted Conformal Prediction Intervals \- arXiv, accessed May 21, 2026, [https://arxiv.org/html/2406.07449v1](https://arxiv.org/html/2406.07449v1)  
14. Conformal Prediction for Time Series \- PubMed, accessed May 21, 2026, [https://pubmed.ncbi.nlm.nih.gov/37819805/](https://pubmed.ncbi.nlm.nih.gov/37819805/)  
15. Transformer Conformal Prediction for Time Series \- arXiv, accessed May 21, 2026, [https://arxiv.org/html/2406.05332v1](https://arxiv.org/html/2406.05332v1)  
16. EnbPI technique for time series — MAPIE 1.4.1.dev3+g59ff1b178 ..., accessed May 21, 2026, [https://mapie.readthedocs.io/en/latest/examples\_regression/2-advanced-analysis/plot\_timeseries\_enbpi.html](https://mapie.readthedocs.io/en/latest/examples_regression/2-advanced-analysis/plot_timeseries_enbpi.html)  
17. Sequential Predictive Conformal Inference for Time Series \- Proceedings of Machine Learning Research, accessed May 21, 2026, [https://proceedings.mlr.press/v202/xu23r/xu23r.pdf](https://proceedings.mlr.press/v202/xu23r/xu23r.pdf)  
18. \[2010.09107\] Conformal prediction for time series \- arXiv, accessed May 21, 2026, [https://arxiv.org/abs/2010.09107](https://arxiv.org/abs/2010.09107)  
19. Conformal Prediction Interval for Dynamic Time-Series \- Proceedings of Machine Learning Research, accessed May 21, 2026, [https://proceedings.mlr.press/v139/xu21h/xu21h.pdf](https://proceedings.mlr.press/v139/xu21h/xu21h.pdf)  
20. Demystifying EnbPI: Mastering Conformal Prediction Forecasting | by Valeriy Manokhin, PhD, MBA, CQF, accessed May 21, 2026, [https://valeman.medium.com/demystifying-enbpi-mastering-conformal-prediction-forecasting-d49e65532416](https://valeman.medium.com/demystifying-enbpi-mastering-conformal-prediction-forecasting-d49e65532416)  
21. Tutorial for time series — MAPIE 0.8.1 documentation \- Read the Docs, accessed May 21, 2026, [https://mapie.readthedocs.io/en/v0.8.1/examples\_regression/4-tutorials/plot\_ts-tutorial.html](https://mapie.readthedocs.io/en/v0.8.1/examples_regression/4-tutorials/plot_ts-tutorial.html)  
22. Adaptive Conformal Inference for Computing Market Risk Measures: An Analysis with Four Thousand Crypto-Assets \- MDPI, accessed May 21, 2026, [https://www.mdpi.com/1911-8074/17/6/248](https://www.mdpi.com/1911-8074/17/6/248)  
23. herbps10/AdaptiveConformal: R package for Adaptive Conformal Inference \- GitHub, accessed May 21, 2026, [https://github.com/herbps10/AdaptiveConformal](https://github.com/herbps10/AdaptiveConformal)  
24. Conformal Inference for Online Prediction with Arbitrary Distribution Shifts \- Journal of Machine Learning Research, accessed May 21, 2026, [https://www.jmlr.org/papers/volume25/22-1218/22-1218.pdf](https://www.jmlr.org/papers/volume25/22-1218/22-1218.pdf)  
25. Tutorial for conformalized quantile regression (CQR) — MAPIE 0.8.5 ..., accessed May 21, 2026, [https://mapie.readthedocs.io/en/v0.8.5/examples\_regression/4-tutorials/plot\_cqr\_tutorial.html](https://mapie.readthedocs.io/en/v0.8.5/examples_regression/4-tutorials/plot_cqr_tutorial.html)  
26. Seminar on StatiSticS and data Science \- HKUST Math Department, accessed May 21, 2026, [https://www.math.hkust.edu.hk/intranet/file/?c=seminar\_abstract\&f=20231120115614\_7\_Chen%20XU%20and%20Prof.%20Yao%20XIE\_1120.pdf](https://www.math.hkust.edu.hk/intranet/file/?c=seminar_abstract&f=20231120115614_7_Chen+XU+and+Prof.+Yao+XIE_1120.pdf)  
27. MLBoost Seminars (7): Sequential Conformal Prediction for Time Series \- YouTube, accessed May 21, 2026, [https://www.youtube.com/watch?v=QEkR1xTiOWc](https://www.youtube.com/watch?v=QEkR1xTiOWc)  
28. Statistical, Machine Learning and Neural Forecasting methods | StatsForecast \- Nixtla, accessed May 21, 2026, [https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/statisticalneuralmethods.html](https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/statisticalneuralmethods.html)  
29. Time Series Forecasting Using Foundation Models 1 \- DOKUMEN.PUB, accessed May 21, 2026, [https://dokumen.pub/time-series-forecasting-using-foundation-models-1.html](https://dokumen.pub/time-series-forecasting-using-foundation-models-1.html)  
30. Transformer-based conformal predictors for paraphrase detection, accessed May 21, 2026, [https://proceedings.mlr.press/v152/giovannotti21a.html](https://proceedings.mlr.press/v152/giovannotti21a.html)  
31. Jayaos/TCPTS: Transformer Conformal Prediction for Time Series \- GitHub, accessed May 21, 2026, [https://github.com/Jayaos/TCPTS](https://github.com/Jayaos/TCPTS)  
32. Yao Xie \- Publication, accessed May 21, 2026, [https://www2.isye.gatech.edu/\~yxie77/Pub\_timeseries.html](https://www2.isye.gatech.edu/~yxie77/Pub_timeseries.html)  
33. \[2406.05332\] Transformer Conformal Prediction for Time Series \- arXiv, accessed May 21, 2026, [https://arxiv.org/abs/2406.05332](https://arxiv.org/abs/2406.05332)  
34. Regression prediction intervals with MAPIE \- Kaggle, accessed May 21, 2026, [https://www.kaggle.com/code/carlmcbrideellis/regression-prediction-intervals-with-mapie](https://www.kaggle.com/code/carlmcbrideellis/regression-prediction-intervals-with-mapie)  
35. Conformal Prediction in Time Series Forecasting \- Data Science With Marco, accessed May 21, 2026, [https://www.datasciencewithmarco.com/blog/conformal-prediction-in-time-series-forecasting](https://www.datasciencewithmarco.com/blog/conformal-prediction-in-time-series-forecasting)  
36. Lightweight, useful implementation of conformal prediction on real data. \- GitHub, accessed May 21, 2026, [https://github.com/aangelopoulos/conformal-prediction](https://github.com/aangelopoulos/conformal-prediction)  
37. Conformal PID Control for Time-Series Prediction \- GitHub, accessed May 21, 2026, [https://github.com/aangelopoulos/conformal-time-series](https://github.com/aangelopoulos/conformal-time-series)  
38. Rose-STL-Lab/CPTC: Conformal Prediction for Time-series Forecasting with Change Points, accessed May 21, 2026, [https://github.com/Rose-STL-Lab/CPTC](https://github.com/Rose-STL-Lab/CPTC)  
39. andreacini/corel: Official repository for the paper "Relational Conformal Prediction for Correlated Time Series" (ICML 2025\) \- GitHub, accessed May 21, 2026, [https://github.com/andreacini/corel](https://github.com/andreacini/corel)  
40. MAPIE \- Model Agnostic Prediction Interval Estimator — MAPIE 0.1.dev73+g8cfcee401 documentation, accessed May 21, 2026, [https://mapie.readthedocs.io/](https://mapie.readthedocs.io/)  
41. Conformal Prediction forecasting with MAPIE | by Valeriy Manokhin, PhD, MBA, CQF, accessed May 21, 2026, [https://valeman.medium.com/conformal-prediction-forecasting-with-mapie-library-for-conformal-prediction-7aac033ae3ef](https://valeman.medium.com/conformal-prediction-forecasting-with-mapie-library-for-conformal-prediction-7aac033ae3ef)  
42. mapie.regression.MapieTimeSeriesRegressor — MAPIE 0.8.3 documentation, accessed May 21, 2026, [https://mapie.readthedocs.io/en/v0.8.3/generated/mapie.regression.MapieTimeSeriesRegressor.html](https://mapie.readthedocs.io/en/v0.8.3/generated/mapie.regression.MapieTimeSeriesRegressor.html)  
43. Machine Learning Forecast \- Nixtla \- Nixtlaverse, accessed May 21, 2026, [https://nixtlaverse.nixtla.io/mlforecast/index.html](https://nixtlaverse.nixtla.io/mlforecast/index.html)  
44. Uncertainty quantification with Conformal Prediction \- Nixtla, accessed May 21, 2026, [https://nixtlaverse.nixtla.io/neuralforecast/docs/tutorials/conformal\_prediction.html](https://nixtlaverse.nixtla.io/neuralforecast/docs/tutorials/conformal_prediction.html)  
45. neuro-divergent 0.1.0 \- Docs.rs, accessed May 21, 2026, [https://docs.rs/crate/neuro-divergent/latest/source/README.md](https://docs.rs/crate/neuro-divergent/latest/source/README.md)  
46. neuro-divergent \- crates.io: Rust Package Registry, accessed May 21, 2026, [https://crates.io/crates/neuro-divergent](https://crates.io/crates/neuro-divergent)  
47. Conformal Prediction forecasting with Nixtla's statsforecast \- Valeriy Manokhin, PhD, MBA, CQF, accessed May 21, 2026, [https://valeman.medium.com/conformal-prediction-forecasting-with-nixtlas-statsforecast-cc39b9e30b36](https://valeman.medium.com/conformal-prediction-forecasting-with-nixtlas-statsforecast-cc39b9e30b36)  
48. v1.6.0 · Nixtla statsforecast · Discussion \#696 \- GitHub, accessed May 21, 2026, [https://github.com/Nixtla/statsforecast/discussions/696](https://github.com/Nixtla/statsforecast/discussions/696)  
49. Probabilistic forecasting | MLForecast \- Nixtla, accessed May 21, 2026, [https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/prediction\_intervals.html](https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/prediction_intervals.html)  
50. GitHub \- Nixtla/mlforecast: Scalable machine learning for time series forecasting., accessed May 21, 2026, [https://github.com/Nixtla/mlforecast](https://github.com/Nixtla/mlforecast)  
51. neuralforecast/neuralforecast/core.py at main · Nixtla/neuralforecast \- GitHub, accessed May 21, 2026, [https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/core.py](https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/core.py)  
52. Conformal Forecast Prediction Intervals in Modeltime \- GitHub Pages, accessed May 21, 2026, [https://business-science.github.io/modeltime/articles/modeltime-conformal-prediction.html](https://business-science.github.io/modeltime/articles/modeltime-conformal-prediction.html)  
53. Extreme Gradient Boosting Combined with Conformal Predictors for Informative Solubility Estimation \- PMC, accessed May 21, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10779886/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10779886/)  
54. How to obtain prediction intervals from XGBoost regression made with tidymodels?, accessed May 21, 2026, [https://forum.posit.co/t/how-to-obtain-prediction-intervals-from-xgboost-regression-made-with-tidymodels/171766](https://forum.posit.co/t/how-to-obtain-prediction-intervals-from-xgboost-regression-made-with-tidymodels/171766)  
55. Estimating prediction intervals of time series forecast — MAPIE 0.8.0 documentation, accessed May 21, 2026, [https://mapie.readthedocs.io/en/v0.8.0/examples\_regression/1-quickstart/plot\_timeseries\_example.html](https://mapie.readthedocs.io/en/v0.8.0/examples_regression/1-quickstart/plot_timeseries_example.html)  
56. Stock Market Forecasting with Nixtla Conformal Prediction | by Claudio Giorgio Giancaterino, accessed May 21, 2026, [https://medium.com/@c.giancaterino/stock-market-forecasting-with-nixtla-conformal-prediction-49874285634e](https://medium.com/@c.giancaterino/stock-market-forecasting-with-nixtla-conformal-prediction-49874285634e)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAaCAYAAAB7GkaWAAAAZ0lEQVR4XmNgGHigAMT30QVh4C0Q/0cXpAx0AnECuiAI/IDSIPsckSVmAjETlA2SdEWSY6iF0v0MeFwKkriALggCIgwQSTF0CRA4z4AwshyIpZHkwBI7oOxXyBIg4MwAUfAHXWLEAwCaDRQuuqoUtAAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAZCAYAAAA4/K6pAAAAtElEQVR4XmNgGAWjADtIBOJlQGyDLkEM+A/EClB2JRBXIaQIg31AfAKJDzKsE4lPEHxigGiaD8SyaHIwcBKI+dAFYUCTAWIADL9HlQaDVnQBGJgLxB1IfDkGiCHIwA2NjwJAircj8dWhYjBwBUr/QhJDAd+BOB+IhRgg0QjSDGLDQAQQZzGgWoIBeBggCtXQJaAAZCgHuiCxgJkB4aUkZAligS4QrwLiOegSpAAldIFRQCUAAA+DHqGMY9G4AAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAaCAYAAAC+aNwHAAAA2ElEQVR4XmNgGAWjABXMAuJUdEEgMEcXwAZ+ATEzEP8HYkck8RSoGF4wjwGiGQRAir2Q5N5AxfCCWijdzYCpGMRfj8S3AeKNSHwUAFL8AokP85IhkpgxEEsh8VEASHEQEr8MKkYUEGfAVPwRTQwUHn+Q+BgApFgPytaB8jdA+f1QGt0SFACLMhCeBKW1kOQFgfgvEh8FoAfMTgZM204CcSSaGBiATAYpngblM0L5HnAVEAAzcB+KKBBIA/F3KJufAaIQm003gPg+uiAMWALxGiBuQ5cYBXQAAJQ1L2Q7w2ndAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAZCAYAAAA4/K6pAAABAklEQVR4XmNgGAXYABMQ/wfim+gSxABmBohmfiDOBeLHqNKEQQIa3xyIOdDEaAd6gLgGXRAIzNAFsIE/DBB/g/zfiCSuAhXDC+KAWBvKBinuQpJbDxXDC+KhtDcDpmIQ/y2a2Ac0Phx8YcBuQCESnxGIg5H4KACkeBISHxR4IDFQwiIKgBSrI/G3QsVgYBsDxDssSGIoAKQ4D8oWgPLfQfliQKwMxGeB2B8qhgGsGCCaQHgTlM5CUYEZRnDAxQAJIBioZMBU7MmAJ4OBFF9F43cg8UHgBxDLAvE+NHEwgNkGyo3fgbgbSQ4GQNF5Cohl0CVAQBGIFwDxRCBmRZUaBYQAAPNQNCL7EkdnAAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHAAAAAaCAYAAABvj9h3AAADFElEQVR4Xu2ZW8gNURTH/26JIvcQfScplxeKRyQpSblErg9S8oCiKHfmxQMelBflmnt4cIskefBEyitFfIkkpORBclvL7N2sWXvm+2afjn2+Oc2v/s2s/+7MWWf/57LPOUBFRUVFEXppo8UZqo2ys43UXZslor82OuGhNsrOdpQ3wLWkP6RZeqADWi7AHShngBtJ7Wb/M2lhMtQhLRfgTpQzwAuqvq7qPFouwF0oZ4D1UneAa0gXSdP0QJPZDTfAIaTLpPHKZ2raCExP0nnSHD1ATNZGBnUFyA/amtnnZw7ftroKe5AOcDjpnvG4b8nbDC8kPUivzT730VuMcahFevMO8AHpkaj5TQ6I2odzOTpLOkM6TTpFOkE6bl7TGXuRDvCX2a6GOyFcX1FeSF6abQ1xL/IrBNcfRJ2Hd4BfER+cJ3e0GrM8hv/3mUaxD+kAOTjmB+m38O0VOUl460lbRJ3FYNLUghpnXpOHvXM9QfbJtUnU/Ki6IWqLd4ATEB/c6kt6+B/7tRGQCO4zkOFeN4h6q/EkS0l9lKdpI80vqOnmNZ3BfVwVNYfPnvwcU0gjRW3xDtAylvQe7iT4cNBTRYjgBrgIbp/2TtJs+IrmPnhruWO8IngFyAf9luFZ+OzkZ+RJ4YUmghvgUbgTwvVNUb+D+9lCsAzZvck72yfST1FLvAMcKOr7SF/6L0iDkCwcmkEEN8AlSE+SDdQu05+arZ7IEIxA+n1XmXqzqQ+bbV5vXgHORrwQ4IOxsh74vIBZrs2ARHADZO4i6fua2Urmkp4pLxRHkPRmT65uYpwvmryLwivAIuiJCU0EN0BeFEi4xzfK+04apbwQTFT1R7hh8UWxQnmWhgY4E/FttB9pQHooGBHSAR5CHNgwUy8wtTzDGXvi8ffIUMxD/L68+mXs7dT2arG98fpC09AAGT6TL2kzIBHSAR4j3TL7MxBPhlzxWfjMb9fmf2Yl6ZXZH4O4t6yfz54j+cVG0/AAm00E9xbK/7PdJq1TfldgMeITjP/HrIeWC7CvNlqcNm1UVFRUVFRUJPwFY+GyUdjcRncAAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAbCAYAAABIpm7EAAAAiklEQVR4XmNgGAVDGnwH4k9AfBjKfwrEJ4D4PxCfhimCgTQgVgdiGwaIgn9IcspQMRTwB0pPZ4BIMiLJaUHFsAKQxC80sfVQcawAJHEMTewjVBwrAEl4YBGrRBMDg3AGTJNCkcSEgHgKkhzDZQZMDVuQxF4gS4AAKKQmo4mxMkA0gLAAmtwoGGoAAFUtI0LZEJOuAAAAAElFTkSuQmCC>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC0AAAAaCAYAAAAjZdWPAAABVklEQVR4Xu2WPUsDQRCGRwULbSzzAwJiJ2ktAiIphOAPsLWztBDrtP4CGxub9IoIljZilUpQsFMU/ESMqCHxHfeOm3vVEM2Kq+wDD+zM3LGzmxy7IpHI/2YePsGO8dLUn6l2bGq/TkVcU6eUH4VtOEL5YEh3k3NBUxfXZC2JdTyUlcNkQLLdbsKxfDlcXsQ1XeZCyOyLa/qEC6GyBpfl4w/yJ9niRK8swo1knH6QugAfTHEiYRXOyTc3aAYemNh+kD6ocoL48jzj8JyTkp2EBS6AXdiCg/ACNuBK7ok83poehuviXtAxsySudkZ5/RUmxC100+S7Teyl6SN4C6/gHXzIl+U6yWtdx/fyfiftRLPw0MSPpN5tbDyZPfpGT037wE6kiyuZmPGy0/2yAHdMnE66bXKWIJreg0UT63W122H0WdO68BtxV2H9G+ppHAzTnIhEIn+QV9MgXSpUnHDcAAAAAElFTkSuQmCC>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACYAAAAaCAYAAADbhS54AAABNUlEQVR4XmNgGAWjYBSMguEFmIG4EogLoHxeIDZGSA8MaAfiX1C2KhD/AOL/DBDHDhiIZoA4ggNJ7BJUbEAByAHPsYh9QxPbisanKQhhgDgiHU0cJFYLZfcAsT9UjG5gBwOmhSpQMXY0cXR1NAVTGDAtXIJFDASwiZECzgPxciD+A8SMUDHkdI0CuBlQLQyG8kG5Eh1Q4jCQXlgOB8XEUyCWBGJxuAoswJkBohGEfaB0A7ICKEB3GBMQf8eDS6HqJgLxbSgbBkBmvUUTwwvUGCCaONElGDAdRiwA6fPFIiaBJoYXrGfA7QBc4oTAGSAOQuKDahOizQLFdTEDIkozkeR2AvF7IH4DxO+A+DeSHLHgJQPEnDtAzAPEkxkgDiZYs4ASojsQOwGxCxAHoEqPglEwCugCAFurSFkNZTV4AAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJMAAAAaCAYAAACzWm4FAAAD70lEQVR4Xu2aW6gOURTHl/stx6XkEuWSXBOlyIOSQooHJbyRJOJFyQOlEFJKSZRLnnjwoDwIL0oePUgkl1LuSiEHyXX9mz3n22edPWuv+c6cc2ZqfvXvm++/9l6z9zd7Zvbsb4hqampqys5a1gandaxZ7cNt9GeNlGaF6Ev52j+W1VuaFWOuNDwWU3K802O/sn24Of6xZjhNY7W0D7fxWxoV5DNrqDQDDGO9lmZFwfENMYo1nRrHPqtcLixJ/lB1ztLhpPdJi6XEynyQRomZxPokzQCxPpuIJbnAuibNkoGz7CUlfUmVxWHWPWl6PGftkiZzn2z5y8g31ippCgrpUyxJLF4kRezLcrAR7yVNR6wuTqxYmbIxleJtjsVNaEkwOdfiRVPEviyDCbftQ9JkTrFapSmo4mACaPNgaXoU0ictCW4dd6Xp2M464H2fyTrH2ud5edHaYsUymM5Q+IEC9UKDzKcsgwlP1ycoOQHGiVgItPmYND0K6ZOWBLEt0mS2sRaxnlJyP37M2uRimJz+cNt50dpixTKY0PZQGXi4JWiUYTBhXveQ1c99R3sONsJ01NtOucX6KU2PQvqkJUFsqTSpUeeZ2/bPjOPOa4Zm6/lYBtMYCpeBh/UojZ4eTN9ZL4S3nhpt2ska78VScDXW2q3FzGhJEMOjpWS1+0Rc3gZxZdJygims+QGhnvSgCUk1E5bBBEJlQp4k72CSfdEUIz1R5TJNH+fPdp8h9lB2DGgxM1oSxCZK0wPxZQHvivAkCygZkFKoKz1oTlLNRNkGk+yLphha3+Bfoo7HI2U3ZdcFWsyMlgSxJdJ0YCle1k3PjIHCtyLzNYP2g6eMpnAZeDjLNfIOpiLBfrHeFQKxL9L0OE16u7WYGS0JYpul6cCkW9Z95HmYBFqeMnxkvmawDKaFFC4DD7dgjZ4eTGel6UBshzQ9blAy38qikD5pSd6ybkvTgXp3At5Nt/3XDxjR2mLFMphOUrgMvP3SFGCOiHIDZKAbuE4dn8hWsL5S0qbzlFxZ8bQqQRxLCVmEfo/caEk2UnYc/mThLXe+7LCVrH1ZwJ+4GPyvnLANL8Qv1hFpMhcpuw6WO95T8gcw8r9hfaTsK3dXkd6uoFZKphZgnvOwVBMCsRHS9OjMb99GLEksXiTdta+s/Qyh7FiVwVJBrF+xuIlYkqusy9LsImKT3yLYy3ogTQ9c0bZKs+LgrYE10hTExoEJSxJLmaoQ6wtWlWNlqgSeXHF7jlFIny1JMNls9i+SMvGOkhffYmCh9ok0K4rl+AJrORW8vmmhhWwHoqwMonwr6XjrNOs1lapgPbYgaz2xpqampqampqamk/wH+b4WY/sTFNUAAAAASUVORK5CYII=>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFYAAAAbCAYAAADmkHhFAAADHklEQVR4Xu2Zy+tNURTHl7ek5G2ATDzyGJFXBgop8hhIYqII+WUgM8XYgLxTBjIQGUjJwMg/II+iMCDySiLv92t9rbP7rbOcffY+595z7/3V+dS3fvu79t5nn/U7Zz/OJcqnN+sP674N1JSnD0lSh7B2sJ6kwzVl2WjKc1gDjVdT01kMZW1hnWTNVv5o9XcRBlHPnFKus8Zaswx7SObXV6y1rAmJ9401KomVIa/dW5K40y8Ve2RiH1WsVeC6fa1ZhK8knUy0AWYeSey3DUSAxGX1aXHJs8DTb02rwduWNa4oXlK4MeJHrBlgBIX7dbwmqTtXeW9Y41W5XXxnrbZmiFskNzTZBgyoM8yaATCv7rKmh6kk1/iUlO+wFnSH28pSin9A/jGApMFPG8igUMcJaNPfmjm46eAia5OJtZtC9/+YpMEiG2gCZRa74yRtrtpAB4BxrbSmD/eEVMFOKt73Gqp2TI2AMR2wpo8qb+IwFet7OushyYKFdtPS4crYzlpmzQwwpsvW9BGbWKyKRblAcX2D4az3yd9uEUOCmwEWwSywMG4luVZMYrFzumdNHzGJxSnstDUj2E/hvkE/+n9/HDOuWLA/zyM2sagXPfefImmQt1fUJyGwjnWTNYt1g3WOZMtm2Uxxycmqc4jEP2oDzAzWFZIj91mSe3iWqpEm9LYVSSyuFc0XkkZ4HTVjWD+MB86ztlF6wFnJca+0j/Uk8S4bIPlk6XtqsTeeQulYVj1HTGKXWzMD1FtizRAnqPtG3Nl9b6pGmqes+arsuzH49py9gWQ+xUkL18L5f6SKH2N9IJljUecd67mKgzOs3aqsr4+P83hYnBDTZYxdg/gK42Xhu8emoi+Ck9UlVdagHj6WNxv06w4eM0n+AT5intjQ/nQhxR2iGkYnFgsPFrgHynPgpqv4T+s+Mb+vIv8CG5PY0HeAzyTJrZTFrIOqvI91jTVYeRokfpw1GwC/ZuA7qQOHitusScrT+BKLqQzTDJ52yH2jsODNqOLhaJhe1N6BvbBGQbArwo+rHQl2HHet2QPAvjX01a+mpqampkX8BW6i0YPwfWlYAAAAAElFTkSuQmCC>

[image11]: <data:image/png;base64,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>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAaCAYAAADFTB7LAAABTklEQVR4Xu2Wvy4EURTGjxBR67UbjUonQk8WhXgEEo8gHkEiOpUg4gFWo/QCEgmNGlEI4l+IYvmOO5M598vMmjFrZov7S37Jznfu7nx7M5u7IoFA/RzCD/hlvPRWiPSZmfrqj6vhRNzN1ymP2YZHHFbJsCQ7xEzBUw7rIK2gFn+hrDb2xRXcMhkXrpUB8XexDfuTcW8QF3yEIzTrCTbFFVziwT+wCmc4/I076d5zd8FBxCRcEXefwgXTfsl/RQ+AThQuGJ8YLR5EjMFjuCzuBNqBN94Kn08OiMIFN8S9aZoHEVdwVPwd7rTbeQrOcpjGHnyGD/AePkn2eXsA18y1Lahn+LtRZ/b6Oln6g86blJVGP3Qwej0u7gtlkWcH5zgsi92xMzgPd01myVNwgcMyDIn/p2ERnsOGySxZBSfEPUa6++qbP66OWw4CgUCX+QZc9lj8rXctygAAAABJRU5ErkJggg==>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADEAAAAaCAYAAAAe97TpAAABIUlEQVR4Xu2WMUtCYRSGzyIkuARpEE7NtfUHXHRzcpOGxoyiwH+Rg6OTmziI+AvcW/wBDk5FEI5uhdh7OCe8Hvhud/pSOg88eO/7evy+D/QikeM4v9GGtzY8BIbwE27U1m59ePgh9oV9OMQx7MNnmDddJv76ECs40usjkv3U9D4HG3qdCg/d2TASvPbEZD3NmXGySIMH7m0Y4BJeZbSgMyFmtN1skgpt83WySIMHHmwYoArrGS3qTAhe98uG4JykG8BT0wXhgUcbRoDX7doQlEi6F1ukwQNPNowAr9u0IcmTirsLW4Q4IRno2CICr3BuMv5GLEj2dA3L8GznHQn4kbaEbyQfxq8fJH9FYjIl2TDLe/j5Ddxo9q73juM4jvO/+AaQP0ALBKWTTgAAAABJRU5ErkJggg==>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAcCAYAAACtQ6WLAAAAe0lEQVR4XmNgGOTAEYht0QVh4D8Qr0UXhAGQZAG6IAjoM0AkmZAFbYDYC4h3QyV9oXwwKALiEqjEWygfhFEASDIXXRAEdBkgkozoEiCwhgEiiRW8ZsAjCZLYhMTfhsQGS1pA2RlIbDAASYK8UwfEK5AlYADkeQl0wREPAGL/GMEfWDMiAAAAAElFTkSuQmCC>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACsAAAAaCAYAAAAue6XIAAAA1klEQVR4Xu2TPQoCQQyFo4gWXsNWLLT0BIJewloQ8RgexcrW1srKO6iwglr6U+kLMwtjEMkimyoffDCTvIXHskvkOM4/nOTgFze4lcOS2cFXohoOT+XQiBUVKNuhEK7KhRGqsn04gGsK4WG8W6MqO4NzCsFLvLPWqMrmcHAih4aoy7YpBCty8YU67Cntxmc0qMsuSRkETThSyt+/FnVZDl3l0JhCZfkny9kkZysKlW3F8zNdGMIviHs05EKyoBB8wJrYlc0dZvAA9/AIz3CchhzHcRzngzcjETkwh7m/rQAAAABJRU5ErkJggg==>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAaCAYAAADxNd/XAAABUUlEQVR4Xu2WsS8EQRTGv6gUOgWXKESUEqG7IBq1QiJK/gl6F4WoECUhRBQaCp2I0Eqo9KIRoVVo+N6+2WRmdnYvJ+wo5pf8crffzOTem925OyCR+HWW6Be9orPu0P9nns6Z92/QRupmjF7ScdpFh+gmPbMnlbFDp/ywZmagG2f76syoYBhxdt1mml7QXdqiPe5we6SBez+skUm66oedcA5tQm5lDCbwwwakcyl81LzGepSa9Bj6+Sf0g944MwIMwi04vwvtWKBHJR7SA7pP96DP9Fa2qhr59nnyMqnl1sscZMK6dS23UbJ+K4vJCyo2dATFwYFAFpNraD19Xp6xjWKxK4EshBz0jQ5c02WVhM7fncm6vTxjGcUFz9ADFAOp5dTLPk1eit1dr7mOxQP0DOY0oPUsWlkB+c+RH5R3bywGj9Ba8p2P9ZuUSCQSicTf8g107lOhPIWSjgAAAABJRU5ErkJggg==>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEkAAAAaCAYAAAD7aXGFAAACrUlEQVR4Xu2XS8hOURSGl1suSa7JSBKSKMSAkkyYKBMT5FLCgKEyQBm5DeQ2cykMjMRAlPgzMGCkSFH8PyaSa3KLsF9rbd8679nnQvn/yX5qdb71rnefb5/97b3P/kQymUwfMSlEfxYzytMQPy0GU623GReiW7Qvj0JMLZZrmRDivmjbE1TznBb1vAoxkmq1nBVt2JfMDtHl8o2ifdrmtCo2i3rjSkDbH53yb/qJetZYjgmBfOIfRwMwf2Wxl/ki2o+hToszvA4MDDyHSYd21OWfTfMcS2iVwLiHxRbg11lo1zrGsJDgvJQ73GaQVop6VpHObTkHy01r3GYmixqHhNgtumabHhp0ibZ7adfXUt3uAwstWCB63wNcINBf+FaQjuXWNEiLTNtCeokzosZ3oiM60/KqBwZzpbxXzBBtd4j0pSEukdbEfNF73eRCgnWi3n+ZSRtM88syCUzfEtpF0jxPWHCclE6HEG+L5UYOSudFggFoA7xHEpofFMx0HqQHpl0mvQRMexPaC9I8dbMsMpCFv2SQaD96SE+xQ4oDgIeOL4LIAMuXWT42xHXTdkVTCpwtYBrutPiq5LcFM0x0BsKLN+OsYrnADRZaEmfDNC4kwBHidohbIUZLeSZFcIZ6H2JriLWiniUFB7FJyjfabtoU0hkMzHj7jD0Km/PzTrkADm5NfAxxh7Tvon3ZSXob0K6qP5H9Un7+EnF39yB/RhpzlQUDxwG0fxhisXQGfJTzpMARIfXLR2260+aFmONyAI/f++JSjT8iOGWaB/ld0pL4hvsor+IcCwReqfdCXBDdC9qA78X/x8gI07qdBqoG0/veiC49zzUptltNeS3oWPziK1TrTTCY8VTcY9dUf45beNaL+h/bteq/G2pY1p9E96VMJpPJZDKZTOb/8Qv+SsL/WWkcGgAAAABJRU5ErkJggg==>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEwAAAAaCAYAAAAdQLrBAAACUklEQVR4Xu2XTYhOURjH/z6z8BEm2fgoW1KUUoxIaJqtyMKKlA0rypaVmsJ2LGSBheSrUBZqFjZjpUTKAtNE8s2CfDz/ec51n/u8533Pve80manzq3/d83+ee+97/u99zz0vkMlkMpmJZqbohzdrcF70R/RStNzVCo6JfkP7LlRLU4+n0IkUagJD2GHGPH+XGZNnos1m/A3N7zMp+YxmEzmJ1v7+iOe/iCNhfNl4E856UY83x0nTwNj7xJtQf5kb86kqOB68AeMlmS/aKepzSrEVejOuNT/D8ZZKR4n9GdShm8DuexPqn/Om4RO0Z5ovxFiB8hGN6VDZGoU387wQvRfNdX6TyZNuArvpTah/x5uB09D6Gl+IMQ/afNR4b6ELZx34BlrgzcBqtIa/tNKRpklg06G913wB6j933hLRGdGQ6LVocbUchxe67rwDwZ8MNAmMxOZD6DOYdpyC9vT6goXrU+zDXEHc78QNlE/RWVezzBLN8GYHugnsnjeh/qA3DVy7is/flruIN9D76M0OfBUdDMcM4xb0Gmv/dZTc9kaCbgJr95bcE443hvGGsjxGMrBLaG3gtoAe35h1mA3d58R4B73WftFe0Rvo09uEVGAn3Jhh+X4GY73HYfzAeCQZGBc521AsmvuMlyIV7ErRVdGwaFu1VIvvaD+JD9DaYeMtCt4c430RPTLjddDtj+Ui9DxuqzqyCWWy/LuwsFr+b3CSo6JXQSPQbcoq08NtALcvnu3Q+XBdZagPq+UxdkN7uJz8CsfcT2YymUwmk8lkMlObv/MmqwTyek++AAAAAElFTkSuQmCC>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACUAAAAZCAYAAAC2JufVAAABpklEQVR4Xu2VPyhGYRTGj5QMSqTI6E9RSjYpKYuULBJKIqVMFotNjAbJpqRkMTGZ7AYLgzCgKLL4tyjlzznfuZfzPs69H4My3F89fd953uc9nfu9t/cjysj4O8pZ16x31jmrNFzOSxvriXT/IqwtsS5YU6wR1hBrkDUQyaWIdKCYAtLmLcZLY491a+pj1oyp5SGlX5JcHtBgmlivaDrMU9i4Iqp3jCd1J2nPelZNpBdWsckFyKZJ8JojPx+SWQWvAeo3qIUF1jialnvS5rvGk1+v3dQeXaT7+qK616ylIe/vGZpIIYVnLAP1BAmfTdL8MGuWVca6Yt2YjMdPTiBHNYWDHYXLLqek2TvwxVsGL2aM9MXPixzDc/S9g74GO/xM+OyT5ibAj/d7iD+KpofX4JJ837JGmqkFP2moOlK/BBeQfvIbCOK3omnoJs00gp801Ar5/jfkgkwKoi8PUAmeZORFR8+7BsTDnolIcA48qQ9MHd/y2HSdwktWjkYyVcaL8fan8ki64ST63AiXc2yzptFktkj3xP+dcnN7yPC/GiojIyPjP/IBpaV0gVuFEVkAAAAASUVORK5CYII=>

[image20]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAkAAAAaCAYAAABl03YlAAAAk0lEQVR4XmNgGFCgCMRM6IIw8BCI/0MxO5ocCljMAFGEF4AU/EQXRAcgRY3ogshAmQGiiAOI64B4PhAzoqgAgkUMEEUfGCAO14XyURSCBH4jC0DFNqALtCMLQMVewDiSUAEeuDTEGpDYRJhAGlQAGZRCxVRhAnZQAWQA4j9CE0NR1IHGhwNQxIIkQHg7mtwooCYAANAqI3RvAHqJAAAAAElFTkSuQmCC>

[image21]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACUAAAAZCAYAAAC2JufVAAABjElEQVR4Xu2UvyuFYRTHv0hikEgGk4UsRDFgUDallMwWg8Vmkr8AC2UyMshkEKNkkdUiy70igxhF+XlO5zw697zve5/XYFDvp77d5/l+zzmd3vfeCxQU/B39pCfSF+mc1FIZ52aTdOlNop1Ugsy/JnVXxkkWIMMCu5DmQePloQnS55caIJ2Y+zykbtF4CbiAFfNifCJ9qVf1G40XnX+HZEG0ybFOGkL6UnvqW347H8uQhkkfZNBAutJz2lKeEUjdqg+ymIY0bPigCm/mHFtqGFJz6oMs+BXskz5IEy7LYoU0bu7Vlloj7UBq5lwWpRPSeOgDRy3pxnnVlgrUQ+rKzo+S54v47A3kW4oJ83t8EODXte280DTmfMtZiriHl+XzqNbx/ULPgXdILb/+BDNIfyrBqzPeLKnD3NPwT6pNvaz5vc7/gUP+WQf61DsyXo16friHc/7f816XuTerVzJeglbIvzHrEdKwVVEhHJCWvKnwK7on3UKWeiBNacZP+wUyt6yfx5oVFBQU/Du+Ae1fd9du3bT0AAAAAElFTkSuQmCC>

[image22]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAaCAYAAAD1wA/qAAABa0lEQVR4Xu2VOy8FURSFl1eh0ZJotUKi1Xsn/gJRScSN0FAotKKS+Ac0Sp1WRTT8DUQEQQh722di7jLX3XPmRnKT8yUrM7P2rD2Tk/MAEolEu/BZQh8h05Bn0Tmb/0AH7AcHc15n8K5znnIoOiHvFxqssVlAr6ibzQrsicbIW4X9zzz5M6JF8uoYgQV1JDwcie5FA1yIQL/L3KHY3xH1samMi6ZFp7DgXHj2si96F41yoQQ6yky2Hpiid79ZE63DQrfhWVWWTfwMRFW6YL0uueBBgytsRrAA67XMhRJswHrMcqEZw7Cg7h6tYgrWc4sLDh5QPK2acozI4B/o5qH7/QEXHDRaH03RkO4SrWAC1m+bC056YPkLLnjQoC74jLPcvZdsbSxxoSQ6FaPWh6LBoXD/mi84yD48yYVI3mD9vOdZHbuw8Av8J7aexnp+6EZRlSvRI2x634TrE6x/zOxwo6PVz2YikUgkEu3KFz8ZV26ZgOEGAAAAAElFTkSuQmCC>

[image23]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAaCAYAAABozQZiAAAAlklEQVR4XmNgGLlgMxD/JwGjAJBAGBYxdIUa6GJCDBCbkQETA0TRBTRxEHiEzNkKxIzIAkBQwADR7I8mzgbEfcgC+cgcKHjPgOlkEBAAYnF0QXSAzb9EAWYGiMYz6BLEgHIGiGZvdAliwGcGMp0MAmT7FxQVZPt3NgNEcwKaOE4QBMTfGCBx+xaKQf7+xUCm80fBKIADAO8/LWwyw7tTAAAAAElFTkSuQmCC>

[image24]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAbCAYAAACeA7ShAAAAwklEQVR4XmNgGAWjYIiD2UD8H4gPQOk6FFkSwC8gvoAmBjLQEcr+jSyBD9xggGhEByAxkCUgsBtZAhdgYoBomoouAQRvGSByh4GYHU1uPxofDBwYIBrc0cRB4BEDRA6XqzHAagYcEkBwjQEiJ4kuwYBDTwMDDgkguMiAKQdzKS4XgwVV0cTuAfFaqBwI9CHJnQNifyQ+ChAC4r8MCNtmIMndh4rFI4lhdRG5gGqGOQHxdSgb5GqKASghn0cXHAXDCQAAwUAztFt2XBsAAAAASUVORK5CYII=>

[image25]: <data:image/png;base64,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>

[image26]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAWCAYAAAD0OH0aAAAAkUlEQVR4XmNgGAWDFbgC8QYgzkOXQAcmQPwfiB2g/CooHwamIbEZdBkgkkLIglCx1VD2X5ggE1RiKkwACbxlgMgdBmJ2mOAEqCAvTAAJ3GaAyK1AFtwEFcQGLjFgkTPAJggEnAwQcWxyDA+B+A4DxFmmDBCTQTYzM0A0iALxL7hqKACFUDwQa6BLAIEfusCIAgDN0B5os0BKpwAAAABJRU5ErkJggg==>

[image27]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAaCAYAAACKER0bAAAAaklEQVR4XmNgGBwgCYhnAvE8IM4D4pNA3AzETsiKQKAfSt9CEUUCU6D0bRRRJDADSt9HEUUCMAWPUUSRAB0VPEQRRQLToPRdFFEkMAlK4wyodih9GUUUCF4C8T8g/o+E/wLxPmRFo4AAAADysBoipyYEIQAAAABJRU5ErkJggg==>

[image28]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAYCAYAAADKx8xXAAAA00lEQVR4XmNgGAWjAAjmAfF9IP4PxIxI4oJQMazAEIjLoWyQogwkudVQMazgO5Q2ZoAoYkeSA/FfI/FRQC2U7mbANB3EX48mhgFAil4g8ZmhYiCvwMAfIBZC4oMBSFEQEr8MKoYM0PkM4lgEPyKJvYKyYRgFgAT0oGwdKH8DQpqhD4j7kfhwkMKAMHESlNZCkv/LgMV/Umj8nQyYTkLnw1PHNCgflHJAfA+4CgYGFqgYCFyBCUozIBIBPwNEQSRMEgl8AeLL6IKWQLwGiNvQJQYPAABzHTSCnI5ZIAAAAABJRU5ErkJggg==>

[image29]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAYCAYAAAD3Va0xAAAA20lEQVR4XmNgGAWUgHYg/o+GRaFyn2CK8IEQBoimJ+gSQPAZiD8C8T90CXSQygAxxAVdAgr4GCDyk9AlkIEdA0TRQnQJNABSI4wuiAxg4UAI4FWzkgGiIB9dAguIRRdABsS6hiAYFAaxwBjMDBBDXiLkcAJ0y3qAeAKyADEusgDiBDQxUMJESQp3GSAGgVyHDYDEXyHxsxgQloNwBpIcWOAPA6ZhRkD8Gk0MBAQY8GSV3QwIW75CaVC2wQb6GNDCh1zwl4FAViEWwCInHkWUDLAXiA8CsQ66xDADAHw4OQNZKvANAAAAAElFTkSuQmCC>

[image30]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAaCAYAAACD+r1hAAAAgUlEQVR4XmNgGAWDDUgD8TIgzkWXAAJbdIFjQPwfCb9BlWa4gczJA+K3QMwK5UsyQDTJQvkLgZgNygYDkCQ6cADiA1A2yDCiAMigLCAWQZfABWD+IRq8B+IadEF84Am6ACFAknOYgPgVuiA+UAfE3eiC+MA7IOZDF8QHfqMLjGgAAJinF9CI2KPUAAAAAElFTkSuQmCC>

[image31]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACYAAAAaCAYAAADbhS54AAABKElEQVR4Xu2UwUqCQRSF7zO46wFaugvaShBuBB+gRW/gwmWr1kbSSnwHEXqFlm0KIQw3BiK4qU0ujCD1HmeGLsf/l4H+DGI+ODBzjnIPd1CRRCKR+H+cqm5VDQ7+iiPVSlXx9wt/D3TMeW+UxZUokQ+v589fNtgXKDBjU1mKy45VZ5T9hJqqxSZTETe8Sj6YiMvskwayvMATGx68SFN1LxHF8FR5Q4bisgMOJP874IMN4k4iil1K/pCBbGdhg3mbBJ9sEFHFAAYckjdW9X0G2iZ7UNXNnYkpdsVmFnh7/OrCFrome/HeufF4UyPVwgi5vU+/P7oBxa7JKwQuxsRszL5AIZyonv0Z28wiptgNm0WAwY9sGnYVe1e9qV5Vc8p+naw/60QiEcMad9VN1+GaR5sAAAAASUVORK5CYII=>

[image32]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAZCAYAAABOxhwiAAAB60lEQVR4Xu2XvS8tQRjGXx/XR6FwVQpRaCS0XJHbCEGtoZCcKCQ6hQRRK24kGgX1/Tvu1SpUQiPREQQhvhJBfLyP2XMy+5jZnd3IieL8kqfY37w7O7s7590ckQrfg2EWZaJVU80ylBnNIssy8sYihDbNCcuIVc2T5lLzm8ay0qk5YxlRp3llmQbutoGlcqNZto4fNH+s4xCw/S7EXAM5jw/H2BLzoILo1zyyVAbl8+v76XBZSFt4jWSY/1ncexvbwzUJ3CTLQNIWDlAzxNIFChtZivG4KQZ+l2UgIQvf12yzZJrE/VQB/C1LMR57PQ8hC18S/5pKDIi/CP6KpRjvOycNnIcfahITEjD/lPiL4K9ZivGZ21YEzkVbTeKX+NdUoiD+InhXt4E/YBmI7y3a9Ih/TSX6xF/k2xJwGywD8b1Fm3FxXzdGs/iL1uXzWFXkflgOHWnWOk4C5+KjlgRaM1/XCYrwuXWBsS7rGG2KO03xzfSSd4G6tI60p9lh6QKTzbGMaBczvqk51BzHhz8YE+P/8kBEt5h9jZqjKKeaO7vIAtcbYeliXnPPMiP4HiywzEFxKwaD4lqWGfA97az806yxTGJU8re4Fs1/ljnAH4kXliGsaKZZBlDPIie5Fl2kwKJMdMjXPYAKFcA7GD18IGogCNoAAAAASUVORK5CYII=>

[image33]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAZCAYAAAA4/K6pAAAA/ElEQVR4XmNgGAXYABMQ/wfim+gSxABmBohmfiDOBeLHqNKEQQIa3xyIOdDEaAd6gLgGXRAIzNAFsIE/DBB/g/zfiCSuAhXDC+KAWBvKBinuQpJbDxXDC+KhtDcDpmIQ/y2aGAsaHw6+MGA3oBCJDwqnCUh8FABSPAmJDwo8kBgoYcHAPyAWRuKjAJBidST+VqgYCGRB2TCcAVOEDEASeVC2AJT/DiENFgO5ACewYkDYsAlKg2yGgT4GPP7nAmJGJH4lA2aA/mUg4P+raPwOJD5MDATiUUShACYJyo3fgbgbSQ4G9gLxQSDWQZcAAUUgXgDEE4GYFVVqFBACACfkNtu6zD7AAAAAAElFTkSuQmCC>

[image34]: <data:image/png;base64,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>

[image35]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFUAAAAaCAYAAADG+xDjAAACKklEQVR4Xu2XPUgdQRDHJyp+EBFBUOyeBps0EbVQEAtBU4imEIyYOliFpAoiImIlYhFtRERRNCJoJ1inD0lAEkghSiqRFEIICgli5u/O+ca9OzyNb48H+4M/Nx97b+f27e3tEnk8Ho8nIRV2wPN/tLAuWG/thOdutLFOxf7CmlY5zx3Ztvx5y/e4ZJb1UvlTrDHlu6SItc56aieYRjvggAzrHeuZ+J3ZVDQ1rD2xX7H+kFmQwSfWjNiuKGQdio06SlQOAx3U5oofrI9iv2YdUYIadINy8Z+wWsUeVPkoFllrMVplrbCWWUvStufyrnj25Zoh07/exsA/Vn6uwWSzBxD+bysWoknZ2C7oHylVtitG5YrZEfVAmC1xFJPZ+iRRs9wTB95g9DdsxRG71bL4i8IPkhaoY0v5wT6xQMVsHrL6EqpX7onjK4XHokFi+PMSgxvw2t6GSTL7tqS66WFAFZlacA3YlZgr0JfdX6I1vZJMo0eUXU8fq/w3ZbvkOYWLh39ixXIJ+vsbETuzYiEWyDQsY30Wu05y+Fhtiu2aWro+qC/Ef6NiuQZru67hg/gTKhbJA8pO8y4yMzbwx1W7NJijbC04weCKel3ynky/OKK2i53Gx/te0EsQ+Mk6t2Ku2aDrMzevwD4WxQ+IHywF1Vct0gE14GCUlwyxDsSuJ/MwaRxNA7AsYoeDOr5TeN+aN/SzdlgjdiIFOljdZM77GGDU5vF4PB5PfvIPzn6BQdqSQhcAAAAASUVORK5CYII=>

[image36]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAALAAAAAaCAYAAAAXMNbWAAAGV0lEQVR4Xu2aZ6jlRBiGP9desGBZFWUt2H6IYsOGGysqiqJiF3fF3v6IDf21KhYQFRFFLFgXEVEUe1lsP1QQFMGGuvbesHfncTJ75747k5Pck5y7K3ng42TemWSSyXemfBOznp6enp7J57DSttOMBYTNVBiSzVXo+Y/9VRiSA1SICD43VTMmwu/ONnG2pmZETFFhRGzh7CEVW+AfFYZkeRUWMt50toqKQ3Kgs5tULMHfjna2i2ZMhF9VEF42/8KX0IyOWdzZnypGfGX+vmLbvsw7TvS/nC1Z5sG6zr6N0sNwrPk6WnkZk8DFzs5XseQIZ7/Z+Lak3QN0fnHe21EevOjscNEChbXUZlUO/JqzS807Lze47PjsTsHptlQxQWi8xUSnMWeJFvOTs31VbMipzuaWx19b+8Nw1zCy0naD2MN8uY9Fxx/+draM6DG56xfWsQMvYvO/4Isk3RU0TO7BlTPNl30k0q5zdkGUTrGB1a8jxx2Svk/SCzqPObtKxQyho1BtEHOdXa2ijcCBJxN6/tghBxE37vHO7o3yquCcqt7j/05q5Mpxl/nyoWPgeNGx7Cz7WNrRCxuBAx/s7HpnK5fpa6K8LuGBd1OxgnfMn3Ohs1clrwrOuUzFBmxlvn12KNPMvRcW1rO0Y+VgRA4dxc/OVhyfXUmqnsI6dOAwhO9cpn80Pyd9cF6Jbkk9cBVr2PheuC4MoSxSJgLtdkt5fJuzT615/ZMJf9ym9/uH+XOma8YAOEfDtIV16MBUGM/vCLGkbqIuTAnqwrDUtGGXtjEHpqeoC3PlpnUBf2heZgzXeVS0UXKQCgN42po/+wvmz3lXMwbAOTNFK6wjBw5hsxjmlarVZSln60fpl8xf6/lIiyHE1bQuyp9V/t4teVWcbc3r2s/8ORuJjrZjlCauzqg1Cs6Jjmnv78zfT9Wi+/3S6sJUKbRX0zaj/CWiFdaRA6dukPCJanV5IDqOY4i32vy9GEyzZnVRNvS6qXuvIkQwmpCqI8SdAwzPqnUJ05cAHVCA9mXxleI9GwsBDuIUZ7eXx2ExhzPXhfKXi1ZYhw48O6E9JVpd7o+OuU74J4ZFQYqcrjB/XS5KE/vl3F0jrYprrX5dgZQDf5TQWD+o1gVEEdixDFBnaJNzy3QKojzEcAfBYppRMxAv5upC2dNFK6xDBw6r6libLtqHzm509kOk6fyTLUOGtBRbW74RcnrMl+ZX0jGrmT+XHaI68BJZUcfwDLmdKeD6jye0OaIN68BrO/ve/ELx2VJLha1yPSyw6M4tUon/Dro/pkmfqWhjO3Cra0YGyoZd0kBhHTrwDZLWB43TODa9bOrjjQ9UiGB4y20zcn2um4LehnzdFQqE+4175hyUu1I0dtTQmVOnIHIRP3/ofYtIg2EceFNnX0TpOeZDdq9EWoC8HNSf2z3d0PL3x87rzebzU58QnGE+7xPNyJCqp7COHBjYZqXSe2z++S+NGDs4kP+NaJBbRNBzH6pixFwbP3cO0FtSD98x0POzjx9Yx3x0gDyckIUMz1EF972SaHz9xvlVC5ywEOUFsp2cekEpB2Yx/EuFhdGK86aVx8B8k9FCdxcpk/uQiN57BRUF6pki2lvm24424BrahrQ/Ovkc8x6YquQI29BKYR06cAyVPyNpbTQ0nT4cIunAMc62LY91qhLYxtIP3SZrWXUdD6uQASdOXSflwHXR83JRoNdVKGEzJ+ywVUVl6OVznUxbcI+nqWgjcmD+nTRcvCvG0L9qlOaLpVTj6twSGAZZxM0wv0rnM74cLDA2VrFF6KlT0x6Yad7B68CzP6eitevAT1o6YqDTH2ABTicxw9lJVv1+Q/y8S3LXL6xjB+bhQ0z4PBub4/HPZmHAXJDFBT0vn82xnRvDNrTCtdRysIjJ3duwTLX04iRQlRc4ytkV5p+B3vqEKI/pAMMrYUOG4ybb4sAfnRgy7RrCT1zzjXklzPaOjmO0fZkGVkF0iXfdBU8421PFksJacmCckbmSftSyl/mGJyRFYxGYr8ssFSYIX8PpF19tUPXHqQsvZnfzL4F5HsejJI7+DMvnVm/R2wTa5U4VS/A3PvJpxYFZUWI4bFs03WqsomnvNYidVFhImaHCkBypwpCcrEJE8DlG2QUOphTx1nFP+5yoQk9PT09PT09Pz0D+BX4MriMCfIscAAAAAElFTkSuQmCC>