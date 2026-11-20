# Practical Assignment 05: Ensemble Learning & Performance Evaluation

**Course**: Machine Learning Laboratory (MDM-AIML)  
**Academic Year**: 2026–2027 | Third Year B.Tech (Computer Science & Engineering - Software Engineering)  
**Batch**: B2 | **Roll No**: 48  
**Student Name**: Aditya Iche  
**PRN**: 202401100136  
**Faculty In-Charge**: Mr. Khushal Khairnar  
**Date Performed**: 13-11-2026 | **Date of Submission**: 13-11-2026  

---

## 1. Executive Summary & Repository Overview

This repository provides an end-to-end, mathematically rigorous implementation, evaluation, and benchmark of **Ensemble Learning Paradigms** applied to high-stakes diagnostic clinical classification on the **Breast Cancer Wisconsin Diagnostic Benchmark Dataset** ($N=569, d=30$).

Ensemble learning combines predictions from multiple individual machine learning models to synthesize a meta-predictor that achieves superior generalization, reduced error variance, and mitigated inductive bias relative to any constituent base estimator. In this practical assignment, three foundational ensemble paradigms are implemented from mathematical first principles and standard frameworks:
1. **Voting Ensemble (Majority & Soft Probability Averaging)**: Aggregating diverse base hypotheses (Logistic Regression, Decision Tree, Support Vector Classifier, $k$-NN, Gaussian Naïve Bayes) to suppress uncorrelated individual error variance.
2. **AdaBoost Ensemble (Adaptive Boosting via Sequential Reweighting)**: Constructing an additive linear combination of weak learners (Decision Stumps) by iteratively penalizing misclassified instances via exponential loss minimization.
3. **Stacking Ensemble (Stacked Generalization via Meta-Learning)**: Blending heterogeneous base learners (Random Forest, RBF-SVC, $k$-NN, Decision Tree) through out-of-fold $5$-fold cross-validation to construct a level-$1$ meta-feature matrix, mapped by a Logistic Regression meta-classifier.

All ensemble architectures are rigorously benchmarked against five standalone base models across classification accuracy, precision, recall (sensitivity), $F_1$-score, Area Under the ROC Curve (ROC-AUC), zero-one classification loss, and training/inference computational latency.

---

## 2. Theoretical Formulations & Mathematical Principles

### 2.1 Voting Classifier (Hard & Soft Ensembles)
Given a set of $M$ trained base classifiers $\mathcal{H} = \{h_1, h_2, \dots, h_M\}$ operating on feature vector $\mathbf{x} \in \mathbb{R}^d$:

- **Hard (Majority) Voting**:
  $$\hat{y}_{\text{hard}} = \text{mode}\left( \{ h_m(\mathbf{x}) \}_{m=1}^M \right) = \arg\max_{c \in \{0, 1\}} \sum_{m=1}^M \mathbb{I}(h_m(\mathbf{x}) = c)$$
  where $\mathbb{I}(\cdot)$ is the indicator function. By the Condorcet Jury Theorem, if each base classifier operates with an independent error rate $\epsilon < 0.5$, the collective ensemble error rate drops monotonically as $M \to \infty$:
  $$P(\text{Ensemble Error}) = \sum_{k=\lceil (M+1)/2 \rceil}^M \binom{M}{k} \epsilon^k (1-\epsilon)^{M-k} < \epsilon$$

- **Soft (Argmax Probability) Voting**:
  $$\hat{y}_{\text{soft}} = \arg\max_{c \in \{0, 1\}} \frac{1}{M} \sum_{m=1}^M P_{h_m}(Y = c \mid \mathbf{x})$$
  Soft voting incorporates classifier certainty, giving higher effective influence to confident predictions and softening threshold boundary sensitivity.

---

### 2.2 AdaBoost (Adaptive Boosting)
AdaBoost fits a sequence of weak base learners $h_t(\mathbf{x})$ iteratively on weighted versions of the training dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ with $y_i \in \{-1, +1\}$:

1. **Initialization**: Uniform sample weights $w_1(i) = \frac{1}{N}$ for all $i \in \{1, \dots, N\}$.
2. **Sequential Iteration for $t = 1, \dots, T$**:
   - Fit weak learner $h_t(\mathbf{x})$ to minimize weighted training error:
     $$\epsilon_t = \sum_{i=1}^N w_t(i) \mathbb{I}(y_i \neq h_t(\mathbf{x}_i))$$
   - Compute estimator weight (stage significance):
     $$\alpha_t = \frac{1}{2} \ln\left( \frac{1 - \epsilon_t}{\epsilon_t} \right)$$
   - Update sample weights:
     $$w_{t+1}(i) = \frac{w_t(i) \exp\left( -\alpha_t y_i h_t(\mathbf{x}_i) \right)}{Z_t}$$
     where $Z_t = \sum_{i=1}^N w_t(i) \exp(-\alpha_t y_i h_t(\mathbf{x}_i))$ is the partition normalization constant ensuring $\sum_i w_{t+1}(i) = 1$.
3. **Ensemble Final Prediction**:
   $$H(\mathbf{x}) = \text{sign}\left( \sum_{t=1}^T \alpha_t h_t(\mathbf{x}) \right)$$
   AdaBoost minimizes exponential loss $L(y, f(\mathbf{x})) = \exp(-y f(\mathbf{x}))$, functioning as a stage-wise gradient descent optimization in functional space.

---

### 2.3 Stacking Classifier (Stacked Generalization)
Stacking addresses the limitation of fixed voting rules by training a meta-classifier $g(\cdot)$ on level-$1$ meta-features:
1. Divide training data $\mathcal{D}_{\text{train}}$ into $K = 5$ stratified cross-validation folds.
2. For each fold $k \in \{1, \dots, K\}$, train base learners $h_m$ on remaining $K-1$ folds and generate out-of-fold predictions $\hat{y}_{i, m}^{\text{OOF}}$ on fold $k$.
3. Construct level-$1$ meta-feature matrix:
   $$\mathbf{Z} = \begin{bmatrix} \hat{y}_{1, 1}^{\text{OOF}} & \hat{y}_{1, 2}^{\text{OOF}} & \dots & \hat{y}_{1, M}^{\text{OOF}} \\ \vdots & \vdots & \ddots & \vdots \\ \hat{y}_{N, 1}^{\text{OOF}} & \hat{y}_{N, 2}^{\text{OOF}} & \dots & \hat{y}_{N, M}^{\text{OOF}} \end{bmatrix}$$
4. Train meta-classifier (Logistic Regression) on $(\mathbf{Z}, \mathbf{y})$:
   $$\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \left\{ -\sum_{i=1}^N \left[ y_i \ln \sigma(\boldsymbol{\beta}^T \mathbf{z}_i) + (1-y_i) \ln(1 - \sigma(\boldsymbol{\beta}^T \mathbf{z}_i)) \right] + \frac{\lambda}{2} \|\boldsymbol{\beta}\|_2^2 \right\}$$
5. For test observation $\mathbf{x}_{\text{test}}$, pass level-$0$ predictions through meta-learner $\hat{y}_{\text{stack}} = g\left( h_1(\mathbf{x}_{\text{test}}), \dots, h_M(\mathbf{x}_{\text{test}}) \right)$.

---

## 3. Quantitative Benchmark Results

Evaluated on the standardized stratified test partition ($N_{\text{test}} = 114$):

| Model Architecture | Model Paradigm | Accuracy | Precision | Recall (Sens.) | $F_1$-Score | ROC-AUC | Zero-One Loss | Training (ms) | Inference (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear Base Model | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9954 | 0.0175 | 261.51 | 0.60 |
| **Decision Tree (CART)** | Non-Linear Base | 0.9386 | 0.9577 | 0.9444 | 0.9510 | 0.9342 | 0.0614 | 42.96 | 0.83 |
| **Support Vector Machine (SVC)** | Kernel Margin Base | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9950 | 0.0175 | 480.53 | 17.57 |
| **$k$-Nearest Neighbors ($k$-NN)** | Instance-Based Base | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9788 | 0.0439 | 0.98 | 1958.73 |
| **Gaussian Naïve Bayes** | Probabilistic Base | 0.9298 | 0.9444 | 0.9444 | 0.9444 | 0.9868 | 0.0702 | 7.35 | 14.75 |
| **Voting Ensemble (Hard)** | Homogeneous Voting | **0.9737** | 0.9726 | **0.9861** | **0.9793** | 0.9692 | 0.0263 | 770.55 | 1810.71 |
| **Voting Ensemble (Soft)** | Probabilistic Voting | **0.9649** | 0.9595 | **0.9861** | **0.9726** | **0.9904** | 0.0351 | 450.35 | 2164.42 |
| **AdaBoost Ensemble** | Sequential Boosting | **0.9561** | 0.9467 | **0.9861** | **0.9660** | **0.9831** | 0.0439 | 11925.39 | 448.77 |
| **Stacking Ensemble** | Stacked Meta-Learner | **0.9737** | **0.9859** | 0.9722 | **0.9790** | **0.9940** | **0.0263** | 10270.54 | 45.58 |

---

## 4. Key Empirical Insights & Comparative Analysis

1. **Variance Suppression via Voting & Stacking**:
   - Standalone Decision Trees suffer from high variance and hierarchical sensitivity, achieving only $93.86\%$ accuracy and $0.0614$ zero-one loss.
   - When combined in the **Voting** and **Stacking** ensembles, the individual decision errors of trees are smoothed out by linear and margin-based learners, lifting accuracy to **$97.37\%$** and reducing misclassification by more than $57\%$.
2. **Clinical Sensitivity (Recall) Maximization**:
   - In medical oncology diagnostics, false negatives (misclassifying a malignant lesion as benign) carry catastrophic clinical consequences.
   - All three ensemble architectures (Voting Hard, Voting Soft, and AdaBoost) achieved an exceptional **Recall of $98.61\%$**, capturing $71$ out of $72$ malignant cases with near-perfect sensitivity.
3. **Meta-Learner Weight Distribution in Stacking**:
   - Inspection of the level-$1$ meta-learner coefficients reveals that the Logistic Regression meta-classifier assigned the highest positive predictive weight to the **Random Forest** base model ($+2.14$), followed by **Support Vector Machines** ($+1.86$), while dampening individual decision tree noise.
4. **AdaBoost Iteration Error Dynamics**:
   - Staged error tracking demonstrates rapid exponential loss reduction during the first $25$ iterations, after which training zero-one error drops toward zero while testing error stabilizes at $0.0439$ without severe degradation, highlighting AdaBoost's margin-maximizing behavior.

---

## 5. Architectural Trade-offs: When to Use Each Ensemble Method

| Criteria | Voting Ensemble | AdaBoost Ensemble | Stacking Ensemble |
| :--- | :--- | :--- | :--- |
| **Primary Error Reduction Target** | Variance | Bias (and Margin Maximization) | Both Bias and Variance |
| **Base Learner Diversity** | Heterogeneous or Homogeneous | Homogeneous Weak Learners (e.g. Stumps) | Heterogeneous Base Learners |
| **Computational Complexity** | Moderate (Parallelizable) | High (Strictly Sequential) | High (Requires $K$-Fold CV) |
| **Sensitivity to Outliers / Noise** | Highly Robust | Sensitive (Penalizes hard outliers) | Moderately Robust (Regularized meta-learner) |
| **Inference Latency** | Sum of all base models | Moderate ($T$ simple evaluations) | Fast ($O(M)$ meta-prediction) |
| **Ideal Real-World Use Case** | Competitive Kaggle / Fast baseline aggregation | Structured tabular data with weak signals | Production predictive pipelines requiring maximal accuracy |

---

## 6. Generated Visual Artifacts

1. `ensemble_metrics_comparison.png`: 4-panel comparative bar chart across Accuracy, Recall, $F_1$-score, and ROC-AUC.
2. `confusion_matrices_ensemble.png`: Complete grid comparing True/False Positive/Negative distributions for all key models.
3. `roc_curves_ensemble.png`: Multi-model ROC curves plotting sensitivity vs specificity with AUC annotations.
4. `decision_boundaries_2d_ensemble.png`: 2D PCA decision boundary contour plots demonstrating ensemble boundary smoothing.
5. `adaboost_staged_performance.png`: Zero-one training vs testing error progression across 100 boosting iterations.
6. `stacking_meta_weights.png`: Logistic Regression level-$1$ meta-learner weight attribution bar chart.
7. `terminal_output.png`: Simulated high-resolution console terminal log.

---

## 7. Ethical, Societal & Green AI Considerations

- **Diagnostic Autonomy vs Human-in-the-Loop**: While ensemble models achieve over $97\%$ accuracy, critical medical decisions should utilize ensembles as diagnostic decision-support systems rather than autonomous black-box arbiters.
- **Explainability & Model Transparency**: Ensembles increase predictive accuracy at the cost of interpretability. In clinical healthcare, stacking and voting obscure individual feature contributions. Utilizing SHAP (SHapley Additive exPlanations) or TreeSHAP is essential for clinical auditability.
- **Sustainable Green AI**: Sequential boosting (AdaBoost) and $K$-fold stacking require substantially higher compute cycles (10–12 seconds training time) compared to standalone linear models (0.2 seconds). For edge devices and embedded point-of-care medical hardware, soft voting or distilled ensemble models provide the optimal balance between diagnostic power and energy consumption.

---

## 8. Author & Academic Verification

- **Author**: Aditya Iche
- **PRN**: 202401100136
- **Branch**: Computer Science (Software Engineering) - Batch B2 | Roll No: 48
- **Institution**: Department of Computer Engineering
- **Faculty In-Charge**: Mr. Khushal Khairnar
- **Repository**: [https://github.com/aniche59/MDM-AIML-05](https://github.com/aniche59/MDM-AIML-05)
