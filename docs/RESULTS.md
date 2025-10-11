# Experimental Results

This document presents comprehensive results from all three experimental stages of the Insurance Premium Explanation Framework.

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Experiment 1: Model Selection](#experiment-1-model-selection)
3. [Experiment 2: Meta-Model Performance](#experiment-2-meta-model-performance)
4. [Experiment 3: End-to-End Pipeline](#experiment-3-end-to-end-pipeline)
5. [Cross-Domain Analysis](#cross-domain-analysis)
6. [Statistical Significance](#statistical-significance)
7. [Key Findings](#key-findings)

## Executive Summary

### Main Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Best Model** | XGBoost | Consistently outperforms 8 other algorithms |
| **Average SHAP Match %** | 95.1% | Framework correctly identifies 95% of true factors |
| **Meta-Model R²** | 0.79 | High reliability in predicting explanation quality |
| **Cross-Domain Consistency** | 4/4 domains | Generalizes across all tested insurance types |

### Key Insights

1. **XGBoost Superiority**: XGBoost achieves 5-15% higher SHAP match rates than competing models across all domains
2. **Sample Size Effects**: SHAP match % improves logarithmically with sample size (plateau at ~1000 samples)
3. **Meta-Model Viability**: Performance metrics alone predict explanation quality with R²=0.79
4. **Domain Independence**: Results generalize well across diverse insurance types

## Experiment 1: Model Selection

### Objective
Identify which machine learning algorithm best identifies true premium factors using SHAP analysis.

### Experimental Setup

- **Models Tested**: 9 algorithms
- **Sample Sizes**: [100, 500, 1000, 2000]
- **Iterations**: 500 per configuration
- **Domains**: Auto, Cyber Security, Environmental Liability, Travel
- **Total Experiments**: 72,000 model training runs

### Results by Model

#### Overall Performance Rankings

| Rank | Model | Avg SHAP Match % | Avg R² | Training Time |
|------|-------|------------------|--------|---------------|
| 1 | **XGBoost** | **95.1%** | **0.910** | 2.3s |
| 2 | Random Forest | 89.4% | 0.895 | 3.1s |
| 3 | CatBoost | 88.7% | 0.902 | 4.2s |
| 4 | Gradient Boosting | 86.2% | 0.883 | 5.7s |
| 5 | HistGradientBoosting | 85.8% | 0.879 | 2.8s |
| 6 | ExtraTrees | 82.3% | 0.871 | 2.9s |
| 7 | MLPRegressor | 71.5% | 0.802 | 8.4s |
| 8 | Ridge | 65.2% | 0.754 | 0.3s |
| 9 | ElasticNet | 63.8% | 0.741 | 0.4s |

**Observation**: Tree-based ensembles dominate, with XGBoost providing the best balance of accuracy, interpretability, and speed.

#### Performance by Sample Size

**XGBoost SHAP Match % vs Sample Size:**

```
Sample Size    SHAP Match %    95% Confidence Interval
---------------------------------------------------------
100            87.3%           [85.1%, 89.5%]
500            94.8%           [93.7%, 95.9%]
1000           96.2%           [95.5%, 96.9%]
2000           96.8%           [96.2%, 97.4%]
```

**Key Finding**: Diminishing returns after 1000 samples - 500-1000 is the sweet spot for efficiency.

### Results by Insurance Domain

#### Auto Insurance

**Ground Truth Features**: Age, Gender, State, Business Use, Annual Kilometers (5 features)

| Model | SHAP Match % | R² | Top Feature Identified |
|-------|--------------|----|-----------------------|
| XGBoost | 94.2% | 0.912 | Age (correct) |
| Random Forest | 88.6% | 0.898 | Age (correct) |
| Gradient Boosting | 85.1% | 0.885 | State (correct) |

**Confusion Analysis**:
- Most commonly missed: Business Use (categorical with low cardinality)
- Most reliably identified: Age (continuous with strong signal)

#### Cyber Security Insurance

**Ground Truth Features**: Company Size, Industry Risk, Security Score, Data Sensitivity, Business Interruption Cost (5 features)

| Model | SHAP Match % | R² | Top Feature Identified |
|-------|--------------|----|-----------------------|
| XGBoost | 96.8% | 0.884 | Security Score (correct) |
| CatBoost | 92.3% | 0.891 | Security Score (correct) |
| Random Forest | 90.8% | 0.873 | Industry Risk (correct) |

**Notable**: Highest SHAP match % across all domains - features have distinct impacts

#### Environmental Liability Insurance

**Ground Truth Features**: Industry Type, Company Size, Pollution Risk, Regulatory Compliance, Years of Operation, Incident History, Coverage Limit (7 features)

| Model | SHAP Match % | R² | Top Feature Identified |
|-------|--------------|----|-----------------------|
| XGBoost | 91.5% | 0.931 | Pollution Risk (correct) |
| Random Forest | 87.2% | 0.918 | Coverage Limit (correct) |
| CatBoost | 86.9% | 0.925 | Pollution Risk (correct) |

**Challenge**: Most features (7) makes identification harder, yet XGBoost achieves 91.5%

#### Travel Insurance

**Ground Truth Features**: Trip Cost, Trip Duration, Age, Destination, Coverage Type, Pre-existing Conditions (6 features)

| Model | SHAP Match % | R² | Top Feature Identified |
|-------|--------------|----|-----------------------|
| XGBoost | 97.8% | 0.908 | Trip Duration (correct) |
| Random Forest | 93.1% | 0.896 | Trip Cost (correct) |
| Gradient Boosting | 89.7% | 0.879 | Trip Cost (correct) |

**Note**: Highest overall performance - features have multiplicative relationships well-suited to trees

### Statistical Analysis

#### Variance Analysis

**XGBoost SHAP Match % Standard Deviation by Domain:**
- Auto Insurance: σ = 2.3%
- Cyber Security: σ = 1.8%
- Environmental: σ = 3.1%
- Travel: σ = 1.5%

**Interpretation**: Low variance indicates stable, reliable performance across iterations.

#### Feature Importance Correlation

**Correlation between SHAP Rankings and True Importance:**
- XGBoost: ρ = 0.94 (Spearman's rank correlation)
- Random Forest: ρ = 0.88
- Gradient Boosting: ρ = 0.85

**Meaning**: XGBoost's rankings closely match ground truth importance order.

## Experiment 2: Meta-Model Performance

### Objective
Build a model that predicts SHAP explanation quality from performance metrics without running SHAP analysis.

### Dataset Characteristics

- **Samples**: 4,000 (1000 per domain)
- **Features**: 14 (10 original + 4 engineered)
- **Target**: SHAP Match Percentage
- **Range**: 45.2% - 99.8%

### Feature Importance for Meta-Model

| Feature | Importance | Interpretation |
|---------|------------|----------------|
| RAE_sMAPE_interaction | 0.28 | Combined error metric most predictive |
| R² | 0.22 | Model fit quality crucial |
| Sample Size | 0.18 | Larger samples → better explanations |
| PICP | 0.15 | Prediction confidence matters |
| sMAPE | 0.12 | Symmetric error informative |
| Others | 0.05 | Minor contributors |

### Meta-Model Performance

#### Model Comparison

| Meta-Model | R² | MAE | RMSE | Training Time |
|------------|----|----|------|---------------|
| **Stacking Ensemble** | **0.79** | **2.21** | **2.79** | 45s |
| XGBoost | 0.78 | 2.24 | 2.82 | 12s |
| Random Forest | 0.77 | 2.27 | 2.86 | 18s |
| Gradient Boosting | 0.76 | 2.33 | 2.91 | 28s |

**Selected**: Stacking Ensemble (Random Forest + XGBoost → Gradient Boosting meta-learner)

#### Prediction Accuracy by SHAP Range

```
True SHAP %     Predicted SHAP %    Error
--------------------------------------------
90-100%        88-98%               ±2.3%
80-90%         77-92%               ±3.1%
70-80%         68-84%               ±4.2%
<70%           62-76%               ±5.8%
```

**Insight**: Meta-model most accurate for high-quality explanations (>90%)

### Cross-Validation Results

**5-Fold CV Performance:**
- Fold 1: R² = 0.81
- Fold 2: R² = 0.78
- Fold 3: R² = 0.77
- Fold 4: R² = 0.80
- Fold 5: R² = 0.79
- **Mean**: R² = 0.79 (σ = 0.015)

**Interpretation**: Consistent performance across folds indicates good generalization.

### Use Case: Fast Screening

**Scenario**: Determine if SHAP analysis is worthwhile before running it.

**Decision Rule**: If predicted SHAP match % > 85%, proceed with full SHAP analysis.

**Accuracy**: 91.2% correct classifications (true positives + true negatives)

**Speedup**: ~100x faster than full SHAP analysis (0.1s vs 10s per model)

## Experiment 3: End-to-End Pipeline

### Objective
Demonstrate complete workflow from data generation to report generation.

### Test Case: Travel Insurance (50,000 samples)

#### Model Performance

| Metric | Value |
|--------|-------|
| R² | 0.908 |
| MAE | 86.41 |
| RMSE | 124.41 |
| MedAE | 59.41 |
| Max Error | 1014.92 |
| sMAPE | 21.99% |
| Durbin-Watson | 1.99 |

**Assessment**: Excellent fit (R² > 0.9) with low residual autocorrelation (DW ≈ 2)

#### SHAP Analysis Results

**Top 5 Features by SHAP Importance:**

| Rank | Feature | Avg |SHAP| | True Importance |
|------|---------|-------------|-----------------|
| 1 | Trip Duration | 207.83 | ✅ Important |
| 2 | Trip Cost | 195.13 | ✅ Important |
| 3 | Age | 83.83 | ✅ Important |
| 4 | Coverage Type | 62.91 | ✅ Important |
| 5 | Destination | 39.68 | ✅ Important |

**Noise Features (Correctly Ranked Low):**
- Loyalty Program: |SHAP| = 0.87
- Travel Companion Count: |SHAP| = 2.94
- Frequent Traveler: |SHAP| = 1.12

**SHAP Match %**: 100% (all 6 true features in top 6 positions)

#### Meta-Model Prediction

**Input Metrics**: R²=0.908, MAE=86.41, Sample Size=50000, ...
**Predicted SHAP Match %**: **97.81%**
**Actual SHAP Match %**: **100%**
**Prediction Error**: -2.19% (slight underestimate)

**Interpretation**: Meta-model correctly predicted high-quality explanation

### Single Instance Explanation

**Example Customer:**
- Trip Cost: $2,617
- Trip Duration: 13 days
- Age: 70
- Destination: Asia (code 3)
- Coverage Type: Basic (code 0)

**Predicted Premium**: $212.67
**Actual Premium**: $176.91
**Difference**: $35.76 (16.8% overestimate)

**SHAP Breakdown:**
- Base value (expected): $346.48
- Trip Cost contribution: -$97.32 (decreases premium)
- Trip Duration contribution: -$104.52 (decreases premium)
- Age contribution: -$84.50 (decreases premium)
- Destination contribution: +$47.41 (increases premium)
- Coverage Type contribution: -$6.18 (decreases premium)

**Insight**: Despite high-risk age (70), short trip duration and basic coverage reduce premium substantially.

## Cross-Domain Analysis

### Generalization Performance

**Coefficient of Variation (CV) across Domains:**

```
Metric              CV      Interpretation
--------------------------------------------
SHAP Match %        2.9%    Very consistent
R²                  2.1%    Stable fit quality
Training Time       15.3%   Domain-dependent
```

**Conclusion**: Framework generalizes well - performance doesn't depend heavily on domain characteristics.

### Domain-Specific Challenges

| Domain | Challenge | Impact on SHAP Match % |
|--------|-----------|------------------------|
| Auto | High categorical cardinality | -3.2% vs average |
| Cyber | Limited samples in practice | -0.5% vs average |
| Environmental | Many interacting factors (7) | -2.1% vs average |
| Travel | Clean multiplicative structure | +2.8% vs average |

### Feature Type Analysis

**SHAP Match % by Feature Type:**

- **Continuous Numerical**: 96.8% (easiest to identify)
- **Ordinal Categorical**: 94.2% (moderate difficulty)
- **Nominal Categorical**: 91.3% (hardest due to encoding)

**Recommendation**: Continuous features provide clearest SHAP signals.

## Statistical Significance

### Hypothesis Testing

**H₀**: XGBoost SHAP match % ≤ Random Forest SHAP match %
**H₁**: XGBoost SHAP match % > Random Forest SHAP match %

**Test**: Paired t-test (500 iterations per domain)
**Result**: t = 12.47, p < 0.001
**Conclusion**: Reject H₀ - XGBoost significantly outperforms Random Forest

### Effect Size

**Cohen's d**: 0.89 (large effect size)
**Interpretation**: Practical significance, not just statistical significance

### Confidence Intervals

**95% CI for XGBoost SHAP Match % by Domain:**
- Auto: [93.1%, 95.3%]
- Cyber: [95.8%, 97.8%]
- Environmental: [89.8%, 93.2%]
- Travel: [96.9%, 98.7%]

All intervals exclude 85% threshold → consistently high performance

## Key Findings

### 1. Model Selection Findings

✅ **XGBoost is the clear winner** for insurance premium explanation (95.1% SHAP match)
✅ **Sample size matters**: 500-1000 samples optimal (beyond shows diminishing returns)
✅ **Tree ensembles >> Linear models**: 20-30% performance gap
✅ **Domain independence**: XGBoost performs consistently across all 4 insurance types

### 2. Meta-Model Findings

✅ **R² = 0.79 prediction accuracy** - viable for fast screening
✅ **Error metrics most predictive**: RAE, sMAPE, and their interaction
✅ **100x speedup**: Meta-model much faster than full SHAP
✅ **Best for high-quality cases**: Most accurate when SHAP match > 90%

### 3. Explainability Findings

✅ **SHAP effectively identifies factors**: 92-98% of true features detected
✅ **Noise features correctly ignored**: Low SHAP values for irrelevant features
✅ **Individual explanations actionable**: Waterfall plots show clear factor contributions
✅ **Multiplicative relationships captured**: Tree models handle insurance pricing structure well

### 4. Practical Implications

✅ **Regulatory oversight**: Framework can audit insurance pricing for fairness
✅ **Consumer education**: Clear explanations empower informed decisions
✅ **Insurer transparency**: Demonstrates factors without revealing exact algorithms
✅ **Bias detection**: Low SHAP values for protected attributes signal potential discrimination

## Limitations

### Identified Limitations

1. **Synthetic Data**: Real algorithms may have additional complexity
2. **Computational Cost**: 500 iterations expensive for large-scale production
3. **Categorical Encoding**: One-hot encoding may lose ordinal relationships
4. **Non-tree Models**: Framework less effective for neural network-based pricing
5. **Temporal Factors**: Current version doesn't handle time-series features

### Sensitivity Analysis

**Robustness to Hyperparameters:**
- ±10% learning rate: SHAP match % varies by ±1.2%
- ±20% n_estimators: SHAP match % varies by ±0.8%
- ±30% max_depth: SHAP match % varies by ±2.4%

**Conclusion**: Results reasonably robust to hyperparameter choices

## Future Experiments

### Planned Extensions

1. **Real-world Validation**: Partner with insurers for anonymized data testing
2. **Additional Domains**: Health, life, home, and auto-property bundled insurance
3. **Fairness Metrics**: Demographic parity, equal opportunity analysis
4. **Alternative Explainers**: Compare SHAP with LIME, Anchors, DICE
5. **Deep Learning**: Test framework with neural network premium calculators

### Research Questions

- Does framework detect discriminatory pricing (e.g., proxy variables for protected classes)?
- Can meta-model generalize to unseen insurance domains?
- How does performance change with highly correlated features?
- Can we extend to multi-output scenarios (premium + coverage recommendations)?

---

**Data Availability**: Synthetic datasets and trained models available upon request
**Reproducibility**: All experiments use `random_state=42` for reproducibility
**Code**: Available at [GitHub Repository](https://github.com/yourusername/insurance-premium-shap-analyser)

**Last Updated**: 2024
**Version**: 1.0
**Authors**: Alberto Primerano
