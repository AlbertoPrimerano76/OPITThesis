# Methodology

This document details the research methodology, theoretical foundation, and experimental design of the Insurance Premium Explanation Framework.

## Table of Contents

1. [Research Problem](#research-problem)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Experimental Design](#experimental-design)
4. [Data Generation Strategy](#data-generation-strategy)
5. [Model Selection Process](#model-selection-process)
6. [SHAP Analysis Approach](#shap-analysis-approach)
7. [Meta-Model Development](#meta-model-development)
8. [Validation Strategy](#validation-strategy)

## Research Problem

### Background

Insurance premium pricing algorithms are often proprietary "black boxes" that lack transparency. This creates several challenges:

- **Consumer Trust**: Customers cannot understand why they pay specific premiums
- **Regulatory Oversight**: Regulators struggle to detect discriminatory pricing
- **Market Efficiency**: Lack of transparency reduces competition and informed choice
- **Ethical Concerns**: Algorithmic pricing may perpetuate biases

### Research Questions

1. Can machine learning models accurately reverse-engineer insurance premium calculations?
2. Which ML algorithms best identify truly influential pricing factors?
3. Can we predict the reliability of explainability analysis before running it?
4. Does the approach generalize across different insurance domains?

### Novel Contribution

While existing work focuses on either **prediction** or **explanation**, this framework combines both and adds a **meta-level reliability assessment**, creating a three-tier system:

1. **Prediction Layer**: XGBoost mimics premium calculations
2. **Explanation Layer**: SHAP identifies influential factors
3. **Validation Layer**: Meta-model predicts explanation quality

## Theoretical Foundation

### SHAP (SHapley Additive exPlanations)

SHAP values are based on cooperative game theory, specifically Shapley values from coalitional game theory.

**Key Properties:**
- **Local Accuracy**: Explanation model matches original model locally
- **Missingness**: Features not used have zero attribution
- **Consistency**: If a model changes to rely more on a feature, attribution shouldn't decrease

**Why SHAP for Insurance Pricing?**
- Provides feature-level contribution to individual predictions
- Satisfies fairness axioms from game theory
- Works with complex non-linear models (tree-based ensembles)
- Generates both local (per-instance) and global (aggregate) explanations

### XGBoost for Premium Mimicking

XGBoost (Extreme Gradient Boosting) is chosen for several reasons:

1. **Non-linear Relationships**: Insurance pricing involves complex factor interactions
2. **Categorical Handling**: Naturally handles mixed data types
3. **Regularization**: Prevents overfitting with L1/L2 penalties
4. **SHAP Integration**: TreeExplainer provides exact SHAP values efficiently
5. **Performance**: Consistently outperforms other models in tabular data tasks

### Meta-Learning for Reliability Prediction

The meta-model concept is inspired by **learning to learn** paradigms:

- **Input**: Model performance metrics (R², MAE, RMSE, etc.)
- **Output**: Predicted SHAP match percentage (explanation quality)
- **Advantage**: Fast reliability assessment without expensive SHAP computation

## Experimental Design

### Three-Stage Pipeline

#### Stage 1: Model Selection
**Objective**: Identify which ML algorithm best identifies true premium factors

**Methodology:**
- Test 9 algorithms: XGBoost, Random Forest, Gradient Boosting, CatBoost, ElasticNet, Ridge, MLP, ExtraTrees, HistGradientBoosting
- Sample sizes: [100, 500, 1000, 2000]
- Iterations: 500 per configuration
- Metric: SHAP Match Percentage = |true_features ∩ top_SHAP_features| / |true_features|

**Hypothesis**: Tree-based ensembles will outperform linear and neural models due to:
- Ability to capture feature interactions
- Robustness to feature scaling
- Natural categorical variable handling

#### Stage 2: Meta-Model Training
**Objective**: Build a model that predicts SHAP explanation quality from performance metrics

**Methodology:**
- Generate comprehensive dataset with varied sample sizes and iterations
- Input features: R², MAE, RMSE, MedAE, Max Error, RAE, sMAPE, Bias, PICP, Sample Size
- Target: SHAP Match Percentage
- Model: Stacking ensemble (Random Forest + XGBoost + Ridge)
- Validation: 5-fold cross-validation

**Feature Engineering:**
- `RAE_sMAPE_interaction`: Product of relative and symmetric errors
- `log_MedAE`, `log_Max_Error`: Log-transformed error metrics
- `Sample_Size_R2`: Interaction between sample size and model fit
- `negative_r2`: Boolean flag for negative R² (severe overfitting)

**Hypothesis**: Model performance metrics contain sufficient information to predict explanation quality

#### Stage 3: Report Generation
**Objective**: Create end-to-end pipeline for stakeholder-specific explanations

**Components:**
1. Data generation (synthetic insurance cases)
2. Model training (XGBoost with optimal hyperparameters)
3. SHAP analysis (TreeExplainer with interventional perturbation)
4. Meta-model reliability prediction
5. LLM prompt generation for natural language reports

## Data Generation Strategy

### Synthetic Data Rationale

**Why Synthetic Data?**
- **Ground Truth**: We know exactly which features influence premiums
- **Controlled Experiments**: Can systematically vary factors
- **Privacy**: No real customer data exposure
- **Reproducibility**: Anyone can replicate experiments

**Limitations:**
- May not capture real-world complexity
- Cannot validate against actual insurance company behavior
- Simplified factor relationships

### Insurance Domain Simulators

Each simulator follows a consistent pattern:

```python
premium = base_rate × factor₁ × factor₂ × ... × factorₙ
```

**Auto Insurance Example:**
```python
premium = 723 × gender_factor × territory_factor × age_group_factor
          × annual_km_factor × business_use_factor
```

**Design Principles:**
1. **Influencing Features**: Actually used in premium calculation
2. **Noise Features**: Random variables not affecting premium
3. **Realistic Ranges**: Values based on real-world insurance data
4. **Factor Interactions**: Multiplicative relationships (common in insurance)

### Data Generation Process

1. **Sample Specification**: User defines sample size (n)
2. **Random Feature Generation**:
   - Influencing: Drawn from domain-appropriate distributions
   - Noise: Random categorical or continuous variables
3. **Premium Calculation**: Apply domain-specific formula
4. **Dataset Assembly**: Combine features + calculated premium
5. **Export**: Return as pandas DataFrame

## Model Selection Process

### Evaluation Metrics

**Primary Metric: SHAP Match Percentage**
```
SHAP Match % = (|Important Features ∩ Top SHAP Features| / |Important Features|) × 100
```

**Secondary Metrics:**
- **R² Score**: Model fit quality
- **MAE/RMSE**: Prediction error
- **Cross-validation R²**: Generalization ability

### Hyperparameter Tuning

**XGBoost Configuration:**
```python
{
    'n_estimators': 100,
    'max_depth': 3,
    'learning_rate': 0.1,
    'subsample': 1.0,
    'colsample_bytree': 1.0
}
```

Chosen through RandomizedSearchCV with 5-fold cross-validation.

### Train/Test Split

- **Ratio**: 80/20
- **Stratification**: Not applicable (regression task)
- **Random State**: 42 (for reproducibility)

## SHAP Analysis Approach

### TreeExplainer Configuration

```python
explainer = shap.TreeExplainer(
    model,
    feature_perturbation='interventional'
)
shap_values = explainer.shap_values(
    X_test,
    check_additivity=False
)
```

**Key Parameters:**
- `feature_perturbation='interventional'`: Uses feature distribution from training data
- `check_additivity=False`: Avoids numerical precision errors

### Feature Importance Ranking

1. Calculate absolute SHAP values: `|SHAP(fᵢ)|` for each feature
2. Average across all instances: `mean(|SHAP(fᵢ)|)`
3. Sort features by average absolute SHAP value (descending)
4. Select top k features (k = |true_important_features|)
5. Compare with ground truth using set intersection

### Visualization Methods

- **Summary Plot**: Global feature importance
- **Waterfall Plot**: Single prediction breakdown
- **Force Plot**: Individual contribution visualization
- **Dependence Plot**: Feature interaction effects

## Meta-Model Development

### Architecture

**Stacking Ensemble:**
```
Base Models:
├── Random Forest (n_estimators=300, max_depth=None)
├── XGBoost (n_estimators=500, max_depth=10)
└── Final Estimator: Gradient Boosting Regressor
```

### Training Process

1. **Data Collection**: Run Experiment 1 across all domains and sample sizes
2. **Feature Selection**: Use SelectKBest (k=5) + engineered features
3. **Model Training**: Fit stacking ensemble with 5-fold CV
4. **Validation**: Test on held-out data from new experiment runs
5. **Serialization**: Save with joblib for deployment

### Input Features (Final Selection)

- `RAE`: Relative Absolute Error
- `sMAPE`: Symmetric Mean Absolute Percentage Error
- `PICP`: Prediction Interval Coverage Probability
- `RAE_sMAPE_interaction`: Product term
- `Sample_Size_R2`: Sample size × R²

**Rationale**: These features capture both prediction accuracy and uncertainty

## Validation Strategy

### Internal Validation

**Synthetic Ground Truth:**
- Each insurance simulator explicitly defines `important_features`
- SHAP rankings compared against this ground truth
- Match percentage quantifies explanation quality

**Cross-Domain Validation:**
- Test framework across 4 insurance types
- Ensures generalizability beyond single domain

### Robustness Testing

**Sample Size Sensitivity:**
- Test with [100, 500, 1000, 2000] samples
- Expect SHAP match % to increase with sample size

**Iteration Consistency:**
- Run 500 iterations per configuration
- Statistical analysis of mean and variance

### Limitations and Threats to Validity

1. **Synthetic Data Bias**: Results may not transfer to real algorithms
2. **Domain Coverage**: Limited to 4 insurance types
3. **Feature Complexity**: Simplified factor relationships
4. **Computational Cost**: Full experiments require significant compute time
5. **Generalization**: No validation on proprietary insurance algorithms

## Reproducibility

### Code Organization
- Modular design with clear separation of concerns
- Configuration constants defined at module level
- Random seeds set for all stochastic operations

### Documentation
- Inline comments explaining non-obvious logic
- Function docstrings with parameter descriptions
- README with step-by-step reproduction instructions

### Dependencies
- `requirements.txt` with pinned versions
- Python 3.9+ compatibility
- Cross-platform support (Unix, Windows, macOS)

## Ethical Considerations

### Responsible Use

This framework should be used to:
- ✅ Increase transparency in insurance pricing
- ✅ Detect potential discrimination in algorithms
- ✅ Educate consumers about premium factors

Should NOT be used to:
- ❌ Reverse-engineer proprietary algorithms for competitive advantage
- ❌ Manipulate data to game insurance pricing
- ❌ Violate data privacy or intellectual property

### Bias Detection

Future work should incorporate:
- Fairness metrics (demographic parity, equal opportunity)
- Disparate impact analysis
- Protected attribute monitoring (age, gender, race)

## References

1. **SHAP**: Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS.
2. **XGBoost**: Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD.
3. **Shapley Values**: Shapley, L. S. (1953). A value for n-person games. Contributions to the Theory of Games.
4. **Insurance Pricing**: Frees, E. W. (2014). Regression Modeling with Actuarial and Financial Applications. Cambridge University Press.

---

**Last Updated**: 2024
**Version**: 1.0
**Maintainer**: Alberto Primerano
