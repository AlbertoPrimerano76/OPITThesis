# Experiment 2: SHAP Precision Meta-Model

## Purpose

Train a meta-model that predicts SHAP explanation quality from standard ML performance metrics without running expensive SHAP analysis.

## Research Question

Can we predict how well SHAP will identify important features by only looking at model performance metrics (R², MAE, etc.)?

## Key Innovation

This meta-learning approach enables **100x speedup** in assessing explanation quality—predict in 0.1s instead of running full SHAP analysis in 10s.

## Files

### Data Collection

- **`01 - Collect Experiment 3 Data.py`**: Runs XGBoost across varied sample sizes and domains, collecting both performance metrics AND SHAP match percentages for meta-model training.

### Analysis & Training

- **`02 - Data Analysis.ipynb`**: Exploratory data analysis of meta-model training data. Identifies correlations between metrics and SHAP quality.

- **`03 - Machine Learning Predictor.ipynb`**: Trains and evaluates multiple meta-models. Tests Linear Regression, Random Forest, XGBoost, and Stacking Ensemble. Saves best model to `saved_models/`.

### Output Files

- **`Experiment_3_data.csv`**: Raw data from all experiments
- **`Experiment_3_filtered_data.csv`**: Cleaned data used for training
- **`saved_models/stacking_model.joblib`**: Trained meta-model (used in Experiment 3)

## How to Run

### Step 1: Generate Training Data

```bash
cd "Experiment 2 - SHAP Precision ML"
python "01 - Collect Experiment 3 Data.py"
```

When prompted:
- **Sample size**: e.g., `1000`
- **Number of iterations**: e.g., `100` (500 for publication-quality)
- **Output filename**: e.g., `meta_training_data`

**Note**: This is the most computationally intensive step. Use smaller iterations for testing.

### Step 2: Explore Data

```bash
jupyter notebook "02 - Data Analysis.ipynb"
```

Examine:
- Feature distributions
- Correlations with SHAP match %
- Outlier detection
- Sample size effects

### Step 3: Train Meta-Model

```bash
jupyter notebook "03 - Machine Learning Predictor.ipynb"
```

This notebook:
1. Tests 4 different meta-model architectures
2. Performs feature engineering
3. Tunes hyperparameters via RandomizedSearchCV
4. Trains final stacking ensemble
5. Saves best model to `saved_models/`

## Meta-Model Architecture

### Input Features (10)
- R²: Model fit quality
- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- MedAE: Median Absolute Error
- Max Error: Largest prediction error
- RAE: Relative Absolute Error
- sMAPE: Symmetric Mean Absolute Percentage Error
- Bias: Average prediction bias
- PICP: Prediction Interval Coverage Probability
- Sample Size: Training data size

### Engineered Features (5)
- `RAE_sMAPE_interaction`: Product of error metrics
- `log_MedAE`: Log-transformed median error
- `log_Max_Error`: Log-transformed max error
- `Sample_Size_R2`: Interaction term
- `negative_r2`: Boolean flag for overfitting

### Model: Stacking Ensemble
```
Base Models:
├── Random Forest (n_estimators=300, max_depth=None)
├── XGBoost (n_estimators=500, max_depth=10)
└── Final Estimator: Gradient Boosting Regressor

Performance: R² = 0.79, MAE = 2.21
```

## Key Findings

### Performance by Metric

| Metric | Importance | Interpretation |
|--------|------------|----------------|
| RAE_sMAPE_interaction | 0.28 | Combined error most predictive |
| R² | 0.22 | Model fit crucial |
| Sample Size | 0.18 | Larger samples → better explanations |
| PICP | 0.15 | Confidence matters |
| sMAPE | 0.12 | Symmetric error informative |

### Prediction Accuracy

**Meta-model is most accurate for high-quality explanations:**
- SHAP Match 90-100%: Error ±2.3%
- SHAP Match 80-90%: Error ±3.1%
- SHAP Match 70-80%: Error ±4.2%
- SHAP Match <70%: Error ±5.8%

### Use Case Example

```python
# Fast screening: Should we run full SHAP analysis?
if predicted_shap_match_pct > 85:
    run_full_shap_analysis()
else:
    try_different_model_or_more_data()
```

## Computational Requirements

### Data Generation (Step 1)
- **Time**: 4-6 hours for 100 iterations
- **Memory**: 4GB RAM
- **Output**: ~200MB CSV file

### Meta-Model Training (Step 3)
- **Time**: 30-45 minutes for hyperparameter tuning
- **Memory**: 2GB RAM
- **Output**: ~5MB model file

## Expected Results

Meta-model should achieve:
- R² = 0.75-0.80 (predicting SHAP match %)
- MAE = 2-3 percentage points
- 5-fold CV consistency (σ < 0.02)

## Cross-Validation Results

```
Fold 1: R² = 0.81
Fold 2: R² = 0.78
Fold 3: R² = 0.77
Fold 4: R² = 0.80
Fold 5: R² = 0.79
────────────────────
Mean:   R² = 0.79 (σ = 0.015)
```

## Loading the Trained Model

```python
import joblib

# Load meta-model
meta_model = joblib.load('saved_models/stacking_model.joblib')

# Prepare input (model performance metrics)
metrics_df = pd.DataFrame([{
    'R²': 0.91,
    'MAE': 86.41,
    'RMSE': 124.41,
    'Sample Size': 50000,
    # ... other metrics
}])

# Predict SHAP match percentage
predicted_shap_quality = meta_model.predict(metrics_df)
print(f"Expected SHAP Match %: {predicted_shap_quality[0]:.1f}%")
```

## Troubleshooting

**Issue**: Low meta-model R² (<0.70)
**Solution**: Increase training data (more iterations in Step 1)

**Issue**: `FileNotFoundError: saved_models/stacking_model.joblib`
**Solution**: Run notebook 03 to completion to generate model file

**Issue**: High prediction variance
**Solution**: Check for outliers in training data, consider more feature engineering

## Next Steps

After completing this experiment:
1. Verify meta-model achieves R² > 0.75
2. Check saved model exists in `saved_models/`
3. Proceed to **Experiment 3** which uses this meta-model

## Practical Applications

- **Fast Model Screening**: Test 100 models, only run SHAP on top 10
- **Hyperparameter Optimization**: Choose configs likely to yield good explanations
- **Sample Size Planning**: Predict if more data will improve explainability
- **Quality Assurance**: Flag models with predicted low SHAP quality

## References

- See `docs/METHODOLOGY.md` section "Meta-Model Development" for details
- See `docs/RESULTS.md` section "Experiment 2" for comprehensive results
