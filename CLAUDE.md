# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research framework for explaining insurance premium calculations using machine learning and SHAP (SHapley Additive exPlanations) analysis. The system reverse-engineers proprietary insurance pricing algorithms by training XGBoost models to mimic them, then uses SHAP to identify which factors most influence premiums.

**Key Innovation**: Includes a meta-model that predicts SHAP analysis effectiveness before running the full analysis, adding reliability estimation to the explanation framework.

## Quick Start for New Users

For portfolio reviewers or first-time users:

1. **Read First**: Start with `README.md` for project overview and `PORTFOLIO.md` for context
2. **Quick Demo**: Run `python scripts/generate_demo_results.py` (5 minutes)
3. **Documentation**: Read `docs/METHODOLOGY.md` for approach, `docs/RESULTS.md` for findings
4. **Deep Dive**: Explore experiment folders in order (1 → 2 → 3), each has its own README

## Architecture

The repository is structured around three sequential experiments:

### 1. Insurance Premium Simulators (`Calculations/`)

Simulated premium calculation functions for different insurance domains:
- `auto_insurance_premium.py` - Auto insurance with factors like age, gender, state, business use, annual kilometers
- `cybersecurity_insurance_premium.py` - Cyber insurance with company size, security score, data sensitivity
- `env_liability_insurance_premium.py` - Environmental liability with industry type, pollution risk, compliance
- `travel_insurance_premium.py` - Travel insurance with trip cost, duration, age, destination, coverage type

Each module contains:
- Premium calculation logic with domain-specific factors
- `generate_test_data(n)` function to create synthetic datasets
- `important_features` set listing which features actually influence the premium

**Common Pattern**: Each generator creates datasets with both influencing features (used in premium calculation) and non-influencing features (noise variables) to test if ML models can correctly identify important factors.

### 2. Data Preprocessing (`Calculations/utilities.py`)

Core preprocessing functions used across all experiments:

- `pre_process_data(sample_data, target_column)` - Main preprocessing pipeline:
  - Separates features (X) and target (y)
  - Drops 'Name' column and date columns
  - Splits into categorical and numerical features
  - Creates mappings for categorical → numeric transformation
  - Returns: `(X_preprocessed, y, mappings)`

- `map_dataset(df, mappings, target_column)` - Applies pre-created mappings to new data (for prediction on single instances)

- `_create_mappings_and_transform(df)` - Internal function that creates categorical → integer mappings

- `_identify_columns_containing_date(df)` - Internal helper to find and remove date columns

### 3. Three-Stage Experiment Pipeline

**Experiment 1: Model Selection** (`Experiment 1 - Determine Best Model/`)
- Tests 9 ML models (Random Forest, XGBoost, Gradient Boosting, CatBoost, ElasticNet, MLPRegressor, Ridge, ExtraTrees, HistGradientBoosting)
- Evaluates how well each model's SHAP values identify true important features
- Key Finding: XGBoost consistently outperforms others

Scripts:
- `01 - Filter Models - Collect Data.py` - Runs all models across sample sizes [100, 500, 1000, 2000], saves SHAP match percentages
- `03 - Determine Best Model - Collect Data.py` - More detailed analysis on selected models

**Experiment 2: Meta-Model Training** (`Experiment 2 - SHAP Precision ML/`)
- Trains a stacking ensemble meta-model to predict SHAP accuracy from model performance metrics
- Allows estimating explanation reliability without running expensive SHAP analysis
- Uses metrics: R², MedAE, Max Error, RAE, sMAPE, Bias, PICP, Sample Size

Scripts:
- `01 - Collect Experiment 3 Data.py` - Generates comprehensive training data with performance metrics and SHAP match percentages
- `02 - Data Analysis.ipynb` - Exploratory analysis of the meta-model training data
- `03 - Machine Learning Predictor.ipynb` - Trains stacking ensemble to predict SHAP match percentage

**Experiment 3: Report Generation** (`Experiment 3 - Generate Report/`)
- Complete end-to-end pipeline: train model → SHAP analysis → reliability prediction → report generation
- Generates prompts for LLMs to create stakeholder-specific explanations

Pipeline (`01 - Generate Pipeline.ipynb`):
1. Generate synthetic insurance data
2. Train XGBoost model
3. Evaluate model performance (R², MAE, RMSE, etc.)
4. Run SHAP analysis to identify important features
5. Use meta-model to predict SHAP reliability
6. Generate LLM prompts for regulatory reports and customer explanations

## Development Workflow

### Quick Demo (Recommended First Step)

```bash
# 5-minute demonstration of full capabilities
python scripts/generate_demo_results.py
```

### Running Experiments

**Experiment 1** (Model Comparison):
```bash
cd "Experiment 1 - Determine Best Model"
python "01 - Filter Models - Collect Data.py"
# Will prompt for output filename
# Outputs: CSV with columns [Experiment, Model, Sample Size, SHAP Match %, Top Features]
# See Experiment 1/README.md for details
```

**Experiment 2** (Meta-Model Data Collection):
```bash
cd "Experiment 2 - SHAP Precision ML"
python "01 - Collect Experiment 3 Data.py"
# Will prompt for: sample size, iterations, output filename
# Generates comprehensive metrics for meta-model training
# See Experiment 2/README.md for details
```

**Experiment 3** (Full Pipeline):
```bash
cd "Experiment 3 - Generate Report"
jupyter notebook "01 - Generate Pipeline.ipynb"
# Interactive notebook - run cells sequentially
# See Experiment 3/README.md for details
```

### Working with Notebooks

All notebooks are designed to be run sequentially cell-by-cell. Key notebooks:
- Use `sys.path.append(os.path.abspath('../Calculations'))` to import from Calculations module
- SHAP visualizations: `.summary_plot()`, `.waterfall_plot()`, `.force_plot()`
- Meta-model loading: `joblib.load('../Experiment 2 - SHAP Precision ML/saved_models/stacking_model.joblib')`

### Adding New Insurance Domains

To add a new insurance type:

1. Create `Calculations/new_insurance_premium.py` with:
   - Factor calculation functions (e.g., `age_factor()`, `risk_factor()`)
   - `calculate_premium()` function
   - `generate_test_data(n)` that returns DataFrame with 'Premium' column
   - `important_features` set listing truly influencing factors

2. Add experiment config to `EXPERIMENTS` list in experiment scripts:
```python
{
    'name': 'New Insurance Type',
    'module': 'new_insurance_premium',
    'prefix': 'new_prefix',
    'important_features': {'Feature1', 'Feature2', ...}
}
```

3. Add to `EXPERIMENTS_TO_RUN` list to include in experiments

## Repository Structure

```
.
├── Calculations/                  # Insurance premium simulators
├── Experiment 1 - Determine Best Model/    # Model comparison
├── Experiment 2 - SHAP Precision ML/       # Meta-model training
├── Experiment 3 - Generate Report/         # End-to-end pipeline
├── docs/                         # Comprehensive documentation
│   ├── METHODOLOGY.md           # Research approach
│   └── RESULTS.md               # Experimental findings
├── scripts/                      # Automation scripts
│   └── generate_demo_results.py  # Quick demo
├── examples/                     # Sample outputs (generated)
├── config.py                     # Centralized configuration
├── requirements.txt              # Python dependencies
├── CLAUDE.md                     # This file (dev guide)
├── PORTFOLIO.md                  # Project context & achievements
└── README.md                     # Main documentation
```

Each experiment folder contains its own README.md with detailed instructions.

## Key Dependencies

- `xgboost` - Primary model for premium mimicking
- `shap` - SHAP value calculation and visualization
- `sklearn` - Model training, evaluation, preprocessing
- `pandas`, `numpy` - Data manipulation
- `tqdm` - Progress bars for long-running data generation
- `catboost` - Alternative gradient boosting model
- `statsmodels` - Statistical tests (Durbin-Watson)
- `joblib` - Model serialization
- `jupyter` - Interactive notebooks

**Installation**: `pip install -r requirements.txt`

## Important Implementation Details

**SHAP Analysis Configuration**:
- Tree-based models use `shap.TreeExplainer(model, feature_perturbation='interventional')`
- Non-tree models use `shap.KernelExplainer(model.predict, X_train[:10])` with small background set
- Always use `check_additivity=False` for shap_values() to handle numerical precision issues

**Model Training**:
- Standard train/test split: 80/20, `random_state=42`
- XGBoost config: `n_estimators=100, max_depth=3, learning_rate=0.1`
- All models use `random_state=42` for reproducibility

**Performance Metrics**:
The framework tracks extensive metrics beyond standard R²/MAE:
- R², MAE, RMSE, MSE (standard regression metrics)
- MedAE (Median Absolute Error), Max Error, RAE (Relative Absolute Error)
- sMAPE (Symmetric Mean Absolute Percentage Error)
- Bias, PICP (Prediction Interval Coverage Probability), CV (Coefficient of Variation)
- Quantile Loss, Durbin-Watson statistic
- **SHAP Match Percentage**: Core metric = (|important_features ∩ top_SHAP_features|) / |important_features| × 100

**Data Generation Pattern**:
All `generate_test_data(n)` functions use `tqdm` progress bars with `leave=False` and include both:
- Influencing features: Actually used in premium calculation
- Non-influencing features: Random noise to test feature selection

**Path Management**:
Scripts in experiment folders import from Calculations using:
```python
sys.path.append(os.path.abspath('../Calculations'))
```

## Testing and Validation

The framework is self-validating:
- Ground truth important features are known (defined in each insurance module)
- SHAP match percentage measures how well the explanation identifies ground truth
- Meta-model validation: Train on comprehensive metrics, test if predicted SHAP accuracy matches actual

No traditional unit tests - validation is built into the experimental design through SHAP match percentage comparison.
