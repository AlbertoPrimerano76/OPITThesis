# Experiment 1: Determine Best Model

## Purpose

This experiment evaluates 9 different machine learning algorithms to determine which best identifies truly influential insurance premium factors using SHAP analysis.

## Research Question

Which ML algorithm most accurately identifies important pricing factors across different insurance domains?

## Files

### Data Collection Scripts

- **`01 - Filter Models - Collect Data.py`**: Tests all 9 models across sample sizes [100, 500, 1000, 2000] with 500 iterations each. Outputs SHAP match percentages for initial model filtering.

- **`03 - Determine Best Model - Collect Data.py`**: More comprehensive analysis with additional metrics (15 different performance measures) for promising models identified in step 01.

### Analysis Notebooks

- **`02 - Filter Models - Result Analysis.ipynb`**: Analyzes results from script 01. Creates visualizations comparing SHAP match percentages across models and sample sizes.

- **`04 - Determine Best Model - Result Analysis.ipynb`**: Deep dive into comprehensive metrics from script 03. Statistical significance testing, effect size calculations, and final model selection.

## How to Run

### Quick Start

```bash
cd "Experiment 1 - Determine Best Model"
python "01 - Filter Models - Collect Data.py"
```

When prompted:
- **Output filename**: e.g., `filter_results`
- Script will generate `filter_results.csv` with SHAP match percentages

### Full Analysis

```bash
python "03 - Determine Best Model - Collect Data.py"
```

When prompted:
- **Output filename**: e.g., `comprehensive_results`
- Script will generate `comprehensive_results.csv` with all 15 metrics

### View Results

```bash
jupyter notebook "02 - Filter Models - Result Analysis.ipynb"
# OR
jupyter notebook "04 - Determine Best Model - Result Analysis.ipynb"
```

## Models Tested

1. **XGBoost** (Winner - 95.1% SHAP match)
2. Random Forest (89.4%)
3. CatBoost (88.7%)
4. Gradient Boosting (86.2%)
5. HistGradientBoosting (85.8%)
6. ExtraTrees (82.3%)
7. MLPRegressor (71.5%)
8. Ridge (65.2%)
9. ElasticNet (63.8%)

## Key Findings

- **XGBoost significantly outperforms all other models** (p < 0.001)
- **Sample size matters**: 500-1000 samples optimal (diminishing returns beyond)
- **Tree ensembles >> Linear models**: 20-30% performance gap
- **Results generalize** across all 4 insurance domains

## Output Format

CSV files contain:
- `Experiment`: Insurance domain name
- `Model`: ML algorithm name
- `Sample Size`: Number of training samples
- `SHAP Match %`: Percentage of true features correctly identified
- `Top Features`: Ranked list of SHAP-identified features
- Additional metrics (R², MAE, RMSE, etc.) in comprehensive version

## Computational Requirements

- **Time**: ~6-8 hours for full run (all models, all domains)
- **Memory**: ~4GB RAM recommended
- **Storage**: ~500MB for output files

## Expected Results

XGBoost should achieve:
- Auto Insurance: ~94% SHAP match
- Cyber Security: ~97% SHAP match
- Environmental: ~92% SHAP match
- Travel: ~98% SHAP match

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'auto_insurance_premium'`
**Solution**: Run from project root or ensure `sys.path` includes `Calculations/`

**Issue**: Script runs slowly
**Solution**: Reduce `N_ITERATIONS` or `SAMPLE_SIZES` in script for faster testing

**Issue**: Memory errors
**Solution**: Process one domain at a time by modifying `EXPERIMENTS_TO_RUN`

## Next Steps

After completing this experiment:
1. Review analysis notebooks to confirm XGBoost is best
2. Proceed to **Experiment 2** to train meta-model
3. Use XGBoost configuration in **Experiment 3** for report generation

## References

- See `docs/METHODOLOGY.md` for theoretical foundation
- See `docs/RESULTS.md` for comprehensive results and statistical analysis
