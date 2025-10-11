# Example Outputs

This directory contains sample outputs from running the framework.

## Generated Files

When you run `scripts/generate_demo_results.py`, it will optionally save results here:

- **`demo_shap_results.csv`**: SHAP feature importance rankings with ground truth labels
- **`demo_metrics.csv`**: Model performance metrics (R², MAE, RMSE, SHAP match %)

## Sample Data Format

### demo_shap_results.csv
```csv
Feature,SHAP_Importance,Is_Important
Trip Duration,207.83,True
Trip Cost,195.13,True
Age,83.83,True
Coverage Type,62.91,True
Destination,39.68,True
Loyalty Program,0.87,False
...
```

### demo_metrics.csv
```csv
Metric,Value
R²,0.908
MAE,86.41
RMSE,124.41
SHAP_Match_%,97.8
Sample_Size,1000
```

## Visualization Assets

SHAP visualizations (plots) should be saved to `assets/figures/` directory for inclusion in documentation or presentations.

## Adding Your Own Examples

Feel free to add:
- CSV outputs from experiments
- SHAP visualization images (PNG/SVG)
- Report templates
- LLM-generated explanations

**Note**: CSV files in this directory are excluded from `.gitignore` so examples can be committed to the repository.
