# Experiment 3: Generate Report Pipeline

## Purpose

Demonstrate end-to-end pipeline from data generation to stakeholder-specific report generation, integrating all components from Experiments 1 and 2.

## Research Question

Can we create a complete workflow that:
1. Trains an accurate premium prediction model
2. Generates SHAP explanations
3. Validates explanation quality with meta-model
4. Produces natural language reports for different stakeholders?

## Files

- **`01 - Generate Pipeline.ipynb`**: Complete interactive pipeline showcasing all framework capabilities

## Pipeline Stages

### Stage 1: Data Generation
- Generate synthetic insurance data (default: travel insurance)
- Configurable sample size (demo: 50,000 quotes)
- Includes both influencing and noise features

### Stage 2: Model Training
- Train XGBoost model with optimized hyperparameters
- Calculate comprehensive performance metrics (15 metrics)
- Evaluate prediction accuracy (R², MAE, RMSE, etc.)

### Stage 3: SHAP Analysis
- Run TreeExplainer on test set
- Calculate global feature importance
- Generate individual quote explanations
- Create visualizations (summary plot, waterfall plot, force plot)

### Stage 4: Meta-Model Validation
- Load trained meta-model from Experiment 2
- Predict SHAP explanation quality
- Compare prediction with actual SHAP match %
- Assess reliability of explanations

### Stage 5: Report Generation
- Generate LLM prompts for:
  - **Regulatory reports**: Technical details, fairness analysis
  - **Consumer explanations**: Plain language, actionable insights
- Include SHAP visualizations and confidence scores

## How to Run

### Prerequisites

Ensure you've completed:
- ✅ Experiment 1 (to know XGBoost is best model)
- ✅ Experiment 2 (to have trained meta-model in `saved_models/`)

### Launch Notebook

```bash
cd "Experiment 3 - Generate Report"
jupyter notebook "01 - Generate Pipeline.ipynb"
```

### Run Cells Sequentially

1. **Cell 1-2**: Imports and SHAP initialization
2. **Cell 3-10**: Generate data and train model
3. **Cell 11-14**: SHAP analysis and visualization
4. **Cell 15-18**: Meta-model prediction
5. **Cell 19**: Generate regulatory report prompt
6. **Cell 20-27**: Single instance explanation for consumer

## Example Output

### Model Performance (Travel Insurance, 50k samples)

```
R² Score:              0.908
MAE:                   $86.41
RMSE:                  $124.41
Predicted SHAP Match:  97.8%
Actual SHAP Match:     100%
```

### Top 5 SHAP Features

```
1. Trip Duration:    207.83 (important ✓)
2. Trip Cost:        195.13 (important ✓)
3. Age:              83.83  (important ✓)
4. Coverage Type:    62.91  (important ✓)
5. Destination:      39.68  (important ✓)
```

### Single Quote Explanation

```
Customer: Age 70, Trip Cost $2,617, 13 days, Asia, Basic Coverage
Predicted Premium: $212.67

SHAP Breakdown:
- Base value:        $346.48
- Trip Duration:     -$104.52 (reduces premium)
- Age:               -$84.50  (reduces premium)
- Trip Cost:         -$97.32  (reduces premium)
- Destination:       +$47.41  (increases premium)
- Coverage Type:     -$6.18   (reduces premium)
```

## Visualizations Generated

### Global Explanations
- **SHAP Summary Plot**: Feature importance across all quotes
  - Dots represent individual instances
  - Color shows feature value (red=high, blue=low)
  - X-axis shows SHAP impact on premium

### Local Explanations
- **Waterfall Plot**: Single quote breakdown
  - Shows how each feature pushes premium up/down
  - Starts from base value, ends at final prediction

- **Force Plot**: Interactive explanation
  - Red arrows: Features increasing premium
  - Blue arrows: Features decreasing premium
  - Hover for exact SHAP values

## Report Templates

### Regulatory Report (Technical)

Includes:
- Executive summary of model performance
- Explanation of top 5 features with fairness considerations
- Analysis of accuracy and reliability metrics
- Comparison with meta-model predictions
- Recommendations for regulatory oversight
- Ethical considerations and risks

### Consumer Report (Plain Language)

Includes:
- Introduction to premium calculation
- Summary of customer's risk factors
- Comparison: actual vs predicted premium
- SHAP explanation in accessible language
- Recommendations for reducing premium
- Conclusion and next steps

## Customization Options

### Change Insurance Domain

```python
# In notebook cell, modify:
from Calculations.auto_insurance_premium import generate_test_data, important_features
# Instead of travel_insurance_premium
```

### Adjust Sample Size

```python
# Generate more/fewer quotes
test_data = generate_test_data(10000)  # Instead of 50000
```

### Modify Visualizations

```python
# SHAP summary plot with different options
shap.summary_plot(shap_values, X_test, max_display=10, plot_type='violin')
```

## Integration with LLMs

The generated prompts are designed for:
- **ChatGPT/GPT-4**: Paste prompt for detailed report
- **Claude**: Paste prompt for regulatory analysis
- **Local LLMs**: Modify prompt format as needed

Example workflow:
1. Run notebook to generate metrics and SHAP values
2. Copy generated prompt from output
3. Paste into LLM with appropriate role (regulator, consumer, insurer)
4. Review and refine generated report
5. Include SHAP visualizations as figures

## Use Cases

### For Insurers
- Transparency reports for customers
- Regulatory compliance documentation
- Internal model auditing
- Customer service explanations

### For Regulators
- Algorithmic pricing audits
- Bias detection in premium calculations
- Fairness assessments
- Market surveillance

### For Consumers
- Understanding personal premiums
- Identifying ways to reduce costs
- Challenging potentially unfair pricing
- Comparing across insurers

## Computational Requirements

- **Time**: 5-10 minutes for full pipeline
- **Memory**: 4GB RAM (for 50k sample size)
- **Storage**: ~100MB for notebook with outputs

## Expected Outputs

### Model Metrics
- R² > 0.90 (excellent fit)
- SHAP Match > 95% (high quality explanations)
- Meta-model prediction within ±3% of actual

### Visualizations
- Clear SHAP summary plot showing feature importance
- Waterfall plot with sensible factor contributions
- Force plot highlighting key drivers

## Troubleshooting

**Issue**: `FileNotFoundError: saved_models/stacking_model.joblib`
**Solution**: Complete Experiment 2 first to train meta-model

**Issue**: SHAP visualizations not displaying
**Solution**: Run `shap.initjs()` in early cell

**Issue**: Memory error with large sample sizes
**Solution**: Reduce sample size or increase system RAM

**Issue**: Meta-model prediction far from actual
**Solution**: Check if model performance is within training distribution

## Next Steps

After completing this experiment:
1. Test with different insurance domains
2. Experiment with LLM report generation
3. Customize visualizations for specific stakeholders
4. Consider deploying as web application (Streamlit)

## Extending the Pipeline

### Add Fairness Analysis

```python
# Check if protected attributes have high SHAP values
protected_attrs = ['Age', 'Gender']
for attr in protected_attrs:
    if attr in top_shap_features:
        print(f"Warning: Protected attribute {attr} influential in pricing")
```

### Batch Processing

```python
# Explain multiple quotes at once
for idx in range(len(X_test)):
    instance = X_test.iloc[[idx]]
    explanation = generate_explanation(instance)
    save_to_database(explanation)
```

### Real-time API

```python
# Flask endpoint for live quote explanations
@app.route('/explain', methods=['POST'])
def explain_premium():
    quote = request.json
    prediction = model.predict(quote)
    shap_values = explainer.shap_values(quote)
    return jsonify({
        'premium': prediction,
        'explanation': shap_values
    })
```

## References

- See `docs/METHODOLOGY.md` for complete pipeline details
- See `docs/RESULTS.md` section "Experiment 3" for example outputs
- See `PORTFOLIO.md` for real-world applications

## Output Examples

The notebook generates example files that can be saved:
- `regulatory_report_prompt.txt`: LLM prompt for technical report
- `consumer_explanation_prompt.txt`: LLM prompt for plain language
- `shap_summary.png`: Feature importance visualization
- `waterfall_example.png`: Single quote explanation

These can be found in the notebook outputs or exported to `examples/` directory.
