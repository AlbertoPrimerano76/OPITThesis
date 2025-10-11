#!/usr/bin/env python3
"""
Quick Demo Script for Insurance Premium Explanation Framework

This script runs a simplified version of the full pipeline to demonstrate
the framework's capabilities in ~5 minutes.

Usage:
    python scripts/generate_demo_results.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'Calculations'))

import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import custom modules
from Calculations.travel_insurance_premium import generate_test_data, important_features
from Calculations.utilities import pre_process_data
from config import TARGET_COLUMN, XGBOOST_CONFIG, DEMO_CONFIG

print("=" * 70)
print("Insurance Premium Explanation Framework - Quick Demo")
print("=" * 70)
print()

# =============================================================================
# Step 1: Generate Synthetic Data
# =============================================================================
print("Step 1: Generating synthetic travel insurance data...")
print(f"  Sample size: {DEMO_CONFIG['sample_size']}")

data = generate_test_data(DEMO_CONFIG['sample_size'])
print(f"  ✓ Generated {len(data)} insurance quotes")
print(f"  Features: {list(data.columns)}")
print()

# =============================================================================
# Step 2: Preprocess Data
# =============================================================================
print("Step 2: Preprocessing data...")

X_processed, y, mappings = pre_process_data(data, target_column=TARGET_COLUMN)
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

print(f"  ✓ Training samples: {len(X_train)}")
print(f"  ✓ Test samples: {len(X_test)}")
print(f"  ✓ Features after preprocessing: {list(X_processed.columns)}")
print()

# =============================================================================
# Step 3: Train XGBoost Model
# =============================================================================
print("Step 3: Training XGBoost model...")

model = XGBRegressor(**XGBOOST_CONFIG)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"  ✓ Model trained successfully")
print(f"  Performance Metrics:")
print(f"    - R² Score: {r2:.4f}")
print(f"    - MAE: {mae:.2f}")
print(f"    - RMSE: {rmse:.2f}")
print()

# =============================================================================
# Step 4: SHAP Analysis
# =============================================================================
print("Step 4: Running SHAP analysis...")

explainer = shap.TreeExplainer(model, feature_perturbation='interventional')
shap_values = explainer.shap_values(X_test, check_additivity=False)

# Calculate feature importance
shap_feature_importance = dict(zip(
    X_processed.columns,
    np.mean(np.abs(shap_values), axis=0)
))

# Sort features by importance
sorted_features = sorted(
    shap_feature_importance.items(),
    key=lambda item: item[1],
    reverse=True
)

print("  ✓ SHAP analysis complete")
print(f"  Top 5 Features by SHAP Importance:")
for i, (feature, importance) in enumerate(sorted_features[:5], 1):
    print(f"    {i}. {feature}: {importance:.2f}")
print()

# =============================================================================
# Step 5: Validation Against Ground Truth
# =============================================================================
print("Step 5: Validating SHAP results against ground truth...")

# Get top k features (k = number of important features)
k = len(important_features)
top_shap_features = set([feat for feat, _ in sorted_features[:k]])

# Calculate SHAP match percentage
matches = len(important_features.intersection(top_shap_features))
shap_match_pct = (matches / len(important_features)) * 100

print(f"  Ground Truth Important Features: {important_features}")
print(f"  Top {k} SHAP Features: {top_shap_features}")
print(f"  ✓ SHAP Match Percentage: {shap_match_pct:.1f}%")
print()

# Identify correctly identified and missed features
correctly_identified = important_features.intersection(top_shap_features)
missed_features = important_features - top_shap_features
false_positives = top_shap_features - important_features

print(f"  Correctly Identified: {correctly_identified}")
if missed_features:
    print(f"  Missed Features: {missed_features}")
if false_positives:
    print(f"  False Positives: {false_positives}")
print()

# =============================================================================
# Step 6: Single Instance Explanation
# =============================================================================
print("Step 6: Explaining a single insurance quote...")

# Select a random instance
idx = np.random.randint(0, len(X_test))
instance = X_test.iloc[[idx]]
actual_premium = y_test.iloc[idx]
predicted_premium = model.predict(instance)[0]

print(f"  Selected Quote #{idx}")
print(f"  Feature Values:")
for feat, val in instance.iloc[0].items():
    print(f"    - {feat}: {val}")
print(f"  Actual Premium: ${actual_premium:.2f}")
print(f"  Predicted Premium: ${predicted_premium:.2f}")
print(f"  Prediction Error: ${abs(actual_premium - predicted_premium):.2f}")
print()

# SHAP values for this instance
instance_shap = shap_values[idx]
base_value = explainer.expected_value

print(f"  SHAP Explanation:")
print(f"    Base value (average premium): ${base_value:.2f}")
print(f"    Feature Contributions:")
for feat, shap_val in zip(X_processed.columns, instance_shap):
    sign = "+" if shap_val >= 0 else ""
    print(f"      - {feat}: {sign}${shap_val:.2f}")
print()

# =============================================================================
# Summary
# =============================================================================
print("=" * 70)
print("Demo Complete - Summary")
print("=" * 70)
print()
print(f"✓ Model Performance: R² = {r2:.3f}, MAE = ${mae:.2f}")
print(f"✓ SHAP Explanation Quality: {shap_match_pct:.1f}% match with ground truth")
print(f"✓ Framework successfully identified {matches}/{len(important_features)} important factors")
print()
print("Key Insights:")
print("  1. XGBoost accurately mimics insurance premium calculations")
print("  2. SHAP correctly identifies truly influential pricing factors")
print("  3. Framework provides actionable explanations for individual quotes")
print()
print("Next Steps:")
print("  - Explore demo.ipynb for interactive visualizations")
print("  - Run full experiments with: cd 'Experiment 1 - Determine Best Model'")
print("  - Read docs/METHODOLOGY.md for theoretical foundation")
print("  - Read docs/RESULTS.md for comprehensive experimental results")
print()
print("=" * 70)

# =============================================================================
# Optional: Save Results
# =============================================================================
save_results = input("\nSave demo results to examples/ folder? (y/n): ").lower().strip()

if save_results == 'y':
    # Create examples directory if it doesn't exist
    examples_dir = Path(__file__).parent.parent / 'examples'
    examples_dir.mkdir(exist_ok=True)

    # Save top features
    results_df = pd.DataFrame(sorted_features, columns=['Feature', 'SHAP_Importance'])
    results_df['Is_Important'] = results_df['Feature'].isin(important_features)
    results_df.to_csv(examples_dir / 'demo_shap_results.csv', index=False)

    # Save model metrics
    metrics_df = pd.DataFrame({
        'Metric': ['R²', 'MAE', 'RMSE', 'SHAP_Match_%', 'Sample_Size'],
        'Value': [r2, mae, rmse, shap_match_pct, len(data)]
    })
    metrics_df.to_csv(examples_dir / 'demo_metrics.csv', index=False)

    print(f"✓ Results saved to {examples_dir}/")
else:
    print("Results not saved.")

print("\nThank you for trying the Insurance Premium Explanation Framework!")
