# Insurance Premium Explanation Framework

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![SHAP](https://img.shields.io/badge/explainability-SHAP-orange)
![Status](https://img.shields.io/badge/status-active-success)

## Overview

This repository contains a research framework designed to **explain insurance premium calculations without direct access to proprietary algorithms**. By combining machine learning with explainable AI techniques, this project addresses the critical need for transparency in the insurance industry.

The framework reverse-engineers insurance pricing by training XGBoost models to mimic premium calculations, then uses SHAP (SHapley Additive exPlanations) to identify which factors most influence premiums. A unique **meta-model** predicts the reliability of SHAP explanations before running expensive analyses.

### Key Innovation

Unlike traditional ML approaches that simply predict premiums, this framework:
- ✅ Identifies which factors are truly important (not just correlated)
- ✅ Validates explanation quality through meta-model predictions
- ✅ Works across multiple insurance domains (auto, cyber, environmental, travel)
- ✅ Generates stakeholder-specific reports for regulators and consumers

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/AlbertoPrimerano76/insurance-premium-shap-analyser.git
cd insurance-premium-shap-analyser

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Demo (5 minutes)

```bash
# Quick demo with travel insurance
python scripts/generate_demo_results.py
```

Or explore the interactive notebook:
```bash
jupyter notebook demo.ipynb
```

**Expected Output**: Model performance metrics, SHAP feature importance rankings, and explanation reliability scores showing ~90% accuracy in identifying true premium factors.

## Key Results

| Insurance Domain | Model | SHAP Match % | R² Score |
|-----------------|-------|--------------|----------|
| Auto Insurance | XGBoost | 94.2% | 0.91 |
| Cyber Security | XGBoost | 96.8% | 0.88 |
| Environmental Liability | XGBoost | 91.5% | 0.93 |
| Travel Insurance | XGBoost | 97.8% | 0.91 |

**Key Finding**: XGBoost consistently outperforms other ML models (Random Forest, Gradient Boosting, CatBoost, etc.) in both premium prediction accuracy and feature importance identification.

The meta-model achieves **R² = 0.79** in predicting SHAP explanation quality from performance metrics alone, enabling fast reliability assessment.

## Repository Structure

```
.
├── Calculations/                  # Insurance premium simulators
│   ├── auto_insurance_premium.py
│   ├── cybersecurity_insurance_premium.py
│   ├── env_liability_insurance_premium.py
│   ├── travel_insurance_premium.py
│   └── utilities.py              # Data preprocessing utilities
│
├── Experiment 1 - Determine Best Model/
│   ├── 01 - Filter Models - Collect Data.py
│   ├── 02 - Filter Models - Result Analysis.ipynb
│   ├── 03 - Determine Best Model - Collect Data.py
│   └── 04 - Determine Best Model - Result Analysis.ipynb
│
├── Experiment 2 - SHAP Precision ML/
│   ├── 01 - Collect Experiment 3 Data.py
│   ├── 02 - Data Analysis.ipynb
│   └── 03 - Machine Learning Predictor.ipynb
│
├── Experiment 3 - Generate Report/
│   └── 01 - Generate Pipeline.ipynb
│
├── docs/                         # Detailed documentation
│   ├── METHODOLOGY.md
│   └── RESULTS.md
│
├── scripts/                      # Automation scripts
│   └── generate_demo_results.py
│
├── examples/                     # Sample outputs
│
├── demo.ipynb                    # Quick start notebook
├── config.py                     # Centralized configuration
├── requirements.txt
└── README.md
```

## Core Components

### 1. Insurance Premium Simulators (`Calculations/`)

Domain-specific modules simulating real-world insurance pricing:
- **Auto Insurance**: Factors include age, gender, state, business use, annual kilometers
- **Cyber Security**: Company size, security score, data sensitivity, breach history
- **Environmental Liability**: Industry type, pollution risk, regulatory compliance
- **Travel Insurance**: Trip cost, duration, age, destination, coverage type

Each simulator generates synthetic datasets with both **influencing features** (used in calculations) and **noise features** (not used) to test if ML models correctly identify important factors.

### 2. Three-Stage Experimental Pipeline

#### Experiment 1: Model Selection
Tests 9 ML algorithms across multiple sample sizes to determine which best identifies true premium factors using SHAP analysis. XGBoost emerges as the clear winner.

#### Experiment 2: Meta-Model Training
Trains a stacking ensemble that predicts SHAP explanation quality from model performance metrics (R², MAE, RMSE, etc.), enabling fast reliability assessment without running expensive SHAP calculations.

#### Experiment 3: Report Generation
End-to-end pipeline: data generation → model training → SHAP analysis → reliability prediction → automated report generation for regulators and consumers.

## Usage Examples

### Running Full Experiments

```bash
# Experiment 1: Compare all models
cd "Experiment 1 - Determine Best Model"
python "01 - Filter Models - Collect Data.py"
# Enter filename when prompted: model_comparison_results

# Experiment 2: Train meta-model
cd "../Experiment 2 - SHAP Precision ML"
python "01 - Collect Experiment 3 Data.py"
# Sample size: 1000, Iterations: 100, Filename: meta_model_data

# Experiment 3: Generate reports
cd "../Experiment 3 - Generate Report"
jupyter notebook "01 - Generate Pipeline.ipynb"
```

### Adding a New Insurance Domain

```python
# Create Calculations/health_insurance_premium.py
def calculate_premium(age, bmi, smoking_status, ...):
    base_rate = 500
    return base_rate * age_factor(age) * bmi_factor(bmi) * ...

def generate_test_data(n):
    # Generate synthetic data
    ...
    return pd.DataFrame(data, columns=[...])

important_features = {'age', 'bmi', 'smoking_status', ...}
```

Then add to experiment configuration:
```python
EXPERIMENTS.append({
    'name': 'Health Insurance',
    'module': 'health_insurance_premium',
    'important_features': important_features
})
```

## Technical Highlights

### Skills Demonstrated
- **Machine Learning**: XGBoost, ensemble methods, hyperparameter tuning, cross-validation
- **Explainable AI**: SHAP analysis, feature importance interpretation
- **Meta-Learning**: Building models that predict model behavior
- **Research Methodology**: Controlled experiments, reproducibility, systematic validation
- **Domain Knowledge**: Multi-domain insurance pricing understanding
- **Software Engineering**: Modular design, documentation, version control

### Novel Contributions
1. **Meta-model for SHAP reliability**: First framework to predict explanation quality before analysis
2. **Cross-domain validation**: Demonstrates generalizability across 4 insurance types
3. **Ground truth comparison**: Uses synthetic data with known important features for validation

## Documentation

- **[METHODOLOGY.md](docs/METHODOLOGY.md)**: Detailed research approach and theoretical foundation
- **[RESULTS.md](docs/RESULTS.md)**: Comprehensive experimental results with visualizations
- **[CLAUDE.md](CLAUDE.md)**: Developer guide for code navigation and architecture
- **[PORTFOLIO.md](PORTFOLIO.md)**: Project context, challenges, and technical achievements

## Limitations and Future Work

- **Synthetic Data**: Current implementation uses simulated premiums. Validation with real-world data needed.
- **Regulatory Compliance**: Framework requires adaptation to meet specific regulatory requirements
- **Real-time Analysis**: Current pipeline not optimized for production deployment

### Planned Enhancements
- [ ] Integration with real insurance datasets (with privacy protections)
- [ ] Additional insurance domains (health, life, property)
- [ ] Web interface for interactive exploration
- [ ] Production-ready API for regulatory compliance tools
- [ ] Bias detection and fairness analysis

## Results

Our findings reveal that:
- **XGBoost achieves 90%+ accuracy** in mimicking insurance premium algorithms
- **SHAP correctly identifies 92-98%** of truly influential pricing factors
- **Meta-model predictions correlate 0.79 (R²)** with actual SHAP performance
- Framework demonstrates **robust cross-domain performance**

These results provide valuable insights for consumers (understanding premiums), regulators (oversight), and insurers (transparency).

## Contributing

Contributions are welcome! Areas of interest:
- Additional insurance domain implementations
- Real-world dataset integration (with privacy compliance)
- Alternative explainability methods (LIME, Anchors)
- Performance optimizations

## License

Apache License 2.0

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{primerano2024insurance,
  author = {Primerano, Alberto},
  title = {Insurance Premium Explanation Framework},
  year = {2024},
  url = {https://github.com/AlbertoPrimerano76/insurance-premium-shap-analyser}
}
```

## Contact

**Alberto Primerano**
📧 alberto.primerano@gmail.com
🔗 [LinkedIn](https://linkedin.com/in/albertoprimerano) | [GitHub](https://github.com/AlbertoPrimerano76)

---

⭐ **Star this repo if you find it useful!** Feedback and contributions are greatly appreciated.
