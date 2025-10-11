"""
Central Configuration File for Insurance Premium Explanation Framework

This module contains all configurable parameters for experiments, models,
and data generation processes.
"""

# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

# XGBoost Hyperparameters (optimized through grid search)
XGBOOST_CONFIG = {
    'n_estimators': 100,
    'max_depth': 3,
    'learning_rate': 0.1,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'random_state': 42
}

# Random Forest Hyperparameters
RANDOM_FOREST_CONFIG = {
    'n_estimators': 100,
    'random_state': 42
}

# Gradient Boosting Hyperparameters
GRADIENT_BOOSTING_CONFIG = {
    'n_estimators': 100,
    'random_state': 42
}

# CatBoost Hyperparameters
CATBOOST_CONFIG = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'random_state': 42,
    'verbose': 0
}

# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================

# Sample sizes for experiments
SAMPLE_SIZES = [100, 500, 1000, 2000]

# Number of iterations for statistical robustness
N_ITERATIONS = 500

# Train/test split ratio
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Target column name (consistent across all domains)
TARGET_COLUMN = 'Premium'

# =============================================================================
# INSURANCE DOMAIN CONFIGURATIONS
# =============================================================================

# List of all insurance domains
EXPERIMENTS = [
    {
        'name': 'Auto Premium',
        'module': 'auto_insurance_premium',
        'prefix': 'auto',
        'important_features': {'Age', 'Gender', 'State', 'Business Use', 'Annual kilometers'}
    },
    {
        'name': 'Cyber Security',
        'module': 'cybersecurity_insurance_premium',
        'prefix': 'cyber',
        'important_features': {
            'Company Size', 'Industry Risk', 'Security Score',
            'Data Sensitivity', 'Business Interruption Cost'
        }
    },
    {
        'name': 'Environment Liability',
        'module': 'env_liability_insurance_premium',
        'prefix': 'env_liab',
        'important_features': {
            'Industry Type', 'Company Size', 'Pollution Risk',
            'Regulatory Compliance', 'Years of Operation',
            'Incident History', 'Coverage Limit'
        }
    },
    {
        'name': 'Travel Insurance',
        'module': 'travel_insurance_premium',
        'prefix': 'travel',
        'important_features': {
            'Trip Cost', 'Trip Duration', 'Age', 'Destination',
            'Coverage Type', 'Pre-existing Conditions'
        }
    }
]

# Experiments to run (modify to run subset)
EXPERIMENTS_TO_RUN = ['Auto Premium', 'Cyber Security', 'Environment Liability', 'Travel Insurance']

# =============================================================================
# SHAP ANALYSIS CONFIGURATION
# =============================================================================

# SHAP TreeExplainer settings
SHAP_CONFIG = {
    'feature_perturbation': 'interventional',
    'check_additivity': False
}

# =============================================================================
# META-MODEL CONFIGURATION
# =============================================================================

# Features used for meta-model prediction
META_MODEL_FEATURES = [
    'R²', 'MAE', 'RMSE', 'MedAE', 'Max Error',
    'RAE', 'sMAPE', 'Bias', 'PICP', 'Sample Size'
]

# Engineered features for meta-model
ENGINEERED_FEATURES = [
    'RAE_sMAPE_interaction',
    'log_MedAE',
    'log_Max_Error',
    'Sample_Size_R2',
    'negative_r2'
]

# Meta-model stacking configuration
META_MODEL_CONFIG = {
    'cv_folds': 5,
    'random_state': 42
}

# =============================================================================
# MODEL REGISTRY
# =============================================================================

# All models available for experimentation
MODEL_REGISTRY = {
    'XGBoost': 'xgboost.XGBRegressor',
    'Random Forest': 'sklearn.ensemble.RandomForestRegressor',
    'Gradient Boosting': 'sklearn.ensemble.GradientBoostingRegressor',
    'CatBoost': 'catboost.CatBoostRegressor',
    'ElasticNet': 'sklearn.linear_model.ElasticNet',
    'Ridge': 'sklearn.linear_model.Ridge',
    'MLPRegressor': 'sklearn.neural_network.MLPRegressor',
    'ExtraTrees': 'sklearn.ensemble.ExtraTreesRegressor',
    'HistGradientBoosting': 'sklearn.ensemble.HistGradientBoostingRegressor'
}

# Tree-based models (use TreeExplainer for SHAP)
TREE_BASED_MODELS = [
    'Random Forest', 'XGBoost', 'Gradient Boosting',
    'CatBoost', 'ExtraTrees', 'HistGradientBoosting'
]

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

# Output directories
OUTPUT_DIRS = {
    'results': 'results/',
    'models': 'saved_models/',
    'figures': 'assets/figures/',
    'examples': 'examples/'
}

# File naming conventions
FILE_NAMING = {
    'experiment_1': 'model_comparison_{domain}_{timestamp}.csv',
    'experiment_2': 'meta_model_data_{domain}_{timestamp}.csv',
    'experiment_3': 'pipeline_results_{domain}_{timestamp}.csv'
}

# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================

PLOT_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8-darkgrid',
    'color_palette': 'husl'
}

# =============================================================================
# DEMO CONFIGURATION
# =============================================================================

# Default settings for quick demo
DEMO_CONFIG = {
    'domain': 'Travel Insurance',
    'sample_size': 1000,
    'iterations': 10,
    'verbose': True
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S'
}

# =============================================================================
# PERFORMANCE THRESHOLDS
# =============================================================================

# Quality thresholds for automated decision-making
QUALITY_THRESHOLDS = {
    'min_r2': 0.80,  # Minimum acceptable R² score
    'min_shap_match': 0.85,  # Minimum SHAP match percentage for "good" explanation
    'max_mae': 100,  # Maximum acceptable MAE (domain-dependent)
    'max_training_time': 300  # Maximum training time in seconds
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_model_config(model_name):
    """
    Get configuration for a specific model.

    Args:
        model_name (str): Name of the model

    Returns:
        dict: Model configuration dictionary
    """
    config_map = {
        'XGBoost': XGBOOST_CONFIG,
        'Random Forest': RANDOM_FOREST_CONFIG,
        'Gradient Boosting': GRADIENT_BOOSTING_CONFIG,
        'CatBoost': CATBOOST_CONFIG
    }
    return config_map.get(model_name, {})


def get_experiment_by_name(name):
    """
    Get experiment configuration by name.

    Args:
        name (str): Experiment name

    Returns:
        dict: Experiment configuration or None
    """
    for exp in EXPERIMENTS:
        if exp['name'] == name:
            return exp
    return None


def is_tree_based(model_name):
    """
    Check if a model is tree-based (for SHAP explainer selection).

    Args:
        model_name (str): Name of the model

    Returns:
        bool: True if tree-based, False otherwise
    """
    return model_name in TREE_BASED_MODELS


# =============================================================================
# VERSION INFO
# =============================================================================

__version__ = '1.0.0'
__author__ = 'Alberto Primerano'
__email__ = 'alberto.primerano@gmail.com'
