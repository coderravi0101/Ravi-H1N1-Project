"""
Configuration module for H1N1 Vaccine Prediction App
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    "model_path": MODELS_DIR / "h1n1_model.pkl",
    "cache_duration": 3600,  # 1 hour
    "version": "1.0.0"
}

# Logging configuration
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "app.log"
}

# Feature configuration
FEATURE_CONFIG = {
    "numeric_features": [
        "H1N1_Worry",
        "H1N1_Awareness",
        "Doctor_Rec_H1N1",
        "Chronic_Med_Condition",
        "Opinion_H1N1_Vacc_Effective",
        "Opinion_H1N1_Risk",
        "Opinion_H1N1_Sick_From_Vacc",
        "Health_Worker",
        "No_Of_Adults"
    ],
    "categorical_features": [
        "Age_Group",
        "Education",
        "Sex"
    ]
}

# Prediction thresholds
PREDICTION_THRESHOLDS = {
    "high_confidence": 0.75,
    "medium_confidence": 0.50,
    "low_confidence": 0.25
}

# UI Configuration
UI_CONFIG = {
    "page_title": "H1N1 Vaccine Prediction",
    "page_icon": "💉",
    "layout": "wide",
    "theme": "light"
}
