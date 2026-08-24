"""
Utility functions for H1N1 Vaccine Prediction App
"""

import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate and process input data for predictions."""
    
    @staticmethod
    def validate_ranges(data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate that all numeric inputs are within acceptable ranges.
        
        Args:
            data: Dictionary of input values
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        validations = [
            (0 <= data.get('H1N1_Worry', 0) <= 3, "H1N1 Worry Level must be 0-3"),
            (0 <= data.get('H1N1_Awareness', 0) <= 2, "H1N1 Awareness Level must be 0-2"),
            (1 <= data.get('Opinion_H1N1_Vacc_Effective', 1) <= 5, "Opinion values must be 1-5"),
            (1 <= data.get('Opinion_H1N1_Risk', 1) <= 5, "Opinion values must be 1-5"),
            (1 <= data.get('Opinion_H1N1_Sick_From_Vacc', 1) <= 5, "Opinion values must be 1-5"),
            (0 <= data.get('No_Of_Adults', 0) <= 3, "Number of adults must be 0-3"),
            (data.get('Doctor_Rec_H1N1') in [0, 1], "Doctor recommendation must be 0 or 1"),
            (data.get('Chronic_Med_Condition') in [0, 1], "Medical condition must be 0 or 1"),
            (data.get('Health_Worker') in [0, 1], "Healthcare worker must be 0 or 1"),
        ]
        
        for condition, message in validations:
            if not condition:
                return False, message
        
        return True, ""
    
    @staticmethod
    def validate_categories(data: Dict[str, Any], 
                           age_groups: list, 
                           education_levels: list,
                           genders: list) -> Tuple[bool, str]:
        """
        Validate categorical inputs against allowed values.
        
        Args:
            data: Dictionary of input values
            age_groups: List of valid age groups
            education_levels: List of valid education levels
            genders: List of valid genders
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if data.get('Age_Group') not in age_groups:
            return False, "Invalid age group selected"
        if data.get('Education') not in education_levels:
            return False, "Invalid education level selected"
        if data.get('Sex') not in genders:
            return False, "Invalid gender selected"
        
        return True, ""


class FeatureEncoder:
    """Encode categorical features for model prediction."""
    
    ENCODINGS = {
        'Age_Group': {
            "18 - 34 Years": 0,
            "35 - 44 Years": 1,
            "45 - 54 Years": 2,
            "55 - 64 Years": 3,
            "65+ Years": 4
        },
        'Education': {
            "< 12 Years": 0,
            "12 Years": 1,
            "Some College": 2,
            "College Graduate": 3
        },
        'Sex': {
            "Female": 0,
            "Male": 1
        }
    }
    
    @classmethod
    def encode(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features in the dataframe.
        
        Args:
            data: Input DataFrame with categorical columns
            
        Returns:
            DataFrame with encoded features
        """
        data_encoded = data.copy()
        
        for column, mapping in cls.ENCODINGS.items():
            if column in data_encoded.columns:
                data_encoded[column] = data_encoded[column].map(mapping)
        
        return data_encoded
    
    @classmethod
    def decode(cls, encoded_value: int, column: str) -> str:
        """
        Decode numeric value back to categorical label.
        
        Args:
            encoded_value: Numeric encoded value
            column: Column name
            
        Returns:
            Original categorical label
        """
        mapping = cls.ENCODINGS.get(column, {})
        reverse_mapping = {v: k for k, v in mapping.items()}
        return reverse_mapping.get(encoded_value, "Unknown")


class PredictionInterpreter:
    """Interpret and explain prediction results."""
    
    @staticmethod
    def get_confidence_level(probability: float) -> str:
        """
        Categorize confidence level based on probability.
        
        Args:
            probability: Prediction probability (0-100)
            
        Returns:
            Confidence level string
        """
        if probability >= 75:
            return "Very High"
        elif probability >= 60:
            return "High"
        elif probability >= 50:
            return "Medium"
        elif probability >= 35:
            return "Low"
        else:
            return "Very Low"
    
    @staticmethod
    def get_recommendation(prediction: int, probability: float) -> str:
        """
        Generate actionable recommendation based on prediction.
        
        Args:
            prediction: Binary prediction (0 or 1)
            probability: Prediction probability (0-100)
            
        Returns:
            Recommendation text
        """
        if prediction == 1:
            if probability >= 85:
                return "Strong candidate for vaccination campaign targeting"
            else:
                return "Likely to receive vaccine with proper education"
        else:
            if probability <= 30:
                return "Requires targeted intervention and education"
            else:
                return "May need additional information about vaccine benefits"
    
    @staticmethod
    def get_key_factors(data: Dict[str, Any]) -> Dict[str, str]:
        """
        Identify key factors influencing the prediction.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Dictionary of key factors
        """
        factors = {}
        
        if data.get('Doctor_Rec_H1N1') == 1:
            factors['Doctor Recommendation'] = "✅ Positive influence"
        
        if data.get('Opinion_H1N1_Vacc_Effective') >= 4:
            factors['Vaccine Effectiveness'] = "✅ Positive influence"
        
        if data.get('H1N1_Worry') >= 2:
            factors['Health Concern'] = "✅ Positive influence"
        
        if data.get('Health_Worker') == 1:
            factors['Healthcare Professional'] = "✅ Positive influence"
        
        if data.get('Opinion_H1N1_Sick_From_Vacc') >= 4:
            factors['Vaccine Safety Concern'] = "⚠️ Negative influence"
        
        return factors


class ModelMonitor:
    """Monitor and log model predictions for audit trail."""
    
    @staticmethod
    def log_prediction(input_data: Dict[str, Any], 
                      prediction: int, 
                      probability: float) -> None:
        """
        Log prediction details for monitoring and audit.
        
        Args:
            input_data: Input features
            prediction: Prediction result
            probability: Prediction probability
        """
        logger.info(
            f"Prediction: {prediction} | Probability: {probability:.2f}% | "
            f"Age: {input_data.get('Age_Group')} | Education: {input_data.get('Education')}"
        )


# Utility functions for common operations
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default on division by zero."""
    return numerator / denominator if denominator != 0 else default


def normalize_probability(value: float) -> float:
    """Normalize value to probability range (0-100)."""
    return max(0, min(100, value))
