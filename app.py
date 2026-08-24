"""
H1N1 Vaccine Prediction Application
Production-ready Streamlit application for H1N1 vaccine usage prediction.
"""

import logging
import os
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== Constants ====================
MODEL_PATH = Path("models/h1n1_model.pkl")
CACHE_DURATION = 3600  # 1 hour
PAGE_CONFIG = {
    "page_title": "H1N1 Vaccine Prediction",
    "page_icon": "💉",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Feature Ranges
FEATURE_RANGES = {
    'H1N1_Worry': (0, 3),
    'H1N1_Awareness': (0, 2),
    'Opinion_H1N1_Vacc_Effective': (1, 5),
    'Opinion_H1N1_Risk': (1, 5),
    'Opinion_H1N1_Sick_From_Vacc': (1, 5),
    'No_Of_Adults': (0, 3)
}

AGE_GROUPS = ["18 - 34 Years", "35 - 44 Years", "45 - 54 Years", "55 - 64 Years", "65+ Years"]
EDUCATION_LEVELS = ["< 12 Years", "12 Years", "Some College", "College Graduate"]
GENDER_OPTIONS = ["Female", "Male"]

# ==================== Model Loading ====================
@st.cache_resource(ttl=CACHE_DURATION)
def load_model() -> Optional[Pipeline]:
    """
    Load the trained model from disk with error handling.
    
    Returns:
        Optional[Pipeline]: Loaded model or None if not found
    """
    try:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            logger.info("Model loaded successfully")
            return model
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}")
            return None
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"❌ Error loading model: {str(e)}")
        return None


# ==================== Data Validation ====================
def validate_input(data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate input data before prediction.
    
    Args:
        data: Input dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check numeric ranges
        if not (0 <= data['H1N1_Worry'] <= 3):
            return False, "H1N1 Worry Level must be between 0-3"
        if not (0 <= data['H1N1_Awareness'] <= 2):
            return False, "H1N1 Awareness Level must be between 0-2"
        if not (1 <= data['Opinion_H1N1_Vacc_Effective'] <= 5):
            return False, "Opinion values must be between 1-5"
        if not (0 <= data['No_Of_Adults'] <= 3):
            return False, "Number of adults must be between 0-3"
        
        # Check categorical values
        if data['Age_Group'] not in AGE_GROUPS:
            return False, "Invalid age group"
        if data['Education'] not in EDUCATION_LEVELS:
            return False, "Invalid education level"
        if data['Sex'] not in GENDER_OPTIONS:
            return False, "Invalid gender"
        
        return True, ""
    except KeyError as e:
        return False, f"Missing required field: {str(e)}"


# ==================== Feature Engineering ====================
def encode_categorical_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features for model prediction.
    
    Args:
        data: Input DataFrame
        
    Returns:
        DataFrame with encoded features
    """
    data_encoded = data.copy()
    
    # Age group encoding
    age_mapping = {
        "18 - 34 Years": 0,
        "35 - 44 Years": 1,
        "45 - 54 Years": 2,
        "55 - 64 Years": 3,
        "65+ Years": 4
    }
    data_encoded['Age_Group'] = data_encoded['Age_Group'].map(age_mapping)
    
    # Education encoding
    education_mapping = {
        "< 12 Years": 0,
        "12 Years": 1,
        "Some College": 2,
        "College Graduate": 3
    }
    data_encoded['Education'] = data_encoded['Education'].map(education_mapping)
    
    # Gender encoding
    gender_mapping = {
        "Female": 0,
        "Male": 1
    }
    data_encoded['Sex'] = data_encoded['Sex'].map(gender_mapping)
    
    return data_encoded


# ==================== Prediction Logic ====================
def make_prediction(model: Pipeline, input_data: pd.DataFrame) -> Tuple[int, float]:
    """
    Make prediction using the trained model.
    
    Args:
        model: Trained model
        input_data: Input features DataFrame
        
    Returns:
        Tuple of (prediction, probability)
    """
    try:
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1] * 100
        return prediction, prediction_proba
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise


# ==================== UI Components ====================
def render_header():
    """Render application header."""
    st.title("💉 H1N1 Vaccine Usage Prediction")
    st.write("Advanced ML-based prediction system for H1N1 vaccine adoption likelihood")
    st.markdown("---")


def render_sidebar_info():
    """Render sidebar information."""
    with st.sidebar:
        st.header("📊 About This App")
        st.info(
            """
            This application uses machine learning to predict the likelihood 
            of an individual receiving the H1N1 vaccine based on survey responses.
            
            **Features:**
            - Real-time predictions
            - Confidence scoring
            - Input validation
            - Production-ready infrastructure
            """
        )
        st.markdown("---")
        st.subheader("📈 Model Information")
        st.write("- **Algorithm**: Ensemble Learning")
        st.write("- **Accuracy Metric**: Confidence Score")
        st.write("- **Last Updated**: 2026-08-24")


def render_input_form() -> Dict[str, Any]:
    """
    Render user input form and return collected data.
    
    Returns:
        Dictionary of form inputs
    """
    with st.form("prediction_form", clear_on_submit=False):
        st.subheader("📋 Patient Survey Information")
        
        col1, col2, col3 = st.columns(3)
        
        input_data = {}
        
        with col1:
            st.markdown("### Health Concerns")
            input_data['H1N1_Worry'] = st.selectbox(
                "H1N1 Worry Level (0-3)",
                [0, 1, 2, 3],
                index=1,
                help="How worried are you about H1N1?"
            )
            input_data['H1N1_Awareness'] = st.selectbox(
                "H1N1 Awareness Level (0-2)",
                [0, 1, 2],
                index=1,
                help="How aware are you about H1N1?"
            )
            input_data['Doctor_Rec_H1N1'] = st.selectbox(
                "Doctor Recommended H1N1 Vaccine?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
            input_data['Chronic_Med_Condition'] = st.selectbox(
                "Chronic Medical Condition?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
        
        with col2:
            st.markdown("### Medical Opinions")
            input_data['Opinion_H1N1_Vacc_Effective'] = st.selectbox(
                "Opinion: H1N1 Vaccine Effective? (1-5)",
                [1, 2, 3, 4, 5],
                index=3,
                help="1=Strongly Disagree, 5=Strongly Agree"
            )
            input_data['Opinion_H1N1_Risk'] = st.selectbox(
                "Opinion: Risk of Getting H1N1? (1-5)",
                [1, 2, 3, 4, 5],
                index=2
            )
            input_data['Opinion_H1N1_Sick_From_Vacc'] = st.selectbox(
                "Opinion: Sick from Vaccine? (1-5)",
                [1, 2, 3, 4, 5],
                index=1
            )
            input_data['Health_Worker'] = st.selectbox(
                "Healthcare Worker?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
        
        with col3:
            st.markdown("### Demographics")
            input_data['Age_Group'] = st.selectbox(
                "Age Group",
                AGE_GROUPS
            )
            input_data['Education'] = st.selectbox(
                "Education Level",
                EDUCATION_LEVELS
            )
            input_data['Sex'] = st.selectbox(
                "Sex",
                GENDER_OPTIONS
            )
            input_data['No_Of_Adults'] = st.slider(
                "Number of Adults in Household",
                min_value=0,
                max_value=3,
                value=1
            )
        
        submit_button = st.form_submit_button(
            "🔮 Predict Vaccination Likelihood",
            use_container_width=True
        )
        
        return input_data, submit_button


def render_results(prediction: int, probability: float):
    """
    Render prediction results with visualization.
    
    Args:
        prediction: Binary prediction (0 or 1)
        probability: Probability percentage
    """
    st.markdown("---")
    st.subheader("🎯 Prediction Result")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.success(f"✅ **Likely to Receive Vaccine**")
            st.metric("Confidence Score", f"{probability:.2f}%", delta="High Risk")
        else:
            st.warning(f"⚠️ **Unlikely to Receive Vaccine**")
            st.metric("Confidence Score", f"{probability:.2f}%", delta="Low Risk")
    
    with col2:
        # Probability bar chart
        fig_data = {
            'Outcome': ['No Vaccine', 'Receives Vaccine'],
            'Probability': [100-probability, probability]
        }
        st.bar_chart(pd.DataFrame(fig_data).set_index('Outcome'))
    
    # Additional insights
    st.markdown("---")
    st.subheader("💡 Insights")
    
    if prediction == 1:
        st.info(
            f"""
            Based on the survey responses, this individual shows a **{probability:.1f}% probability** 
            of receiving the H1N1 vaccine. Key positive indicators include positive opinions 
            about vaccine effectiveness and health awareness.
            """
        )
    else:
        st.warning(
            f"""
            Based on the survey responses, this individual shows a **{100-probability:.1f}% probability** 
            of NOT receiving the H1N1 vaccine. Consider addressing concerns about vaccine safety 
            and effectiveness through targeted health education.
            """
        )


def render_model_error():
    """Render model loading error message."""
    st.error("❌ Model File Missing")
    st.markdown("""
    The trained model file is not available. Please:
    
    1. **Train your model** using the training notebook
    2. **Save the model** using:
       ```python
       import joblib
       joblib.dump(model, 'models/h1n1_model.pkl')
       ```
    3. **Ensure the file** is in the `models/` directory
    
    For setup instructions, check the README.md file.
    """)


# ==================== Main Application ====================
def main():
    """Main application entry point."""
    st.set_page_config(**PAGE_CONFIG)
    
    # Custom styling
    st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stMetric {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Render components
    render_header()
    render_sidebar_info()
    
    # Load model
    model = load_model()
    
    # Render input form
    input_data, submit_button = render_input_form()
    
    # Process prediction
    if submit_button:
        # Validate input
        is_valid, error_msg = validate_input(input_data)
        
        if not is_valid:
            st.error(f"❌ Validation Error: {error_msg}")
        elif model is None:
            render_model_error()
        else:
            try:
                # Prepare data
                df = pd.DataFrame([input_data])
                
                # Make prediction
                prediction, probability = make_prediction(model, df)
                
                # Render results
                render_results(prediction, probability)
                
                logger.info(f"Prediction made: {prediction}, Probability: {probability:.2f}%")
                
            except Exception as e:
                logger.error(f"Application error: {str(e)}")
                st.error(f"❌ An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
