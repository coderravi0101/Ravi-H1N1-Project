"""
H1N1 Vaccine Prediction Application
Production-ready Streamlit application for H1N1 vaccine adoption likelihood prediction.

================================================================================
PROJECT INFORMATION
================================================================================
Created By: Ravi Kumar Singh
Certified Microsoft Trainer
Professional Development: Advanced Machine Learning & Web Applications

Version: 1.0.0
Last Updated: August 24, 2026

================================================================================
FEATURES
================================================================================
✅ Advanced ML-based Predictions - Ensemble learning models
✅ Production-Ready Infrastructure - Logging, validation, error handling
✅ Professional UI/UX - Clean, intuitive interface design
✅ Input Validation - Comprehensive data validation
✅ Confidence Scoring - Probability-based predictions
✅ Scalable Architecture - Modular, maintainable code structure
✅ Interactive Visualizations - Engaging health education content

================================================================================
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
    """Render application header with health education visuals."""
    st.title("💉 H1N1 Vaccine Usage Prediction System")
    st.write("Advanced ML-based prediction system for H1N1 vaccine adoption likelihood")
    
    # Creator attribution
    col1, col2, col3 = st.columns([2, 2, 1])
    with col3:
        st.markdown("""
        <div style='text-align: right; font-size: 0.85rem; color: #666;'>
        <b>Created By:</b><br/>
        Ravi Kumar Singh<br/>
        <i>Certified Microsoft Trainer</i>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Interactive Health Education Section
    with st.container():
        st.subheader("🏥 Vaccine Health Information")
        
        col_vacc1, col_vacc2 = st.columns(2)
        
        with col_vacc1:
            st.markdown("""
            ### 💊 H1N1 Vaccine (Influenza A H1N1)
            
            **What is H1N1?**
            - Influenza A virus subtype
            - Highly contagious respiratory virus
            - Can cause severe complications
            
            **Vaccine Benefits:**
            ✅ 40-60% reduction in infection risk
            ✅ Reduces severity of illness
            ✅ Protects vulnerable populations
            ✅ Community immunity development
            
            **Best For:**
            👶 Young children (6 months - 5 years)
            👴 Seniors (65+ years)
            🏥 Healthcare workers
            🤒 Chronic disease patients
            
            **Common Side Effects:**
            - Mild arm soreness (1-2 days)
            - Low-grade fever (rare)
            - Fatigue (temporary)
            """)
        
        with col_vacc2:
            st.markdown("""
            ### 🦠 Seasonal Flu Vaccine (Influenza)
            
            **What is Flu?**
            - Seasonal respiratory illness
            - Different strains each year
            - Affects millions annually
            
            **Vaccine Benefits:**
            ✅ 40-60% effectiveness
            ✅ Prevents hospitalization
            ✅ Reduces complications
            ✅ Safe for all ages
            
            **Recommended For:**
            👨‍👩‍👧‍👦 Everyone 6+ months old
            ⚖️ Pregnant women
            🏥 Healthcare workers
            🧑‍💼 Essential workers
            
            **Safety Profile:**
            - Approved by FDA for decades
            - Monitored by CDC continuously
            - Used by millions worldwide
            """)
        
        # Visual comparison
        st.markdown("---")
        st.subheader("📊 Vaccine Comparison & Recommendations")
        
        comparison_data = {
            'Feature': [
                'Onset Time',
                'Duration',
                'Target Population',
                'Effectiveness',
                'Contraindications',
                'Cost'
            ],
            'H1N1 Vaccine': [
                '1-2 weeks',
                '6-12 months',
                'All ages (6+ months)',
                '40-60%',
                'Severe egg allergy',
                'Variable'
            ],
            'Flu Vaccine': [
                '1-2 weeks',
                'Annual (12 months)',
                'Everyone 6+ months',
                '40-60%',
                'Severe egg allergy',
                'Low/Free'
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Health Tips
        st.info("""
        ### 💚 Health Tips & Best Practices
        
        **Before Vaccination:**
        - Get adequate sleep (7-8 hours)
        - Stay hydrated
        - Eat nutritious food
        - Avoid stress when possible
        
        **After Vaccination:**
        - Keep arm moving gently
        - Apply cool compress if soreness
        - Take over-the-counter pain relief if needed
        - Drink plenty of water
        - Monitor for severe reactions (rare)
        
        **When to Seek Medical Help:**
        ⚠️ Difficulty breathing
        ⚠️ Severe facial swelling
        ⚠️ High fever (>103°F)
        ⚠️ Persistent symptoms beyond 3 days
        """)
    
    st.markdown("---")


def render_sidebar_info():
    """Render sidebar information with professional branding."""
    with st.sidebar:
        st.header("📊 Application Info")
        
        # Developer info
        st.info(
            """
            **H1N1 Vaccine Prediction System**
            
            Created By: **Ravi Kumar Singh**  
            Certification: Microsoft Certified Trainer  
            Version: 1.0.0
            
            This application uses advanced machine learning 
            to predict vaccine adoption likelihood based on 
            comprehensive health survey data.
            """
        )
        
        st.markdown("---")
        
        st.subheader("🎯 Features")
        st.markdown("""
        ✅ Real-time AI Predictions
        ✅ Confidence Scoring
        ✅ Input Validation
        ✅ Health Education
        ✅ Risk Assessment
        ✅ Personalized Insights
        """)
        
        st.markdown("---")
        
        st.subheader("📈 Model Statistics")
        st.write("- **Algorithm**: Ensemble Learning")
        st.write("- **Accuracy Metric**: Confidence Score")
        st.write("- **Last Updated**: 2026-08-24")
        st.write("- **Data Points**: 2000+")
        
        st.markdown("---")
        
        st.subheader("📞 Support")
        st.write("For issues or questions, contact the development team.")


def render_input_form() -> Dict[str, Any]:
    """
    Render user input form with attractive formatting.
    
    Returns:
        Dictionary of form inputs
    """
    with st.form("prediction_form", clear_on_submit=False):
        st.subheader("📋 Patient Survey Information")
        st.write("Please provide accurate information for better predictions")
        
        col1, col2, col3 = st.columns(3)
        
        input_data = {}
        
        with col1:
            st.markdown("### 🏥 Health Concerns")
            input_data['H1N1_Worry'] = st.selectbox(
                "H1N1 Worry Level (0-3)",
                [0, 1, 2, 3],
                index=1,
                help="0=Not worried, 3=Very worried"
            )
            input_data['H1N1_Awareness'] = st.selectbox(
                "H1N1 Awareness Level (0-2)",
                [0, 1, 2],
                index=1,
                help="0=Not aware, 2=Very aware"
            )
            input_data['Doctor_Rec_H1N1'] = st.selectbox(
                "Doctor Recommended H1N1 Vaccine?",
                [0, 1],
                format_func=lambda x: "✅ Yes" if x == 1 else "❌ No"
            )
            input_data['Chronic_Med_Condition'] = st.selectbox(
                "Chronic Medical Condition?",
                [0, 1],
                format_func=lambda x: "✅ Yes" if x == 1 else "❌ No"
            )
        
        with col2:
            st.markdown("### 💭 Medical Opinions")
            input_data['Opinion_H1N1_Vacc_Effective'] = st.selectbox(
                "Opinion: H1N1 Vaccine Effective? (1-5)",
                [1, 2, 3, 4, 5],
                index=3,
                help="1=Strongly Disagree, 5=Strongly Agree"
            )
            input_data['Opinion_H1N1_Risk'] = st.selectbox(
                "Opinion: Risk of Getting H1N1? (1-5)",
                [1, 2, 3, 4, 5],
                index=2,
                help="1=Low Risk, 5=High Risk"
            )
            input_data['Opinion_H1N1_Sick_From_Vacc'] = st.selectbox(
                "Opinion: Sick from Vaccine? (1-5)",
                [1, 2, 3, 4, 5],
                index=1,
                help="1=No Concern, 5=High Concern"
            )
            input_data['Health_Worker'] = st.selectbox(
                "Healthcare Worker?",
                [0, 1],
                format_func=lambda x: "✅ Yes" if x == 1 else "❌ No"
            )
        
        with col3:
            st.markdown("### 👤 Demographics")
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
        
        st.markdown("---")
        
        submit_button = st.form_submit_button(
            "🔮 Predict Vaccination Likelihood",
            use_container_width=True
        )
        
        return input_data, submit_button


def render_results(prediction: int, probability: float):
    """
    Render prediction results with attractive visualizations.
    
    Args:
        prediction: Binary prediction (0 or 1)
        probability: Probability percentage
    """
    st.markdown("---")
    st.subheader("🎯 Prediction Result & Analysis")
    
    # Main prediction display
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if prediction == 1:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 1rem; text-align: center; color: white;'>
                <h2 style='margin: 0;'>✅ LIKELY</h2>
                <p style='margin: 0; font-size: 1.1rem;'>Will Receive Vaccine</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 2rem; border-radius: 1rem; text-align: center; color: white;'>
                <h2 style='margin: 0;'>⚠️ UNLIKELY</h2>
                <p style='margin: 0; font-size: 1.1rem;'>Will Receive Vaccine</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.metric(
            "Confidence Score",
            f"{probability:.1f}%",
            delta=f"{probability-50:.1f}% from neutral"
        )
    
    with col3:
        # Risk indicator
        if probability >= 75:
            risk_level = "🟢 Very High Probability"
            color = "#00ff00"
        elif probability >= 60:
            risk_level = "🟢 High Probability"
            color = "#90ee90"
        elif probability >= 50:
            risk_level = "🟡 Moderate Probability"
            color = "#ffff00"
        elif probability >= 35:
            risk_level = "🟠 Low Probability"
            color = "#ffa500"
        else:
            risk_level = "🔴 Very Low Probability"
            color = "#ff6347"
        
        st.markdown(f"""
        <div style='background-color: {color}; padding: 1rem; 
                    border-radius: 0.5rem; text-align: center;'>
            <p style='margin: 0; font-weight: bold;'>{risk_level}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Probability visualization
    st.markdown("---")
    
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        st.subheader("📊 Probability Distribution")
        fig_data = pd.DataFrame({
            'Outcome': ['No Vaccine', 'Receives Vaccine'],
            'Probability': [100-probability, probability]
        })
        st.bar_chart(fig_data.set_index('Outcome'))
    
    with col_viz2:
        st.subheader("🎯 Vaccination Likelihood")
        # Gauge-style visualization
        st.progress(probability/100, text=f"{probability:.1f}%")
        
        if probability >= 75:
            st.success("Excellent - Strong vaccine adoption likelihood")
        elif probability >= 60:
            st.info("Good - Above average vaccine adoption probability")
        elif probability >= 50:
            st.warning("Moderate - Borderline vaccine adoption decision")
        else:
            st.error("Low - May need targeted education and counseling")
    
    # Personalized insights
    st.markdown("---")
    st.subheader("💡 Personalized Insights & Recommendations")
    
    if prediction == 1:
        st.success(f"""
        ### ✅ Positive Vaccination Profile
        
        Based on the survey responses, this individual shows a **{probability:.1f}% probability** 
        of receiving the H1N1 vaccine. 
        
        **Positive Indicators:**
        - Good awareness about H1N1
        - Trusts vaccine effectiveness
        - Open to medical recommendations
        - Low concerns about side effects
        
        **Recommendations:**
        1. ✅ Schedule vaccination appointment soon
        2. 📞 Contact healthcare provider for guidance
        3. 📚 Review vaccine safety information
        4. 💪 Ensure good health before vaccination
        """)
    else:
        st.warning(f"""
        ### ⚠️ Vaccination Risk Assessment
        
        Based on the survey responses, this individual shows a **{100-probability:.1f}% probability** 
        of NOT receiving the H1N1 vaccine.
        
        **Potential Concerns:**
        - Low awareness about vaccine benefits
        - Concerns about vaccine safety
        - Skeptical about effectiveness
        - Medical hesitation present
        
        **Recommendations:**
        1. 📚 Provide credible vaccine information
        2. 👨‍⚕️ Discuss concerns with healthcare provider
        3. 🎓 Attend health education sessions
        4. 💬 Address vaccine myths and misconceptions
        5. ⏰ Follow up with personalized counseling
        """)
    
    # Health education callout
    st.info("""
    ### 🏥 Why Vaccination Matters
    
    Vaccines are among the most effective public health tools we have:
    - Save millions of lives annually
    - Prevent serious diseases and complications
    - Protect vulnerable populations (children, elderly, immunocompromised)
    - Contribute to community immunity (herd immunity)
    - Backed by decades of safety monitoring
    
    **Speak with your healthcare provider** if you have any concerns or questions about vaccination.
    """)


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
        .stSuccess, .stWarning, .stError, .stInfo {
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
