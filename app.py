import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="H1N1 Vaccine Prediction App",
    page_icon="💉",
    layout="wide"
)

# Title & Description
st.title("💉 H1N1 Vaccine Usage Prediction")
st.write("Enter the survey details below to predict the likelihood of receiving the H1N1 vaccine.")

st.markdown("---")

# Load trained pipeline/model (Ensure model.pkl is saved in your directory)
@st.cache_resource
def load_model():
    try:
        model = joblib.load("h1n1_model.pkl")
        return model
    except FileNotFoundError:
        return None

model = load_model()

# Form Layout for Input Features
with st.form("prediction_form"):
    st.subheader("📋 Patient Survey Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        h1n1_worry = st.selectbox("H1N1 Worry Level (0-3)", [0, 1, 2, 3], index=1)
        h1n1_awareness = st.selectbox("H1N1 Awareness Level (0-2)", [0, 1, 2], index=1)
        doctor_recc_h1n1 = st.selectbox("Doctor Recommended H1N1 Vaccine?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        chronic_med_condition = st.selectbox("Chronic Medical Condition?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    with col2:
        opinion_h1n1_vacc_effective = st.selectbox("Opinion: H1N1 Vaccine Effective? (1-5)", [1, 2, 3, 4, 5], index=3)
        opinion_h1n1_risk = st.selectbox("Opinion: Risk of Getting H1N1? (1-5)", [1, 2, 3, 4, 5], index=2)
        opinion_h1n1_sick_from_vacc = st.selectbox("Opinion: Sick from Vaccine? (1-5)", [1, 2, 3, 4, 5], index=1)
        health_worker = st.selectbox("Healthcare Worker?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    with col3:
        age_group = st.selectbox("Age Group", ["18 - 34 Years", "35 - 44 Years", "45 - 54 Years", "55 - 64 Years", "65+ Years"])
        education = st.selectbox("Education Level", ["< 12 Years", "12 Years", "Some College", "College Graduate"])
        sex = st.selectbox("Sex", ["Female", "Male"])
        no_of_adults = st.slider("Number of Adults in Household", 0, 3, 1)

    submit_button = st.form_submit_button("🔮 Predict Vaccination Likelihood")

# Prediction logic
if submit_button:
    # Prepare input dictionary (Adjust feature names according to your trained model columns)
    input_data = pd.DataFrame([{
        'H1N1_Worry': h1n1_worry,
        'H1N1_Awareness': h1n1_awareness,
        'Doctor_Rec_H1N1': doctor_recc_h1n1,
        'Chronic_Med_Condition': chronic_med_condition,
        'Opinion_H1N1_Vacc_Effective': opinion_h1n1_vacc_effective,
        'Opinion_H1N1_Risk': opinion_h1n1_risk,
        'Opinion_H1N1_Sick_From_Vacc': opinion_h1n1_sick_from_vacc,
        'Health_Worker': health_worker,
        'Age_Group': age_group,
        'Education': education,
        'Sex': sex,
        'No_Of_Adults': no_of_adults
    }])

    if model is not None:
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1] * 100

        st.markdown("---")
        st.subheader("🎯 Result")
        
        if prediction == 1:
            st.success(f"**High Probability of Vaccination!** (Probability: **{prediction_proba:.2f}%**)")
        else:
            st.warning(f"**Low Probability of Vaccination.** (Probability: **{prediction_proba:.2f}%**)")
    else:
        st.info("💡 **Model file missing:** Save your trained scikit-learn model using `joblib.dump(model, 'h1n1_model.pkl')` in your notebook to activate live inference.")