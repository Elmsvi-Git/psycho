# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 21:54:49 2025

@author: ElaheMousavi
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# App title
st.title("🔮 Psychosomatic Susceptibility Profile")
st.write("Answer all the questions below carefully:")

# Define answer options
neo_options = ["Strongly agree", "Agree", "No idea", "Disagree", "Strongly Disagree"]
cope_options = ["Usually", "Often", "Never"]

# Mapping for encoding
neo_mapping = {
    "Strongly Disagree": 0,
    "Disagree": 1,
    "No idea": 2,
    "Agree": 3,
    "Strongly agree": 4
}
cope_mapping = {
    "Never": 1,
    "Often": 2,
    "Usually": 3
}

# Preprocessing function
def preprocess_inputs(q1, q2, q3, q4, q5, q6, q7, q8):
    if None in [q1, q2, q3, q4, q5, q6, q7, q8]:
        st.warning("⚠️ Please answer all questions before proceeding.")
        st.stop()  # <--- This stops the code immediately after the warning.

    
    if None in [q1, q2, q3, q4, q5, q6, q7, q8]:
        st.warning("Please answer all questions before proceeding.")
        return None

    q1_encoded = neo_mapping.get(q1, 2)
    q2_encoded = neo_mapping.get(q2, 2)
    q3_encoded = 4-neo_mapping.get(q3, 2)
    q4_encoded = neo_mapping.get(q4, 2)
    q5_encoded = cope_mapping.get(q5, 2)
    q6_encoded = cope_mapping.get(q6, 2)
    q7_encoded = cope_mapping.get(q7, 2)
    q8_encoded = cope_mapping.get(q8, 2)

    data = np.array([[q5_encoded, q6_encoded, q4_encoded, q3_encoded,
                      q2_encoded, q7_encoded, q8_encoded, q1_encoded]])

    selected_features = ['Q5', 'Q7', 'question_51', 'question_42',
                         'question_26', 'Q8', 'Q18', 'question_21']

    return pd.DataFrame(data=data, columns=selected_features)

# Function to display progress
def progress_text(current, total):
    return f"**Question {current}/{total}**"

# User input for 8 questions
st.subheader("📝 Questionnaire")

with st.form("questionnaire_form"):
    st.markdown(progress_text(1, 8))
    q1 = st.radio("I often feel tense and jittery.", neo_options, horizontal=True, index=None)

    st.markdown(progress_text(2, 8))
    q2 = st.radio("Sometimes I feel completely worthless.", neo_options, horizontal=True, index=None)

    st.markdown(progress_text(3, 8))
    q3 = st.radio("I am not a cheerful optimist.", neo_options, horizontal=True, index=None)

    st.markdown(progress_text(4, 8))
    q4 = st.radio("I often feel helpless and want someone else to solve my problems.", neo_options, horizontal=True, index=None)

    st.markdown(progress_text(5, 8))
    q5 = st.radio("I take direct action to get around the problem.", cope_options, horizontal=True, index=None)

    st.markdown(progress_text(6, 8))
    q6 = st.radio("I look for something good in what is happening.", cope_options, horizontal=True, index=None)

    st.markdown(progress_text(7, 8))
    q7 = st.radio("I try to see it in a different light, to make it more positive.", cope_options, horizontal=True, index=None)

    st.markdown(progress_text(8, 8))
    q8 = st.radio("I learn to live with it.", cope_options, horizontal=True, index=None)

    submitted = st.form_submit_button("🚀 Predict")

# Prediction after submission
if submitted:
    try:
        inputs = preprocess_inputs(q1, q2, q3, q4, q5, q6, q7, q8)
        prediction = model.predict(inputs)
        prediction_proba = model.predict_proba(inputs)

        pred_map = {0: "DISORDER PRONE", 1: "EXTREMITY RESILIENT", 2: "MODERATE"}
        prediction_label = pred_map.get(np.argmax(prediction_proba[0]), "Unknown")

        st.success(f"🎯 Predicted Label: **{prediction_label}**")
        st.write(f"🔍 Confidence: **{np.max(prediction_proba[0]):.2f}**")
    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {e}")
