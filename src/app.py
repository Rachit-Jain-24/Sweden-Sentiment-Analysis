# Simple Streamlit App for Sweden Sentiment Analysis
# --------------------------------------------------
# Features:
# - Text input
# - Predict sentiment using Random Forest model
# - Show confidence, probability bar, LIME explanation
# - Display word cloud and model accuracy images
# - Display model metrics for all models in the main content area


import streamlit as st
import joblib
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
import pandas as pd

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="Sweden Sentiment Analyzer", page_icon="🇸🇪", layout="centered")

# Custom CSS to center the content
st.markdown(
    """
    <style>
    .main {
        max-width: 800px;
        margin: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Step 1: Load model, vectorizer, and metrics
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
model = joblib.load("models/random_forest_model.joblib")
pipeline = make_pipeline(vectorizer, model)
explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])

# Load metrics from JSON if available
metrics = {}
metrics_path = "models/model_metrics.json"
if os.path.exists(metrics_path):
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

# Step 2: Main content layout
st.title("🇸🇪 Sweden Sentiment Analyzer")
st.markdown("Analyze sentiment in Wikipedia content using the Random Forest model and LIME explanations.")

# Step 3: Text input with example sentence
example_sentence = "Sweden is known for its beautiful landscapes and high quality of life."
user_input = st.text_area("Enter a sentence or paragraph to analyze:", value=example_sentence, height=150)

# Step 4: Analyze Button
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Predict sentiment
        prediction = pipeline.predict([user_input])[0]
        probabilities = pipeline.predict_proba([user_input])[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = probabilities[prediction]
        color = "#28a745" if prediction == 1 else "#dc3545"

        # Show result
        st.header("Prediction Result")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(f"<div style='background-color:{color};padding:20px;border-radius:10px;text-align:center;color:white;font-size:24px'>{sentiment}</div>", unsafe_allow_html=True)
            st.metric("Confidence", f"{confidence*100:.1f}%")

        with col2:
            labels = ['Negative', 'Positive']
            plt.figure(figsize=(6, 2.5))
            plt.barh(labels, probabilities, color=['#dc3545', '#28a745'])
            plt.xlim(0, 1)
            plt.title("Sentiment Probabilities")
            st.pyplot(plt.gcf())
            plt.close()

# Step 5: How it works
with st.expander("ℹ️ How it works"):
    st.markdown("""
    1. This app uses TF-IDF to convert your sentence into numerical features.
    2. The Random Forest model predicts whether your input is positive or negative.
    3. The model was trained using SMOTE-balanced sentiment-labeled Wikipedia sentences from Sweden.
    """)

# Step 6: Show Word Cloud
wordcloud_path = "data/processed/wordcloud.png"
if os.path.exists(wordcloud_path):
    st.subheader("☁️ Word Cloud from Wikipedia Content")
    st.image(wordcloud_path, caption="Word Cloud of Sweden Wikipedia Article")

# Step 7: Show performance chart
chart_path = "models/cross_validation_results.png"
if os.path.exists(chart_path):
    st.subheader("📈 Cross-Validation Accuracy")
    st.image(chart_path, caption="Model comparison using 5-fold cross-validation")

# Step 8: Show model metrics comparison chart
chart_path1 = "models/model_metrics_comparison.png"
if os.path.exists(chart_path1):
    st.subheader("📈 Model Metrics Comparison")
    st.image(chart_path1, caption="Comparison of model metrics")

# Step 9: Display Model Metrics in a structured format
st.subheader("📊 Model Performance Metrics")
st.write("Below are the performance metrics for all trained models:")

# Create a table to display metrics
metrics_table = []
for model_name, performance in metrics.items():
    metrics_table.append([model_name, 
                          f"{performance['precision']*100:.2f}%",
                          f"{performance['recall']*100:.2f}%",
                          f"{performance['f1-score']*100:.2f}%"])

st.table(pd.DataFrame(metrics_table, columns=["Model", "Precision", "Recall", "F1-score"]))

# Footer
st.markdown("---")
st.markdown("Made by Rachit Jain | Applied AI Project | NMIMS | April 2025")