import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import re
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentimentApp:
    
    def __init__(self, model_dir: str):
       
        self.model_dir = model_dir
        self.model = None
        self.vectorizer = None
        self.example_sentences = [
            "Sweden has one of the highest standards of living in the world with excellent healthcare and education.",
            "The cold, dark winters in Sweden can lead to seasonal depression and isolation.",
            "Swedish cuisine features delicious meatballs, lingonberries, and cinnamon buns.",
            "The high taxes in Sweden make it difficult for some people to save money.",
            "Sweden's commitment to sustainability and renewable energy is impressive.",
            "Some areas in Swedish cities have problems with crime and social segregation.",
            "I love the beautiful natural landscapes and the northern lights in Sweden.",
            "Swedish design is minimalist, functional and highly influential around the world.",
            "Sweden's long waiting times for healthcare specialists can be frustrating for patients."
        ]
    
    def load_model(self):
        
        try:
            # Load the Logistic Regression model
            model_path = os.path.join(self.model_dir, 'logistic_regression_model.joblib')
            self.model = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")
            
            # Load the TF-IDF vectorizer
            vectorizer_path = os.path.join(self.model_dir, 'tfidf_vectorizer.joblib')
            self.vectorizer = joblib.load(vectorizer_path)
            logger.info(f"Loaded vectorizer from {vectorizer_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading model or vectorizer: {e}")
            return False
    
    def predict_sentiment(self, text: str):
        
        try:
            # Clean the text (similar to the preprocessing step)
            cleaned_text = re.sub(r'[^\w\s.,!?;:]', ' ', text)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            
            # Transform the text using the TF-IDF vectorizer
            text_vector = self.vectorizer.transform([cleaned_text])
            
            # Predict the sentiment
            prediction = self.model.predict(text_vector)[0]
            
            # Get the probability
            probabilities = self.model.predict_proba(text_vector)[0]
            confidence = probabilities[prediction]
            
            # Map the prediction to a label and color
            sentiment = "Positive" if prediction == 1 else "Negative"
            color = "#28a745" if prediction == 1 else "#dc3545"  # Green for positive, red for negative
            
            return {
                "text": text,
                "sentiment": sentiment,
                "confidence": confidence,
                "color": color,
                "probabilities": probabilities.tolist()
            }
        except Exception as e:
            logger.error(f"Error predicting sentiment: {e}")
            return None
    
    def run(self):
        
        # Set page configuration
        st.set_page_config(
            page_title="Sweden Sentiment Analysis",
            page_icon="🇸🇪",
            layout="wide"
        )
        
        # App title and description
        st.title("Sweden Sentiment Analysis")
        st.markdown("""
        This application analyzes the sentiment of text about Sweden. 
        Enter a sentence or choose an example, and the model will classify it as positive or negative.
        """)
        
        # Load model
        if self.model is None or self.vectorizer is None:
            with st.spinner("Loading model..."):
                success = self.load_model()
                if not success:
                    st.error("Failed to load the model or vectorizer. Please check the logs for details.")
                    return
        
        # Create sidebar
        st.sidebar.title("About Project")
        st.sidebar.info("""
        This application uses a Logistic Regression model trained on sentiment-labeled text about Sweden.
        
        The model achieved:
        - Accuracy: 93.24% (cross-validation)
        - Precision: 81.82%
        - Recall: 86.54%
        - F1 Score: 84.11%
        
        Logistic Regression was chosen as the final model because it showed:
        - Highest cross-validation accuracy (93.24%)
        - More stable performance across folds (±3.10%)
        - Better generalization on unseen data
        - Good balance between precision and recall
        """)
        # Create tabs
        tab1, tab2 = st.tabs(["Sentiment Analysis", "Model Insights"])
        
        with tab1:
            # Input section
            st.header("Input Text")
            
            # Option to use example or custom text
            input_option = st.radio(
                "Choose input method:",
                ["Enter your own text", "Use an example sentence"]
            )
            
            input_text = ""
            
            if input_option == "Enter your own text":
                input_text = st.text_area("Enter text about Sweden:", height=150)
            else:
                example_index = st.selectbox(
                    "Select an example sentence:",
                    range(len(self.example_sentences)),
                    format_func=lambda i: self.example_sentences[i]
                )
                input_text = self.example_sentences[example_index]
                st.text_area("Example text:", value=input_text, height=150, disabled=True)
            
            # Analyze button
            if st.button("Analyze Sentiment"):
                if not input_text or len(input_text.strip()) < 5:
                    st.error("Please enter a valid text with at least 5 characters.")
                else:
                    with st.spinner("Analyzing sentiment..."):
                        # Add a small delay to show the spinner
                        time.sleep(0.5)
                        
                        # Predict sentiment
                        result = self.predict_sentiment(input_text)
                        
                        if result:
                            # Display result
                            st.header("Analysis Result")
                            
                            # Create columns for displaying result
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                # Display sentiment badge
                                st.markdown(
                                    f"""
                                    <div style="
                                        background-color: {result['color']}; 
                                        padding: 20px; 
                                        border-radius: 10px; 
                                        text-align: center;
                                        color: white;
                                        font-size: 24px;
                                    ">
                                        {result['sentiment']}
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                
                                # Display confidence
                                st.metric(
                                    label="Confidence",
                                    value=f"{result['confidence']*100:.1f}%"
                                )
                            
                            with col2:
                                # Create a more visually appealing horizontal bar chart for probabilities
                                fig, ax = plt.subplots(figsize=(8, 2.5))
                                
                                labels = ['Negative', 'Positive']
                                probabilities = result['probabilities']
                                colors = ['#dc3545', '#28a745']
                                
                                # Create the horizontal bar chart
                                bars = ax.barh(labels, probabilities, color=colors, height=0.5, alpha=0.8)
                                ax.set_xlim(0, 1)
                                ax.set_xlabel('Probability', fontweight='bold')
                                ax.set_title('Sentiment Probabilities', fontweight='bold', fontsize=14)
                                
                                # Add text annotations with more formatting
                                for i, prob in enumerate(probabilities):
                                    text_color = 'white' if prob > 0.4 else 'black'
                                    text_pos = prob/2 if prob > 0.4 else prob + 0.02
                                    ax.text(text_pos, i, f"{prob:.2f}", va='center', ha='center', 
                                           color=text_color, fontweight='bold')
                                
                                # Add grid lines for better readability
                                ax.grid(axis='x', linestyle='--', alpha=0.7)
                                ax.spines['top'].set_visible(False)
                                ax.spines['right'].set_visible(False)
                                
                                st.pyplot(fig)
                                
                                # Additional explanation
                                if result['sentiment'] == 'Positive':
                                    st.info("The model detected a positive sentiment in the text.")
                                else:
                                    st.info("The model detected a negative sentiment in the text.")
                        else:
                            st.error("An error occurred during analysis. Please try again.")
            
            # Add some space
            st.markdown("---")
            
            # How it works section
            with st.expander("How it works"):
                st.markdown("""
                1. Your text is preprocessed to clean special characters and standardize formatting.
                2. The cleaned text is transformed into a numerical representation using TF-IDF vectorization.
                3. The trained Logistic Regression model analyzes the vectorized text.
                4. The model predicts whether the sentiment is positive or negative and provides a confidence score.
                
                The model was trained on text extracted from Wikipedia articles about Sweden.
                """)
        
        with tab2:
            # Display model insights
            st.header("Model Performance")
            
            # Create columns for key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(label="Accuracy (CV)", value="93.24%", help="Cross-validation accuracy")
            with col2:
                st.metric(label="Precision", value="81.82%", help="Precision on test data")
            with col3:
                st.metric(label="Recall", value="86.54%", help="Recall on test data")
            with col4:
                st.metric(label="F1 Score", value="84.11%", help="F1 Score on test data")
            
            # Add explanation of metrics
            st.info("""
            **Why Logistic Regression?** 
            
            Our Logistic Regression model showed the best cross-validation accuracy (93.24%) with low 
            variance (±3.10%), indicating stable and reliable performance across different data subsets.
            This model offers the best balance between precision and recall for our specific task.
            """)
            
            # Load and display metrics comparison image
            metrics_img_path = os.path.join(self.model_dir, 'model_metrics_comparison.png')
            if os.path.exists(metrics_img_path):
                st.image(metrics_img_path, caption="Model Performance Metrics Comparison")
            
            # Load and display cross-validation results
            cv_img_path = os.path.join(self.model_dir, 'cross_validation_results.png')
            if os.path.exists(cv_img_path):
                st.image(cv_img_path, caption="Cross-Validation Results")
            
            # Load and display word cloud if available
            wordcloud_path = os.path.join('data', 'processed', 'wordcloud.png')
            if os.path.exists(wordcloud_path):
                st.header("Word Cloud from Sweden Wikipedia Content")
                st.image(wordcloud_path, caption="Word Cloud of Sweden Wikipedia Content")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        © Rachit Jain | 70572200036 | L040 | 2025 Sweden Sentiment Analysis Project | Applied Artificial Intelligence """)

def main():
    """
    Main function to run the Streamlit app.
    """
    try:
        # Set the model directory
        model_dir = os.path.join('models')
        
        # Initialize and run the app
        app = SentimentApp(model_dir)
        app.run()
        
    except Exception as e:
        logger.error(f"An error occurred in the Streamlit app: {e}")
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

