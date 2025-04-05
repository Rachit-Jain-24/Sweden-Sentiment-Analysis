#!/usr/bin/env python
"""
Machine Learning Model Trainer for Sentiment Analysis

This script loads the processed sentiment data, applies TF-IDF vectorization,
filters out neutral sentiments, applies SMOTE for class balancing, and implements
six different machine learning models for sentiment classification.
"""

import os
import logging
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
from pprint import pprint

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.feature_selection import SelectKBest, chi2

# SMOTE for class balancing
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline

# NLP tools
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentimentModelTrainer:
    """
    A class for training and evaluating machine learning models for sentiment analysis.
    """
    
    def __init__(self, data_file: str, output_dir: str):
        """
        Initialize the model trainer with the input data file and output directory.
        
        Args:
            data_file (str): Path to the CSV file containing sentence sentiments.
            output_dir (str): Directory to save the trained models and evaluation results.
        """
        self.data_file = data_file
        self.output_dir = output_dir
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer = None
        self.model = None
        self.models = {}
        self.model_metrics = {}
        self.best_params = {}
        self.ensemble_model = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def load_data(self):
        """
        Load the sentence sentiment data from a CSV file.
        
        Raises:
            FileNotFoundError: If the data file does not exist.
        """
        try:
            logger.info(f"Loading data from {self.data_file}")
            self.data = pd.read_csv(self.data_file)
            logger.info(f"Loaded data with shape: {self.data.shape}")
            # No return value needed, data is stored in self.data
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_file}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def filter_neutral_sentiments(self) -> pd.DataFrame:
        """
        Filter out neutral sentiments, keeping only positive and negative.
        
        Returns:
            pd.DataFrame: DataFrame with only positive and negative sentiments.
        """
        if self.data is None:
            self.load_data()
        
        try:
            logger.info("Filtering out neutral sentiments")
            filtered_data = self.data[self.data['sentiment'] != 'neutral']
            logger.info(f"After filtering neutral sentiments, data shape: {filtered_data.shape}")
            
            # Update the data
            self.data = filtered_data
            return filtered_data
        except Exception as e:
            logger.error(f"Error filtering neutral sentiments: {e}")
            raise
    
    def prepare_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare the data for model training by splitting into train and test sets.
        
        Args:
            test_size (float): Proportion of data to use for testing. Default is 0.2.
            random_state (int): Random state for reproducibility. Default is 42.
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test arrays.
        """
        if self.data is None or 'neutral' in self.data['sentiment'].values:
            self.filter_neutral_sentiments()
        
        try:
            logger.info("Preparing data for model training")
            
            # Convert sentiments to binary labels (1 for positive, 0 for negative)
            self.data['sentiment_binary'] = (self.data['sentiment'] == 'positive').astype(int)
            
            # Split the data into training and testing sets
            X = self.data['sentence'].values
            y = self.data['sentiment_binary'].values
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            logger.info(f"Data split: X_train: {self.X_train.shape}, X_test: {self.X_test.shape}")
            
            return self.X_train, self.X_test, self.y_train, self.y_test
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def preprocess_text(self, texts: List[str]) -> List[str]:
        """
        Apply advanced text preprocessing including lemmatization.
        
        Args:
            texts (List[str]): List of text sentences.
            
        Returns:
            List[str]: List of preprocessed text sentences.
        """
        preprocessed_texts = []
        
        for text in texts:
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs, email addresses, and special characters
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            text = re.sub(r'\S*@\S*\s?', '', text)
            text = re.sub(r'[^\w\s.,!?;:]', ' ', text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and lemmatize
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
            
            # Rejoin tokens
            preprocessed_text = ' '.join(tokens)
            
            preprocessed_texts.append(preprocessed_text)
        
        return preprocessed_texts
    
    def apply_tfidf(self, tune_params: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply TF-IDF vectorization to the sentence data with optimized parameters.
        
        Args:
            tune_params (bool): Whether to tune the TF-IDF parameters. Default is True.
            
        Returns:
            Tuple: TF-IDF vectors for training and testing sets.
        """
        if self.X_train is None or self.X_test is None:
            self.prepare_data()
        
        try:
            logger.info("Preprocessing text with advanced techniques")
            
            # Apply advanced preprocessing
            X_train_preprocessed = self.preprocess_text(self.X_train)
            X_test_preprocessed = self.preprocess_text(self.X_test)
            
            # Define TF-IDF parameters for grid search
            if tune_params:
                logger.info("Tuning TF-IDF parameters using grid search")
                
                # Define parameter grid for TF-IDF
                param_grid = {
                    'max_features': [3000, 5000, 7000],
                    'ngram_range': [(1, 1), (1, 2), (1, 3)],
                    'min_df': [2, 3],
                    'max_df': [0.9, 0.95],
                    'use_idf': [True, False]
                }
                
                # Initialize TF-IDF vectorizer
                tfidf = TfidfVectorizer()
                
                # Initialize simple model for parameter tuning
                simple_model = LogisticRegression(max_iter=1000, random_state=42)
                
                # Create pipeline
                pipeline = Pipeline([
                    ('tfidf', tfidf),
                    ('classifier', simple_model)
                ])
                
                # Initialize grid search
                grid_search = GridSearchCV(
                    pipeline, 
                    {f'tfidf__{key}': val for key, val in param_grid.items()},
                    cv=3,
                    scoring='f1',
                    n_jobs=-1,
                    verbose=1
                )
                
                # Fit grid search
                grid_search.fit(X_train_preprocessed, self.y_train)
                
                # Get best parameters
                best_params = {key.replace('tfidf__', ''): val for key, val in grid_search.best_params_.items()}
                logger.info(f"Best TF-IDF parameters: {best_params}")
                
                # Initialize vectorizer with best parameters
                self.vectorizer = TfidfVectorizer(**best_params)
            else:
                # Use default parameters
                logger.info("Using default TF-IDF parameters (max_features=5000, ngram_range=(1, 3))")
                self.vectorizer = TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 3),
                    min_df=2,
                    max_df=0.95,
                    use_idf=True
                )
            
            # Transform the training and testing data
            X_train_tfidf = self.vectorizer.fit_transform(X_train_preprocessed)
            X_test_tfidf = self.vectorizer.transform(X_test_preprocessed)
            
            logger.info(f"TF-IDF vectors: X_train_tfidf: {X_train_tfidf.shape}, X_test_tfidf: {X_test_tfidf.shape}")
            
            return X_train_tfidf, X_test_tfidf
        except Exception as e:
            logger.error(f"Error applying TF-IDF: {e}")
            raise
    
    def apply_smote(self, X_train_tfidf: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE to balance the classes in the training data.
        
        Args:
            X_train_tfidf (np.ndarray): TF-IDF vectors for the training set.
            random_state (int): Random state for reproducibility. Default is 42.
            
        Returns:
            Tuple: Balanced TF-IDF vectors and labels for the training set.
        """
        try:
            logger.info("Applying SMOTE to balance classes")
            
            # Check class distribution before SMOTE
            logger.info(f"Class distribution before SMOTE: {np.bincount(self.y_train)}")
            
            # Apply SMOTE
            smote = SMOTE(random_state=random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, self.y_train)
            
            # Check class distribution after SMOTE
            logger.info(f"Class distribution after SMOTE: {np.bincount(y_train_resampled)}")
            
            return X_train_resampled, y_train_resampled
        except Exception as e:
            logger.error(f"Error applying SMOTE: {e}")
            raise
    
    def tune_hyperparameters(self, X_train_resampled: np.ndarray, y_train_resampled: np.ndarray) -> Dict[str, Any]:
        """
        Tune hyperparameters for different models using GridSearchCV.
        
        Args:
            X_train_resampled (np.ndarray): TF-IDF vectors for the balanced training set.
            y_train_resampled (np.ndarray): Labels for the balanced training set.
            
        Returns:
            Dict[str, Any]: Dictionary of best parameters for each model.
        """
        try:
            logger.info("Tuning hyperparameters for machine learning models")
            
            # Define parameter grids for each model
            param_grids = {
                'logistic_regression': {
                    'C': [0.1, 1.0, 10.0],
                    'solver': ['liblinear', 'saga'],
                    'penalty': ['l1', 'l2']
                },
                'decision_tree': {
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                },
                'gradient_boosting': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                },
                'naive_bayes': {
                    'alpha': [0.1, 0.5, 1.0, 2.0]
                },
                'knn': {
                    'n_neighbors': [3, 5, 7, 10],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                }
            }
            
            # Initialize models with best parameters
            self.best_params = {}
            tuned_models = {}
            
            # Tune each model
            for name, param_grid in param_grids.items():
                logger.info(f"Tuning hyperparameters for {name}")
                
                # Initialize base model based on name
                if name == 'logistic_regression':
                    model = LogisticRegression(random_state=42, max_iter=1000)
                elif name == 'decision_tree':
                    model = DecisionTreeClassifier(random_state=42)
                elif name == 'random_forest':
                    model = RandomForestClassifier(random_state=42)
                elif name == 'gradient_boosting':
                    model = GradientBoostingClassifier(random_state=42)
                elif name == 'naive_bayes':
                    model = MultinomialNB()
                elif name == 'knn':
                    model = KNeighborsClassifier()
                
                # Initialize grid search
                grid_search = GridSearchCV(
                    model, 
                    param_grid, 
                    cv=3, 
                    scoring='f1', 
                    n_jobs=-1,
                    verbose=1
                )
                
                # Fit grid search
                grid_search.fit(X_train_resampled, y_train_resampled)
                
                # Store best parameters and model
                self.best_params[name] = grid_search.best_params_
                tuned_models[name] = grid_search.best_estimator_
                
                logger.info(f"Best parameters for {name}: {grid_search.best_params_}")
                logger.info(f"Best score for {name}: {grid_search.best_score_:.4f}")
            
            logger.info("Hyperparameter tuning complete")
            return tuned_models
        except Exception as e:
            logger.error(f"Error tuning hyperparameters: {e}")
            raise
    
    def train_models(self, X_train_resampled: np.ndarray, y_train_resampled: np.ndarray, use_tuned: bool = True) -> Dict[str, Any]:
        """
        Train various machine learning models for sentiment classification.
        
        Args:
            X_train_resampled (np.ndarray): TF-IDF vectors for the balanced training set.
            y_train_resampled (np.ndarray): Labels for the balanced training set.
            use_tuned (bool): Whether to use hyperparameter tuning. Default is True.
            
        Returns:
            Dict[str, Any]: Dictionary of trained models.
        """
        try:
            logger.info("Training machine learning models")
            
            # If using tuned models, tune hyperparameters first
            if use_tuned:
                logger.info("Using hyperparameter tuning")
                tuned_models = self.tune_hyperparameters(X_train_resampled, y_train_resampled)
                self.models = tuned_models
            else:
                # Define the models with default parameters
                logger.info("Using default model parameters")
                models = {
                    'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
                    'decision_tree': DecisionTreeClassifier(random_state=42),
                    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                    'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                    'naive_bayes': MultinomialNB(),
                    'knn': KNeighborsClassifier(n_neighbors=5)
                }
                
                # Train each model
                for name, model in models.items():
                    logger.info(f"Training {name}")
                    model.fit(X_train_resampled, y_train_resampled)
                    self.models[name] = model
            
            # Create a voting classifier ensemble
            if len(self.models) >= 3:
                logger.info("Creating voting classifier ensemble")
                
                # Use the best performing models for the ensemble
                estimators = []
                for name, model in self.models.items():
                    if name != 'knn':  # Exclude KNN as it tends to perform poorly
                        estimators.append((name, model))
                
                # Create and train the voting classifier
                voting_clf = VotingClassifier(
                    estimators=estimators,
                    voting='soft'  # Use probability estimates for voting
                )
                voting_clf.fit(X_train_resampled, y_train_resampled)
                
                # Add the voting classifier to the models dictionary
                self.models['voting_ensemble'] = voting_clf
                self.ensemble_model = voting_clf
            
            logger.info("Model training complete")
            return self.models
        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise
    
    def evaluate_models(self, X_test_tfidf: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the trained models on the test set.
        
        Args:
            X_test_tfidf (np.ndarray): TF-IDF vectors for the test set.
            
        Returns:
            Dict[str, Dict[str, float]]: Dictionary of model metrics.
        """
        if not self.models:
            logger.error("No trained models found. Train models first.")
            raise ValueError("No trained models found. Train models first.")
        
        try:
            logger.info("Evaluating models on test data")
            
            for name, model in self.models.items():
                # Make predictions
                y_pred = model.predict(X_test_tfidf)
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred)
                recall = recall_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred)
                
                # Store metrics
                self.model_metrics[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
                
                # Log results
                logger.info(f"Model: {name}")
                logger.info(f"  Accuracy: {accuracy:.4f}")
                logger.info(f"  Precision: {precision:.4f}")
                logger.info(f"  Recall: {recall:.4f}")
                logger.info(f"  F1 Score: {f1:.4f}")
                
                # Generate classification report
                report = classification_report(self.y_test, y_pred, target_names=['negative', 'positive'])
                logger.info(f"Classification Report for {name}:\n{report}")
                
                # Generate confusion matrix
                cm = confusion_matrix(self.y_test, y_pred)
                logger.info(f"Confusion Matrix for {name}:\n{cm}")
            
            return self.model_metrics
        except Exception as e:
            logger.error(f"Error evaluating models: {e}")
            raise
    
    def cross_validate_models(self, X_train_resampled: np.ndarray, y_train_resampled: np.ndarray, cv: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Perform cross-validation on the trained models.
        
        Args:
            X_train_resampled (np.ndarray): TF-IDF vectors for the balanced training set.
            y_train_resampled (np.ndarray): Labels for the balanced training set.
            cv (int): Number of cross-validation folds. Default is 5.
            
        Returns:
            Dict[str, Dict[str, float]]: Dictionary of cross-validation results.
        """
        try:
            logger.info(f"Performing {cv}-fold cross-validation")
            
            cv_results = {}
            
            for name, model in self.models.items():
                logger.info(f"Cross-validating {name}")
                
                # Perform cross-validation
                cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='accuracy')
                
                # Store results
                cv_results[name] = {
                    'mean_accuracy': cv_scores.mean(),
                    'std_accuracy': cv_scores.std(),
                    'all_scores': cv_scores.tolist()
                }
                
                # Log results
                logger.info(f"  Mean Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            
            # Add cross-validation results to model metrics
            for name, cv_result in cv_results.items():
                self.model_metrics[name]['cv_mean_accuracy'] = cv_result['mean_accuracy']
                self.model_metrics[name]['cv_std_accuracy'] = cv_result['std_accuracy']
            
            return cv_results
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            raise
    
    def save_models(self) -> Dict[str, str]:
        """
        Save the trained models and vectorizer to disk.
        
        Returns:
            Dict[str, str]: Dictionary mapping model names to their file paths.
        """
        if not self.models or not self.vectorizer:
            logger.error("No trained models or vectorizer found.")
            raise ValueError("No trained models or vectorizer found.")
        
        try:
            logger.info(f"Saving models to {self.output_dir}")
            
            # Ensure the output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Save the vectorizer
            vectorizer_path = os.path.join(self.output_dir, 'tfidf_vectorizer.joblib')
            joblib.dump(self.vectorizer, vectorizer_path)
            logger.info(f"Saved vectorizer to {vectorizer_path}")
            
            # Save each model
            model_paths = {}
            for name, model in self.models.items():
                model_path = os.path.join(self.output_dir, f'{name}_model.joblib')
                joblib.dump(model, model_path)
                model_paths[name] = model_path
                logger.info(f"Saved {name} model to {model_path}")
            
            # Save model metrics
            metrics_path = os.path.join(self.output_dir, 'model_metrics.json')
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(self.model_metrics, f, indent=2)
            logger.info(f"Saved model metrics to {metrics_path}")
            
            return {
                'vectorizer': vectorizer_path,
                'models': model_paths,
                'metrics': metrics_path
            }
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise
    
    def plot_metrics(self) -> Dict[str, str]:
        """
        Plot model metrics for comparison and save the plots.
        
        Returns:
            Dict[str, str]: Dictionary mapping plot types to their file paths.
        """
        if not self.model_metrics:
            logger.error("No model metrics found. Evaluate models first.")
            raise ValueError("No model metrics found. Evaluate models first.")
        
        try:
            logger.info("Plotting model performance metrics")
            
            # Ensure the output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            plot_paths = {}
            
            # Extract metrics for plotting
            model_names = list(self.model_metrics.keys())
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            
            # Plot comparison of metrics
            plt.figure(figsize=(12, 8))
            
            for i, metric in enumerate(metrics, 1):
                plt.subplot(2, 2, i)
                values = [self.model_metrics[model][metric] for model in model_names]
                
                # Create bar plot
                bars = plt.bar(model_names, values, color='steelblue')
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                             f'{height:.3f}', ha='center', va='bottom', rotation=0)
                
                plt.title(f'Model Comparison - {metric.replace("_", " ").title()}')
                plt.xticks(rotation=45, ha='right')
                plt.ylim(0, 1.0)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            metrics_plot_path = os.path.join(self.output_dir, 'model_metrics_comparison.png')
            plt.savefig(metrics_plot_path, dpi=300)
            plt.close()
            
            plot_paths['metrics_comparison'] = metrics_plot_path
            logger.info(f"Saved metrics comparison plot to {metrics_plot_path}")
            
            # Plot cross-validation results
            if 'cv_mean_accuracy' in next(iter(self.model_metrics.values())):
                plt.figure(figsize=(10, 6))
                
                model_names = list(self.model_metrics.keys())
                mean_values = [self.model_metrics[model]['cv_mean_accuracy'] for model in model_names]
                std_values = [self.model_metrics[model]['cv_std_accuracy'] for model in model_names]
                
                # Create bar plot with error bars
                bars = plt.bar(model_names, mean_values, yerr=std_values, 
                               capsize=5, color='lightgreen', edgecolor='darkgreen')
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                             f'{height:.3f}', ha='center', va='bottom', rotation=0)
                
                plt.title('Cross-Validation Results - Mean Accuracy with Standard Deviation')
                plt.xticks(rotation=45, ha='right')
                plt.ylim(0, 1.0)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                cv_plot_path = os.path.join(self.output_dir, 'cross_validation_results.png')
                plt.savefig(cv_plot_path, dpi=300)
                plt.close()
                
                plot_paths['cross_validation'] = cv_plot_path
                logger.info(f"Saved cross-validation results plot to {cv_plot_path}")
            
            return plot_paths
        except Exception as e:
            logger.error(f"Error plotting metrics: {e}")
            raise
    
    def run_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete model training pipeline.
        
        Returns:
            Dict[str, Any]: Dictionary containing all results and file paths.
        """
        try:
            logger.info("Starting the complete model training pipeline")
            
            # Load and prepare data
            self.load_data()
            self.filter_neutral_sentiments()
            self.prepare_data()
            
            # Apply TF-IDF vectorization
            X_train_tfidf, X_test_tfidf = self.apply_tfidf()
            
            # Apply SMOTE for class balancing
            X_train_resampled, y_train_resampled = self.apply_smote(X_train_tfidf)
            
            # Train models
            self.train_models(X_train_resampled, y_train_resampled)
            
            # Evaluate models
            self.evaluate_models(X_test_tfidf)
            
            # Cross-validate models
            self.cross_validate_models(X_train_resampled, y_train_resampled)
            
            # Save models
            saved_paths = self.save_models()
            
            # Plot metrics
            plot_paths = self.plot_metrics()
            
            logger.info("Model training pipeline completed successfully")
            
            # Return all results
            return {
                'metrics': self.model_metrics,
                'saved_paths': saved_paths,
                'plot_paths': plot_paths,
                'best_model': self._get_best_model()
            }
        except Exception as e:
            logger.error(f"Error in model training pipeline: {e}")
            raise
    
    def _get_best_model(self) -> str:
        """
        Get the name of the best performing model based on F1 score.
        
        Returns:
            str: Name of the best model.
        """
        if not self.model_metrics:
            return None
        
        # Find the model with the highest F1 score
        best_model = max(self.model_metrics.items(), key=lambda x: x[1]['f1_score'])
        return best_model[0]

def main():
    """
    Main function to execute the model training.
    """
    try:
        # Define the input file and output directory
        data_file = os.path.join('data', 'processed', 'sentences_sentiment.csv')
        models_dir = os.path.join('models')
        
        # Initialize the model trainer
        trainer = SentimentModelTrainer(data_file, models_dir)
        
        # Run the complete pipeline
        results = trainer.run_pipeline()
        
        # Log the best model
        best_model = results['best_model']
        logger.info(f"Best performing model: {best_model}")
        logger.info(f"Best model metrics: {trainer.model_metrics[best_model]}")
        
        # Create a summary file
        summary_path = os.path.join(models_dir, 'training_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Sentiment Analysis Model Training Summary\n")
            f.write("========================================\n\n")
            f.write(f"Best model: {best_model}\n")
            f.write(f"Best model metrics: {trainer.model_metrics[best_model]}\n\n")
            f.write("All models performance:\n")
            for model_name, metrics in trainer.model_metrics.items():
                f.write(f"- {model_name}:\n")
                for metric_name, value in metrics.items():
                    f.write(f"  {metric_name}: {value}\n")
                f.write("\n")
        logger.info(f"Saved training summary to {summary_path}")
        
    except Exception as e:
        logger.error(f"An error occurred during model training: {e}")

if __name__ == "__main__":
    main()
