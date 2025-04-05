#!/usr/bin/env python
"""
Text Preprocessor for Wikipedia Content

This script handles text preprocessing, sentence tokenization, sentiment analysis,
word tokenization, stopword removal, and word frequency analysis for the scraped
Wikipedia content about Sweden.
"""

import os
import re
import logging
import json
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from textblob import TextBlob
from pathlib import Path
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Optional
import string

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')

class TextPreprocessor:
    """
    A class for preprocessing text, tokenizing sentences, analyzing sentiment,
    and generating word frequency counts.
    """
    
    def __init__(self, input_file: str):
        """
        Initialize the preprocessor with the input file path.
        
        Args:
            input_file (str): Path to the input text file.
        """
        self.input_file = input_file
        self.raw_text = ""
        self.cleaned_text = ""
        self.sentences = []
        self.sentence_sentiments = pd.DataFrame()
        self.words = []
        self.words_no_stopwords = []
        self.word_freq = None
        
    def load_text(self) -> str:
        """
        Load text from the input file.
        
        Returns:
            str: The loaded text content.
        
        Raises:
            FileNotFoundError: If the input file does not exist.
        """
        try:
            logger.info(f"Loading text from {self.input_file}")
            with open(self.input_file, 'r', encoding='utf-8') as f:
                self.raw_text = f.read()
            logger.info(f"Loaded {len(self.raw_text)} characters of text")
            return self.raw_text
        except FileNotFoundError:
            logger.error(f"Input file not found: {self.input_file}")
            raise
        except Exception as e:
            logger.error(f"Error loading text: {e}")
            raise
    
    def clean_text(self) -> str:
        """
        Clean the loaded text by removing special characters, extra whitespace, etc.
        
        Returns:
            str: The cleaned text.
        """
        if not self.raw_text:
            self.load_text()
        
        try:
            logger.info("Cleaning text")
            
            # Remove URLs
            cleaned = re.sub(r'http\S+', '', self.raw_text)
            
            # Remove citations
            cleaned = re.sub(r'\[\d+\]', '', cleaned)
            
            # Remove special characters but keep sentence punctuation
            cleaned = re.sub(r'[^\w\s.,!?;:]', ' ', cleaned)
            
            # Remove extra whitespace and normalize
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            self.cleaned_text = cleaned
            logger.info(f"Text cleaned. New length: {len(self.cleaned_text)} characters")
            return self.cleaned_text
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            raise
    
    def tokenize_sentences(self) -> List[str]:
        """
        Tokenize the cleaned text into sentences using TextBlob.
        
        Returns:
            List[str]: List of sentences.
        """
        if not self.cleaned_text:
            self.clean_text()
        
        try:
            logger.info("Tokenizing sentences")
            blob = TextBlob(self.cleaned_text)
            self.sentences = [str(sentence) for sentence in blob.sentences]
            logger.info(f"Tokenized {len(self.sentences)} sentences")
            return self.sentences
        except Exception as e:
            logger.error(f"Error tokenizing sentences: {e}")
            raise
    
    def analyze_sentiment(self) -> pd.DataFrame:
        """
        Analyze sentiment of each sentence using TextBlob.
        
        Returns:
            pd.DataFrame: DataFrame with sentences and their sentiment scores.
        """
        if not self.sentences:
            self.tokenize_sentences()
        
        try:
            logger.info("Analyzing sentiment of sentences")
            sentiments = []
            
            for sentence in self.sentences:
                blob = TextBlob(sentence)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                # Determine sentiment label
                if polarity > 0.05:
                    sentiment = "positive"
                elif polarity < -0.05:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
                
                sentiments.append({
                    'sentence': sentence,
                    'polarity': polarity,
                    'subjectivity': subjectivity,
                    'sentiment': sentiment
                })
            
            self.sentence_sentiments = pd.DataFrame(sentiments)
            logger.info(f"Sentiment analysis complete. DataFrame shape: {self.sentence_sentiments.shape}")
            return self.sentence_sentiments
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            raise
    
    def tokenize_words(self) -> List[str]:
        """
        Tokenize the cleaned text into words using NLTK.
        
        Returns:
            List[str]: List of words.
        """
        if not self.cleaned_text:
            self.clean_text()
        
        try:
            logger.info("Tokenizing words")
            # Convert to lowercase for better word frequency analysis
            lowercase_text = self.cleaned_text.lower()
            
            # Remove punctuation
            no_punct_text = lowercase_text.translate(str.maketrans('', '', string.punctuation))
            
            # Tokenize words
            self.words = word_tokenize(no_punct_text)
            logger.info(f"Tokenized {len(self.words)} words")
            return self.words
        except Exception as e:
            logger.error(f"Error tokenizing words: {e}")
            raise
    
    def remove_stopwords(self) -> List[str]:
        """
        Remove stopwords from the tokenized words.
        
        Returns:
            List[str]: List of words with stopwords removed.
        """
        if not self.words:
            self.tokenize_words()
        
        try:
            logger.info("Removing stopwords")
            
            # Get English stopwords
            stop_words = set(stopwords.words('english'))
            
            # Remove stopwords
            self.words_no_stopwords = [word for word in self.words if word not in stop_words]
            
            logger.info(f"Removed stopwords. {len(self.words_no_stopwords)} words remaining")
            return self.words_no_stopwords
        except Exception as e:
            logger.error(f"Error removing stopwords: {e}")
            raise
    
    def generate_word_frequency(self, n: int = 50) -> Dict[str, int]:
        """
        Generate word frequency counts for the top N words.
        
        Args:
            n (int): Number of top words to include. Default is 50.
            
        Returns:
            Dict[str, int]: Dictionary of words and their frequencies.
        """
        if not self.words_no_stopwords:
            self.remove_stopwords()
        
        try:
            logger.info(f"Generating word frequency for top {n} words")
            
            # Calculate word frequencies
            self.word_freq = FreqDist(self.words_no_stopwords)
            
            # Get top N words
            top_words = dict(self.word_freq.most_common(n))
            
            logger.info(f"Generated frequency counts for top {len(top_words)} words")
            return top_words
        except Exception as e:
            logger.error(f"Error generating word frequency: {e}")
            raise
    
    def generate_wordcloud(self, output_file: str) -> None:
        """
        Generate a WordCloud visualization and save it to a file.
        
        Args:
            output_file (str): Path to save the WordCloud image.
        """
        if not self.words_no_stopwords:
            self.remove_stopwords()
        
        try:
            logger.info(f"Generating WordCloud and saving to {output_file}")
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Create a string of all words for WordCloud
            text = ' '.join(self.words_no_stopwords)
            
            # Generate WordCloud
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                max_words=200,
                contour_width=3, 
                contour_color='steelblue'
            ).generate(text)
            
            # Save the image
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(output_file, dpi=300)
            plt.close()
            
            logger.info(f"WordCloud saved to {output_file}")
        except Exception as e:
            logger.error(f"Error generating WordCloud: {e}")
            raise
    
    def save_processed_data(self, output_dir: str) -> Dict[str, str]:
        """
        Save all processed data to files in the output directory.
        
        Args:
            output_dir (str): Directory to save the processed data.
            
        Returns:
            Dict[str, str]: Dictionary mapping data type to output file paths.
        """
        try:
            # Ensure all processing steps have been completed
            if not self.sentence_sentiments.empty or not self.word_freq:
                self.analyze_sentiment()
                self.generate_word_frequency()
            
            logger.info(f"Saving processed data to {output_dir}")
            
            # Ensure the output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # File paths
            sentences_csv = os.path.join(output_dir, 'sentences_sentiment.csv')
            word_freq_json = os.path.join(output_dir, 'word_frequency.json')
            cleaned_text_file = os.path.join(output_dir, 'cleaned_text.txt')
            wordcloud_image = os.path.join(output_dir, 'wordcloud.png')
            
            # Save sentence sentiments DataFrame to CSV
            self.sentence_sentiments.to_csv(sentences_csv, index=False)
            logger.info(f"Saved sentence sentiments to {sentences_csv}")
            
            # Save word frequency to JSON
            with open(word_freq_json, 'w', encoding='utf-8') as f:
                json.dump(dict(self.word_freq.most_common(100)), f, ensure_ascii=False, indent=2)
            logger.info(f"Saved word frequency to {word_freq_json}")
            
            # Save cleaned text
            with open(cleaned_text_file, 'w', encoding='utf-8') as f:
                f.write(self.cleaned_text)
            logger.info(f"Saved cleaned text to {cleaned_text_file}")
            
            # Generate and save WordCloud
            self.generate_wordcloud(wordcloud_image)
            
            return {
                'sentences_sentiment': sentences_csv,
                'word_frequency': word_freq_json,
                'cleaned_text': cleaned_text_file,
                'wordcloud': wordcloud_image
            }
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise
    
    def process_all(self, output_dir: str) -> Dict[str, Any]:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            output_dir (str): Directory to save the processed data.
            
        Returns:
            Dict[str, Any]: Dictionary containing all processed data and file paths.
        """
        try:
            logger.info("Starting complete preprocessing pipeline")
            
            # Execute all processing steps
            self.load_text()
            self.clean_text()
            self.tokenize_sentences()
            self.analyze_sentiment()
            self.tokenize_words()
            self.remove_stopwords()
            self.generate_word_frequency()
            
            # Save all processed data
            output_files = self.save_processed_data(output_dir)
            
            logger.info("Preprocessing pipeline completed successfully")
            
            # Return processed data and file paths
            return {
                'text_stats': {
                    'raw_text_length': len(self.raw_text),
                    'cleaned_text_length': len(self.cleaned_text),
                    'sentence_count': len(self.sentences),
                    'word_count': len(self.words),
                    'word_count_no_stopwords': len(self.words_no_stopwords)
                },
                'sentence_sentiments': self.sentence_sentiments,
                'word_freq': dict(self.word_freq.most_common(100)),
                'output_files': output_files
            }
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {e}")
            raise

def main():
    """
    Main function to execute the text preprocessing.
    """
    try:
        # Define the input and output file paths
        input_file = os.path.join('data', 'sweden_wikipedia.txt')
        output_dir = os.path.join('data', 'processed')
        
        # Initialize the preprocessor
        preprocessor = TextPreprocessor(input_file)
        
        # Run the complete preprocessing pipeline
        results = preprocessor.process_all(output_dir)
        
        # Log some statistics
        stats = results['text_stats']
        logger.info(f"Text processing complete. Stats: {stats}")
        
        # Log sentiment distribution
        sentiment_counts = results['sentence_sentiments']['sentiment'].value_counts()
        logger.info(f"Sentiment distribution: {sentiment_counts.to_dict()}")
        
    except Exception as e:
        logger.error(f"An error occurred during preprocessing: {e}")

if __name__ == "__main__":
    main()

