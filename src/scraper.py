#!/usr/bin/env python
"""
Wikipedia Scraper for Sweden Content

This script scrapes the Wikipedia page for Sweden and extracts the text content
from the main article body. It saves the raw text to a file in the data directory.
"""

import os
import logging
import requests
from bs4 import BeautifulSoup
import re
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WikipediaScraper:
    """
    A class for scraping text content from Wikipedia pages.
    """
    
    def __init__(self, url: str):
        """
        Initialize the scraper with the URL to scrape.
        
        Args:
            url (str): The URL of the Wikipedia page to scrape.
        """
        self.url = url
        
    def fetch_page(self) -> str:
        """
        Fetch the HTML content of the Wikipedia page.
        
        Returns:
            str: HTML content of the page.
            
        Raises:
            requests.exceptions.RequestException: If there is an error fetching the page.
        """
        try:
            logger.info(f"Fetching content from {self.url}")
            response = requests.get(self.url, timeout=10)
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching page: {e}")
            raise
    
    def extract_text(self, html_content: str) -> str:
        """
        Extract text content from the HTML of a Wikipedia page.
        
        Args:
            html_content (str): The HTML content of the Wikipedia page.
            
        Returns:
            str: Extracted text content.
        """
        try:
            logger.info("Parsing HTML content")
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find the main content div
            content_div = soup.find('div', {'id': 'mw-content-text'})
            if not content_div:
                logger.error("Could not find main content div")
                return ""
            
            # Get all paragraphs within the content div
            paragraphs = content_div.find_all('p')
            
            # Extract text from each paragraph and join with newlines
            text_content = '\n\n'.join([p.get_text() for p in paragraphs])
            
            # Clean up the text
            # Remove citations like [1], [2], etc.
            text_content = re.sub(r'\[\d+\]', '', text_content)
            # Remove extra whitespace
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            
            logger.info(f"Extracted {len(text_content)} characters of text")
            return text_content
            
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return ""
    
    def save_to_file(self, text_content: str, file_path: str) -> None:
        """
        Save the extracted text content to a file.
        
        Args:
            text_content (str): The text content to save.
            file_path (str): The path to the file where the content will be saved.
        """
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            logger.info(f"Saving content to {file_path}")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            logger.info("Content saved successfully")
        except Exception as e:
            logger.error(f"Error saving to file: {e}")
            raise
    
    def scrape_and_save(self, file_path: str) -> str:
        """
        Scrape the Wikipedia page and save the content to a file.
        
        Args:
            file_path (str): The path to the file where the content will be saved.
            
        Returns:
            str: The extracted text content.
        """
        html_content = self.fetch_page()
        text_content = self.extract_text(html_content)
        self.save_to_file(text_content, file_path)
        return text_content

def main():
    """
    Main function to execute the Wikipedia scraping.
    """
    try:
        # URL of the Wikipedia page for Sweden
        url = "https://en.wikipedia.org/wiki/Sweden"
        
        # Initialize the scraper
        scraper = WikipediaScraper(url)
        
        # Define the output file path
        output_dir = Path("data")
        output_file = output_dir / "sweden_wikipedia.txt"
        
        # Scrape and save the content
        text_content = scraper.scrape_and_save(str(output_file))
        
        logger.info(f"Successfully scraped and saved {len(text_content)} characters of text")
        
    except Exception as e:
        logger.error(f"An error occurred during scraping: {e}")

if __name__ == "__main__":
    main()

