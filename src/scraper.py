# Simple Wikipedia Scraper for Sweden Content
# ------------------------------------------

import requests
from bs4 import BeautifulSoup
import re
import os

# Step 1: Define the Wikipedia URL
url = "https://en.wikipedia.org/wiki/Sweden"

# Step 2: Send GET request to fetch the page content
response = requests.get(url)

# Step 3: Parse HTML using BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Step 4: Locate the main content section
content_div = soup.find('div', {'id': 'mw-content-text'})
parser_output = content_div.find('div', {'class': 'mw-parser-output'})

# Step 5: Extract and clean text from paragraphs
paragraphs = parser_output.find_all('p')
text = ""
for para in paragraphs:
    text += para.get_text()

# Step 6: Clean up the text
text = re.sub(r'\[\d+\]', '', text)              # Remove references like [1]
text = re.sub(r'\s+', ' ', text).strip()           # Remove extra whitespace
text = re.sub(r'\s+([.,;:!?])', r'\1', text)       # Fix punctuation spacing

# Step 7: Save text to a file
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, r"C:\Data\Rachit\NMIMS\Sem 6\Applied Artificial Intelligence\sweden_sentiment\data\sweden_wikipedia.txt"), "w", encoding="utf-8") as f:
    f.write(text)

print("Wikipedia text scraped and saved to 'data/sweden_wikipedia.txt'")

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load cleaned text file
with open("data/sweden_wikipedia.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Generate word cloud
wordcloud = WordCloud(width=1000, height=500, background_color='white').generate(text)

# Display it
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout()
plt.show()
