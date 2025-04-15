# Simple Preprocessing Script for Sweden Sentiment Analysis
# --------------------------------------------------------

import re
import os
import pandas as pd
import nltk
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')

# Step 1: Load raw scraped text
with open("data/sweden_wikipedia.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Step 2: Clean the text (✔️ Instruction 3)
clean_text = re.sub(r'\[\d+\]', '', raw_text)           # remove references
clean_text = re.sub(r'\s+', ' ', clean_text).strip()     # remove extra whitespace

# Step 3: Sentence tokenization (✔️ Instruction 4)
sentences = sent_tokenize(clean_text)

# Step 4: Sentiment analysis using TextBlob (✔️ Instruction 5)
sentence_data = []
for sentence in sentences:
    blob = TextBlob(sentence)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        sentiment = 'positive'
    elif polarity < 0:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    sentence_data.append((sentence, sentiment))

# Step 5: Create a DataFrame (✔️ Instruction 6)
df_sentiment = pd.DataFrame(sentence_data, columns=['sentence', 'sentiment'])

# Count the number of positive, negative, and neutral sentences
print(df_sentiment['sentiment'].value_counts())

df_sentiment.to_csv("data/sentences_sentiment.csv", index=False)

# Step 6: Word tokenization (✔️ Instruction 7)
words = word_tokenize(clean_text)

# Step 7: Remove stopwords (✔️ Instruction 8)
stop_words = set(stopwords.words('english'))
filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]

# Step 8: WordCloud generation (✔️ Instruction 9)
wordcloud_text = ' '.join(filtered_words)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of Sweden Wikipedia Text")
plt.tight_layout()
os.makedirs("data/processed", exist_ok=True)
plt.savefig("data/processed/wordcloud.png")
plt.close()

# Step 9: Frequency of top words (✔️ Instruction 10)
word_counts = Counter(filtered_words)
most_common_words = word_counts.most_common(20)

# Plot and save frequency chart
words, freqs = zip(*most_common_words)
plt.figure(figsize=(10, 5))
plt.bar(words, freqs, color='skyblue')
plt.xticks(rotation=45)
plt.title("Top 20 Frequent Words")
plt.tight_layout()
plt.savefig("data/processed/frequent_words.png")
plt.close()

print("✔️ Preprocessing complete. Outputs saved in 'data/' and 'data/processed/'")
