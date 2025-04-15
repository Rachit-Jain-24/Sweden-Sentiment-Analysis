# Simple Model Trainer for Sweden Sentiment Analysis
# --------------------------------------------------
# Instructions Covered:
# 11) Apply TF-IDF Vectorizer on sentence/sentiment
# 12) Filter to only positive & negative
# 13) Apply SMOTE
# 14a-f) Train all six ML models

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import joblib
import os
import json

# Step 1: Load sentiment-labeled sentences
print("Loading sentiment data...")
df = pd.read_csv("data/sentences_sentiment.csv")

# Step 2: Filter only positive & negative (✔️ Instruction 12)
df = df[df['sentiment'] != 'neutral']
df = df.reset_index(drop=True)

# Step 3: Apply TF-IDF Vectorizer (✔️ Instruction 11)
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['sentence'])
y = df['sentiment'].map({'negative': 0, 'positive': 1})

# Step 4: Apply SMOTE (✔️ Instruction 13)
print("Applying SMOTE to balance classes...")
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Step 5: Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 6: Train all required models (✔️ Instruction 14a-f)
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Naive Bayes": MultinomialNB(),
    "KNN": KNeighborsClassifier()
}

metrics = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics[name] = report['weighted avg']

# Step 7: Create and save Voting Ensemble model
ensemble = VotingClassifier(estimators=[
    ('lr', models['Logistic Regression']),
    ('rf', models['Random Forest']),
    ('gb', models['Gradient Boosting']),
    ('nb', models['Naive Bayes']),
    ('knn', models['KNN']),
    ('dt', models['Decision Tree'])
], voting='soft')
ensemble.fit(X_train, y_train)

# Step 8: Save best models and vectorizer
os.makedirs("models", exist_ok=True)
for name, model in models.items():
    joblib.dump(model, f"models/{name.lower().replace(' ', '_')}_model.joblib")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

# Step 9: Save model performance report
with open("models/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Step 10: Plot cross-validation results
model_names = list(models.keys())
cv_scores = [cross_val_score(m, X_resampled, y_resampled, cv=5).mean() for m in models.values()]

plt.figure(figsize=(10, 5))
plt.bar(model_names, cv_scores, color='lightgreen')
plt.title("Model Cross-Validation Accuracy")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("models/cross_validation_results.png")
plt.close()

print("✔️ Model training complete. Artifacts saved in 'models/' folder.")