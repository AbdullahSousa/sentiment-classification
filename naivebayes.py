import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
# Dataset Source: IMDB Large Movie Review Dataset (50,000 reviews labeled as positive/negative)
data = pd.read_csv("IMDB Dataset.csv")

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)  # remove HTML tags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove non-letter characters
    return text

# Apply text cleaning
data['clean_review'] = data['review'].apply(clean_text)

# Label encoding: positive = 1, negative = 0
data['label'] = data['sentiment'].map({'positive': 1, 'negative': 0})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data['clean_review'], data['label'], test_size=0.2, random_state=42
)

# Build pipeline: TF-IDF Vectorizer + Naive Bayes classifier
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('nb', MultinomialNB())
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Evaluate on test set
y_pred = model_pipeline.predict(X_test)
accuracy = model_pipeline.score(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Classification metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix heatmap
labels = ['Negative', 'Positive']
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Sample predictions
sample_reviews = [
    "This movie was awful",
    "One of the best movies I've seen"
]
sample_preds = model_pipeline.predict(sample_reviews)
print("\nSample Predictions:")
for s, p in zip(sample_reviews, sample_preds):
    print(f"Review: {s}\nPredicted Sentiment: {'Positive' if p == 1 else 'Negative'}\n")

# Optional: Save model for future use
joblib.dump(model_pipeline, 'sentiment_model.pkl')
