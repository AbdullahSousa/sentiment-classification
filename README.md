# Sentiment Classification (IMDB Reviews)

A machine-learning model that classifies movie reviews as **positive** or **negative**, trained on the IMDB dataset of 50,000 labelled reviews.

## Approach

1. **Cleaning:** lower-case the text, strip HTML tags and remove non-letter characters.
2. **Features:** TF-IDF vectorisation, limited to the 5,000 most informative words.
3. **Model:** Multinomial Naive Bayes, wrapped together with TF-IDF in a scikit-learn `Pipeline`.
4. **Evaluation:** 80/20 train-test split, with accuracy, a full classification report (precision, recall, F1) and a confusion-matrix heatmap.
5. **Export:** the trained pipeline is saved to `sentiment_model.pkl` with `joblib`, so it can be reused without retraining.

## Usage

```bash
pip install pandas scikit-learn matplotlib seaborn joblib
python naivebayes.py
```

Loading the saved model in your own code:

```python
import joblib

model = joblib.load("sentiment_model.pkl")
model.predict(["One of the best movies I've seen"])  # -> [1] (positive)
```

## Files

| File | Description |
|---|---|
| `naivebayes.py` | Training, evaluation and sample predictions |
| `IMDB Dataset.csv` | IMDB Large Movie Review dataset (50,000 reviews) |
| `sentiment_model.pkl` | Trained TF-IDF + Naive Bayes pipeline |

## Tech stack

Python · pandas · scikit-learn · Matplotlib · Seaborn · joblib
