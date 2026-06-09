# train_model.py

import pandas as pd
import numpy as np
import re
import string
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# =====================================================
# 1. LOAD DATASET
# =====================================================

print("=" * 60)
print("Loading dataset...")
print("=" * 60)

df = pd.read_csv("data/AI_Human.csv")

print(f"Original Dataset Shape: {df.shape}")

# =====================================================
# 2. SAMPLE DATASET
# =====================================================

# Use 100k rows for faster training
df = df.sample(n=100000, random_state=42)

print(f"Sampled Dataset Shape: {df.shape}")

# =====================================================
# 3. HANDLE MISSING VALUES
# =====================================================

df.dropna(inplace=True)

# =====================================================
# 4. TEXT CLEANING FUNCTION
# =====================================================

def clean_text(text):
    """
    Basic NLP preprocessing
    """

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


print("\nCleaning text...")

df["clean_text"] = df["text"].apply(clean_text)

# =====================================================
# 5. FEATURES & TARGET
# =====================================================

X = df["clean_text"]
y = df["generated"]

# =====================================================
# 6. TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# =====================================================
# 7. TF-IDF VECTORIZATION
# =====================================================

print("\nApplying TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF Shape:", X_train_tfidf.shape)

# =====================================================
# 8. MODELS
# =====================================================

models = {
    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Linear SVM": LinearSVC(
        random_state=42,
    )
}

# =====================================================
# 9. TRAIN & EVALUATE
# =====================================================

results = []

best_model = None
best_model_name = ""
best_f1 = 0

print("\n" + "=" * 60)
print("TRAINING MODELS")
print("=" * 60)

for name, model in models.items():

    print(f"\n{name}")

    # Train
    model.fit(X_train_tfidf, y_train)

    # Predict
    y_pred = model.predict(X_test_tfidf)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 4),
        "F1 Score": round(f1, 4)
    })

    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    # Save best model
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

# =====================================================
# 10. RESULTS TABLE
# =====================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df)

# =====================================================
# 11. SAVE BEST MODEL
# =====================================================

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(best_model_name)
print(f"Best F1 Score: {best_f1:.4f}")

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

print("\nModel Saved Successfully!")

# =====================================================
# 12. SAVE RESULTS
# =====================================================

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

print("Comparison Table Saved!")

from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC

model = LinearSVC(random_state=42)

scores = cross_val_score(
    model,
    X_train_tfidf,
    y_train,
    cv=5,
    scoring="f1"
)

print(scores)
print("Mean F1:", scores.mean())