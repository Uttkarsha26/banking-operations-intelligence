
from pathlib import Path
import time

import joblib
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

X_train = train_df["text"].fillna("").astype(str)
y_train = train_df["category"]
X_test = test_df["text"].fillna("").astype(str)
y_test = test_df["category"]

print(f"Training examples: {len(X_train)}")
print(f"Test examples: {len(X_test)}")
print(f"Categories: {y_train.nunique()}")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True,
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

base_model = LinearSVC(random_state=42)

model = CalibratedClassifierCV(
    estimator=base_model,
    method="sigmoid",
    cv=3,
)

print("\nTraining calibrated Linear SVM...")
start = time.perf_counter()
model.fit(X_train_tfidf, y_train)
training_seconds = time.perf_counter() - start

predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(
    y_test, predictions, average="macro", zero_division=0
)
weighted_f1 = f1_score(
    y_test, predictions, average="weighted", zero_division=0
)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print(f"Training time: {training_seconds:.2f} seconds")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save only after successful training and evaluation.
joblib.dump(model, MODEL_DIR / "intent_model.joblib")
joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")

print("\nCalibrated Linear SVM and vectorizer saved.")
