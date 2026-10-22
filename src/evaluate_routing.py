
from pathlib import Path
import time

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df["text"].fillna("").astype(str)
y = df["category"]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Training calibrated Linear SVM on the training split...")
print(f"Training examples: {len(X_train)}")
print(f"Validation examples: {len(X_val)}")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True,
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

model = CalibratedClassifierCV(
    estimator=LinearSVC(random_state=42),
    method="sigmoid",
    cv=3,
)

start_time = time.perf_counter()
model.fit(X_train_vec, y_train)
training_seconds = time.perf_counter() - start_time

probabilities = model.predict_proba(X_val_vec)
predicted_indices = probabilities.argmax(axis=1)
predictions = model.classes_[predicted_indices]
confidence = probabilities.max(axis=1)

accuracy = accuracy_score(y_val, predictions)
macro_f1 = f1_score(
    y_val, predictions, average="macro", zero_division=0
)
weighted_f1 = f1_score(
    y_val, predictions, average="weighted", zero_division=0
)
validation_log_loss = log_loss(
    y_val,
    probabilities,
    labels=model.classes_,
)

print("\nVALIDATION RESULTS")
print("=" * 50)
print(f"Validation examples: {len(y_val)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print(f"Log loss: {validation_log_loss:.4f}")
print(f"Training time: {training_seconds:.2f} seconds")

print("\nClassification report:")
print(
    classification_report(
        y_val,
        predictions,
        zero_division=0,
    )
)

print("\nROUTING THRESHOLD ANALYSIS")
print("=" * 65)

threshold_rows = []

for threshold in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    accepted = confidence >= threshold
    accepted_count = int(accepted.sum())
    coverage = float(accepted.mean())

    if accepted_count:
        accepted_accuracy = accuracy_score(
            y_val.to_numpy()[accepted],
            predictions[accepted],
        )
        accepted_macro_f1 = f1_score(
            y_val.to_numpy()[accepted],
            predictions[accepted],
            average="macro",
            zero_division=0,
        )
    else:
        accepted_accuracy = float("nan")
        accepted_macro_f1 = float("nan")

    row = {
        "threshold": threshold,
        "validation_examples": len(y_val),
        "accepted_examples": accepted_count,
        "coverage": coverage,
        "accepted_accuracy": accepted_accuracy,
        "accepted_macro_f1": accepted_macro_f1,
    }
    threshold_rows.append(row)

    print(
        f"Threshold: {threshold:.0%} | "
        f"Accepted: {accepted_count}/{len(y_val)} | "
        f"Coverage: {coverage:.1%} | "
        f"Accepted accuracy: "
        f"{accepted_accuracy:.1%}"
        if accepted_count
        else
        f"Threshold: {threshold:.0%} | "
        f"Accepted: 0/{len(y_val)} | "
        f"Coverage: 0.0% | "
        "Accepted accuracy: N/A"
    )

threshold_path = RESULTS_DIR / "routing_thresholds.csv"
pd.DataFrame(threshold_rows).to_csv(
    threshold_path,
    index=False,
)

summary = pd.DataFrame([{
    "model": "Calibrated Linear SVM",
    "validation_examples": len(y_val),
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "weighted_f1": weighted_f1,
    "log_loss": validation_log_loss,
    "training_seconds": training_seconds,
}])

summary_path = RESULTS_DIR / "routing_validation_summary.csv"
summary.to_csv(summary_path, index=False)

print(f"\nThreshold results saved to: {threshold_path}")
print(f"Validation summary saved to: {summary_path}")
