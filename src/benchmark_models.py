
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test.csv"
RESULTS_DIR = ROOT / "results"

def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train_text = train_df["text"].fillna("").astype(str)
    y_train = train_df["category"]
    X_test_text = test_df["text"].fillna("").astype(str)
    y_test = test_df["category"]

    print(f"Training examples: {len(train_df)}")
    print(f"Test examples: {len(test_df)}")
    print(f"Categories: {y_train.nunique()}")

    # Fit TF-IDF on training text only to prevent test-data leakage.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
        "Linear SVM": LinearSVC(
            random_state=42,
        ),
        "Complement Naive Bayes": ComplementNB(
            alpha=0.1,
        ),
    }

    results = []

    for name, model in models.items():
        print(f"\n{'=' * 60}")
        print(f"Training: {name}")

        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        training_seconds = time.perf_counter() - start_time

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(
            y_test, predictions, average="macro", zero_division=0
        )
        weighted_f1 = f1_score(
            y_test, predictions, average="weighted", zero_division=0
        )

        print(f"Accuracy:    {accuracy:.4f}")
        print(f"Macro F1:    {macro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")
        print(f"Training time: {training_seconds:.2f} seconds")

        print("\nClassification report:")
        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0,
            )
        )

        results.append({
            "model": name,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "training_seconds": training_seconds,
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results).sort_values(
        "macro_f1", ascending=False
    )
    output_path = RESULTS_DIR / "model_benchmark.csv"
    results_df.to_csv(output_path, index=False)

    print("\nFinal comparison:")
    print(results_df.to_string(index=False))
    print(f"\nSaved benchmark results to: {output_path}")

if __name__ == "__main__":
    main()
