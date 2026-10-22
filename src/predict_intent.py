
from pathlib import Path

import joblib

# Locate the saved model files.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

model = joblib.load(MODEL_DIR / "intent_model.joblib")
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")

# This is a provisional threshold, not a validated business rule.
CONFIDENCE_THRESHOLD = 0.60


def predict_intent(message):
    """Predict a customer's intent and flag uncertain predictions."""

    message = message.strip()

    if not message:
        raise ValueError("Please enter a non-empty message.")

    # Convert the message into the same features used during training.
    message_tfidf = vectorizer.transform([message])

    # Obtain class probabilities from Logistic Regression.
    probabilities = model.predict_proba(message_tfidf)[0]

    # Find the three categories with the highest scores.
    top_indices = probabilities.argsort()[-3:][::-1]

    top_predictions = [
        {
            "category": model.classes_[index],
            "score": float(probabilities[index]),
        }
        for index in top_indices
    ]

    best_prediction = top_predictions[0]

    # Route uncertain predictions to human review.
    if best_prediction["score"] < CONFIDENCE_THRESHOLD:
        status = "HUMAN_REVIEW"
    else:
        status = "AUTO_ROUTE_CANDIDATE"

    return {
        "message": message,
        "predicted_category": best_prediction["category"],
        "confidence": best_prediction["score"],
        "status": status,
        "top_3_predictions": top_predictions,
    }


if __name__ == "__main__":
    customer_message = input("Enter a customer message: ")

    result = predict_intent(customer_message)

    print("\nPREDICTION RESULT")
    print("-" * 40)
    print("Message:", result["message"])
    print("Predicted intent:", result["predicted_category"])
    print(f"Confidence score: {result['confidence']:.2%}")
    print("Routing status:", result["status"])

    print("\nTop 3 predictions:")

    for prediction in result["top_3_predictions"]:
        print(
            f"- {prediction['category']}: "
            f"{prediction['score']:.2%}"
        )
