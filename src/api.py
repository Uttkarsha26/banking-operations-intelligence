from src.answer_generator import generate_grounded_answer
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag_retriever import BankingRetriever

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

model = joblib.load(MODEL_DIR / "intent_model.joblib")
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")

# Experimental threshold; validate before operational use.
CONFIDENCE_THRESHOLD = 0.60

SENSITIVE_INTENTS = {
    "cash_withdrawal_not_recognised",
    "card_payment_not_recognised",
    "compromised_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "transaction_charged_twice",
    "direct_debit_payment_not_recognised",
    "terminate_account",
    "unable_to_verify_identity",
    "verify_source_of_funds",
}

app = FastAPI(
    title="Banking Operations AI Copilot API",
    description=(
        "Banking message classification, review routing, "
        "and evidence retrieval from a curated knowledge base."
    ),
    version="2.1.0",
)

# Initialize the retriever once when the API process starts.
retriever = BankingRetriever()


class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=6)


def determine_routing(category: str, confidence: float) -> dict:
    if category in SENSITIVE_INTENTS:
        return {
            "routing_status": "ESCALATE_SENSITIVE_INTENT",
            "routing_reason": "Sensitive intent requires human review.",
        }

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "routing_status": "HUMAN_REVIEW",
            "routing_reason": (
                "Confidence is below the experimental threshold."
            ),
        }

    return {
        "routing_status": "AUTO_ROUTE_CANDIDATE",
        "routing_reason": (
            "Confidence meets the experimental threshold and "
            "no configured sensitive-intent rule matched."
        ),
    }


@app.get("/")
def home():
    return {
        "message": "Banking Operations AI Copilot API is running.",
        "docs": "/docs",
        "version": app.version,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "retriever_loaded": retriever is not None,
        "routing_threshold": CONFIDENCE_THRESHOLD,
    }


@app.post("/predict")
def predict(request: MessageRequest):
    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=422,
            detail="Message cannot contain only whitespace.",
        )

    features = vectorizer.transform([message])
    probabilities = model.predict_proba(features)[0]
    top_indices = probabilities.argsort()[-3:][::-1]

    top_predictions = [
        {
            "category": str(model.classes_[index]),
            "score": float(probabilities[index]),
        }
        for index in top_indices
    ]

    best = top_predictions[0]
    routing = determine_routing(
        category=best["category"],
        confidence=best["score"],
    )

    return {
        "message": message,
        "predicted_category": best["category"],
        "confidence": best["score"],
        **routing,
        "top_3_predictions": top_predictions,
    }



@app.post("/ask")
def ask(request: AskRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot contain only whitespace.",
        )

    try:
        documents = retriever.search(
            query=question,
            top_k=request.top_k,
        )

        grounded_result = generate_grounded_answer(
            question=question,
            documents=documents,
        )

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "question": question,
        "answer": grounded_result["answer"],
        "answer_status": grounded_result["answer_status"],
        "evidence_ids": grounded_result["evidence_ids"],
        "sources": [
            {
                "source_url": document.get("source_url"),
                "source_type": document.get("source_type", "illustrative"),
                "verified_on": document.get("verified_on"),
                "id": document["id"],
                "title": document["title"],
                "source": document["source"],
                "topic": document["topic"],
                "content": document["content"],
                "similarity": document["similarity"],
            }
            for document in documents
        ],
    }
