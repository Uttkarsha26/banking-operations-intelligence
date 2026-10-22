
import re


def generate_grounded_answer(question: str, documents: list[dict]) -> dict:
    """
    Build an extractive answer using retrieved knowledge-base documents.
    This function does not call an LLM or invent additional policy.
    """
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    if not documents:
        return {
            "answer": (
                "I could not find supporting information in the "
                "current knowledge base. Please contact your bank "
                "or a qualified support representative."
            ),
            "answer_status": "INSUFFICIENT_EVIDENCE",
            "evidence_ids": [],
        }

    # Use the highest-ranked retrieved document as the primary evidence.
    best_document = documents[0]
    content = best_document["content"].strip()

    # Split the retrieved passage into readable sentences.
    sentences = re.split(r"(?<=[.!?])\s+", content)
    answer_sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    if not answer_sentences:
        return {
            "answer": (
                "I could not extract a usable answer from the retrieved "
                "evidence. Please contact your bank for assistance."
            ),
            "answer_status": "INSUFFICIENT_EVIDENCE",
            "evidence_ids": [],
        }

    # Keep the answer grounded in the source document's own wording.
    answer = " ".join(answer_sentences[:3])

    return {
        "answer": answer,
        "answer_status": "GROUNDED_EXTRACTIVE",
        "evidence_ids": [best_document["id"]],
    }
