
from src.answer_generator import generate_grounded_answer


def test_generates_answer_from_retrieved_document():
    documents = [
        {
            "id": "faq_002",
            "title": "Lost or stolen card",
            "content": (
                "Report a lost or stolen card to your bank "
                "as soon as possible. Follow your bank's "
                "official instructions."
            ),
        }
    ]

    result = generate_grounded_answer(
        question="Someone stole my card. What should I do?",
        documents=documents,
    )

    assert result["answer_status"] == "GROUNDED_EXTRACTIVE"
    assert result["answer"].startswith("Report a lost or stolen card")
    assert "Report a lost or stolen card" in result["answer"]
    assert result["evidence_ids"] == ["faq_002"]


def test_returns_insufficient_evidence_when_no_documents():
    result = generate_grounded_answer(
        question="How do I check my account?",
        documents=[],
    )

    assert result["answer_status"] == "INSUFFICIENT_EVIDENCE"
    assert result["evidence_ids"] == []
    assert "could not find supporting information" in (
        result["answer"].lower()
    )


def test_rejects_empty_question():
    try:
        generate_grounded_answer(question="   ", documents=[])
        assert False, "Expected ValueError for an empty question"
    except ValueError:
        pass
