
import json
from pathlib import Path

from src.rag_retriever import BankingRetriever


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = PROJECT_ROOT / "data" / "rag_eval.json"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_PATH = RESULTS_DIR / "rag_evaluation_results.json"


def evaluate_retrieval():
    with EVAL_PATH.open("r", encoding="utf-8") as file:
        evaluation_cases = json.load(file)

    if not evaluation_cases:
        raise ValueError("The RAG evaluation dataset is empty.")

    retriever = BankingRetriever()

    hit_at_1 = 0
    hit_at_3 = 0
    reciprocal_ranks = []
    detailed_results = []

    for case in evaluation_cases:
        question = case["question"]
        expected_id = case["expected_document_id"]

        retrieved = retriever.search(question, top_k=3)
        retrieved_ids = [doc["id"] for doc in retrieved]

        if retrieved_ids and retrieved_ids[0] == expected_id:
            hit_at_1 += 1

        if expected_id in retrieved_ids:
            hit_at_3 += 1
            rank = retrieved_ids.index(expected_id) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)

        detailed_results.append(
            {
                "question": question,
                "expected_document_id": expected_id,
                "retrieved_document_ids": retrieved_ids,
                "hit_at_1": bool(
                    retrieved_ids and retrieved_ids[0] == expected_id
                ),
                "hit_at_3": expected_id in retrieved_ids,
            }
        )

    total = len(evaluation_cases)
    metrics = {
        "evaluation_cases": total,
        "hit_rate_at_1": hit_at_1 / total,
        "hit_rate_at_3": hit_at_3 / total,
        "mean_reciprocal_rank": sum(reciprocal_ranks) / total,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "metrics": metrics,
        "results": detailed_results,
    }

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print("\nRAG Retrieval Evaluation")
    print("=" * 40)
    print(f"Evaluation cases: {total}")
    print(f"Hit Rate@1:      {metrics['hit_rate_at_1']:.2%}")
    print(f"Hit Rate@3:      {metrics['hit_rate_at_3']:.2%}")
    print(f"MRR:             {metrics['mean_reciprocal_rank']:.4f}")
    print(f"\nDetailed results saved to: {RESULTS_PATH}")

    print("\nCases where the expected document was not ranked first:")
    for result in detailed_results:
        if not result["hit_at_1"]:
            print(f"\nQuestion: {result['question']}")
            print(f"Expected: {result['expected_document_id']}")
            print(f"Retrieved: {result['retrieved_document_ids']}")


if __name__ == "__main__":
    evaluate_retrieval()
