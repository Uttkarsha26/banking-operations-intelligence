
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_PATH = PROJECT_ROOT / "data" / "banking_knowledge.json"

# Compact, multilingual-capable embedding model.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class BankingRetriever:
    def __init__(self):
        with KNOWLEDGE_PATH.open("r", encoding="utf-8") as file:
            self.documents = json.load(file)

        if not self.documents:
            raise ValueError("The banking knowledge base is empty.")

        self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # Combine title and content so retrieval can use both.
        self.document_texts = [
            (
                f"Title: {doc['title']}. "
                f"Topic: {doc['topic'].replace('_', ' ')}. "
                f"Source: {doc['source']}. "
                f"Content: {doc['content']}"
            )
            for doc in self.documents
        ]


        embeddings = self.encoder.encode(
            self.document_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        # Inner product equals cosine similarity for normalized vectors.
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 3):
        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_embedding = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        k = min(top_k, len(self.documents))
        scores, indices = self.index.search(query_embedding, k)

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            document = self.documents[int(index)]

            results.append({
                "source_url": document.get("source_url"),
                "source_type": document.get("source_type", "illustrative"),
                "verified_on": document.get("verified_on"),
                "id": document["id"],
                "title": document["title"],
                "source": document["source"],
                "topic": document["topic"],
                "content": document["content"],
                "similarity": float(score),
            })

        return results


if __name__ == "__main__":
    retriever = BankingRetriever()

    while True:
        query = input(
            "\nEnter a banking question (or type 'exit'): "
        ).strip()

        if query.lower() == "exit":
            break

        try:
            results = retriever.search(query, top_k=3)

            print("\nRetrieved documents")
            print("=" * 60)

            for rank, result in enumerate(results, start=1):
                print(
                    f"\n{rank}. {result['title']} "
                    f"(similarity: {result['similarity']:.3f})"
                )
                print(f"Topic: {result['topic']}")
                print(f"Source: {result['source']}")
                print(result["content"])

        except ValueError as exc:
            print(f"Invalid query: {exc}")
