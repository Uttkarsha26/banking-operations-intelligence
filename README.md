# Banking Operations AI Copilot

An AI engineering portfolio project combining banking intent classification, confidence-based routing, retrieval-augmented question answering (RAG), and a REST API with a Streamlit dashboard.

## Key Features

- **Intent classification:** TF-IDF features and a calibrated LinearSVC classifier trained on BANKING77.
- **Confidence-based routing:** Low-confidence requests are flagged for review using an experimental 0.60 threshold.
- **Sensitive-intent rules:** Configured sensitive categories are routed for additional review.
- **RAG retrieval:** Sentence Transformers embeddings (`all-MiniLM-L6-v2`) with FAISS similarity search.
- **Grounded answers:** Extractive answers use retrieved knowledge-base text rather than unrestricted generation.
- **Source traceability:** Retrieved documents can expose source URLs, source types, and verification dates.
- **FastAPI:** Prediction, question-answering, and health endpoints.
- **Streamlit:** Interactive interface for intent analysis and knowledge-base questions.
- **Automated tests:** API and answer-generation behavior tested with pytest.

## Architecture

```text
User
 |
 v
Streamlit Dashboard
 |                         |
 v                         v
FastAPI /predict        FastAPI /ask
 |                         |
 v                         v
TF-IDF + Classifier    Sentence Transformers
 |                         |
 v                         v
Confidence Routing     FAISS Retrieval
                           |
                           v
                    Extractive Answer
                           |
                           v
                    Evidence + Sources
```

## Technology Stack

Python, scikit-learn, pandas, NumPy, Sentence Transformers, FAISS, FastAPI, Pydantic, Streamlit, pytest.

## Project Structure

```text
banking-operations-intelligence/
├── data/
│   ├── banking_knowledge.json
│   ├── rag_eval.json
│   └── ...
├── models/
├── results/
│   └── rag_evaluation_results.json
├── src/
│   ├── api.py
│   ├── dashboard.py
│   ├── rag_retriever.py
│   ├── answer_generator.py
│   └── evaluate_rag.py
├── tests/
├── requirements.txt
└── README.md
```

## Setup

Use Python 3.11 or another Python version supported by all installed dependencies. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The embedding model may be downloaded the first time the RAG retriever initializes. Internet access may be needed for this initial download.

## Run the API

From the project root:

```powershell
python -m uvicorn src.api:app --reload
```

API documentation is available at `http://127.0.0.1:8000/docs`.

Endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /` | Basic API information |
| `GET /health` | Component health status |
| `POST /predict` | Classify a banking message and return routing information |
| `POST /ask` | Retrieve knowledge-base evidence and generate a grounded extractive answer |

Example request to `/predict`:

```json
{
  "message": "I do not recognize a card payment."
}
```

Example request to `/ask`:

```json
{
  "question": "How should I report an unauthorized card transaction?",
  "top_k": 3
}
```

## Run the Dashboard

In a second terminal, activate the virtual environment and run:

```powershell
streamlit run src/dashboard.py
```

Use the API URL configured by the dashboard if it differs from the local default.

## Run Tests

```powershell
python -m pytest -v
```

Latest recorded test result: **16 passed**. A Starlette deprecation warning involving the test client was present; it did not fail the test suite.

## RAG Evaluation

The manually curated evaluation set contains 16 questions.

| Metric | Recorded result |
|---|---:|
| Hit Rate@1 | 87.50% |
| Hit Rate@3 | 100.00% |
| Mean Reciprocal Rank (MRR) | 0.9375 |

Detailed results are written to `results/rag_evaluation_results.json`.

These results apply only to this small evaluation set. Some questions have overlapping relevant sources, and the expected document labels are manually assigned. They should not be interpreted as production-level performance guarantees.

## Data and Source Notes

The BANKING77 dataset is used for banking-intent classification. The knowledge base combines illustrative FAQ content with selected RBI reference material.

Illustrative FAQs are not official bank policies. Regulatory documents should be checked against the original RBI publication and applicable current rules before operational use. A recorded source verification date indicates a reference check, not a legal review or guarantee that every summary is complete.

## Limitations and Responsible Use

- This is a portfolio prototype, not a production banking system.
- The confidence threshold is experimental and needs independent validation.
- A classifier can misclassify sensitive requests; predicted-intent routing does not guarantee that every sensitive case is escalated.
- Extractive answers are limited by the retrieved documents and may omit important context.
- RAG metrics are based on a small, manually curated dataset.
- Do not submit real customer account numbers, payment details, passwords, or other confidential banking data.
- Do not use this prototype to make financial, eligibility, fraud, or liability decisions without appropriate controls and expert review.

## Future Improvements

- Expand and independently validate the RAG evaluation dataset.
- Add monitoring, structured logging, and broader negative/error-path tests.
- Validate routing thresholds on a representative held-out dataset.
- Add containerization and deployment automation.
- Improve source freshness checks and document-level provenance.
- Evaluate retrieval quality, answer faithfulness, and escalation behavior separately.

## License

Add a license before publicly redistributing this project. Verify dataset and source-document reuse terms before publication.
