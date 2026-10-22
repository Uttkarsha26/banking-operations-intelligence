
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Banking Operations AI Copilot",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Banking Operations AI Copilot")
st.caption(
    "Intent classification · Review routing · "
    "Evidence-based banking question answering"
)

st.info(
    "Portfolio prototype using BANKING77 and a small knowledge base. "
    "Routing outcomes are recommendations, not production banking controls. "
    "Some knowledge-base documents are illustrative rather than official policy."
)

with st.sidebar:
    st.header("About this prototype")
    st.write("**Classifier:** Calibrated Linear SVM")
    st.write("**Intent categories:** 77")
    st.write("**Routing threshold:** 60% (experimental)")
    st.write("**Retrieval:** Sentence Transformers + FAISS")
    st.write(
        "Sensitive predicted intents receive an escalation status. "
        "A misclassified request may still bypass that rule."
    )

st.divider()
st.header("1. Customer Message Analysis")

message = st.text_area(
    "Customer message",
    placeholder="Example: I am still waiting for my card",
    height=120,
    max_chars=2000,
    key="customer_message",
)

if st.button("Analyze message", type="primary"):
    if not message.strip():
        st.warning("Please enter a customer message.")
    else:
        try:
            with st.spinner("Analyzing the message..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"message": message},
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()

            st.subheader("Analysis results")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Predicted intent",
                    result["predicted_category"],
                )

            with col2:
                st.metric(
                    "Model confidence",
                    f'{result["confidence"]:.2%}',
                )

            status = result["routing_status"]
            reason = result.get(
                "routing_reason",
                "No routing explanation was returned.",
            )

            status = result["routing_status"]
            reason = result.get(
                "routing_reason",
                "No routing explanation was returned.",
            )

            if status == "ESCALATE_SENSITIVE_INTENT":
                st.error(f"Sensitive-intent escalation\n\n{reason}")
            elif status == "HUMAN_REVIEW":
                st.warning(f"Human review recommended\n\n{reason}")
            elif status == "AUTO_ROUTE_CANDIDATE":
                st.success(
                    f"Meets the experimental routing criteria\n\n{reason}"
                )
            else:
                st.warning(f"Unrecognized routing status: {status}")

            st.caption(
                "Model confidence does not establish that a request "
                "is safe or that the prediction is correct."
            )

            st.subheader("Top 3 predicted intents")

            for item in result["top_3_predictions"]:
                st.write(
                    f'**{item["category"]}** — {item["score"]:.2%}'
                )
                st.progress(
                    max(0.0, min(1.0, float(item["score"])))
                )

        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the API. Start FastAPI in another terminal.")
        except requests.exceptions.Timeout:
            st.error("The API request timed out. Please try again.")
        except requests.exceptions.HTTPError as exc:
            st.error(f"The API returned an error: {exc}")
        except requests.exceptions.RequestException as exc:
            st.error(f"Request failed: {exc}")
        except (KeyError, TypeError, ValueError) as exc:
            st.error(f"The API response could not be interpreted: {exc}")

st.divider()
st.header("2. Banking Question Assistant")
st.write(
    "Ask a question to retrieve relevant documents from the knowledge base. "
    "Answers are extracted from retrieved text; this is not a general-purpose "
    "LLM and cannot guarantee that the evidence is complete or authoritative."
)

question = st.text_area(
    "Your banking question",
    placeholder="Example: How do I report an unrecognized card payment?",
    height=100,
    max_chars=2000,
    key="banking_question",
)

top_k = st.selectbox(
    "Number of source documents",
    options=[1, 2, 3, 4, 5, 6],
    index=2,
)

if st.button("Ask question"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Searching the knowledge base..."):
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question, "top_k": top_k},
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()

            answer_status = result.get("answer_status", "UNKNOWN")
            answer = result.get("answer", "")
            sources = result.get("sources", [])

            st.subheader("Answer")

            if answer_status == "INSUFFICIENT_EVIDENCE":
                st.warning(answer or "No supporting evidence was found.")
            elif answer_status == "GROUNDED_EXTRACTIVE":
                st.write(answer)
                st.caption(
                    "Extracted from retrieved text; verify the cited source "
                    "before relying on it."
                )
            else:
                st.info(answer or "The API did not return an answer.")

            st.caption(f"Answer status: {answer_status}")

            evidence_ids = set(result.get("evidence_ids", []))
            if evidence_ids:
                st.caption(
                    "Evidence document IDs: " + ", ".join(sorted(evidence_ids))
                )

            st.subheader("Retrieved evidence")

            if not sources:
                st.warning(
                    "No documents were retrieved. The knowledge base may not "
                    "contain evidence relevant to this question."
                )
            else:
                for index, source in enumerate(sources, start=1):
                    title = source.get("title", "Untitled source")
                    document_id = source.get("id", "unknown")
                    similarity = source.get("similarity", 0.0)

                    with st.expander(
                        f"{index}. {title} · similarity {similarity:.3f}",
                        expanded=(index == 1),
                    ):
                        st.write(f"**Document ID:** {document_id}")
                        st.write(f"**Topic:** {source.get('topic', 'Unknown')}")
                        st.write(f"**Source:** {source.get('source', 'Unknown')}")

                        source_url = source.get("source_url")
                        if source_url:
                            st.markdown(f"[Open source document]({source_url})")

                        st.write(source.get("content", "No content available."))

        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the API. Start FastAPI in another terminal.")
        except requests.exceptions.Timeout:
            st.error("The request timed out. Please try again.")
        except requests.exceptions.HTTPError as exc:
            st.error(f"The API returned an error: {exc}")
        except requests.exceptions.RequestException as exc:
            st.error(f"Request failed: {exc}")
        except (KeyError, TypeError, ValueError) as exc:
            st.error(f"The API response could not be interpreted: {exc}")
