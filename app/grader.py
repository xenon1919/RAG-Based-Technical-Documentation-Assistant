"""
Document Grading Node — The Most Critical Node in the Pipeline.

Purpose:
    After retrieval, we have chunks that are *semantically similar* to the query,
    but semantic similarity ≠ actually relevant. This node uses an LLM to perform
    a true relevance judgment on each chunk.

Why this matters:
    Without grading, a question about FastAPI middleware might retrieve chunks
    about middleware in general (Django, Express) that would confuse the generator.
    Grading filters these out before they pollute the context window.

Design decisions:
    - Grade each doc independently (not batch) for higher accuracy
    - Return True on grading errors to avoid silent data loss
    - Use low temperature (0.1) to get deterministic binary judgments
    - LLM prompt is generous: partial relevance counts as relevant
"""

import logging
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings
from app.prompts import DOCUMENT_GRADER_PROMPT
from app.state import GraphState, RetrievedDoc
from app.utils import clean_json_response, extract_text, truncate

logger = logging.getLogger("rag_assistant.grader")
settings = get_settings()

# Shared LLM instance — initialized once, reused across all nodes
_llm: Optional[ChatGoogleGenerativeAI] = None


def get_llm() -> ChatGoogleGenerativeAI:
    """
    Return the shared Gemini LLM instance.
    All nodes (grader, generator, rewriter) share this single instance
    to avoid creating multiple API connections.
    """
    global _llm
    if _llm is None:
        logger.info("Initializing Gemini LLM: %s", settings.GEMINI_MODEL)
        _llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=settings.GEMINI_TEMPERATURE,
            max_output_tokens=settings.GEMINI_MAX_OUTPUT_TOKENS,
        )
        logger.info("Gemini LLM initialized")
    return _llm


def grade_single_document(question: str, doc: RetrievedDoc) -> tuple[bool, float]:
    """
    Ask the LLM to judge whether a single document chunk is relevant.

    Returns:
        (is_relevant, confidence) — e.g., (True, 0.92)

    Fails safe: returns (True, 0.5) on any error to prevent data loss.
    The risk of including a borderline irrelevant doc is lower than
    silently dropping a relevant one.
    """
    llm = get_llm()
    prompt = DOCUMENT_GRADER_PROMPT.format(
        question=question,
        document=doc["content"],
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        result = clean_json_response(extract_text(response))

        is_relevant = bool(result.get("relevant", True))
        confidence = float(result.get("confidence", 0.5))
        reason = result.get("reason", "")

        logger.debug(
            "Graded doc from '%s': relevant=%s, conf=%.2f | %s",
            doc.get("source"),
            is_relevant,
            confidence,
            truncate(reason, 80),
        )
        return is_relevant, confidence

    except Exception as exc:
        logger.error("Grading error for doc '%s': %s", doc.get("source"), exc)
        return True, 0.5  # fail open — include the doc


def grade_documents_node(state: GraphState) -> dict:
    """
    LangGraph node: Filter retrieved docs down to only the relevant ones.

    Reads:  state["optimized_query"], state["retrieved_docs"]
    Writes: state["relevant_docs"]

    After this node, the conditional router decides:
        - relevant_docs not empty → generate_answer
        - relevant_docs empty + retries left → rewrite_query
        - relevant_docs empty + no retries → fallback
    """
    logger.info("--- NODE: grade_documents ---")

    # Use the optimized query for grading (matches the retrieval context better)
    question = state.get("optimized_query") or state["question"]
    retrieved_docs: list[RetrievedDoc] = state.get("retrieved_docs", [])

    if not retrieved_docs:
        logger.warning("No documents to grade — skipping")
        return {"relevant_docs": []}

    logger.info("Grading %d documents...", len(retrieved_docs))

    relevant_docs: list[RetrievedDoc] = []
    for doc in retrieved_docs:
        is_relevant, confidence = grade_single_document(question, doc)
        if is_relevant:
            # Attach grading metadata for transparency
            enriched = {
                **doc,
                "metadata": {
                    **doc.get("metadata", {}),
                    "grade_confidence": confidence,
                },
            }
            relevant_docs.append(enriched)

    logger.info(
        "Grading complete: %d/%d chunks are relevant",
        len(relevant_docs),
        len(retrieved_docs),
    )

    return {"relevant_docs": relevant_docs}
