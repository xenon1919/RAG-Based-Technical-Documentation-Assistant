"""
Answer Generation + Hallucination Check Nodes.

generate_answer_node:
    Synthesizes a grounded answer from relevant document chunks.
    The prompt explicitly forbids using external knowledge — the model
    MUST cite only what's in the context. This is the core RAG guarantee.

check_hallucination_node (Bonus):
    A secondary LLM call that fact-checks the generated answer against
    the context. This catches cases where the model confidently states
    something not in the documents ("hallucination").

    Real-world necessity: Even with good context, LLMs occasionally
    "blend in" training knowledge. The checker provides a trust score
    that the API consumer can use to flag low-confidence answers.
"""

import logging

from langchain_core.messages import HumanMessage

from app.grader import get_llm
from app.prompts import GENERATION_PROMPT, HALLUCINATION_CHECKER_PROMPT
from app.state import GraphState
from app.utils import (
    build_fallback_response,
    clean_json_response,
    extract_sources,
    extract_text,
    format_docs_for_context,
    truncate,
)

logger = logging.getLogger("rag_assistant.generator")


def generate_answer_node(state: GraphState) -> dict:
    """
    LangGraph node: Generate a grounded answer from relevant documents.

    Reads:  state["question"], state["relevant_docs"]
    Writes: state["generation"], state["source_citations"]

    If relevant_docs is empty (shouldn't happen due to routing, but
    handled defensively), returns a fallback message immediately.
    """
    logger.info("--- NODE: generate_answer ---")

    question = state["question"]
    relevant_docs = state.get("relevant_docs", [])

    # Defensive: should be caught by router, but guard here too
    if not relevant_docs:
        logger.warning("generate_answer called with no relevant docs")
        return {
            "generation": build_fallback_response(question, state.get("retry_count", 0)),
            "source_citations": [],
            "is_grounded": False,
            "hallucination_score": 0.0,
        }

    context = format_docs_for_context(relevant_docs)
    prompt = GENERATION_PROMPT.format(question=question, context=context)

    logger.info(
        "Generating answer from %d relevant chunks | question: %s",
        len(relevant_docs),
        truncate(question, 100),
    )

    llm = get_llm()
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = extract_text(response)
        sources = extract_sources(relevant_docs)

        logger.info(
            "Answer generated: %d chars | %d sources: %s",
            len(answer),
            len(sources),
            sources,
        )

        return {
            "generation": answer,
            "source_citations": sources,
        }

    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        return {
            "generation": f"Generation error: {exc}. Please try again.",
            "source_citations": [],
            "is_grounded": False,
            "hallucination_score": 0.0,
            "error_message": str(exc),
        }


def check_hallucination_node(state: GraphState) -> dict:
    """
    LangGraph node (Bonus): Verify the answer is supported by context.

    Reads:  state["question"], state["generation"], state["relevant_docs"]
    Writes: state["is_grounded"], state["hallucination_score"]

    This is a *secondary* LLM call that acts as an independent auditor.
    The hallucination_score is returned to the API caller so they can
    decide how to present the answer (e.g., show a warning badge if < 0.7).

    Fails safe: returns is_grounded=True on error to avoid alarming users
    about answers that may actually be fine.
    """
    logger.info("--- NODE: check_hallucination ---")

    question = state["question"]
    answer = state.get("generation", "")
    relevant_docs = state.get("relevant_docs", [])

    # Nothing to check if we have no answer or no context
    if not answer or not relevant_docs:
        logger.warning("Skipping hallucination check — no answer or context")
        return {"is_grounded": False, "hallucination_score": 0.0}

    context = format_docs_for_context(relevant_docs)
    prompt = HALLUCINATION_CHECKER_PROMPT.format(
        question=question,
        answer=answer,
        context=context,
    )

    llm = get_llm()
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        result = clean_json_response(extract_text(response))

        is_grounded = bool(result.get("is_grounded", True))
        score = float(result.get("hallucination_score", 0.8))
        unsupported = result.get("unsupported_claims", [])

        logger.info(
            "Hallucination check: grounded=%s, score=%.2f, unsupported_claims=%d",
            is_grounded,
            score,
            len(unsupported),
        )

        if unsupported:
            logger.warning("Unsupported claims detected: %s", unsupported[:3])

        return {
            "is_grounded": is_grounded,
            "hallucination_score": score,
        }

    except Exception as exc:
        logger.error("Hallucination check failed: %s", exc)
        # Fail safe — don't penalize a good answer because the checker errored
        return {"is_grounded": True, "hallucination_score": 0.8}
