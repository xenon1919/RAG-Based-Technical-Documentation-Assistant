"""
Query Analysis + Query Rewriting Nodes.

analyze_query_node (runs first, always):
    - Classifies the user's question (conceptual / troubleshooting / api_reference / how_to)
    - Rewrites it for better vector search recall
    - Initializes retry_count and conversation_history

rewrite_query_node (runs on retry):
    - Called when grading found no relevant documents
    - Applies a different lexical strategy each attempt
    - Increments retry_count (the router checks this to decide when to give up)

Why rewriting matters:
    User question: "My FastAPI app crashes on startup"
    Vector search needs: "FastAPI application startup error exception troubleshooting"
    Without rewriting, the similarity score may be too low to retrieve the right chunk.
"""

import logging

from langchain_core.messages import HumanMessage

from app.grader import get_llm
from app.prompts import QUERY_ANALYSIS_PROMPT, QUERY_REWRITER_PROMPT
from app.state import GraphState
from app.utils import clean_json_response, extract_text, truncate

logger = logging.getLogger("rag_assistant.query_rewriter")


def analyze_query_node(state: GraphState) -> dict:
    """
    LangGraph node: Analyze and optimize the user's question.

    Reads:  state["question"]
    Writes: state["question_type"], state["optimized_query"],
            state["retry_count"] (initialized to 0),
            state["conversation_history"] (initialized if absent)

    This is the entry point of the graph (after START).
    It transforms a colloquial user question into a search-optimized query.
    """
    logger.info("--- NODE: analyze_query ---")

    question = state["question"]
    logger.info("Analyzing question: %s", truncate(question, 120))

    llm = get_llm()
    prompt = QUERY_ANALYSIS_PROMPT.format(question=question)

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        result = clean_json_response(extract_text(response))

        question_type = result.get("question_type", "conceptual")
        optimized_query = result.get("optimized_query", question)
        reasoning = result.get("reasoning", "")

        logger.info(
            "Classified as '%s' | optimized: %s | reasoning: %s",
            question_type,
            truncate(optimized_query, 100),
            truncate(reasoning, 80),
        )

        return {
            "question_type": question_type,
            "optimized_query": optimized_query,
            "retry_count": state.get("retry_count", 0),
            "conversation_history": state.get("conversation_history", []),
        }

    except Exception as exc:
        logger.error("Query analysis failed: %s — using original question", exc)
        # Graceful degradation: use the original question as-is
        return {
            "question_type": "conceptual",
            "optimized_query": question,
            "retry_count": state.get("retry_count", 0),
            "conversation_history": state.get("conversation_history", []),
        }


def rewrite_query_node(state: GraphState) -> dict:
    """
    LangGraph node: Rewrite the query after a failed retrieval attempt.

    Reads:  state["question"], state["optimized_query"], state["retry_count"]
    Writes: state["optimized_query"] (new version), state["retry_count"] (incremented)

    The retry_count is incremented HERE, not in the router.
    The router reads retry_count AFTER grading to decide if we should retry again.

    Strategy escalation by attempt:
        Attempt 1: synonyms and alternate terminology
        Attempt 2: simpler, more general concepts
        Attempt 3: just the core technology name
    """
    current_retry = state.get("retry_count", 0)
    logger.info("--- NODE: rewrite_query (attempt %d/%d) ---", current_retry + 1, 3)

    question = state["question"]
    previous_query = state.get("optimized_query", question)

    llm = get_llm()
    prompt = QUERY_REWRITER_PROMPT.format(
        question=question,
        previous_query=previous_query,
        retry_count=current_retry + 1,
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        result = clean_json_response(extract_text(response))

        rewritten = result.get("rewritten_query", question)
        strategy = result.get("strategy_used", "unknown")

        logger.info(
            "Query rewritten: '%s' | strategy: %s",
            truncate(rewritten, 100),
            strategy,
        )

        return {
            "optimized_query": rewritten,
            "retry_count": current_retry + 1,
            # Clear stale results from previous attempt
            "retrieved_docs": [],
            "relevant_docs": [],
        }

    except Exception as exc:
        logger.error("Query rewrite failed: %s", exc)
        # Even on failure, increment retry_count to avoid infinite loops
        return {
            "retry_count": current_retry + 1,
            "retrieved_docs": [],
            "relevant_docs": [],
        }
