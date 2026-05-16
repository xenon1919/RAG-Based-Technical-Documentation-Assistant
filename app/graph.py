"""
LangGraph StateGraph — Core Workflow Orchestration.

This module assembles all nodes into a directed graph with conditional
routing. It is the "control plane" of the entire RAG pipeline.

Graph structure:

    START
      │
      ▼
  ┌──────────────┐
  │ analyze_query│  → classifies + optimizes the question
  └──────┬───────┘
         │
         ▼
  ┌──────────────────┐
  │retrieve_documents│  → semantic search in ChromaDB
  └──────┬───────────┘
         │
         ▼
  ┌────────────────┐
  │grade_documents │  → LLM filters relevant chunks
  └──────┬─────────┘
         │
    ┌────┴──────────────────────┐
    │                           │
    │ relevant_docs?            │ relevant_docs empty?
    │                           │
    ▼                           ▼
  ┌───────────────┐      retries remaining?
  │generate_answer│        ├── YES → ┌──────────────┐
  └──────┬────────┘        │         │ rewrite_query │ ──► retrieve_documents (loop)
         │                 │         └──────────────┘
         ▼                 └── NO  → ┌──────────┐
  ┌──────────────────┐               │ fallback │
  │check_hallucination│              └─────┬────┘
  └──────┬───────────┘                    │
         │                               ▼
         └───────────────────────────► END

Retry limit is enforced by route_after_grading using settings.MAX_RETRIES.
"""

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.generator import check_hallucination_node, generate_answer_node
from app.grader import grade_documents_node
from app.query_rewriter import analyze_query_node, rewrite_query_node
from app.state import GraphState
from app.utils import build_fallback_response

logger = logging.getLogger("rag_assistant.graph")
settings = get_settings()


# ── Node: Retrieve Documents ──────────────────────────────────────────────────
# Defined here (not retriever.py) to keep retriever.py a pure utility module


def retrieve_documents_node(state: GraphState) -> dict:
    """
    LangGraph node: Retrieve top-k similar documents from ChromaDB.

    Reads:  state["optimized_query"] (falls back to state["question"])
    Writes: state["retrieved_docs"]
    """
    from app.retriever import retrieve_documents

    query = state.get("optimized_query") or state["question"]
    logger.info("--- NODE: retrieve_documents ---")
    logger.info("Query: %s", query[:120])

    docs = retrieve_documents(query)
    logger.info("Retrieved %d raw documents", len(docs))

    return {"retrieved_docs": docs}


# ── Node: Fallback Response ───────────────────────────────────────────────────


def fallback_node(state: GraphState) -> dict:
    """
    LangGraph node: Return a graceful "I couldn't find anything" message.

    Reached when: no relevant docs found AND retry_count >= MAX_RETRIES.
    Writes a structured fallback to state["generation"].
    """
    logger.info("--- NODE: fallback (max retries reached) ---")

    fallback_text = build_fallback_response(
        question=state["question"],
        retry_count=state.get("retry_count", 0),
    )

    return {
        "generation": fallback_text,
        "source_citations": [],
        "is_grounded": False,
        "hallucination_score": 0.0,
    }


# ── Conditional Router ────────────────────────────────────────────────────────


def route_after_grading(
    state: GraphState,
) -> Literal["generate_answer", "rewrite_query", "fallback"]:
    """
    Decide what to do after document grading.

    Decision logic:
        1. relevant_docs is not empty           → proceed to generate_answer
        2. relevant_docs empty, retries left    → rewrite_query and retry
        3. relevant_docs empty, no retries left → fallback response

    This function is passed to add_conditional_edges() and its return
    value is used as the key to look up the next node name.
    """
    relevant_docs = state.get("relevant_docs", [])
    retry_count = state.get("retry_count", 0)

    if relevant_docs:
        logger.info("Router → generate_answer (%d relevant docs)", len(relevant_docs))
        return "generate_answer"

    if retry_count < settings.MAX_RETRIES:
        logger.info(
            "Router → rewrite_query (retry %d/%d, no relevant docs)",
            retry_count + 1,
            settings.MAX_RETRIES,
        )
        return "rewrite_query"

    logger.info("Router → fallback (exhausted %d retries)", settings.MAX_RETRIES)
    return "fallback"


# ── Graph Construction ────────────────────────────────────────────────────────


def build_graph() -> StateGraph:
    """
    Assemble and compile the LangGraph StateGraph.

    Returns a compiled graph ready to call with .invoke(initial_state).
    The graph is compiled once at startup and reused for all requests.
    """
    workflow = StateGraph(GraphState)

    # ── Register all nodes ─────────────────────────────────────────────────
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("retrieve_documents", retrieve_documents_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("check_hallucination", check_hallucination_node)
    workflow.add_node("fallback", fallback_node)

    # ── Fixed edges (always traverse) ─────────────────────────────────────
    workflow.add_edge(START, "analyze_query")
    workflow.add_edge("analyze_query", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "grade_documents")

    # ── Conditional routing after grading ──────────────────────────────────
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query",
            "fallback": "fallback",
        },
    )

    # ── Retry loop: rewrite → retrieve → grade → [conditional] ────────────
    workflow.add_edge("rewrite_query", "retrieve_documents")

    # ── Post-generation hallucination check ────────────────────────────────
    workflow.add_edge("generate_answer", "check_hallucination")

    # ── Terminal nodes ─────────────────────────────────────────────────────
    workflow.add_edge("check_hallucination", END)
    workflow.add_edge("fallback", END)

    compiled = workflow.compile()
    logger.info("LangGraph workflow compiled successfully")
    return compiled


# ── Singleton Graph ───────────────────────────────────────────────────────────

_graph = None


def get_graph():
    """Return the compiled graph (lazy singleton — built on first call)."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── Pipeline Entry Point ──────────────────────────────────────────────────────


def run_rag_pipeline(
    question: str,
    session_id: str | None = None,
    conversation_history: list[dict] | None = None,
) -> dict:
    """
    Execute the full self-corrective RAG pipeline for a single question.

    Args:
        question:             The user's raw question
        session_id:           Optional session ID for multi-turn memory
        conversation_history: Previous Q&A pairs for context (optional)

    Returns:
        The final GraphState dict with keys:
        - generation:          The answer text
        - source_citations:    List of source filenames cited
        - question_type:       Classified question type
        - is_grounded:         Whether the answer is verified against context
        - hallucination_score: Float 0.0–1.0 (1.0 = fully grounded)
        - retry_count:         How many rewrites were needed
        - error_message:       Error info if something failed
    """
    logger.info("=== RAG Pipeline START | question: %s ===", question[:100])

    initial_state = GraphState(
        question=question,
        session_id=session_id,
        question_type="",
        optimized_query="",
        retrieved_docs=[],
        relevant_docs=[],
        generation="",
        source_citations=[],
        is_grounded=False,
        hallucination_score=0.0,
        retry_count=0,
        error_message=None,
        conversation_history=conversation_history or [],
    )

    graph = get_graph()
    final_state = graph.invoke(initial_state)

    logger.info(
        "=== RAG Pipeline END | grounded=%s | sources=%d | retries=%d ===",
        final_state.get("is_grounded"),
        len(final_state.get("source_citations", [])),
        final_state.get("retry_count", 0),
    )

    return final_state
