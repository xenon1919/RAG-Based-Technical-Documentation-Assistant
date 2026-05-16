"""
LangGraph State Schema for the RAG workflow.

The GraphState is the central data structure that flows through every
node in the graph. Think of it as a shared "whiteboard" — each node
reads what it needs and writes back its outputs.

State Flow:
    question  ──► [analyze_query]       ──► optimized_query, question_type
              ──► [retrieve_documents]  ──► retrieved_docs
              ──► [grade_documents]     ──► relevant_docs
              ──► [generate_answer]     ──► generation, source_citations
              ──► [check_hallucination] ──► is_grounded, hallucination_score

Retry Flow (when no relevant docs found):
    relevant_docs empty ──► [rewrite_query] ──► optimized_query (new)
                        ──► [retrieve_documents] (retry)
                        ──► [grade_documents] ...
                        (repeats up to MAX_RETRIES times, then fallback)
"""

from typing import Optional, TypedDict


class RetrievedDoc(TypedDict):
    """
    Represents a single retrieved document chunk with its metadata.
    Produced by the retriever and consumed by grader/generator nodes.
    """

    content: str  # the actual text of the chunk
    source: str  # filename or URL where this content came from
    chunk_id: str  # unique ID for deduplication
    score: float  # cosine similarity score (higher = more similar to query)
    metadata: dict  # arbitrary extra metadata from ChromaDB


class GraphState(TypedDict):
    """
    The complete state object passed between all nodes in the LangGraph.

    Design principles:
    - Every field has a clear owner (which node writes it)
    - Optional fields are marked with Optional[T]
    - Lists start empty and are populated by retrieval/grading nodes
    - Control-flow fields (retry_count) are managed by rewrite + routing
    """

    # ── INPUT (set by the API caller before graph starts) ────────────────────
    question: str  # raw user question, never modified
    session_id: Optional[str]  # for conversation memory, can be None

    # ── QUERY ANALYSIS (written by analyze_query node) ────────────────────────
    question_type: str  # conceptual | troubleshooting | api_reference | how_to
    optimized_query: str  # rewritten query for better vector search

    # ── RETRIEVAL (written by retrieve_documents node) ────────────────────────
    retrieved_docs: list[RetrievedDoc]  # raw top-k chunks from ChromaDB

    # ── GRADING (written by grade_documents node) ─────────────────────────────
    relevant_docs: list[RetrievedDoc]  # subset of retrieved_docs deemed relevant

    # ── GENERATION (written by generate_answer node) ──────────────────────────
    generation: str  # the final answer text
    source_citations: list[str]  # unique source filenames used in the answer

    # ── HALLUCINATION CHECK (written by check_hallucination node) ─────────────
    is_grounded: bool  # True if answer is supported by context
    hallucination_score: float  # 1.0 = fully grounded, 0.0 = hallucinated

    # ── CONTROL FLOW (managed by routing + rewrite nodes) ─────────────────────
    retry_count: int  # number of query rewrites attempted so far
    error_message: Optional[str]  # populated on node errors for debugging

    # ── CONVERSATION MEMORY (optional, for multi-turn sessions) ───────────────
    conversation_history: list[dict]  # list of {"role": ..., "content": ...}
