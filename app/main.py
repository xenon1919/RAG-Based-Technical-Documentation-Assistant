"""
FastAPI Application — REST API for the RAG Documentation Assistant.

Endpoints:
    GET  /health      — Health check + ChromaDB stats
    POST /query       — Ask a question, receive answer + citations
    POST /ingest      — Ingest documents from a directory or raw text
    GET  /documents   — List all indexed sources
    POST /feedback    — Submit thumbs up/down feedback

Design decisions:
    - Lifespan handler warms up models at startup (avoids slow first request)
    - Background tasks used for ingestion so API stays responsive
    - In-memory feedback store (replace with PostgreSQL/SQLite in production)
    - Session memory implemented as a simple dict (replace with Redis in production)
    - All responses use Pydantic models for consistent JSON schema
"""

import logging
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import get_settings
from app.graph import run_rag_pipeline
from app.ingestion import ingest_from_directory, ingest_text
from app.retriever import get_collection_stats, get_embeddings, get_vectorstore, list_indexed_sources
from app.utils import setup_logging

logger = setup_logging()
settings = get_settings()

# ── In-Memory Stores (replace with persistent storage in production) ──────────

# Feedback store: list of feedback dicts
_feedback_store: list[dict] = []

# Conversation memory: session_id → list of {"role": ..., "content": ...}
_session_memory: dict[str, list[dict]] = defaultdict(list)


# ── Application Lifespan ──────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Warm up heavy models before accepting requests.

    Without this, the first request takes 10-30 seconds while the embedding
    model downloads and loads into RAM. Subsequent requests are fast.
    """
    logger.info("Starting RAG Technical Documentation Assistant API...")
    try:
        get_embeddings()   # Download + load sentence-transformer model into RAM
        get_vectorstore()  # Open ChromaDB persistent client
        logger.info("Models warmed up — API ready to serve requests")
    except Exception as exc:
        logger.error("Warm-up failed (API will still start): %s", exc)
    yield
    logger.info("API shutting down")


# ── FastAPI App ───────────────────────────────────────────────────────────────


app = FastAPI(
    title="RAG Technical Documentation Assistant",
    description=(
        "A self-corrective Retrieval-Augmented Generation system built with "
        "LangGraph + Gemini + ChromaDB. Answers technical documentation questions "
        "with source citations and hallucination detection."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Request / Response Models ────────────────────────────────────────


class QueryRequest(BaseModel):
    """Request body for POST /query."""
    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="The technical question to answer",
        examples=["How do I add CORS middleware to FastAPI?"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for conversation memory. Auto-generated if not provided.",
    )


class QueryResponse(BaseModel):
    """Response body for POST /query."""
    question: str
    answer: str
    sources: list[str]
    question_type: str
    is_grounded: bool
    hallucination_score: float
    retry_count: int
    session_id: str
    processing_time_ms: float


class IngestRequest(BaseModel):
    """Request body for POST /ingest."""
    directory: Optional[str] = Field(
        default=None,
        description="Absolute or relative path to a directory of .md/.txt files",
    )
    text: Optional[str] = Field(
        default=None,
        description="Raw text content to ingest directly (alternative to directory)",
    )
    source_name: Optional[str] = Field(
        default="manual_input",
        description="Display name for the source when using raw text input",
    )


class IngestResponse(BaseModel):
    """Response body for POST /ingest."""
    status: str
    documents_loaded: int
    chunks_ingested: int
    message: str


class DocumentsResponse(BaseModel):
    """Response body for GET /documents."""
    collection: str
    total_chunks: int
    sources: list[str]


class FeedbackRequest(BaseModel):
    """Request body for POST /feedback."""
    question: str = Field(..., description="The question that was answered")
    answer: str = Field(..., description="The answer that was given")
    feedback: str = Field(..., pattern="^(positive|negative)$", description="'positive' or 'negative'")
    comment: Optional[str] = Field(default=None, description="Optional explanation")
    session_id: Optional[str] = Field(default=None)


class FeedbackResponse(BaseModel):
    """Response body for POST /feedback."""
    status: str
    message: str


class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status: str
    model: str
    embedding_model: str
    collection_stats: dict
    api_version: str


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Check API health, ChromaDB connectivity, and active model configuration.
    Use this to verify the system is running before sending queries.
    """
    stats = get_collection_stats()
    return HealthResponse(
        status="healthy" if stats.get("status") == "connected" else "degraded",
        model=settings.GEMINI_MODEL,
        embedding_model=settings.EMBEDDING_MODEL,
        collection_stats=stats,
        api_version="1.0.0",
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_documents(request: QueryRequest):
    """
    Ask a question and receive a grounded answer with source citations.

    The self-corrective LangGraph workflow will:
    1. Analyze + optimize your question for vector search
    2. Retrieve the most relevant documentation chunks
    3. Grade each chunk for true relevance (not just similarity)
    4. Generate a citation-backed answer
    5. Check the answer for hallucinations
    6. Retry with rewritten queries if step 3 finds no relevant chunks

    Returns the answer along with trust metadata (is_grounded, hallucination_score).
    """
    start_time = time.perf_counter()

    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())

    # Retrieve conversation history for this session
    history = _session_memory.get(session_id, [])

    try:
        result = run_rag_pipeline(
            question=request.question,
            session_id=session_id,
            conversation_history=history,
        )
    except Exception as exc:
        logger.error("Pipeline failed for question '%s': %s", request.question[:80], exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"RAG pipeline error: {str(exc)}",
        )

    # Store this Q&A in session memory for multi-turn context
    _session_memory[session_id].append({"role": "user", "content": request.question})
    _session_memory[session_id].append({"role": "assistant", "content": result.get("generation", "")})

    # Keep session memory bounded to last 10 exchanges (20 messages)
    if len(_session_memory[session_id]) > 20:
        _session_memory[session_id] = _session_memory[session_id][-20:]

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return QueryResponse(
        question=request.question,
        answer=result.get("generation", "No answer generated."),
        sources=result.get("source_citations", []),
        question_type=result.get("question_type", "unknown"),
        is_grounded=result.get("is_grounded", False),
        hallucination_score=result.get("hallucination_score", 0.0),
        retry_count=result.get("retry_count", 0),
        session_id=session_id,
        processing_time_ms=round(elapsed_ms, 2),
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_documents(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest documents into the ChromaDB vector database.

    Two modes:
    - **Directory mode**: Set `directory` to a path containing .md/.txt files
    - **Text mode**: Set `text` to ingest raw text directly (e.g., pasted docs)

    If neither is provided, defaults to the configured DOCS_DIR (./docs).

    Documents are chunked (800 chars, 150 overlap) and embedded using
    sentence-transformers before storage. Large directories may take a minute.
    """
    try:
        # Mode 1: Ingest raw text directly
        if request.text:
            chunks = ingest_text(request.text, request.source_name or "manual_input")
            return IngestResponse(
                status="success",
                documents_loaded=1,
                chunks_ingested=chunks,
                message=f"Ingested raw text as '{request.source_name}' ({chunks} chunks created)",
            )

        # Mode 2: Ingest from directory (default: ./docs)
        directory = request.directory or settings.DOCS_DIR
        result = ingest_from_directory(directory)

        if result["status"] == "no_documents":
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No supported documents (.md, .txt, .rst) found in '{directory}'. "
                    f"Add documents or specify a different directory."
                ),
            )

        return IngestResponse(
            status="success",
            documents_loaded=result["documents_loaded"],
            chunks_ingested=result["chunks_ingested"],
            message=(
                f"Successfully ingested {result['documents_loaded']} document(s) "
                f"as {result['chunks_ingested']} chunks."
            ),
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(exc)}")


@app.get("/documents", response_model=DocumentsResponse, tags=["RAG"])
async def list_documents():
    """
    List all unique document sources currently indexed in the vector database.
    Use this to verify what topics are available for querying.
    """
    try:
        stats = get_collection_stats()
        sources = list_indexed_sources()
        return DocumentsResponse(
            collection=settings.COLLECTION_NAME,
            total_chunks=stats.get("document_count", 0),
            sources=sources,
        )
    except Exception as exc:
        logger.error("Failed to list documents: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/feedback", response_model=FeedbackResponse, tags=["System"])
async def submit_feedback(request: FeedbackRequest):
    """
    Submit positive or negative feedback for a Q&A pair.

    Feedback is stored in-memory and can be used to:
    - Identify poorly answered questions for investigation
    - Fine-tune prompts based on failure patterns
    - Build a labeled dataset for future model improvement
    """
    _feedback_store.append(
        {
            "id": str(uuid.uuid4()),
            "question": request.question,
            "answer": request.answer[:500],  # truncate for storage
            "feedback": request.feedback,
            "comment": request.comment,
            "session_id": request.session_id,
            "timestamp": time.time(),
        }
    )

    logger.info(
        "Feedback stored: %s | question: %s",
        request.feedback,
        request.question[:80],
    )

    return FeedbackResponse(
        status="success",
        message=f"Thank you for your {request.feedback} feedback! It helps improve the system.",
    )


@app.get("/feedback/stats", tags=["System"])
async def get_feedback_stats():
    """Return aggregate feedback statistics (for monitoring dashboards)."""
    total = len(_feedback_store)
    positive = sum(1 for f in _feedback_store if f["feedback"] == "positive")
    negative = total - positive
    return {
        "total_feedback": total,
        "positive": positive,
        "negative": negative,
        "positive_rate": round(positive / total, 2) if total > 0 else 0.0,
    }


# ── Debug Endpoint (remove in production) ────────────────────────────────────


@app.get("/debug/retrieval", tags=["Debug"])
async def debug_retrieval(q: str = "CORS middleware FastAPI"):
    """
    Directly test the retrieval layer without running the full LangGraph pipeline.
    Use this to confirm ChromaDB similarity search is returning results.
    Visit: http://localhost:8000/debug/retrieval?q=CORS+FastAPI
    """
    from app.retriever import retrieve_documents, get_vectorstore

    vs = get_vectorstore()
    raw_count = vs._collection.count()

    docs = retrieve_documents(q, top_k=5)
    return {
        "query": q,
        "collection_doc_count": raw_count,
        "results_returned": len(docs),
        "results": [
            {
                "source": d["source"],
                "score": round(d["score"], 4),
                "content_preview": d["content"][:200],
            }
            for d in docs
        ],
    }


# ── Entry Point ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )
