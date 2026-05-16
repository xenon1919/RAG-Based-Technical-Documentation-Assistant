"""
Document Retriever — ChromaDB + HuggingFace Sentence Transformers.

Why these choices:
- ChromaDB:           Lightweight, file-based, zero infra needed, great for prototyping
- all-MiniLM-L6-v2:  384-dim embeddings, <1s inference on CPU, excellent recall on
                      English technical text, only ~90MB download
- normalize_embeddings=True: Makes cosine similarity equivalent to dot product,
                              ChromaDB's default distance metric

Singleton pattern: both the embedding model and vectorstore are created once
and reused across all requests to avoid expensive re-initialization.
"""

import logging
from typing import Optional

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import get_settings
from app.state import RetrievedDoc

logger = logging.getLogger("rag_assistant.retriever")
settings = get_settings()

# Module-level singletons — initialized lazily on first use
_embeddings: Optional[HuggingFaceEmbeddings] = None
_vectorstore: Optional[Chroma] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return (or initialize) the sentence-transformer embedding model.
    Downloads the model on first call (~90MB); subsequent calls are instant.
    """
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": settings.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model ready")
    return _embeddings


def get_vectorstore() -> Chroma:
    """
    Return (or initialize) the ChromaDB vectorstore.
    Creates the persistent directory and collection if they don't exist.
    """
    global _vectorstore
    if _vectorstore is None:
        logger.info("Connecting to ChromaDB at: %s", settings.CHROMA_PERSIST_DIR)
        _vectorstore = Chroma(
            collection_name=settings.COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=settings.CHROMA_PERSIST_DIR,
        )
        logger.info("ChromaDB ready — collection: %s", settings.COLLECTION_NAME)
    return _vectorstore


def retrieve_documents(query: str, top_k: Optional[int] = None) -> list[RetrievedDoc]:
    """
    Perform semantic similarity search in ChromaDB.

    Args:
        query:  The search query (usually the optimized query from analyze_query node)
        top_k:  Number of results to return; defaults to settings.TOP_K_RETRIEVAL

    Returns:
        List of RetrievedDoc dicts sorted by similarity (most similar first).
        Returns empty list if the collection is empty or on error.

    Note:
        ChromaDB returns L2 distance by default; lower score = more similar.
        We convert to a similarity score for consistent interpretation.
    """
    k = top_k or settings.TOP_K_RETRIEVAL
    vectorstore = get_vectorstore()

    logger.info("Retrieving top-%d docs | query: %s", k, query[:120])

    try:
        results = vectorstore.similarity_search_with_score(query, k=k)
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        return []

    if not results:
        logger.warning("No results returned from ChromaDB (collection may be empty)")
        return []

    docs: list[RetrievedDoc] = []
    for doc, score in results:
        docs.append(
            RetrievedDoc(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                chunk_id=doc.metadata.get("chunk_id", ""),
                score=float(score),
                metadata=doc.metadata,
            )
        )

    logger.info("Retrieved %d documents (top score: %.4f)", len(docs), docs[0]["score"] if docs else 0)
    return docs


def get_collection_stats() -> dict:
    """
    Return stats about the current ChromaDB collection.
    Used by the health-check and /documents endpoints.
    """
    try:
        client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        collection = client.get_or_create_collection(settings.COLLECTION_NAME)
        return {
            "collection": settings.COLLECTION_NAME,
            "document_count": collection.count(),
            "persist_dir": settings.CHROMA_PERSIST_DIR,
            "status": "connected",
        }
    except Exception as exc:
        logger.error("Failed to get collection stats: %s", exc)
        return {"status": "error", "error": str(exc)}


def list_indexed_sources() -> list[str]:
    """
    Return a sorted list of unique source filenames stored in ChromaDB.
    Useful for the GET /documents endpoint.
    """
    try:
        vs = get_vectorstore()
        result = vs._collection.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []
        sources = sorted({m.get("source", "unknown") for m in metadatas})
        return sources
    except Exception as exc:
        logger.error("Failed to list sources: %s", exc)
        return []
