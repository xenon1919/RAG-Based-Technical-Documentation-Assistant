"""
Document Ingestion Pipeline.

Handles the full ingest flow:
    Load files → chunk → embed → store in ChromaDB

Why RecursiveCharacterTextSplitter?
    It tries to split on increasingly smaller separators:
        paragraphs (\n\n) → lines (\n) → sentences (". ") → words (" ") → chars
    This preserves semantic coherence — a sentence about one concept stays
    in one chunk rather than being cut mid-thought.

Why chunk_size=800 / overlap=150?
    - 800 chars ≈ 150-200 tokens, comfortably under most LLM context windows
    - Large enough to capture a full explanation or code example
    - 150-char overlap prevents answer truncation at boundaries (e.g., a code
      example that starts at the end of one chunk continues in the next)
"""

import logging
import uuid
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.retriever import get_vectorstore

logger = logging.getLogger("rag_assistant.ingestion")
settings = get_settings()


# ── Document Loaders ──────────────────────────────────────────────────────────


def load_file(file_path: str) -> Optional[Document]:
    """
    Load a single text file (.md, .txt, .rst) as a LangChain Document.
    Returns None if the file doesn't exist or can't be read.
    """
    path = Path(file_path)
    if not path.exists():
        logger.warning("File not found: %s", file_path)
        return None

    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.error("Failed to read %s: %s", file_path, exc)
        return None

    return Document(
        page_content=text,
        metadata={
            "source": path.name,
            "file_path": str(path.resolve()),
            "type": "file",
            "extension": path.suffix,
        },
    )


def load_from_directory(directory: Optional[str] = None) -> list[Document]:
    """
    Recursively load all .md, .txt, and .rst files from a directory.
    Skips hidden files and files with unsupported extensions.
    """
    docs_dir = Path(directory or settings.DOCS_DIR)

    if not docs_dir.exists():
        logger.warning("Docs directory not found: %s", docs_dir)
        return []

    supported_extensions = {".md", ".txt", ".rst"}
    documents: list[Document] = []

    for ext in supported_extensions:
        for file_path in sorted(docs_dir.glob(f"**/*{ext}")):
            # Skip hidden files (e.g., .DS_Store)
            if file_path.name.startswith("."):
                continue
            doc = load_file(str(file_path))
            if doc:
                logger.info("Loaded: %s (%d chars)", file_path.name, len(doc.page_content))
                documents.append(doc)

    logger.info("Loaded %d documents from %s", len(documents), docs_dir)
    return documents


def load_text(text: str, source_name: str = "manual_input") -> Document:
    """Create a Document from raw text (e.g., from API input)."""
    return Document(
        page_content=text,
        metadata={"source": source_name, "type": "manual"},
    )


# ── Chunking ──────────────────────────────────────────────────────────────────


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into overlapping chunks using RecursiveCharacterTextSplitter.

    Each chunk gets a unique chunk_id so we can:
    - Deduplicate on re-ingestion (future improvement)
    - Trace answers back to specific chunks for debugging
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        # Try to split at paragraph → line → sentence → word → char boundaries
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Stamp each chunk with a unique ID and its position
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = str(uuid.uuid4())
        chunk.metadata["chunk_index"] = idx

    logger.info("Chunked %d documents → %d chunks", len(documents), len(chunks))
    return chunks


# ── Ingestion Pipeline ────────────────────────────────────────────────────────


def ingest_documents(documents: list[Document]) -> int:
    """
    Run the full ingestion pipeline: chunk → embed → store in ChromaDB.
    Returns the number of chunks successfully ingested.
    """
    if not documents:
        logger.warning("ingest_documents called with empty list")
        return 0

    chunks = chunk_documents(documents)
    if not chunks:
        logger.warning("No chunks produced from %d documents", len(documents))
        return 0

    vectorstore = get_vectorstore()
    logger.info("Ingesting %d chunks into ChromaDB...", len(chunks))

    # LangChain's Chroma.add_documents handles embedding + upsert
    vectorstore.add_documents(chunks)

    logger.info("Ingestion complete: %d chunks stored", len(chunks))
    return len(chunks)


def ingest_from_directory(directory: Optional[str] = None) -> dict:
    """
    Convenience wrapper: load all docs from a directory and ingest them.
    Returns a summary dict for the API response.
    """
    documents = load_from_directory(directory)

    if not documents:
        return {
            "status": "no_documents",
            "documents_loaded": 0,
            "chunks_ingested": 0,
        }

    chunks_ingested = ingest_documents(documents)
    return {
        "status": "success",
        "documents_loaded": len(documents),
        "chunks_ingested": chunks_ingested,
    }


def ingest_text(text: str, source_name: str = "manual_input") -> int:
    """
    Ingest raw text directly (e.g., pasted from API or a URL fetch).
    Returns the number of chunks ingested.
    """
    doc = load_text(text, source_name)
    return ingest_documents([doc])
