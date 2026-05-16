"""
Shared utility functions for the RAG system.

Responsibilities:
- Logging setup (single source of truth)
- Document formatting for LLM context windows
- JSON response cleaning (LLMs sometimes wrap JSON in markdown)
- Citation extraction
- Fallback response generation
"""

import json
import logging
import re
from typing import Optional

from app.config import get_settings

settings = get_settings()


# ── Logging ───────────────────────────────────────────────────────────────────


def setup_logging() -> logging.Logger:
    """
    Configure the application-wide logger.
    Call this once at startup; subsequent calls return the same logger.
    """
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    return logging.getLogger("rag_assistant")


logger = setup_logging()


# ── Document Formatting ───────────────────────────────────────────────────────


def format_docs_for_context(docs: list[dict]) -> str:
    """
    Serialize a list of RetrievedDoc dicts into a single context string
    suitable for injection into LLM prompts.

    Each chunk is labeled with its index and source so the LLM can
    produce accurate citations (e.g., [Document 2] → source.md).
    """
    if not docs:
        return "No relevant documents found."

    sections = []
    for i, doc in enumerate(docs, start=1):
        source = doc.get("source", "unknown")
        content = doc.get("content", "").strip()
        section = f"[Document {i} | Source: {source}]\n{content}"
        sections.append(section)

    return "\n\n---\n\n".join(sections)


def extract_sources(docs: list[dict]) -> list[str]:
    """
    Return a deduplicated, ordered list of source filenames from docs.
    Preserves insertion order (first occurrence wins).
    """
    seen: set[str] = set()
    sources: list[str] = []
    for doc in docs:
        source = doc.get("source", "unknown")
        if source not in seen:
            seen.add(source)
            sources.append(source)
    return sources


# ── JSON Parsing ──────────────────────────────────────────────────────────────


def clean_json_response(raw: str) -> dict:
    """
    Parse JSON from an LLM response that may contain markdown formatting.

    LLMs often wrap JSON in ```json ... ``` code fences. This function
    strips the fences and tries multiple parse strategies before giving up.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    # Attempt 1: direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 2: find the first { ... } block in the response
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse JSON from LLM response: %.200s", raw)
    return {}


# ── Fallback Response ─────────────────────────────────────────────────────────


def build_fallback_response(question: str, retry_count: int) -> str:
    """
    Generate a helpful fallback message when the RAG pipeline exhausts retries.

    This is shown to the user instead of an empty or confusing response
    when the vector database has no relevant documents for the question.
    """
    return (
        f"## Unable to Find Relevant Documentation\n\n"
        f"After **{retry_count} retrieval attempt(s)**, I could not find relevant "
        f"documentation to answer your question:\n\n"
        f"> {question}\n\n"
        f"**Possible reasons:**\n"
        f"- The topic may not be covered in the ingested documentation\n"
        f"- Try rephrasing with different keywords\n"
        f"- Ensure you have ingested the relevant documents via `POST /ingest`\n\n"
        f"**Suggestion:** Run `GET /documents` to see what topics are available."
    )


# ── Text Helpers ──────────────────────────────────────────────────────────────


def truncate(text: str, max_chars: int = 200) -> str:
    """Truncate text for safe logging without losing the log call."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def extract_text(response) -> str:
    """
    Safely extract plain text from a LangChain LLM response.

    In langchain-google-genai >=2.x, response.content may be a list of
    content-part dicts (e.g. [{"type": "text", "text": "..."}]) rather
    than a plain string. This function handles both cases.
    """
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            else:
                parts.append(str(part))
        return "".join(parts).strip()
    return str(content).strip()
