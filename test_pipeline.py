"""
Diagnostic script — run this to isolate exactly which step is failing.

Usage:
    python test_pipeline.py
"""

import sys

print("=" * 60)
print("RAG PIPELINE DIAGNOSTIC")
print("=" * 60)

# ── Step 1: Config ────────────────────────────────────────────
print("\n[1] Loading config...")
try:
    from app.config import get_settings
    s = get_settings()
    print(f"    GOOGLE_API_KEY set: {bool(s.GOOGLE_API_KEY)}")
    print(f"    GEMINI_MODEL: {s.GEMINI_MODEL}")
    print(f"    CHROMA_PERSIST_DIR: {s.CHROMA_PERSIST_DIR}")
    print(f"    EMBEDDING_MODEL: {s.EMBEDDING_MODEL}")
    print("    ✓ Config OK")
except Exception as e:
    print(f"    ✗ Config FAILED: {e}")
    sys.exit(1)

# ── Step 2: ChromaDB raw count ────────────────────────────────
print("\n[2] ChromaDB raw count...")
try:
    import chromadb
    client = chromadb.PersistentClient(path=s.CHROMA_PERSIST_DIR)
    col = client.get_or_create_collection(s.COLLECTION_NAME)
    count = col.count()
    print(f"    Collection '{s.COLLECTION_NAME}' has {count} documents")
    if count == 0:
        print("    ✗ Collection is EMPTY — run POST /ingest first")
        sys.exit(1)
    print("    ✓ ChromaDB raw access OK")
except Exception as e:
    print(f"    ✗ ChromaDB FAILED: {e}")
    sys.exit(1)

# ── Step 3: Embeddings ────────────────────────────────────────
print("\n[3] Loading embedding model...")
try:
    from app.retriever import get_embeddings
    emb = get_embeddings()
    test_vec = emb.embed_query("test query")
    print(f"    Embedding dimensions: {len(test_vec)}")
    print("    ✓ Embeddings OK")
except Exception as e:
    print(f"    ✗ Embeddings FAILED: {e}")
    sys.exit(1)

# ── Step 4: Similarity search ─────────────────────────────────
print("\n[4] Testing similarity search...")
try:
    from app.retriever import get_vectorstore
    vs = get_vectorstore()
    results = vs.similarity_search_with_score("CORS middleware FastAPI", k=3)
    print(f"    Results returned: {len(results)}")
    for doc, score in results:
        src = doc.metadata.get("source", "unknown")
        print(f"    Score={score:.4f} | Source={src} | Content={doc.page_content[:80]!r}")
    if not results:
        print("    ✗ Similarity search returned EMPTY — possible ChromaDB issue")
    else:
        print("    ✓ Similarity search OK")
except Exception as e:
    print(f"    ✗ Similarity search FAILED: {e}")

# ── Step 5: retrieve_documents() wrapper ──────────────────────
print("\n[5] Testing retrieve_documents()...")
try:
    from app.retriever import retrieve_documents
    docs = retrieve_documents("CORS middleware FastAPI", top_k=3)
    print(f"    Docs returned: {len(docs)}")
    for d in docs:
        print(f"    Score={d['score']:.4f} | Source={d['source']} | {d['content'][:80]!r}")
    if not docs:
        print("    ✗ retrieve_documents returned EMPTY")
    else:
        print("    ✓ retrieve_documents OK")
except Exception as e:
    print(f"    ✗ retrieve_documents FAILED: {e}")

# ── Step 6: Gemini LLM ────────────────────────────────────────
print("\n[6] Testing Gemini LLM...")
try:
    from app.grader import get_llm
    from langchain_core.messages import HumanMessage
    llm = get_llm()
    resp = llm.invoke([HumanMessage(content='Reply with exactly: {"status": "ok"}')])
    print(f"    LLM response: {resp.content[:100]}")
    print("    ✓ Gemini LLM OK")
except Exception as e:
    print(f"    ✗ Gemini LLM FAILED: {e}")

# ── Step 7: Grading ───────────────────────────────────────────
print("\n[7] Testing document grader...")
try:
    from app.grader import grade_single_document
    from app.state import RetrievedDoc
    test_doc = RetrievedDoc(
        content=(
            "## CORS Middleware\n\nAdd Cross-Origin Resource Sharing support:\n\n"
            "app.add_middleware(CORSMiddleware, allow_origins=['*'], "
            "allow_credentials=True, allow_methods=['*'], allow_headers=['*'])"
        ),
        source="fastapi_documentation.md",
        chunk_id="test-001",
        score=0.1,
        metadata={},
    )
    is_relevant, confidence = grade_single_document(
        "What is CORS and how do I enable it in FastAPI?", test_doc
    )
    print(f"    is_relevant={is_relevant}, confidence={confidence:.2f}")
    print("    ✓ Grader OK")
except Exception as e:
    print(f"    ✗ Grader FAILED: {e}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
