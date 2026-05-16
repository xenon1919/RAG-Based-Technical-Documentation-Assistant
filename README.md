# RAG-Based Technical Documentation Assistant

> A production-style, self-corrective Retrieval-Augmented Generation (RAG) system that answers technical documentation questions with source citations, LLM-based relevance grading, and hallucination detection — built with **LangGraph**, **FastAPI**, **Gemini**, and **ChromaDB**.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Features](#2-features)
3. [Architecture Diagram](#3-architecture-diagram)
4. [LangGraph Workflow Explanation](#4-langgraph-workflow-explanation)
5. [State Flow Explanation](#5-state-flow-explanation)
6. [Tech Stack](#6-tech-stack)
7. [Project Structure](#7-project-structure)
8. [Setup Instructions](#8-setup-instructions)
9. [Environment Variables](#9-environment-variables)
10. [Installation Steps](#10-installation-steps)
11. [How to Run](#11-how-to-run)
12. [API Reference & Examples](#12-api-reference--examples)
13. [Example Responses](#13-example-responses)
14. [Design Decisions](#14-design-decisions)
15. [Tradeoffs](#15-tradeoffs)
16. [Chunking Strategy](#16-chunking-strategy)
17. [Embedding Strategy](#17-embedding-strategy)
18. [Why ChromaDB](#18-why-chromadb)
19. [Retry Logic Explanation](#19-retry-logic-explanation)
20. [Hallucination Prevention Strategy](#20-hallucination-prevention-strategy)
21. [Future Improvements](#21-future-improvements)

---

## 1. Project Overview

This project implements a **self-corrective RAG pipeline** — a Q&A system that doesn't just retrieve and generate, but actively evaluates the quality of retrieved documents and retries with improved queries when the initial retrieval fails.

**The Problem It Solves:**  
Standard RAG systems retrieve documents by semantic similarity and blindly pass them to a generator, even if the retrieved chunks are tangentially related. This causes hallucinations and off-topic answers.

**The Solution:**  
A LangGraph-powered workflow that:
1. Optimizes your query before searching
2. Retrieves candidate documents
3. **Grades each document with an LLM** (not just similarity score)
4. Retries with a rewritten query if grading fails
5. Generates an answer only from verified, relevant context
6. **Checks the answer for hallucinations** before returning it

---

## 2. Features

| Feature | Description |
|---------|-------------|
| **Self-Corrective Retrieval** | Automatically rewrites queries and retries up to 3 times when no relevant documents are found |
| **LLM Document Grading** | Uses Gemini to evaluate each retrieved chunk for true relevance (not just cosine similarity) |
| **Hallucination Detection** | Secondary LLM call verifies the generated answer is supported by context |
| **Source Citations** | Every answer includes the exact document files it was sourced from |
| **Question Classification** | Automatically classifies questions as conceptual / troubleshooting / API reference / how-to |
| **Conversation Memory** | Session-based memory for multi-turn conversations |
| **FastAPI REST API** | Production-ready API with Swagger UI, validation, and error handling |
| **Streamlit Frontend** | Optional chat UI for interactive testing |
| **Local Embeddings** | Uses sentence-transformers (no embedding API cost) |
| **ChromaDB** | Local persistent vector database (no external service needed) |

---

## 3. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    USER / FRONTEND                          │
│              (Streamlit UI or REST API client)              │
└───────────────────────────┬─────────────────────────────────┘
                            │ POST /query
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    FASTAPI SERVER                           │
│                   (app/main.py)                             │
│  • Request validation (Pydantic)                            │
│  • Session memory management                                │
│  • Response serialization                                   │
└───────────────────────────┬─────────────────────────────────┘
                            │ run_rag_pipeline()
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              LANGGRAPH STATE GRAPH                          │
│                  (app/graph.py)                             │
│                                                             │
│   ┌─────────────┐                                           │
│   │analyze_query│ ◄── Classify + optimize query             │
│   └──────┬──────┘                                           │
│          ▼                                                   │
│   ┌──────────────────┐                                      │
│   │retrieve_documents│ ◄── Semantic search in ChromaDB      │
│   └──────┬───────────┘                                      │
│          ▼                                                   │
│   ┌────────────────┐                                        │
│   │grade_documents │ ◄── LLM evaluates each chunk           │
│   └──────┬─────────┘                                        │
│          │                                                   │
│   ┌──────┴────────────────────────────┐                     │
│   │                                   │                     │
│   │ relevant_docs found?              │ NOT found?          │
│   │                                   │                     │
│   ▼                                   ▼                     │
│ ┌──────────────┐           retry_count < MAX?               │
│ │generate_answer│               │          │                │
│ └──────┬───────┘            YES ▼      NO  ▼                │
│        ▼               ┌──────────┐  ┌──────────┐          │
│ ┌──────────────────┐   │  rewrite │  │ fallback │          │
│ │check_hallucination│  │  _query  │  │ response │          │
│ └──────┬───────────┘   └────┬─────┘  └────┬─────┘          │
│        │                   │              │                 │
│        └──────────────┬────┘              │                 │
│                       ▼                   ▼                 │
│                      END ◄────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┴──────────────┐
              ▼                            ▼
┌─────────────────────┐      ┌─────────────────────────────┐
│      CHROMADB       │      │      GEMINI (Gemini 2.0)     │
│   (app/retriever.py)│      │    (app/grader.py,           │
│                     │      │     generator.py,            │
│  • Vector storage   │      │     query_rewriter.py)       │
│  • Similarity search│      │                              │
│  • Persistent disk  │      │  • Query analysis            │
└─────────────────────┘      │  • Document grading          │
                             │  • Answer generation         │
                             │  • Hallucination checking    │
                             └─────────────────────────────┘
```

---

## 4. LangGraph Workflow Explanation

The LangGraph workflow has **7 nodes** connected by **fixed and conditional edges**:

### Node 1: `analyze_query`
**File**: `app/query_rewriter.py`

Takes the raw user question and:
- Classifies it into one of 4 types: `conceptual`, `troubleshooting`, `api_reference`, `how_to`
- Rewrites it into search-optimized form (e.g., "my API is slow" → "FastAPI endpoint performance optimization techniques")
- Initializes `retry_count = 0`

Why this matters: Users often ask colloquial questions. Vector search needs technical terms that match documentation language.

### Node 2: `retrieve_documents`
**File**: `app/graph.py` (wraps `app/retriever.py`)

- Uses the optimized query to search ChromaDB via cosine similarity
- Returns top-k (default 5) chunks as `RetrievedDoc` objects with content + metadata + score

### Node 3: `grade_documents`
**File**: `app/grader.py`

**The most important node.** For each retrieved chunk:
- Asks Gemini: "Is this chunk relevant to answering [question]?"
- Gets a binary `relevant: true/false` + confidence score back as JSON
- Filters out irrelevant chunks

Result: `relevant_docs` contains only high-confidence relevant chunks.

### Node 4: `rewrite_query` (retry path)
**File**: `app/query_rewriter.py`

Called when `grade_documents` finds no relevant chunks. Uses a different lexical strategy each attempt:
- Attempt 1: synonyms and alternative terminology
- Attempt 2: simpler, more general concepts
- Attempt 3: just the core technology name

Increments `retry_count` and clears stale results for a clean retry.

### Node 5: `generate_answer`
**File**: `app/generator.py`

- Formats relevant chunks into a structured context window
- Sends to Gemini with a strict prompt: "Answer ONLY from this context"
- Extracts source citations from the document metadata

### Node 6: `check_hallucination` (bonus)
**File**: `app/generator.py`

- Makes a secondary Gemini call that acts as an independent auditor
- Checks whether every factual claim in the answer appears in the context
- Returns `is_grounded: bool` and `hallucination_score: float` (0.0-1.0)

### Node 7: `fallback`
**File**: `app/graph.py`

Called when `retry_count >= MAX_RETRIES` and still no relevant docs. Returns a structured "I couldn't find this in the documentation" message with helpful next-step suggestions.

---

## 5. State Flow Explanation

The `GraphState` TypedDict is the shared whiteboard across all nodes:

```
INITIAL STATE (set by API caller)
───────────────────────────────────────────────────────────────────
question:           "How do I add CORS to FastAPI?"
session_id:         "abc-123"
retry_count:        0
[all other fields:  empty/defaults]

AFTER analyze_query
───────────────────────────────────────────────────────────────────
question_type:      "how_to"
optimized_query:    "FastAPI CORS middleware CORSMiddleware setup configuration"

AFTER retrieve_documents
───────────────────────────────────────────────────────────────────
retrieved_docs:     [5 RetrievedDoc objects with content + score]

AFTER grade_documents (scenario: 3 of 5 relevant)
───────────────────────────────────────────────────────────────────
relevant_docs:      [3 RetrievedDoc objects, filtered]

AFTER generate_answer
───────────────────────────────────────────────────────────────────
generation:         "## Adding CORS to FastAPI\n\nUse CORSMiddleware..."
source_citations:   ["fastapi_documentation.md"]

AFTER check_hallucination
───────────────────────────────────────────────────────────────────
is_grounded:        True
hallucination_score: 0.95

FINAL STATE (returned to API)
───────────────────────────────────────────────────────────────────
All of the above, plus processing metadata
```

**Retry scenario**: If `relevant_docs` is empty after grading:
- `rewrite_query` changes `optimized_query` and sets `retry_count = 1`
- `retrieve_documents` runs again with the new query
- `grade_documents` runs again
- This loop continues until `retry_count >= MAX_RETRIES`

---

## 6. Tech Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Graph Orchestration | LangGraph | ≥ 0.2.0 | Self-corrective workflow with cycles |
| LLM | Google Gemini | gemini-2.0-flash | Query analysis, grading, generation |
| Web Framework | FastAPI | ≥ 0.115 | REST API with auto-documentation |
| ASGI Server | Uvicorn | ≥ 0.30 | Production-grade async server |
| Vector Database | ChromaDB | ≥ 0.5.0 | Local persistent embedding storage |
| Embeddings | sentence-transformers | ≥ 3.0.0 | Local text embeddings (no API cost) |
| Embedding Model | all-MiniLM-L6-v2 | — | 384-dim, excellent English recall |
| LangChain Core | langchain | ≥ 0.3.0 | Document loading, text splitting |
| ChromaDB Bridge | langchain-chroma | ≥ 0.1.4 | LangChain ↔ ChromaDB integration |
| HuggingFace Bridge | langchain-huggingface | ≥ 0.1.0 | LangChain ↔ sentence-transformers |
| Data Validation | Pydantic | ≥ 2.7.0 | Request/response models + state schema |
| Config Management | pydantic-settings | ≥ 2.3.0 | Typed env var loading |
| Optional Frontend | Streamlit | ≥ 1.38.0 | Chat UI for interactive testing |
| Package Manager | uv | latest | Fast Python package management |

---

## 7. Project Structure

```
RAG-Based Technical Documentation Assistant/
│
├── app/                        ← Core application package
│   ├── __init__.py
│   ├── config.py               ← Pydantic settings (env vars)
│   ├── state.py                ← GraphState TypedDict schema
│   ├── prompts.py              ← All LLM prompts (centralized)
│   ├── utils.py                ← Logging, formatting, helpers
│   ├── retriever.py            ← ChromaDB queries + singleton management
│   ├── ingestion.py            ← Document loading + chunking pipeline
│   ├── grader.py               ← Document relevance grading node + LLM init
│   ├── generator.py            ← Answer generation + hallucination check
│   ├── query_rewriter.py       ← Query analysis + retry rewriting
│   ├── graph.py                ← LangGraph StateGraph assembly + runner
│   └── main.py                 ← FastAPI app with all endpoints
│
├── docs/                       ← Document corpus (the knowledge base)
│   ├── fastapi_documentation.md
│   ├── langchain_documentation.md
│   └── langgraph_documentation.md
│
├── chroma_db/                  ← ChromaDB persistent storage (auto-created)
│   └── .gitkeep
│
├── streamlit_app.py            ← Optional Streamlit chat frontend
├── pyproject.toml              ← Project config + dependencies (uv)
├── .env.example                ← Environment variable template
├── sample_questions.md         ← Test questions for all topics
└── README.md                   ← This file
```

---

## 8. Setup Instructions

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- A [Google AI Studio](https://aistudio.google.com/app/apikey) API key (free tier available)

### First-time System Check

```bash
python --version    # Should be 3.11+
uv --version        # Should be installed
```

If `uv` is not installed:
```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 9. Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_API_KEY` | **Yes** | — | Google Gemini API key |
| `GEMINI_MODEL` | No | `gemini-2.0-flash` | Gemini model name |
| `GEMINI_TEMPERATURE` | No | `0.1` | LLM temperature (lower = more deterministic) |
| `GEMINI_MAX_OUTPUT_TOKENS` | No | `2048` | Max tokens in LLM response |
| `CHROMA_PERSIST_DIR` | No | `./chroma_db` | Path for ChromaDB storage |
| `COLLECTION_NAME` | No | `technical_docs` | ChromaDB collection name |
| `EMBEDDING_MODEL` | No | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `EMBEDDING_DEVICE` | No | `cpu` | `cpu` or `cuda` for GPU |
| `CHUNK_SIZE` | No | `800` | Characters per document chunk |
| `CHUNK_OVERLAP` | No | `150` | Overlap between adjacent chunks |
| `TOP_K_RETRIEVAL` | No | `5` | Number of chunks to retrieve |
| `MAX_RETRIES` | No | `3` | Max query rewrite attempts |
| `API_HOST` | No | `0.0.0.0` | FastAPI server host |
| `API_PORT` | No | `8000` | FastAPI server port |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity |
| `DOCS_DIR` | No | `./docs` | Default document directory |

---

## 10. Installation Steps

### Step 1: Clone or download the project

```bash
cd "RAG-Based Technical Documentation Assistant"
```

### Step 2: Create virtual environment with uv

```bash
uv venv
```

This creates a `.venv` directory in the project root.

### Step 3: Activate the virtual environment

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# macOS/Linux
source .venv/bin/activate
```

### Step 4: Install dependencies

```bash
uv sync
```

This reads `pyproject.toml` and installs all dependencies into the virtual environment.

> **Note**: The first install downloads PyTorch (~2GB for CPU version) as a dependency of sentence-transformers. This is a one-time download.

### Step 5: Set up environment variables

```bash
# Windows
copy .env.example .env

# macOS/Linux
cp .env.example .env
```

Edit `.env` and set your `GOOGLE_API_KEY`:

```env
GOOGLE_API_KEY=AIza...your-key-here...
```

### Step 6: Verify setup

```bash
python -c "from app.config import get_settings; s = get_settings(); print('Config OK:', s.GEMINI_MODEL)"
```

---

## 11. How to Run

### Terminal 1: Start the FastAPI Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

On startup, you should see:
```
INFO: Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
INFO: Embedding model ready
INFO: Connecting to ChromaDB at: ./chroma_db
INFO: ChromaDB ready — collection: technical_docs
INFO: Models warmed up — API ready to serve requests
INFO: Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Ingest the Sample Documentation

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{}'
```

This ingests all `.md` files from `./docs/` into ChromaDB.

### Step 3: Verify Documents Are Indexed

```bash
curl http://localhost:8000/documents
```

### Step 4: Ask Your First Question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I add CORS middleware to FastAPI?"}'
```

### Optional: Run the Streamlit Frontend

```bash
# Terminal 2 (while FastAPI is running in Terminal 1)
streamlit run streamlit_app.py
```

Open your browser at `http://localhost:8501`

### Explore the Swagger UI

Open `http://localhost:8000/docs` for interactive API documentation.

---

## 12. API Reference & Examples

### `POST /query` — Ask a Question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I add CORS middleware to FastAPI?",
    "session_id": "my-session-123"
  }'
```

**Fields:**
- `question` (required): Your question (5-1000 chars)
- `session_id` (optional): For multi-turn conversation memory. Auto-generated if omitted.

---

### `POST /ingest` — Ingest Documents

**Mode 1: Ingest from directory (default)**
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Mode 2: Ingest from custom directory**
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"directory": "/path/to/your/docs"}'
```

**Mode 3: Ingest raw text**
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "FastAPI is a modern web framework...",
    "source_name": "my_custom_doc"
  }'
```

---

### `GET /documents` — List Indexed Sources

```bash
curl http://localhost:8000/documents
```

---

### `GET /health` — Health Check

```bash
curl http://localhost:8000/health
```

---

### `POST /feedback` — Submit Feedback

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I add CORS middleware?",
    "answer": "Use CORSMiddleware...",
    "feedback": "positive",
    "comment": "Very clear explanation!"
  }'
```

---

## 13. Example Responses

### `GET /health`
```json
{
    "status": "healthy",
    "model": "gemini-2.0-flash",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "collection_stats": {
        "collection": "technical_docs",
        "document_count": 127,
        "persist_dir": "./chroma_db",
        "status": "connected"
    },
    "api_version": "1.0.0"
}
```

### `POST /query` — Successful Response
```json
{
    "question": "How do I add CORS middleware to FastAPI?",
    "answer": "## Adding CORS Middleware to FastAPI\n\nTo enable CORS in FastAPI, use the `CORSMiddleware` from `fastapi.middleware.cors`:\n\n```python\nfrom fastapi import FastAPI\nfrom fastapi.middleware.cors import CORSMiddleware\n\napp = FastAPI()\n\norigins = [\n    \"http://localhost:3000\",\n    \"https://myapp.com\",\n]\n\napp.add_middleware(\n    CORSMiddleware,\n    allow_origins=origins,\n    allow_credentials=True,\n    allow_methods=[\"*\"],\n    allow_headers=[\"*\"],\n)\n```\n\n**Important**: Using `allow_origins=[\"*\"]` with `allow_credentials=True` is not permitted by browsers — specify explicit origins when using credentials.\n\n**Sources**: fastapi_documentation.md",
    "sources": ["fastapi_documentation.md"],
    "question_type": "how_to",
    "is_grounded": true,
    "hallucination_score": 0.97,
    "retry_count": 0,
    "session_id": "my-session-123",
    "processing_time_ms": 4231.5
}
```

### `POST /query` — Fallback Response (topic not in docs)
```json
{
    "question": "How do I configure Kubernetes?",
    "answer": "## Unable to Find Relevant Documentation\n\nAfter **3 retrieval attempt(s)**, I could not find relevant documentation...",
    "sources": [],
    "question_type": "how_to",
    "is_grounded": false,
    "hallucination_score": 0.0,
    "retry_count": 3,
    "session_id": "my-session-123",
    "processing_time_ms": 12840.2
}
```

### `POST /ingest`
```json
{
    "status": "success",
    "documents_loaded": 3,
    "chunks_ingested": 127,
    "message": "Successfully ingested 3 document(s) as 127 chunks."
}
```

### `GET /documents`
```json
{
    "collection": "technical_docs",
    "total_chunks": 127,
    "sources": [
        "fastapi_documentation.md",
        "langchain_documentation.md",
        "langgraph_documentation.md"
    ]
}
```

---

## 14. Design Decisions

### Why LangGraph over a simple LangChain chain?

LangGraph enables **cycles** — the ability to loop back and retry with a rewritten query. A standard LangChain chain is linear (A → B → C) and cannot retry. LangGraph's StateGraph supports conditional routing and cycles, which are essential for self-corrective pipelines.

### Why grade documents at all?

Vector similarity measures *geometric closeness in embedding space*, not semantic relevance to the question. A chunk about "middleware in Express.js" might be geometrically close to "FastAPI middleware" but completely useless for answering the question. LLM grading performs true relevance judgment that similarity scores cannot.

### Why use a separate hallucination check?

Even with grounded context, LLMs occasionally "blend in" training data. The hallucination checker is an independent auditor that verifies every factual claim in the answer against the context. The `hallucination_score` field lets API consumers decide how to present uncertain answers (e.g., show a warning badge).

### Why sentence-transformers over API embeddings?

- **Cost**: Zero per-embedding API cost
- **Privacy**: Data never leaves your machine
- **Latency**: Local inference is faster for batch operations
- **Reproducibility**: Model version is pinned, embeddings are deterministic

### Why separate nodes instead of one big function?

Each node has a single responsibility, making it independently testable, replaceable, and debuggable. Adding a new step (e.g., "rerank documents") only requires adding a new node and edges — no modification to existing nodes.

---

## 15. Tradeoffs

| Decision | Chosen | Alternative | Tradeoff |
|----------|--------|-------------|----------|
| Graph framework | LangGraph | Plain Python | More setup, much more flexibility |
| Vector DB | ChromaDB | Pinecone | No managed service, but free + local |
| Embeddings | sentence-transformers | Google Embeddings API | Slower startup, but free |
| LLM | Gemini | OpenAI GPT-4 | Gemini has a generous free tier |
| Grading | Per-document LLM call | Batch grading | Higher accuracy, more API calls |
| Memory | In-memory dict | Redis/PostgreSQL | Lost on restart, simpler for dev |
| Chunking | RecursiveChar | Token-based | Char-based is less precise but dependency-free |

---

## 16. Chunking Strategy

**Chosen settings**: `chunk_size=800`, `chunk_overlap=150`

**Why RecursiveCharacterTextSplitter?**

This splitter tries a hierarchy of separators before falling back to the next:
```
"\n\n" → paragraph breaks (ideal: each chunk = one concept)
"\n"   → line breaks
". "   → sentence boundaries (preserves complete thoughts)
" "    → word boundaries (never cuts a word)
""     → character fallback (last resort)
```

**Why 800 characters?**
- ~150-200 tokens — well within any LLM's context window
- Large enough to capture a full code example with explanation
- Small enough to be focused (one topic per chunk)

**Why 150 characters overlap?**
- Prevents answer truncation at chunk boundaries
- A code example that starts at the end of chunk N appears at the start of chunk N+1
- Overlap cost is acceptable since we're searching, not storing every character twice

**Alternative considered**: Token-based splitting (requires a tokenizer dependency per model). Character-based is simpler, model-agnostic, and good enough for English technical text.

---

## 17. Embedding Strategy

**Model**: `sentence-transformers/all-MiniLM-L6-v2`

| Property | Value |
|----------|-------|
| Embedding dimensions | 384 |
| Model size | ~90MB |
| Inference speed (CPU) | ~500 sentences/second |
| License | Apache 2.0 (free for commercial use) |
| MTEB benchmark score | 56.26 (excellent for its size) |

**Why normalize embeddings?**  
Setting `normalize_embeddings=True` makes cosine similarity equivalent to dot product. ChromaDB uses L2 distance by default, but normalized embeddings make all distance metrics equivalent — ensuring consistent ranking behavior.

**Why not use a larger model (e.g., all-mpnet-base-v2)?**  
all-MiniLM-L6-v2 is 5x smaller and 2-3x faster than mpnet with only a marginal quality drop (56.26 vs 57.02 MTEB score). For a developer demo, the speed advantage outweighs the tiny accuracy difference.

---

## 18. Why ChromaDB

ChromaDB was chosen over alternatives for the following reasons:

1. **Zero infrastructure**: Runs entirely in-process as a Python library — no Docker, no server, no cloud account required
2. **Persistent storage**: Data is saved to disk (the `chroma_db/` directory) and survives Python restarts
3. **HNSW indexing**: Uses Hierarchical Navigable Small World algorithm for fast approximate nearest neighbor search (logarithmic query time)
4. **Rich metadata filtering**: Can filter by document source, date, or any custom field alongside vector similarity
5. **LangChain native**: The `langchain-chroma` package provides seamless integration
6. **Developer experience**: Clean Python API, excellent error messages, automatic collection creation

**When to graduate from ChromaDB**:
- 10M+ chunks → consider Pinecone, Weaviate, or Qdrant
- Multi-tenant production → Pinecone (managed) or self-hosted Qdrant
- Real-time streaming ingestion → Weaviate with real-time indexing

---

## 19. Retry Logic Explanation

The retry system uses a feedback loop: grading failure → rewrite → retrieve → grade again.

```
Graph state: retry_count = 0

Attempt 1: optimized_query (from analyze_query)
    ↳ retrieve → grade → NO relevant docs
    ↳ rewrite_query: strategy="synonyms", retry_count = 1

Attempt 2: new_query_with_synonyms
    ↳ retrieve → grade → NO relevant docs
    ↳ rewrite_query: strategy="generalize", retry_count = 2

Attempt 3: general_query
    ↳ retrieve → grade → NO relevant docs
    ↳ rewrite_query: strategy="core_term_only", retry_count = 3

retry_count (3) >= MAX_RETRIES (3) → fallback node → END
```

**Key design choices**:
- `MAX_RETRIES = 3` balances thoroughness vs. API cost (each retry = 2 LLM calls: grading + rewriting)
- The rewrite prompt escalates strategy based on attempt number (more aggressive generalization each time)
- `retry_count` is incremented in the rewrite node, not the router — this means the router sees the *updated* count when deciding whether to retry again

**Cost implication**: In the worst case (3 retries, 5 docs graded each), the pipeline makes `3 * (5 grading + 1 rewriting) + 1 generation + 1 hallucination check = 21 LLM calls`. Gemini's free tier (15 RPM) handles this comfortably.

---

## 20. Hallucination Prevention Strategy

This system uses a **multi-layer approach** to prevent hallucinations:

### Layer 1: Grounding through Grading
Only verified-relevant chunks enter the generation context. This removes the most common cause of hallucination: irrelevant context that confuses the model.

### Layer 2: Strict Generation Prompt
The generation prompt explicitly forbids using external knowledge:
```
"Answer based ONLY on the provided context. If context is insufficient,
say: 'Based on the available documentation, I cannot fully answer this question.'"
```

### Layer 3: Post-Generation Hallucination Check
A secondary Gemini call audits the generated answer:
- Identifies specific claims not supported by context
- Returns `hallucination_score` (0.0-1.0) for downstream decisions
- Sets `is_grounded: bool` as a simple threshold flag

### Layer 4: Citation Tracking
Every answer includes the source filenames. Users can verify claims by checking the source documents directly.

### Layer 5: Fallback Over Fabrication
If no relevant context exists after MAX_RETRIES, the system returns a transparent "I don't know" message rather than generating an answer from training data.

**Limitation**: The hallucination checker is itself an LLM and can miss subtle hallucinations or flag valid inferences. For critical applications, add a deterministic fact-checking step (regex matching of key facts against context).

---



## License

MIT License — free to use, modify, and distribute.

---

*Built as an AI/ML internship assignment demonstrating production-style RAG system design with LangGraph.*
