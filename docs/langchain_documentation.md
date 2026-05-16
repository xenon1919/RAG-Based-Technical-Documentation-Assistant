# LangChain Technical Documentation

## Overview

LangChain is a framework for developing applications powered by large language models (LLMs). It provides composable building blocks for creating LLM-based workflows, including:

- **Chains**: Sequences of calls to LLMs and other components
- **Agents**: LLMs that decide which tools to use based on user input
- **Memory**: Mechanisms to persist information across interactions
- **Retrievers**: Interfaces for fetching relevant documents
- **Vector Stores**: Databases for storing and searching embeddings

LangChain follows the "LCEL" (LangChain Expression Language) pattern for composing chains with the pipe operator `|`.

---

## Installation

```bash
pip install langchain langchain-community langchain-openai
# or with uv:
uv add langchain langchain-community langchain-google-genai
```

---

## Core Concept: Language Models

LangChain wraps various LLM providers:

```python
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize the model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key="your-api-key",
    temperature=0.1,
)

# Simple invocation
response = llm.invoke("What is the capital of France?")
print(response.content)

# With messages
from langchain.schema import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is 2+2?"),
]
response = llm.invoke(messages)
```

---

## Prompt Templates

```python
from langchain.prompts import ChatPromptTemplate, PromptTemplate

# Simple string template
prompt = PromptTemplate.from_template(
    "Tell me a joke about {topic}"
)

# Chat prompt template
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
    ("human", "{text}"),
])

# Format the prompt
formatted = chat_prompt.format_messages(
    input_language="English",
    output_language="French",
    text="I love programming."
)
```

---

## LCEL: LangChain Expression Language

LCEL uses the pipe `|` operator to chain components:

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key="...")
output_parser = StrOutputParser()

# Chain: prompt → LLM → parse to string
chain = prompt | llm | output_parser

# Invoke the chain
result = chain.invoke({"topic": "programming"})
print(result)
```

LCEL chains support:
- `invoke()` — single call
- `batch()` — multiple inputs in parallel
- `stream()` — streaming tokens
- `ainvoke()` — async version

---

## Document Loaders

Load documents from various sources:

```python
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader,
    DirectoryLoader,
)

# Load a text file
loader = TextLoader("path/to/file.txt")
documents = loader.load()

# Load a web page
loader = WebBaseLoader("https://example.com/docs")
documents = loader.load()

# Load all .md files in a directory
loader = DirectoryLoader("./docs", glob="**/*.md")
documents = loader.load()

# Each document has:
# document.page_content  — the text
# document.metadata      — dict with source info
```

---

## Text Splitters

Split documents into chunks for vector storage:

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)

# RecursiveCharacterTextSplitter (recommended)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,        # max characters per chunk
    chunk_overlap=150,     # characters shared between adjacent chunks
    separators=["\n\n", "\n", ". ", " ", ""],  # split hierarchy
)

chunks = splitter.split_documents(documents)
print(f"Split {len(documents)} docs into {len(chunks)} chunks")

# Each chunk is a Document with the same metadata structure
# plus chunk-specific metadata if you add it
```

### Why RecursiveCharacterTextSplitter?

It tries to split on natural boundaries in order:
1. `\n\n` — paragraph breaks (ideal)
2. `\n` — line breaks
3. `. ` — sentence boundaries
4. ` ` — word boundaries (last resort)
5. `""` — character by character (emergency)

This preserves semantic coherence better than fixed-size splits.

---

## Embeddings

Convert text to numerical vectors for similarity search:

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Local embeddings with sentence-transformers (free, runs locally)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Generate embedding for a single text
vector = embeddings.embed_query("What is FastAPI?")
print(f"Embedding dimensions: {len(vector)}")  # 384 for MiniLM

# Embed a batch of documents
vectors = embeddings.embed_documents(["doc1 text", "doc2 text"])
```

---

## Vector Stores

Store and search document embeddings:

```python
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a new vectorstore from documents
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="my_docs",
)

# Or connect to an existing one
vectorstore = Chroma(
    collection_name="my_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# Similarity search
results = vectorstore.similarity_search("How do I use FastAPI?", k=5)
for doc in results:
    print(doc.page_content[:200])
    print(f"Source: {doc.metadata['source']}")

# Search with score
results_with_score = vectorstore.similarity_search_with_score("query", k=5)
for doc, score in results_with_score:
    print(f"Score: {score:.4f} | {doc.page_content[:100]}")
```

---

## Retrievers

Retrievers provide a standard interface for fetching documents:

```python
# Convert a vectorstore to a retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)

# Retrieve documents
docs = retriever.invoke("What is dependency injection?")

# Multi-query retriever: generates multiple query variations
from langchain.retrievers import MultiQueryRetriever

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm,
)
```

---

## RAG Chain

A complete Retrieval-Augmented Generation pipeline:

```python
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Setup
vectorstore = Chroma(...)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key="...")

# Prompt
template = """Answer based on context only:
{context}

Question: {question}"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Run
answer = rag_chain.invoke("How do I add authentication to FastAPI?")
print(answer)
```

---

## Memory

Maintain conversation history across turns:

```python
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory

# Buffer memory: stores all messages
memory = ConversationBufferMemory(return_messages=True)
memory.save_context(
    {"input": "What is FastAPI?"},
    {"output": "FastAPI is a modern Python web framework..."},
)

# Get history
history = memory.load_memory_variables({})
print(history["history"])

# Summary memory: summarizes old messages to save context
summary_memory = ConversationSummaryMemory(llm=llm, return_messages=True)
```

---

## Output Parsers

Parse LLM outputs into structured formats:

```python
from langchain.output_parsers import (
    StrOutputParser,
    PydanticOutputParser,
    CommaSeparatedListOutputParser,
    JsonOutputParser,
)
from pydantic import BaseModel

# Parse to string
str_parser = StrOutputParser()

# Parse to Pydantic model
class Movie(BaseModel):
    title: str
    year: int
    director: str

pydantic_parser = PydanticOutputParser(pydantic_object=Movie)
format_instructions = pydantic_parser.get_format_instructions()

# Parse to JSON
json_parser = JsonOutputParser()
chain = prompt | llm | json_parser
```

---

## Agents

Agents use LLMs to decide which tools to use:

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool

def search_web(query: str) -> str:
    # Your search implementation
    return f"Search results for: {query}"

tools = [
    Tool(
        name="WebSearch",
        func=search_web,
        description="Search the web for current information",
    )
]

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({"input": "What is the latest Python version?"})
```

---

## Callbacks and Tracing

Monitor LLM calls with callbacks:

```python
from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

# Print all LLM interactions to stdout
handler = StdOutCallbackHandler()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key="...",
    callbacks=[handler],
)

# Or use LangSmith for production tracing
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-key"
```

---

## Error Handling

```python
from langchain_core.exceptions import OutputParserException

try:
    result = chain.invoke({"question": "..."})
except OutputParserException as e:
    print(f"Failed to parse output: {e}")
    # Fallback to raw string
    result = str(e.llm_output)
```

---

## ChromaDB: Why We Use It

ChromaDB is chosen for this RAG system because:

1. **No server required**: Runs embedded in your Python process or as a file on disk
2. **Persistent storage**: Data survives process restarts
3. **Fast similarity search**: Uses HNSW algorithm for approximate nearest neighbor
4. **Rich filtering**: Filter by metadata alongside vector similarity
5. **LangChain integration**: First-class support via `langchain-chroma`
6. **Open source**: No API costs, full data privacy
7. **Scalable**: Handles millions of vectors on a single machine

Alternative vector databases (and when to use them):
- **Pinecone**: Managed service, great for production at scale
- **Weaviate**: Multi-modal, GraphQL queries
- **Qdrant**: Rust-based, excellent performance
- **FAISS**: Facebook's library, best for pure speed without persistence
