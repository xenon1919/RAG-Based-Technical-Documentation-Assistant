# LangGraph Technical Documentation

## Overview

LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain by allowing you to create graphs where:

- **Nodes** are Python functions that process the current state
- **Edges** define the flow between nodes (fixed or conditional)
- **State** is a shared TypedDict that flows through all nodes
- **Cycles** are supported — enabling retry loops and agent loops

LangGraph solves the main limitation of linear LangChain chains: **the inability to handle conditional logic, loops, and complex multi-step reasoning**.

---

## Installation

```bash
pip install langgraph
# or with uv:
uv add langgraph
```

---

## Core Concepts

### StateGraph

The central class in LangGraph. You define:
1. A **state schema** (TypedDict or dataclass)
2. **Nodes** (functions that read/write state)
3. **Edges** (how nodes connect)

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class MyState(TypedDict):
    message: str
    count: int

# Create the graph
workflow = StateGraph(MyState)
```

### Nodes

Nodes are Python functions that receive the current state and return a dict of updates:

```python
def my_node(state: MyState) -> dict:
    # Read from state
    message = state["message"]
    count = state["count"]
    
    # Process...
    new_count = count + 1
    
    # Return only the fields you want to update
    return {"count": new_count}

workflow.add_node("my_node", my_node)
```

Key rule: Nodes return a **dict of changes**, not the full state. LangGraph merges the returned dict into the existing state automatically.

### Edges

```python
# Fixed edge: always go from A to B
workflow.add_edge("node_a", "node_b")

# Entry point
workflow.add_edge(START, "first_node")

# Terminal edge
workflow.add_edge("last_node", END)
```

### Conditional Edges

```python
def decide_next(state: MyState) -> str:
    if state["count"] > 5:
        return "finish"
    else:
        return "continue"

workflow.add_conditional_edges(
    "decision_node",      # From this node...
    decide_next,          # ...call this function...
    {
        "finish": END,    # ...and go here if return is "finish"
        "continue": "next_node",  # ...or here if "continue"
    }
)
```

---

## Complete Example: Self-Corrective RAG

Here is a simplified version of the workflow in this project:

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class RAGState(TypedDict):
    question: str
    retrieved_docs: list
    relevant_docs: list
    answer: str
    retry_count: int

# ── Nodes ──────────────────────────────────────────────────────

def retrieve(state: RAGState) -> dict:
    """Search vector database for relevant chunks."""
    docs = vector_db.search(state["question"], k=5)
    return {"retrieved_docs": docs}

def grade(state: RAGState) -> dict:
    """Use LLM to filter relevant chunks."""
    relevant = [d for d in state["retrieved_docs"] if llm_grade(d)]
    return {"relevant_docs": relevant}

def generate(state: RAGState) -> dict:
    """Generate answer from relevant context."""
    answer = llm_generate(state["question"], state["relevant_docs"])
    return {"answer": answer}

def rewrite(state: RAGState) -> dict:
    """Rewrite query when no relevant docs found."""
    new_question = llm_rewrite(state["question"])
    return {"question": new_question, "retry_count": state["retry_count"] + 1}

# ── Router ─────────────────────────────────────────────────────

def route(state: RAGState) -> Literal["generate", "rewrite", "fallback"]:
    if state["relevant_docs"]:
        return "generate"
    elif state["retry_count"] < 3:
        return "rewrite"
    else:
        return "fallback"

# ── Build Graph ────────────────────────────────────────────────

workflow = StateGraph(RAGState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade)
workflow.add_node("generate", generate)
workflow.add_node("rewrite", rewrite)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade", route, {
    "generate": "generate",
    "rewrite": "rewrite",
    "fallback": END,
})
workflow.add_edge("rewrite", "retrieve")  # retry loop
workflow.add_edge("generate", END)

graph = workflow.compile()

# ── Run ────────────────────────────────────────────────────────

result = graph.invoke({
    "question": "How do I use FastAPI middleware?",
    "retrieved_docs": [],
    "relevant_docs": [],
    "answer": "",
    "retry_count": 0,
})
```

---

## State Management

### TypedDict State (Simple)

```python
from typing import TypedDict, Optional

class AppState(TypedDict):
    user_input: str
    processed: Optional[str]
    result: str
    error: Optional[str]
```

### Annotated State (Advanced: Reducers)

For list fields that should append instead of replace:

```python
from typing import Annotated, TypedDict
from operator import add

class MessagesState(TypedDict):
    # When two nodes both update "messages", they're merged (appended)
    # instead of one overwriting the other
    messages: Annotated[list, add]
    status: str
```

Without a reducer, the last node to write a field "wins". With `Annotated[list, add]`, all updates are combined.

---

## Checkpointing (Persistence)

LangGraph supports saving graph state between runs:

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

# In-memory checkpointer (lost on restart)
memory_checkpointer = MemorySaver()

# SQLite checkpointer (persists to disk)
sqlite_checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# Compile graph with checkpointer
graph = workflow.compile(checkpointer=sqlite_checkpointer)

# Invoke with a thread_id to resume previous conversation
config = {"configurable": {"thread_id": "user-session-123"}}
result = graph.invoke({"question": "What is FastAPI?"}, config=config)
```

With a checkpointer, the graph can be **paused and resumed** — enabling human-in-the-loop workflows.

---

## Human-in-the-Loop

Interrupt the graph to wait for human input:

```python
from langgraph.graph import StateGraph, END, START, interrupt

class ApprovalState(TypedDict):
    proposal: str
    approved: bool
    result: str

def generate_proposal(state: ApprovalState) -> dict:
    proposal = llm.generate_proposal()
    return {"proposal": proposal}

def human_approval(state: ApprovalState) -> dict:
    # This pauses execution until human provides input
    human_input = interrupt({
        "message": "Please approve or reject this proposal",
        "proposal": state["proposal"],
    })
    return {"approved": human_input.get("approved", False)}

def execute_if_approved(state: ApprovalState) -> dict:
    if state["approved"]:
        result = execute(state["proposal"])
        return {"result": result}
    return {"result": "Proposal rejected by human"}
```

---

## Multi-Agent Graphs

LangGraph supports graphs where nodes can themselves be LLM agents:

```python
from langgraph.graph import StateGraph, END, START

def researcher_agent(state):
    """Agent that searches for information."""
    # Uses tools like web_search, document_retrieval
    return {"research_findings": research_results}

def writer_agent(state):
    """Agent that drafts content based on research."""
    # Uses the research findings to write
    return {"draft": written_content}

def editor_agent(state):
    """Agent that reviews and improves the draft."""
    return {"final_content": edited_content}

workflow = StateGraph(MultiAgentState)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("writer", writer_agent)
workflow.add_node("editor", editor_agent)

workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "editor")
workflow.add_edge("editor", END)
```

---

## Streaming

Stream intermediate state updates to the client:

```python
# Stream node-by-node updates
for chunk in graph.stream(initial_state):
    node_name = list(chunk.keys())[0]
    node_output = chunk[node_name]
    print(f"Node '{node_name}' output: {node_output}")

# Stream token-by-token (if nodes use streaming LLMs)
async for chunk in graph.astream(initial_state):
    print(chunk)
```

---

## Debugging with Visualization

Visualize your graph structure:

```python
# Get a Mermaid diagram of the graph
print(graph.get_graph().draw_mermaid())

# Or draw to a PNG (requires graphviz)
graph.get_graph().draw_png("my_graph.png")
```

---

## Error Handling in Graphs

```python
def safe_node(state: MyState) -> dict:
    try:
        result = risky_operation(state["input"])
        return {"result": result, "error": None}
    except Exception as e:
        # Store error in state for the router to handle
        return {"result": None, "error": str(e)}

def route_after_safe_node(state: MyState) -> str:
    if state.get("error"):
        return "handle_error"
    return "continue"
```

---

## LangGraph vs LangChain Chains

| Feature | LangChain Chain | LangGraph |
|---------|----------------|-----------|
| Structure | Linear | Graph (DAG + cycles) |
| Loops | Not supported | Supported |
| Conditional routing | Limited | Full support |
| State management | Per-chain | Shared typed state |
| Human-in-the-loop | Limited | First-class support |
| Multi-agent | Complex | Built-in |
| Debugging | Callbacks | Graph visualization + checkpoints |

**When to use LangGraph**: Any workflow with retry logic, conditional routing, loops, or multiple specialized agents.

**When to use LangChain chains**: Simple linear pipelines where you don't need loops or complex routing.

---

## Best Practices

1. **Keep nodes focused**: Each node should do exactly one thing
2. **Return minimal state**: Only return the fields your node changed
3. **Fail safe**: Return error info in state rather than raising exceptions (allows routing to error handlers)
4. **Use type hints**: TypedDict state makes your graph self-documenting
5. **Log at node boundaries**: Log the node name and key state values at the start of each node
6. **Test nodes independently**: Each node is just a function — test it without the graph
7. **Use checkpointers in production**: Enables resume after failures
8. **Limit retry loops**: Always have a max retry count to prevent infinite loops

---

## LangGraph State Flow in the RAG System

This project's state object (`GraphState`) contains:

```
question          → Set by user, never modified
session_id        → Optional conversation ID
question_type     → Set by analyze_query node
optimized_query   → Set by analyze_query, updated by rewrite_query
retrieved_docs    → Set by retrieve_documents, cleared on retry
relevant_docs     → Set by grade_documents
generation        → Set by generate_answer or fallback
source_citations  → Set by generate_answer
is_grounded       → Set by check_hallucination
hallucination_score → Set by check_hallucination
retry_count       → Initialized to 0, incremented by rewrite_query
error_message     → Set on node failures
conversation_history → Maintained across turns
```

The state acts as a complete audit trail of the pipeline's execution.
