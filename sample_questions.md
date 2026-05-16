# Sample Questions for Testing the RAG System

These questions are designed to test all aspects of the RAG pipeline.
After ingesting the docs in `./docs/`, use these with `POST /query` or the Streamlit UI.

---

## FastAPI Questions

### Conceptual
1. What is FastAPI and what makes it different from Flask or Django?
2. What is dependency injection in FastAPI and why is it useful?
3. What is the difference between sync and async endpoints in FastAPI?
4. What is a Pydantic model and how does it work with FastAPI?
5. What is CORS and how do I enable it in FastAPI?

### How-To
6. How do I create a FastAPI application with a POST endpoint?
7. How do I add CORS middleware to a FastAPI app?
8. How do I handle file uploads in FastAPI?
9. How do I add OAuth2 authentication to FastAPI?
10. How do I run code at startup in FastAPI?

### API Reference
11. What are the parameters for `add_middleware` in FastAPI?
12. What HTTP status codes are available in FastAPI's `status` module?
13. What is the difference between `File()` and `UploadFile` in FastAPI?
14. How does `response_model` work in FastAPI decorators?
15. What does `BackgroundTasks.add_task()` do?

### Troubleshooting
16. Why does FastAPI return a 422 error?
17. Why is my FastAPI endpoint returning an empty response?
18. How do I debug a slow FastAPI endpoint?
19. What causes a validation error in FastAPI request bodies?
20. How do I fix CORS errors in FastAPI?

---

## LangChain Questions

### Conceptual
21. What is LangChain and what problem does it solve?
22. What is RAG (Retrieval-Augmented Generation) and how does it work?
23. What is LCEL (LangChain Expression Language)?
24. What is the difference between a retriever and a vector store?
25. Why use ChromaDB over other vector databases?

### How-To
26. How do I create a RAG chain with LangChain?
27. How do I split documents using RecursiveCharacterTextSplitter?
28. How do I use HuggingFace embeddings in LangChain?
29. How do I store documents in ChromaDB using LangChain?
30. How do I implement conversation memory in LangChain?

### API Reference
31. What parameters does `RecursiveCharacterTextSplitter` accept?
32. What is the `similarity_search_with_score` method in ChromaDB?
33. What does `normalize_embeddings=True` do in HuggingFaceEmbeddings?
34. How do I use `PydanticOutputParser` in LangChain?
35. What is `StrOutputParser` and when should I use it?

---

## LangGraph Questions

### Conceptual
36. What is LangGraph and how is it different from LangChain?
37. What are nodes and edges in LangGraph?
38. What is the purpose of StateGraph?
39. Why does LangGraph support cycles when LangChain chains don't?
40. What is the difference between fixed edges and conditional edges?

### How-To
41. How do I create a StateGraph in LangGraph?
42. How do I add conditional routing to a LangGraph workflow?
43. How do I implement retry logic in LangGraph?
44. How do I add human-in-the-loop approval to a LangGraph?
45. How do I stream updates from a LangGraph execution?

### Architecture
46. What is the GraphState TypedDict and how should I design it?
47. What are reducers in LangGraph state management?
48. How does checkpointing work in LangGraph?
49. How do I visualize a LangGraph workflow?
50. What are best practices for error handling in LangGraph nodes?

---

## System Integration Questions (tests retrieval across all docs)

51. How does the RAG system in this project use LangGraph and LangChain together?
52. What is the role of the grading node in this RAG pipeline?
53. What happens when the grader finds no relevant documents?
54. How does the hallucination checker work?
55. What is the complete flow from user question to final answer?

---

## Edge Case Questions (tests fallback behavior)

These should trigger the retry + fallback logic:

56. What is the meaning of life according to the documentation?
57. How do I configure Kubernetes with FastAPI?
58. What is the best cryptocurrency to invest in?
59. How do I deploy to AWS Lambda?
60. What is the capital of France?

---

## API Test Commands (copy-paste ready)

```bash
# Health check
curl http://localhost:8000/health

# Ingest documents
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{}'

# List indexed documents
curl http://localhost:8000/documents

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I add CORS middleware to FastAPI?",
    "session_id": "test-session-001"
  }'

# Submit feedback
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I add CORS middleware to FastAPI?",
    "answer": "Use CORSMiddleware...",
    "feedback": "positive",
    "comment": "Very helpful!"
  }'
```
