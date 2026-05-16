"""
All LLM prompts used across the RAG workflow nodes.

Centralizing prompts here makes tuning easy — change one string
and the behavior updates everywhere it's used.

Prompts use Python .format() placeholders: {variable_name}
"""

# ── Query Analysis ────────────────────────────────────────────────────────────

QUERY_ANALYSIS_PROMPT = """You are an expert technical documentation assistant.

Analyze the user's question and perform two tasks:
1. Classify the question type
2. Rewrite it to maximize retrieval quality from a vector database

Question types:
- conceptual: explains what something is (e.g., "What is dependency injection?")
- troubleshooting: solves an error or problem (e.g., "Why does FastAPI return 422?")
- api_reference: asks about specific APIs or parameters (e.g., "What are the parameters for Response?")
- how_to: asks how to accomplish something (e.g., "How do I add middleware to FastAPI?")

Rewriting rules:
- Expand abbreviations and acronyms
- Add relevant technical context
- Use terminology that would appear in documentation
- Keep it concise but specific

User Question: {question}

Respond ONLY with valid JSON, no markdown, no explanation:
{{
    "question_type": "<one of: conceptual | troubleshooting | api_reference | how_to>",
    "optimized_query": "<rewritten query optimized for semantic vector search>",
    "reasoning": "<one sentence explaining your classification>"
}}"""


# ── Document Grader ───────────────────────────────────────────────────────────

DOCUMENT_GRADER_PROMPT = """You are a relevance grader for a technical documentation Q&A system.

Evaluate whether the document chunk below is relevant to answering the user's question.

User Question: {question}

Document Chunk:
\"\"\"
{document}
\"\"\"

Grading rules:
- Mark relevant=true if the chunk directly answers OR provides useful context for the question
- Mark relevant=true even for partial matches — partial relevance is still useful
- Mark relevant=false ONLY if the chunk is completely off-topic
- Be generous: when in doubt, mark as relevant

Respond ONLY with valid JSON, no markdown, no explanation:
{{
    "relevant": true or false,
    "confidence": <float between 0.0 and 1.0>,
    "reason": "<one short sentence explaining your decision>"
}}"""


# ── Answer Generation ─────────────────────────────────────────────────────────

GENERATION_PROMPT = """You are a helpful, precise technical documentation assistant.

Your task: Answer the user's question using ONLY the information in the provided context documents.

STRICT RULES:
1. Only use facts explicitly stated in the context — never add external knowledge
2. If the context is insufficient, say exactly: "Based on the available documentation, I cannot fully answer this question. The documentation covers: [brief summary of what IS in the context]."
3. Format your answer with clear markdown (headers, code blocks, bullet points)
4. End your answer with a "Sources" section listing the document names you used
5. Use code examples from the context when available

User Question: {question}

Context Documents:
{context}

---
Provide a comprehensive, well-structured answer based strictly on the context above:"""


# ── Query Rewriter ────────────────────────────────────────────────────────────

QUERY_REWRITER_PROMPT = """You are a search query optimization specialist.

A previous retrieval attempt failed to find relevant documents for this question.
Your task: rewrite the query using a completely different approach to find matching content.

Original Question: {question}
Previous Query Used: {previous_query}
Retry Attempt Number: {retry_count}

Rewriting strategies (apply based on retry number):
- Attempt 1: Use synonyms and alternative technical terminology
- Attempt 2: Break into simpler, more general concepts
- Attempt 3: Focus on the core technical term or technology name only

Respond ONLY with valid JSON, no markdown, no explanation:
{{
    "rewritten_query": "<new search query using different terminology>",
    "strategy_used": "<brief description of the rewriting strategy applied>"
}}"""


# ── Hallucination Checker ─────────────────────────────────────────────────────

HALLUCINATION_CHECKER_PROMPT = """You are a rigorous fact-checker for an AI Q&A system.

Your task: Verify that every factual claim in the generated answer is directly supported
by the provided context documents. This prevents the AI from "hallucinating" facts.

User Question: {question}

Generated Answer:
\"\"\"
{answer}
\"\"\"

Context Documents:
{context}

Verification rules:
- Check each factual claim: does it appear explicitly in the context?
- Opinions, summaries, and synthesis of context facts are acceptable
- "I don't know" type responses are automatically grounded
- A hallucination_score of 1.0 = fully supported, 0.0 = completely fabricated

Respond ONLY with valid JSON, no markdown, no explanation:
{{
    "is_grounded": true or false,
    "hallucination_score": <float 0.0 to 1.0>,
    "unsupported_claims": ["<claim 1 not in context>", "<claim 2>"],
    "reasoning": "<one sentence explaining your verdict>"
}}"""
