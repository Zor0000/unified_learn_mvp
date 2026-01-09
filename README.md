# Agentic RAG System with PostgreSQL + pgvector

An end-to-end **Retrieval-Augmented Generation (RAG)** system that uses **agentic chunking**, rich metadata, and intent-aware prompt routing to deliver accurate, explainable, and structured answers from large documents.

This project goes beyond naive RAG implementations by focusing on **semantic chunking quality**, **metadata-aware retrieval**, and **answer orchestration**.

---

## 🚀 Key Features

- **Agentic Chunking**

  - Uses an LLM to split documents into semantically meaningful chunks
  - One concept or instruction per chunk (instead of fixed token sizes)

- **Rich Metadata Storage**

  - Page number traceability
  - Chunk type (instruction, definition, explanation)
  - High-level intent tags (`how_to`, `concept`, etc.)
  - Difficulty level and keywords

- **PostgreSQL + pgvector**

  - Embeddings stored directly in Postgres
  - Logical isolation using collections
  - SQL-accessible metadata and vectors

- **Intent-Aware Prompt Routing**

  - Different prompt templates for:
    - HOW / procedural questions
    - WHAT / definition questions
    - WHY / explanatory questions
  - Produces structured, user-friendly answers

- **Ordered Procedural Responses**

  - Retrieved chunks are sorted by page number
  - Ensures step-by-step answers follow correct order

- **Safe & Grounded Answers**
  - Uses only retrieved context
  - Prevents hallucinations
  - Explicit “I don’t know” fallback

---

## 🧠 Architecture Overview

```text
PDF Documents
   ↓
Agentic Chunking (LLM)
   ↓
Chunks + Metadata
   ↓
Embeddings (OpenAI)
   ↓
PostgreSQL + pgvector
   ↓
Retriever (LangChain)
   ↓
Prompt Router (HOW / WHAT / WHY)
   ↓
Chat Interface
```
