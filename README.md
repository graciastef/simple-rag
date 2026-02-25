# LangGraph RAG

**Retrieval-Augmented Generation (RAG)** built with **LangGraph**.  
The workflow supports **tool-based retrieval**, **optional question rewriting**, and **document grading** before generating a concise final answer.

## What this does
- Uses LangGraph to route between:
  - **direct response** (if no retrieval needed)
  - **retrieve relevant chunks from a vector store → grade → (rewrite & retry OR generate answer)**
- Generates a short esponse using retrieved context
