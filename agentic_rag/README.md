# Conditional RAG (Agentic RAG)

This directory contains the implementation of a Conditional Retrieval-Augmented Generation (RAG) system using LangGraph and LangChain. 

## Overview
The Conditional RAG system doesn't just blindly retrieve and generate. Instead, it intelligently routes queries and evaluates the retrieved context to ensure the final answer is both relevant and accurate. If the local vector store (Internal Knowledge) lacks relevant information, the system dynamically falls back to a web search.

### RAG System Structural Diagram

```mermaid
graph TD
    %% Define styles
    classDef inputStyle fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px;
    classDef processStyle fill:#fff3e0,stroke:#ff9800,stroke-width:2px;
    classDef decisionStyle fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px;
    classDef dbStyle fill:#e8f5e9,stroke:#4caf50,stroke-width:2px;
    classDef outputStyle fill:#ffebee,stroke:#f44336,stroke-width:2px;

    %% Nodes
    A([User Query]) ::: inputStyle
    B{Query Router} ::: decisionStyle
    C[Retrieve from Vector Store] ::: processStyle
    D[(FAISS Vector Database)] ::: dbStyle
    E[Web Search] ::: processStyle
    F[Generate Response with LLM] ::: processStyle
    G{Grade Relevance} ::: decisionStyle
    H[Fallback to Web] ::: processStyle
    I([Final Answer]) ::: outputStyle

    %% Connections
    A --> B
    B -- "Internal Knowledge" --> C
    B -- "External/Current Events" --> E
    C --> D
    D --> C
    
    C --> G
    G -- "Relevant" --> F
    G -- "Irrelevant" --> H
    H --> E
    
    E --> F
    F --> I
```

## System Components (Nodes)

- **`retrieve`**: Uses FAISS and HuggingFace Embeddings (`all-MiniLM-L6-v2`) to pull the top `k` most relevant document chunks based on the user's query from the local PDF (`machine-learning.pdf`).
- **`grade_documents`**: An LLM-based grader that evaluates whether the retrieved documents actually answer the user's question. This filters out irrelevant context to prevent hallucinations.
- **`web_search`**: A fallback mechanism (using Tavily) that searches the open internet if the local vector store doesn't have the answer or if the documents are graded as irrelevant.
- **`generate`**: The final generation step where the LLM (`llama3-70b-8192` via Groq) synthesizes a comprehensive answer using the provided context (either from local retrieval or web search).

## State Management
The system uses a LangGraph `StateGraph` with a `TypedDict` (`State`) containing:
- `question`: The user's input query.
- `context`: The aggregated text from retrieved documents or web search results.
- `generation`: The final generated answer.
- `web_fallback`: A boolean flag indicating if web search should be triggered.

## How to Run
1. Ensure your `.env` file has the necessary API keys:
   ```env
   GROQ_API_KEY=your_groq_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```
2. Run the script directly:
   ```bash
   python agentic_rag/conditional_RAG.py
   ```
