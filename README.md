# Multi-Agent AI Research System

## Overview
The **Multi-Agent AI Research System** is an automated, asynchronous pipeline that conducts deep research on any given topic using a multi-agent AI architecture. It leverages different specialized AI agents to search the web, read articles, write structured reports, and critically evaluate the findings. The project provides both a RESTful API and a newly integrated web-based frontend interface.

## Recent Additions & Updates
- **New Web Frontend (`Agent_frontend/`):** Replaced the legacy `frontend/index.html` with a comprehensive, multi-page web UI including `home.html`, `rag.html`, and `research_agent.html`.
- **FastAPI Integration:** Updated `main.py` to seamlessly serve the new `Agent_frontend` static files and route HTML pages dynamically alongside the `/api/research` endpoint.

## What is Included (Features)
This project is built around an asynchronous multi-agent pipeline with the following capabilities:
- **Search Agent:** Uses the Tavily search engine to find the most relevant and up-to-date information and URLs on a given topic.
- **Reader Agent:** Scrapes and extracts main text content from the URLs discovered by the Search Agent.
- **Writer Agent:** Synthesizes the gathered research into a highly structured report, including an Introduction, Key Findings, Conclusion, and Sources.
- **Critic / Checking Agent:** Critically evaluates the final research report, providing a score out of 10, strengths, and areas for improvement.
- **FastAPI Backend:** Exposes a `/api/research` endpoint to trigger the multi-agent pipeline.
- **Web Frontend:** A clean, multi-page web interface (HTML/CSS/JS) to interact with the research agents seamlessly.

## Technologies Used
The system is built using a modern, robust AI and web development stack:

### Backend & AI
- **Python:** Core programming language.
- **LangChain:** Framework for orchestrating the LLMs, tools, and agents.
- **Groq API (`ChatGroq`):** Fast inference engine used for the LLM (`openai/gpt-oss-120b`).
- **Tavily Search API:** Tool for providing real-time, accurate web search results to the agents.
- **BeautifulSoup4 & Requests:** Used for scraping and parsing HTML content from web pages.
- **FastAPI:** High-performance web framework for the backend API and static file serving.
- **Uvicorn:** ASGI server for running the FastAPI app.

### Frontend
- **HTML5, CSS3, JavaScript (Vanilla):** For the interactive web UI served directly by FastAPI.

## Project Structure
- `main.py` - FastAPI application, serves static frontend files and the `/api/research` endpoint.
- `pipeline.py` - The core asynchronous multi-agent pipeline execution logic (Search -> Read -> Write -> Check).
- `agents.py` - Defines the prompts, LLM integration, and LangChain agents (Search, Reader, Writer, Critic).
- `tools.py` - Custom LangChain tools (`web_search` and `get_url_content`).
- `Agent_frontend/` - Contains the newly added frontend HTML UI pages (`home.html`, `rag.html`, `research_agent.html`).
- `requirement.txt` - Python dependencies needed to run the project.

# Research Agent Structural Diagram

<p align="center">
  <img src="images\multi agent.drawio.png" alt="Dashboard" width="700"/>
</p>

# RAG System Structural Diagram (Conditional RAG)

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
    G{Grade Relevance & Hallucinations} ::: decisionStyle
    H[Rewrite Query] ::: processStyle
    I([Final Answer]) ::: outputStyle

    %% Connections
    A --> B
    B -- "Internal Knowledge" --> C
    B -- "External/Current Events" --> E
    C --> D
    D --> C
    C --> F
    E --> F
    F --> G
    
    G -- "Irrelevant or Hallucinated" --> H
    H --> C
    G -- "Relevant & Accurate" --> I
```

## How to Run

1. **Clone the repository.**
2. **Install dependencies:**
   ```bash
   pip install -r requirement.txt
   ```
3. **Set up environment variables:**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   GROQ_API_KEY=your_groq_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```
4. **Start the FastAPI server:**
   ```bash
   fastapi dev main.py
   ```