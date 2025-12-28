# AI Learning RAG Chain

A project for learning and experimenting with AI Agents and RAG (Retrieval-Augmented Generation) chains using LangChain, ChromaDB, and various language models.

## Description

This project demonstrates the implementation of RAG (Retrieval-Augmented Generation) chains and AI agents, enabling intelligent document retrieval and question-answering capabilities.

## Prerequisites

- Python 3.13 or higher
- `uv` package manager
- A `.env` file with your API keys and configuration

## Installation

1. **Create a virtual environment:**
   ```bash
   uv venv .venv
   ```

2. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   uv sync
   ```

4. **Run the specific file:**
   ```bash
   uv run python main.py
   ```
## Environment Setup

Create a `.env` file in the project root with your API keys and configuration. Example:

```env
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
# Add other required environment variables
```

## Usage

Run the project using one of the available scripts:

- `simple_rag.py` - Simple RAG implementation
- `agentic_rag.py` - Agentic RAG implementation
- `main.py` - Main entry point
- Jupyter notebooks for interactive exploration

## Technologies

- **LangChain** - Framework for building LLM applications
- **ChromaDB** - Vector database for embeddings
- **LangGraph** - For building stateful, multi-actor applications
- **Sentence Transformers** - For generating embeddings
- **Various LLM providers** - OpenAI, Groq, HuggingFace

## License

This is a learning project.
